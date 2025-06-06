"""
面部反应生成损失函数 - 适配多输出面部特征
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple


class KLLoss(nn.Module):
    """KL散度损失"""
    def __init__(self):
        super(KLLoss, self).__init__()

    def forward(self, q, p):
        div = torch.distributions.kl_divergence(q, p)
        return div.mean()

    def __repr__(self):
        return "KLLoss()"


class MultiFacialReconstructionLoss(nn.Module):
    """多特征面部重建损失"""
    
    def __init__(self, feature_weights: Dict[str, float] = None):
        super().__init__()
        
        # 默认特征权重
        self.feature_weights = feature_weights or {
            'landmarks': 1.0,  # 结构最重要
            'au': 0.8,         # 表情很重要
            'pose': 0.6,       # 姿态中等重要
            'gaze': 0.4        # 视线相对不太重要
        }
        
        # 不同特征使用不同的损失函数
        self.landmarks_loss = nn.SmoothL1Loss(reduction='mean')
        self.au_loss = nn.MSELoss(reduction='mean') 
        self.pose_loss = nn.SmoothL1Loss(reduction='mean')
        self.gaze_loss = nn.MSELoss(reduction='mean')
    
    def forward(
        self, 
        gt_features: Dict[str, torch.Tensor], 
        pred_features: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            gt_features: 真实特征字典
            pred_features: 预测特征字典
        Returns:
            tuple: (总损失, 各特征损失字典)
        """
        losses = {}
        
        # Landmarks损失（结构损失）
        if 'landmarks' in gt_features and 'landmarks' in pred_features:
            landmarks_loss = self.landmarks_loss(
                pred_features['landmarks'], 
                gt_features['landmarks']
            )
            losses['landmarks'] = landmarks_loss
        
        # AU损失（表情损失）
        if 'au' in gt_features and 'au' in pred_features:
            au_loss = self.au_loss(
                pred_features['au'], 
                gt_features['au']
            )
            losses['au'] = au_loss
        
        # Pose损失（姿态损失）
        if 'pose' in gt_features and 'pose' in pred_features:
            pose_loss = self.pose_loss(
                pred_features['pose'], 
                gt_features['pose']
            )
            losses['pose'] = pose_loss
        
        # Gaze损失（视线损失）
        if 'gaze' in gt_features and 'gaze' in pred_features:
            gaze_loss = self.gaze_loss(
                pred_features['gaze'], 
                gt_features['gaze']
            )
            losses['gaze'] = gaze_loss
        
        # 加权总损失
        total_loss = sum(
            self.feature_weights.get(name, 1.0) * loss 
            for name, loss in losses.items()
        )
        
        return total_loss, losses


class FacialVAELoss(nn.Module):
    """面部VAE损失 - 重建损失 + KL散度"""
    
    def __init__(self, kl_weight: float = 0.0002):
        super().__init__()
        self.reconstruction_loss = MultiFacialReconstructionLoss()
        self.kl_loss = KLLoss()
        self.kl_weight = kl_weight
    
    def forward(
        self, 
        gt_features: Dict[str, torch.Tensor],
        pred_features: Dict[str, torch.Tensor], 
        distributions: List[torch.distributions.Normal]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            gt_features: 真实特征
            pred_features: 预测特征
            distributions: VAE分布列表
        Returns:
            tuple: (总损失, 重建损失, KL损失, 各特征损失)
        """
        # 重建损失
        rec_loss, feature_losses = self.reconstruction_loss(gt_features, pred_features)
        
        # KL散度损失
        if distributions:
            # 参考分布（标准正态分布）
            device = next(iter(pred_features.values())).device
            mu_ref = torch.zeros_like(distributions[0].loc).to(device)
            scale_ref = torch.ones_like(distributions[0].scale).to(device)
            distribution_ref = torch.distributions.Normal(mu_ref, scale_ref)
            
            # 计算平均KL散度
            kld_loss = torch.stack([
                self.kl_loss(dist, distribution_ref) 
                for dist in distributions
            ]).mean()
        else:
            kld_loss = torch.tensor(0.0, device=rec_loss.device)
        
        # 总损失
        total_loss = rec_loss + self.kl_weight * kld_loss
        
        return total_loss, rec_loss, kld_loss, feature_losses


class FacialContrastiveLoss(nn.Module):
    """面部对比损失 - 替代邻居损失的数据增强策略"""
    
    def __init__(self, temperature: float = 0.1, margin: float = 1.0):
        super().__init__()
        self.temperature = temperature
        self.margin = margin
        self.cosine_sim = nn.CosineSimilarity(dim=-1)
    
    def forward(
        self,
        pred_features_1: Dict[str, torch.Tensor],  # 第一次预测
        pred_features_2: Dict[str, torch.Tensor],  # 第二次预测  
        gt_features: Dict[str, torch.Tensor]       # 真实特征
    ) -> torch.Tensor:
        """
        对比损失：鼓励多次预测都接近真实值，但彼此有一定差异
        """
        total_loss = 0.0
        
        for feature_name in pred_features_1:
            if feature_name in pred_features_2 and feature_name in gt_features:
                # 展平特征
                pred1_flat = pred_features_1[feature_name].view(pred_features_1[feature_name].size(0), -1)
                pred2_flat = pred_features_2[feature_name].view(pred_features_2[feature_name].size(0), -1)
                gt_flat = gt_features[feature_name].view(gt_features[feature_name].size(0), -1)
                
                # 预测与真实值的相似度（应该高）
                sim_1_gt = self.cosine_sim(pred1_flat, gt_flat)
                sim_2_gt = self.cosine_sim(pred2_flat, gt_flat)
                
                # 两次预测间的相似度（应该适中，不要完全相同）
                sim_1_2 = self.cosine_sim(pred1_flat, pred2_flat)
                
                # 对比损失：拉近预测与真实值，适度分离两次预测
                contrastive_loss = (
                    -torch.log(torch.sigmoid(sim_1_gt / self.temperature)).mean() +
                    -torch.log(torch.sigmoid(sim_2_gt / self.temperature)).mean() +
                    F.relu(sim_1_2 - self.margin).mean()  # 如果两次预测太相似则惩罚
                )
                
                total_loss += contrastive_loss
        
        return total_loss / len(pred_features_1)


class FacialDiversityLoss(nn.Module):
    """面部多样性损失 - 鼓励生成多样性"""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, pred1: Dict[str, torch.Tensor], pred2: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            pred1, pred2: 两次不同的预测结果
        Returns:
            多样性损失
        """
        total_div_loss = 0.0
        feature_count = 0
        
        for feature_name in pred1:
            if feature_name in pred2:
                # 展平特征
                y1_flat = pred1[feature_name].view(pred1[feature_name].size(0), -1)  # (B, T*feature_dim)
                y2_flat = pred2[feature_name].view(pred2[feature_name].size(0), -1)  # (B, T*feature_dim)
                
                # 计算特征间距离
                batch_size = y1_flat.size(0)
                div_loss = 0.0
                
                for b in range(batch_size):
                    # 计算两个预测的欧式距离
                    dist = torch.norm(y1_flat[b] - y2_flat[b], p=2)
                    # 使用负指数鼓励多样性
                    div_loss += torch.exp(-dist / 100.0)
                
                total_div_loss += div_loss / batch_size
                feature_count += 1
        
        return total_div_loss / max(feature_count, 1)


class FacialSmoothLoss(nn.Module):
    """面部平滑损失 - 确保时序连续性"""
    
    def __init__(self, motion_weight: float = 1.0, expression_weight: float = 0.1):
        super().__init__()
        self.motion_weight = motion_weight      # 运动特征权重（pose, gaze）
        self.expression_weight = expression_weight  # 表情特征权重（landmarks, au）
        self.smooth_loss = nn.SmoothL1Loss(reduction='mean')
    
    def forward(self, predictions: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            predictions: 预测特征字典，每个特征 (B, T, feature_dim)
        Returns:
            平滑损失
        """
        total_smooth_loss = 0.0
        
        for feature_name, pred in predictions.items():
            if pred.size(1) < 3:  # 至少需要3帧计算二阶差分
                continue
                
            # 计算二阶时间差分
            first_diff = pred[:, 1:, :] - pred[:, :-1, :]  # (B, T-1, feature_dim)
            second_diff = first_diff[:, 1:, :] - first_diff[:, :-1, :]  # (B, T-2, feature_dim)
            
            # 平滑损失：鼓励二阶差分接近零
            smooth_loss = torch.mean(second_diff ** 2)
            
            # 根据特征类型加权
            if feature_name in ['pose', 'gaze']:
                weight = self.motion_weight
            else:  # landmarks, au
                weight = self.expression_weight
                
            total_smooth_loss += weight * smooth_loss
        
        return total_smooth_loss


class FacialConsistencyLoss(nn.Module):
    """面部一致性损失 - 确保不同特征间的协调性"""
    
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss(reduction='mean')
    
    def forward(self, predictions: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        计算面部特征间的一致性损失
        """
        consistency_loss = 0.0
        
        # 这里可以定义特定的一致性约束
        # 例如：极端pose时，landmarks的变化应该合理
        if 'pose' in predictions and 'landmarks' in predictions:
            # 简单的一致性检查：pose变化时landmarks也应该相应变化
            pose_var = torch.var(predictions['pose'], dim=1)  # (B, pose_dim)
            landmarks_var = torch.var(predictions['landmarks'], dim=1)  # (B, landmarks_dim)
            
            # 鼓励pose和landmarks变化的相关性
            pose_norm = torch.norm(pose_var, dim=1)  # (B,)
            landmarks_norm = torch.norm(landmarks_var, dim=1)  # (B,)
            
            # 如果pose变化大，landmarks也应该有相应变化
            consistency_loss += F.mse_loss(
                pose_norm / (torch.norm(pose_norm) + 1e-8),
                landmarks_norm / (torch.norm(landmarks_norm) + 1e-8)
            )
        
        return consistency_loss