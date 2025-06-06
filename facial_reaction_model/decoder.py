"""
多分支解码器 - 将潜在表示解码为四种面部特征
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
from .components import (
    PositionalEncoding, init_biased_mask, init_biased_mask2,
    MultiHeadCrossAttention, FeedForward
)


class FeatureBranchDecoder(nn.Module):
    """单一特征分支解码器"""
    
    def __init__(self, feature_dim: int, output_dim: int, branch_name: str):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.output_dim = output_dim
        self.branch_name = branch_name
        
        # Transformer解码器层
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=feature_dim,
            nhead=8,
            dim_feedforward=2 * feature_dim,
            batch_first=True
        )
        
        # 分支特定的解码器
        self.branch_decoder = nn.TransformerDecoder(decoder_layer, num_layers=1)
        
        # 输出映射层
        self.output_projection = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim // 2, output_dim)
        )
        
        # 特征特定的权重初始化
        self._initialize_weights()
    
    def _initialize_weights(self):
        """根据不同特征类型初始化权重"""
        for m in self.output_projection:
            if isinstance(m, nn.Linear):
                if self.branch_name == 'landmarks':
                    # Landmarks需要更精确的初始化
                    nn.init.xavier_normal_(m.weight, gain=0.5)
                elif self.branch_name == 'au':
                    # AU特征通常较小，使用较小的初始化
                    nn.init.xavier_normal_(m.weight, gain=0.3)
                else:
                    # Pose和Gaze使用标准初始化
                    nn.init.xavier_normal_(m.weight, gain=1.0)
                
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(
        self, 
        time_queries: torch.Tensor,
        shared_features: torch.Tensor,
        tgt_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            time_queries: 时间查询 (B, T, feature_dim)
            shared_features: 共享特征表示 (B, T, feature_dim)  
            tgt_mask: 目标掩码
        Returns:
            该分支的预测结果 (B, T, output_dim)
        """
        # Transformer解码
        decoded_features = self.branch_decoder(
            tgt=time_queries,
            memory=shared_features,
            tgt_mask=tgt_mask
        )
        
        # 输出投影
        output = self.output_projection(decoded_features)
        
        return output


class ListenerReactionDecoder(nn.Module):
    """Listener反应多分支解码器"""
    
    def __init__(
        self, 
        feature_dim: int = 256,
        period: int = 25,
        max_seq_len: int = 750,
        device: str = 'cpu',
        window_size: int = 16
    ):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.window_size = window_size
        self.device = device
        self.period = period
        
        # 位置编码
        self.pe = PositionalEncoding(feature_dim)
        
        # 主解码器（生成共享特征）
        main_decoder_layer = nn.TransformerDecoderLayer(
            d_model=feature_dim,
            nhead=8,
            dim_feedforward=2 * feature_dim,
            batch_first=True
        )
        
        self.main_decoder = nn.TransformerDecoder(main_decoder_layer, num_layers=2)
        self.audio_fusion_decoder = nn.TransformerDecoder(main_decoder_layer, num_layers=1)
        self.motion_fusion_decoder = nn.TransformerDecoder(main_decoder_layer, num_layers=1)
        
        # 四个分支解码器
        self.landmarks_decoder = FeatureBranchDecoder(feature_dim, 136, 'landmarks')  # 68*2
        self.au_decoder = FeatureBranchDecoder(feature_dim, 18, 'au')
        self.pose_decoder = FeatureBranchDecoder(feature_dim, 4, 'pose')
        self.gaze_decoder = FeatureBranchDecoder(feature_dim, 2, 'gaze')
        
        # 特征间协调层
        self.feature_coordination = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=8,
            batch_first=True
        )
        
        # 偏置掩码
        self.biased_mask = init_biased_mask(
            n_head=8, 
            max_seq_len=max_seq_len, 
            period=period
        )
    
    def forward(
        self,
        motion_sample: torch.Tensor,
        speaker_motion: torch.Tensor,
        speaker_audio: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            motion_sample: 运动样本 (B, 1, feature_dim) 或 (B, T, feature_dim)
            speaker_motion: Speaker运动特征 (B, T, feature_dim)
            speaker_audio: Speaker音频特征 (B, T_audio, feature_dim)
        Returns:
            包含四种面部特征预测的字典
        """
        batch_size, seq_len, _ = speaker_motion.shape
        
        # 调整为当前窗口大小 - 修复索引计算
        current_window_size = min(self.window_size, seq_len)
        speaker_motion_window = speaker_motion[:, -current_window_size:]
        
        # 确保音频有足够的长度
        audio_window_size = min(current_window_size * 2, speaker_audio.shape[1])
        speaker_audio_window = speaker_audio[:, -audio_window_size:]
        
        # 生成时间查询
        time_queries = torch.zeros(
            batch_size,
            current_window_size,
            self.feature_dim,
            device=speaker_motion.device
        )
        time_queries = self.pe(time_queries)
        
        # 准备掩码 - 确保形状正确
        tgt_mask = None  # 暂时禁用复杂的mask
        if hasattr(self, 'biased_mask') and current_window_size <= self.biased_mask.shape[1]:
            tgt_mask = self.biased_mask[:, :current_window_size, :current_window_size].clone().detach().to(
                device=self.device
            ).repeat(batch_size, 1, 1)
        
        # 主解码器：基于motion sample生成共享特征
        if motion_sample.dim() == 2:
            motion_sample = motion_sample.unsqueeze(1)  # (B, 1, feature_dim)
        
        shared_features = self.main_decoder(
            tgt=time_queries,
            memory=motion_sample
        )
        
        # 音频条件融合 - 简化mask处理
        if speaker_audio_window.shape[1] > 0:  # 确保音频窗口不为空
            shared_features = self.audio_fusion_decoder(
                tgt=shared_features,
                memory=speaker_audio_window,
                tgt_mask=tgt_mask,
                memory_mask=None  # 暂时不使用memory_mask避免形状问题
            )
        
        # 运动条件融合
        if speaker_motion_window.shape[1] > 0:  # 确保运动窗口不为空
            shared_features = self.motion_fusion_decoder(
                tgt=shared_features,
                memory=speaker_motion_window,
                tgt_mask=tgt_mask,
                memory_mask=None  # 暂时不使用memory_mask避免形状问题
            )
        
        # 特征间协调（可选）
        coordinated_features, _ = self.feature_coordination(
            shared_features, shared_features, shared_features
        )
        coordinated_features = shared_features + coordinated_features  # 残差连接
        
        # 多分支解码 - 使用当前窗口大小
        predictions = {}
        
        # 渐进解码：先解码主要结构，再解码依赖特征
        predictions['landmarks'] = self.landmarks_decoder(
            time_queries, coordinated_features, tgt_mask
        )
        
        # AU基于landmarks context
        landmarks_context = coordinated_features + self.landmarks_decoder.branch_decoder(
            time_queries, coordinated_features, tgt_mask
        )
        predictions['au'] = self.au_decoder(
            time_queries, landmarks_context, tgt_mask
        )
        
        # Pose基于整体face context
        face_context = coordinated_features + landmarks_context
        predictions['pose'] = self.pose_decoder(
            time_queries, face_context, tgt_mask
        )
        
        # Gaze基于所有context
        all_context = face_context
        predictions['gaze'] = self.gaze_decoder(
            time_queries, all_context, tgt_mask
        )
        
        return predictions
    
    def reset_window_size(self, window_size: int) -> None:
        """重置窗口大小"""
        self.window_size = window_size


class SpeakerReconstructionDecoder(nn.Module):
    """Speaker重建解码器（用于自监督学习）"""
    
    def __init__(self, feature_dim: int = 256):
        super().__init__()
        
        # Speaker自重建解码器
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=feature_dim,
            nhead=8,
            dim_feedforward=2 * feature_dim,
            batch_first=True
        )
        self.speaker_decoder = nn.TransformerDecoder(decoder_layer, num_layers=1)
        
        # 输出映射层
        self.speaker_output_map = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, 136 + 18 + 4 + 2)  # 重建所有speaker特征
        )
        
        # 初始化权重
        for m in self.speaker_output_map:
            if isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 0)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, speaker_vector: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            speaker_vector: Speaker特征向量 (B, T, feature_dim)
        Returns:
            重建的speaker特征字典
        """
        # 自解码
        reconstructed = self.speaker_decoder(speaker_vector, speaker_vector)
        
        # 输出映射
        output = self.speaker_output_map(reconstructed)
        
        # 分离不同特征
        landmarks_out = output[:, :, :136]
        au_out = output[:, :, 136:154]
        pose_out = output[:, :, 154:158]
        gaze_out = output[:, :, 158:160]
        
        return {
            'landmarks': landmarks_out,
            'au': au_out,
            'pose': pose_out,
            'gaze': gaze_out
        }