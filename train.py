"""
面部反应生成模型训练器
支持tensorboard监控、断点续训、多采样训练
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import numpy as np
import json
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
import json

from facial_reaction_model.model import FacialReactionModel
from facial_reaction_model.losses import (
    FacialVAELoss, FacialContrastiveLoss, FacialDiversityLoss,
    FacialSmoothLoss, FacialConsistencyLoss
)
from Data_Set import get_dataloader


class FacialReactionTrainer:
    """面部反应生成模型训练器"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
        print(f"🚀 使用设备: {self.device}")
        
        # 创建输出目录
        self.setup_directories()
        
        # 初始化模型
        self.model = self.create_model()
        
        # 初始化损失函数
        self.loss_functions = self.create_loss_functions()
        
        # 初始化优化器和调度器
        self.optimizer = self.create_optimizer()
        self.scheduler = self.create_scheduler()
        
        # 初始化数据加载器
        self.train_loader, self.val_loader = self.create_dataloaders()
        
        # 训练状态
        self.start_epoch = 0
        self.best_val_loss = float('inf')
        self.global_step = 0
        
        # Tensorboard
        self.writer = SummaryWriter(self.log_dir)
        
        # 加载检查点（如果存在）
        if config.get('resume_from_checkpoint'):
            self.load_checkpoint(config['resume_from_checkpoint'])
    
    def setup_directories(self):
        """创建必要的目录"""
        self.exp_name = self.config.get('exp_name', f"facial_reaction_{int(time.time())}")
        self.base_dir = os.path.join('experiments', self.exp_name)
        self.checkpoint_dir = os.path.join(self.base_dir, 'checkpoints')
        self.log_dir = os.path.join(self.base_dir, 'logs')
        
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # 保存运行配置为JSON格式（更易读）
        config_path = os.path.join(self.base_dir, 'run_config.json')
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)
        
        print(f"📁 实验目录: {self.base_dir}")
        print(f"💾 配置已保存: {config_path}")
    
    def create_model(self) -> FacialReactionModel:
        """创建模型"""
        model_config = self.config['model']
        model = FacialReactionModel(
            feature_dim=model_config['feature_dim'],
            audio_dim=model_config['audio_dim'],
            period=model_config['period'],
            max_seq_len=model_config['max_seq_len'],
            device=str(self.device),
            window_size=model_config['window_size'],
            momentum=model_config['momentum']
        )
        
        model = model.to(self.device)
        
        # 打印模型信息
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"🔧 模型参数: {total_params:,} (可训练: {trainable_params:,})")
        
        return model
    
    def create_loss_functions(self) -> Dict:
        """创建损失函数"""
        loss_config = self.config['loss']
        return {
            'vae_loss': FacialVAELoss(kl_weight=loss_config['kl_weight']),
            'contrastive_loss': FacialContrastiveLoss(
                temperature=loss_config['contrastive_temperature'],
                margin=loss_config['contrastive_margin']
            ),
            'diversity_loss': FacialDiversityLoss(),
            'smooth_loss': FacialSmoothLoss(
                motion_weight=loss_config['smooth_motion_weight'],
                expression_weight=loss_config['smooth_expression_weight']
            ),
            'consistency_loss': FacialConsistencyLoss()
        }
    
    def create_optimizer(self) -> optim.Optimizer:
        """创建优化器"""
        opt_config = self.config['optimizer']
        
        if opt_config['type'] == 'adam':
            return optim.Adam(
                self.model.parameters(),
                lr=opt_config['lr'],
                betas=opt_config.get('betas', (0.9, 0.999)),
                weight_decay=opt_config.get('weight_decay', 0.01)
            )
        elif opt_config['type'] == 'adamw':
            return optim.AdamW(
                self.model.parameters(),
                lr=opt_config['lr'],
                betas=opt_config.get('betas', (0.9, 0.999)),
                weight_decay=opt_config.get('weight_decay', 0.01)
            )
        else:
            raise ValueError(f"不支持的优化器类型: {opt_config['type']}")
    
    def create_scheduler(self):
        """创建学习率调度器"""
        sched_config = self.config.get('scheduler', {})
        
        if sched_config.get('type') == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['training']['num_epochs'],
                eta_min=sched_config.get('min_lr', 1e-6)
            )
        elif sched_config.get('type') == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=sched_config.get('step_size', 10),
                gamma=sched_config.get('gamma', 0.5)
            )
        else:
            return None
    
    def create_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        """创建数据加载器"""
        data_config = self.config['data']
        
        # 训练数据加载器
        train_loader = get_dataloader(
            mapping_csv=data_config['train_csv'],
            batch_size=data_config['batch_size'],
            num_workers=data_config['num_workers'],
            whisper_model_name=data_config['whisper_model']
        )
        
        # 验证数据加载器
        val_loader = None
        if data_config.get('val_csv'):
            val_loader = get_dataloader(
                mapping_csv=data_config['val_csv'],
                batch_size=data_config['batch_size'],
                num_workers=data_config['num_workers'],
                whisper_model_name=data_config['whisper_model']
            )
        
        print(f"📊 训练样本数: {len(train_loader.dataset)}")
        if val_loader:
            print(f"📊 验证样本数: {len(val_loader.dataset)}")
        
        return train_loader, val_loader
    
    def compute_losses(
        self,
        predictions_list: List[Dict[str, torch.Tensor]],
        distributions_list: List[List],
        gt_listener: Dict[str, torch.Tensor],
        speaker_reconstruction: Optional[Dict[str, torch.Tensor]] = None,
        speaker_data: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """计算所有损失"""
        
        loss_weights = self.config['loss']['weights']
        loss_dict = {}
        
        # 主要的VAE损失（使用第一次预测）
        vae_loss, rec_loss, kld_loss, feature_losses = self.loss_functions['vae_loss'](
            gt_listener, predictions_list[0], distributions_list[0]
        )
        loss_dict['vae_loss'] = vae_loss.item()
        loss_dict['reconstruction_loss'] = rec_loss.item()
        loss_dict['kld_loss'] = kld_loss.item()
        
        # 特征损失
        for feature_name, feature_loss in feature_losses.items():
            loss_dict[f'{feature_name}_loss'] = feature_loss.item()
        
        # 平滑损失
        smooth_loss = self.loss_functions['smooth_loss'](predictions_list[0])
        loss_dict['smooth_loss'] = smooth_loss.item()
        
        # 一致性损失
        consistency_loss = self.loss_functions['consistency_loss'](predictions_list[0])
        loss_dict['consistency_loss'] = consistency_loss.item()
        
        # 多样性损失（需要多次预测）
        diversity_loss = torch.tensor(0.0, device=self.device)
        if len(predictions_list) >= 2:
            diversity_loss = self.loss_functions['diversity_loss'](
                predictions_list[0], predictions_list[1]
            )
            loss_dict['diversity_loss'] = diversity_loss.item()
        
        # 对比损失（需要多次预测）
        contrastive_loss = torch.tensor(0.0, device=self.device)
        if len(predictions_list) >= 3:
            contrastive_loss = self.loss_functions['contrastive_loss'](
                predictions_list[0], predictions_list[1], gt_listener
            )
            loss_dict['contrastive_loss'] = contrastive_loss.item()
        
        # Speaker重建损失
        speaker_rec_loss = torch.tensor(0.0, device=self.device)
        if speaker_reconstruction and speaker_data:
            speaker_rec_loss, _, _, _ = self.loss_functions['vae_loss'](
                speaker_data, speaker_reconstruction, []
            )
            loss_dict['speaker_rec_loss'] = speaker_rec_loss.item()
        
        # 总损失
        total_loss = (
            loss_weights['vae'] * vae_loss +
            loss_weights['smooth'] * smooth_loss +
            loss_weights['diversity'] * diversity_loss +
            loss_weights['contrastive'] * contrastive_loss +
            loss_weights['consistency'] * consistency_loss +
            loss_weights['speaker_rec'] * speaker_rec_loss
        )
        
        loss_dict['total_loss'] = total_loss.item()
        
        return total_loss, loss_dict
    
    def train_step(self, batch_data: Dict) -> Dict[str, float]:
        """单步训练"""
        self.model.train()
        
        # 数据移动到设备
        speaker_data = {k: v.to(self.device) for k, v in batch_data['speaker'].items()}
        listener_data = {k: v.to(self.device) for k, v in batch_data['listener'].items()}
        
        # 多次前向传播用于多样性损失
        predictions_list = []
        distributions_list = []
        speaker_reconstruction = None
        
        num_samples = self.config['training'].get('num_samples', 3)
        
        for i in range(num_samples):
            # 前向传播
            predictions, distributions, speaker_rec = self.model(
                speaker_data,
                speaker_out=(i == 0)  # 只在第一次计算speaker重建
            )
            
            predictions_list.append(predictions)
            distributions_list.append(distributions)
            
            if i == 0:
                speaker_reconstruction = speaker_rec
        
        # 计算损失
        total_loss, loss_dict = self.compute_losses(
            predictions_list, distributions_list, listener_data,
            speaker_reconstruction, speaker_data
        )
        
        # 反向传播
        total_loss.backward()
        
        # 梯度裁剪
        max_grad_norm = self.config['training'].get('max_grad_norm', 1.0)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
        
        return loss_dict
    
    def validate_step(self, batch_data: Dict) -> Dict[str, float]:
        """单步验证"""
        self.model.eval()
        
        with torch.no_grad():
            # 数据移动到设备
            speaker_data = {k: v.to(self.device) for k, v in batch_data['speaker'].items()}
            listener_data = {k: v.to(self.device) for k, v in batch_data['listener'].items()}
            
            # 前向传播
            predictions, distributions, speaker_reconstruction = self.model(
                speaker_data, speaker_out=True
            )
            
            # 计算损失
            _, loss_dict = self.compute_losses(
                [predictions], [distributions], listener_data,
                speaker_reconstruction, speaker_data
            )
        
        return loss_dict
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        
        epoch_losses = {}
        num_batches = 0
        accumulate_steps = self.config['training'].get('accumulate_grad_batches', 1)
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch_data in enumerate(progress_bar):
            if batch_data is None:
                continue
            
            # 训练步骤
            loss_dict = self.train_step(batch_data)
            
            # 累积损失
            for key, value in loss_dict.items():
                if key not in epoch_losses:
                    epoch_losses[key] = 0.0
                epoch_losses[key] += value
            
            num_batches += 1
            
            # 梯度累积
            if (batch_idx + 1) % accumulate_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.global_step += 1
                
                # 记录到tensorboard
                if self.global_step % self.config['training'].get('log_interval', 10) == 0:
                    for key, value in loss_dict.items():
                        self.writer.add_scalar(f'train/{key}', value, self.global_step)
                    
                    self.writer.add_scalar('train/lr', self.optimizer.param_groups[0]['lr'], self.global_step)
            
            # 更新进度条
            progress_bar.set_postfix({
                'loss': f"{loss_dict['total_loss']:.4f}",
                'rec': f"{loss_dict['reconstruction_loss']:.4f}",
                'kld': f"{loss_dict['kld_loss']:.4f}"
            })
        
        # 平均损失
        for key in epoch_losses:
            epoch_losses[key] /= max(num_batches, 1)
        
        return epoch_losses
    
    def validate_epoch(self, epoch: int) -> Dict[str, float]:
        """验证一个epoch"""
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        
        epoch_losses = {}
        num_batches = 0
        
        progress_bar = tqdm(self.val_loader, desc=f"Validation {epoch}")
        
        for batch_data in progress_bar:
            if batch_data is None:
                continue
            
            # 验证步骤
            loss_dict = self.validate_step(batch_data)
            
            # 累积损失
            for key, value in loss_dict.items():
                if key not in epoch_losses:
                    epoch_losses[key] = 0.0
                epoch_losses[key] += value
            
            num_batches += 1
            
            progress_bar.set_postfix({
                'val_loss': f"{loss_dict['total_loss']:.4f}"
            })
        
        # 平均损失
        for key in epoch_losses:
            epoch_losses[key] /= max(num_batches, 1)
        
        return epoch_losses
    
    def save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'val_loss': val_loss,
            'config': self.config,
            'global_step': self.global_step
        }
        
        # 保存最新检查点
        latest_path = os.path.join(self.checkpoint_dir, 'latest.pth')
        torch.save(checkpoint, latest_path)
        
        # 保存最佳检查点
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best.pth')
            torch.save(checkpoint, best_path)
            print(f"💾 保存最佳模型: val_loss={val_loss:.4f}")
        
        # 定期保存
        if epoch % self.config['training'].get('save_interval', 10) == 0:
            epoch_path = os.path.join(self.checkpoint_dir, f'epoch_{epoch}.pth')
            torch.save(checkpoint, epoch_path)
    
    def load_checkpoint(self, checkpoint_path: str):
        """加载检查点"""
        if not os.path.exists(checkpoint_path):
            print(f"⚠️ 检查点不存在: {checkpoint_path}")
            return
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # 加载模型状态
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_val_loss = checkpoint.get('val_loss', float('inf'))
        self.global_step = checkpoint.get('global_step', 0)
        
        print(f"✅ 从检查点恢复: epoch={self.start_epoch}, val_loss={self.best_val_loss:.4f}")
    
    def train(self):
        """主训练循环"""
        print(f"🎯 开始训练: {self.config['training']['num_epochs']} epochs")
        
        start_time = time.time()
        
        for epoch in range(self.start_epoch, self.config['training']['num_epochs']):
            # 训练
            train_losses = self.train_epoch(epoch)
            
            # 验证
            val_losses = self.validate_epoch(epoch)
            
            # 学习率调度
            if self.scheduler:
                self.scheduler.step()
            
            # 记录到tensorboard
            for key, value in train_losses.items():
                self.writer.add_scalar(f'epoch/train_{key}', value, epoch)
            
            for key, value in val_losses.items():
                self.writer.add_scalar(f'epoch/val_{key}', value, epoch)
            
            # 打印结果
            train_loss = train_losses.get('total_loss', 0.0)
            val_loss = val_losses.get('total_loss', 0.0)
            
            print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            
            # 保存检查点
            is_best = val_loss < self.best_val_loss if val_loss > 0 else False
            if is_best:
                self.best_val_loss = val_loss
            
            self.save_checkpoint(epoch, val_loss, is_best)
            
            # 早停检查
            if self.config['training'].get('early_stopping'):
                patience = self.config['training'].get('patience', 20)
                if epoch - self.start_epoch > patience and val_loss > self.best_val_loss:
                    print(f"🛑 早停: {patience} epochs 无改善")
                    break
        
        # 训练完成
        total_time = time.time() - start_time
        print(f"🎉 训练完成! 总时间: {total_time/3600:.2f} 小时")
        print(f"📊 最佳验证损失: {self.best_val_loss:.4f}")
        
        self.writer.close()