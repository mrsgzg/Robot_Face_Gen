"""
面部反应VAE模块 - 潜在空间采样和生成多样性
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
from .components import lengths_to_mask, PositionalEncoding


class FacialVAE(nn.Module):
    """面部反应变分自编码器"""
    
    def __init__(self, feature_dim: int = 256, latent_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.feature_dim = feature_dim
        
        # 特征投影
        self.input_projection = nn.Linear(feature_dim, latent_dim)
        
        # Transformer编码器用于序列建模
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=8,
            dim_feedforward=2 * latent_dim,
            dropout=dropout,
            batch_first=True  # 设置为True以保持一致性
        )
        self.sequence_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # 分布参数token（可学习的特殊token）
        self.mu_token = nn.Parameter(torch.randn(latent_dim))
        self.logvar_token = nn.Parameter(torch.randn(latent_dim))
        
        # 位置编码 (不使用batch_first参数)
        self.pe = PositionalEncoding(latent_dim, dropout=dropout)
        
    def forward(self, fused_features: torch.Tensor) -> Tuple[torch.Tensor, torch.distributions.Normal]:
        """
        Args:
            fused_features: 融合的面部特征 (B, T, feature_dim)
        Returns:
            tuple: (motion_sample, distribution)
        """
        batch_size, seq_len, _ = fused_features.shape
        
        # 投影到潜在维度
        x = self.input_projection(fused_features)  # (B, T, latent_dim)
        
        # 准备分布token
        mu_token = self.mu_token.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1)  # (B, 1, latent_dim)
        logvar_token = self.logvar_token.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1)  # (B, 1, latent_dim)
        
        # 拼接特殊token和序列特征
        x_with_tokens = torch.cat([mu_token, logvar_token, x], dim=1)  # (B, T+2, latent_dim)
        
        # 位置编码
        x_with_tokens = self.pe(x_with_tokens)  # (B, T+2, latent_dim)
        
        # 生成掩码 - 确保正确的掩码处理
        batch_size, total_len, _ = x_with_tokens.shape
        # 创建简单的掩码：前两个token(mu, logvar)不掩码，后面的根据实际序列长度
        mask = torch.ones(batch_size, total_len, dtype=torch.bool, device=fused_features.device)
        # 这里我们假设所有序列都是满长度的，如果需要可以加入真实长度处理
        
        # Transformer编码
        encoded = self.sequence_encoder(
            x_with_tokens,  # (B, T+2, latent_dim)
            src_key_padding_mask=None  # 暂时不使用mask，避免复杂性
        )  # (B, T+2, latent_dim)
        
        # 提取分布参数
        mu = encoded[:, 0, :]  # (B, latent_dim) - 第一个token
        logvar = encoded[:, 1, :]  # (B, latent_dim) - 第二个token
        
        # 构建分布
        std = torch.exp(0.5 * logvar)
        distribution = torch.distributions.Normal(mu, std)
        
        # 重参数化采样
        motion_sample = self.reparameterize(distribution)
        
        return motion_sample, distribution
    
    def reparameterize(self, distribution: torch.distributions.Normal) -> torch.Tensor:
        """重参数化采样"""
        return distribution.rsample()
    
    def sample_prior(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """从先验分布采样"""
        mu_prior = torch.zeros(batch_size, self.latent_dim, device=device)
        std_prior = torch.ones(batch_size, self.latent_dim, device=device)
        prior_dist = torch.distributions.Normal(mu_prior, std_prior)
        return prior_dist.rsample()


class MotionSampleGenerator(nn.Module):
    """运动样本生成器 - 结合历史状态和当前输入生成运动样本"""
    
    def __init__(self, feature_dim: int = 256, period: int = 25, device: str = 'cpu'):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.period = period
        self.device = device
        
        # 历史状态处理
        self.past_motion_linear = nn.Linear(feature_dim, feature_dim)
        
        # 融合transformer
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=feature_dim,
            nhead=8,
            dim_feedforward=2 * feature_dim,
            batch_first=True
        )
        self.audio_fusion = nn.TransformerDecoder(decoder_layer, num_layers=1)
        self.motion_fusion = nn.TransformerDecoder(decoder_layer, num_layers=1)
        
        # VAE模块
        self.vae = FacialVAE(feature_dim, feature_dim)
        
        # 位置编码
        from .components import PeriodicPositionalEncoding
        self.ppe = PeriodicPositionalEncoding(feature_dim, period=period, max_seq_len=750)
        
    def forward(
        self, 
        speaker_motion: torch.Tensor,
        speaker_audio: torch.Tensor, 
        past_listener_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.distributions.Normal]:
        """
        Args:
            speaker_motion: Speaker运动特征 (B, T, feature_dim)
            speaker_audio: Speaker音频特征 (B, T_audio, feature_dim)
            past_listener_features: 历史listener特征 (B, T, feature_dim)
        Returns:
            tuple: (motion_sample, distribution)
        """
        batch_size, frame_num, _ = past_listener_features.shape
        
        # 确保输入尺寸匹配
        speaker_audio = speaker_audio[:, :2 * frame_num]  # 音频是2倍长度
        speaker_motion = speaker_motion[:, :frame_num]
        
        # 处理历史状态
        processed_past = self.past_motion_linear(past_listener_features)
        processed_past = processed_past + self.ppe(processed_past)
        
        # 生成掩码
        from .components import init_biased_mask, init_biased_mask2
        
        tgt_mask = init_biased_mask(
            n_head=8, max_seq_len=750, period=self.period
        )[:, :frame_num, :frame_num].clone().detach().to(
            device=self.device
        ).repeat(batch_size, 1, 1)
        
        # 音频融合
        audio_memory_mask = init_biased_mask2(
            n_head=8,
            window_size=frame_num,
            max_seq_len=speaker_audio.shape[1],
            period=self.period
        ).clone().detach().to(device=self.device).repeat(batch_size, 1, 1)
        
        fused_with_audio = self.audio_fusion(
            tgt=processed_past,
            memory=speaker_audio,
            tgt_mask=tgt_mask,
            memory_mask=audio_memory_mask
        )
        
        # 运动融合
        motion_memory_mask = init_biased_mask2(
            n_head=8,
            window_size=frame_num, 
            max_seq_len=speaker_motion.shape[1],
            period=self.period
        ).clone().detach().to(device=self.device).repeat(batch_size, 1, 1)
        
        fused_with_motion = self.motion_fusion(
            tgt=fused_with_audio,
            memory=speaker_motion,
            tgt_mask=tgt_mask,
            memory_mask=motion_memory_mask
        )
        
        # VAE采样
        motion_sample, distribution = self.vae(fused_with_motion)
        
        return motion_sample, distribution