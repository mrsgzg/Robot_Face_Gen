"""
多模态编码器 - 处理面部特征和音频特征的融合
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
from .components import (
    FeatureProjection, PeriodicPositionalEncoding, PositionalEncoding,
    MultiHeadCrossAttention, FeedForward, init_biased_mask, init_biased_mask2
)


class FacialFeatureEncoder(nn.Module):
    """面部特征编码器"""
    def __init__(self, feature_dim=256, dropout=0.1):
        super().__init__()
        self.feature_dim = feature_dim
        
        # 各种面部特征的投影层
        self.landmarks_proj = FeatureProjection(136, feature_dim, dropout)  # 68*2 landmarks
        self.au_proj = FeatureProjection(18, feature_dim, dropout)          # AU features
        self.pose_proj = FeatureProjection(4, feature_dim, dropout)         # Head pose
        self.gaze_proj = FeatureProjection(2, feature_dim, dropout)         # Gaze direction
        
        # 面部特征内部融合
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=8,
            dim_feedforward=2 * feature_dim,
            dropout=dropout,
            batch_first=True
        )
        self.face_fusion_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # 特征权重学习
        self.feature_weights = nn.Parameter(torch.ones(4))  # landmarks, au, pose, gaze
        
    def forward(self, facial_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            facial_features: 包含 'landmarks', 'au', 'pose', 'gaze' 的字典
        Returns:
            融合后的面部特征 (B, T, feature_dim)
        """
        batch_size, seq_len, _ = facial_features['landmarks'].shape
        
        # 投影到统一维度
        landmarks_feat = self.landmarks_proj(facial_features['landmarks'])  # (B, T, feature_dim)
        au_feat = self.au_proj(facial_features['au'])
        pose_feat = self.pose_proj(facial_features['pose'])
        gaze_feat = self.gaze_proj(facial_features['gaze'])
        
        # 加权融合
        weights = F.softmax(self.feature_weights, dim=0)
        fused_features = (
            weights[0] * landmarks_feat +
            weights[1] * au_feat +
            weights[2] * pose_feat +
            weights[3] * gaze_feat
        )
        
        # 堆叠所有特征用于自注意力
        stacked_features = torch.stack([
            landmarks_feat, au_feat, pose_feat, gaze_feat
        ], dim=2)  # (B, T, 4, feature_dim)
        
        # 重塑为序列形式进行self-attention
        stacked_features = stacked_features.view(batch_size, seq_len * 4, self.feature_dim)
        
        # 面部特征内部自注意力
        encoded_features = self.face_fusion_encoder(stacked_features)
        
        # 重塑回原始形状并平均
        encoded_features = encoded_features.view(batch_size, seq_len, 4, self.feature_dim)
        encoded_features = encoded_features.mean(dim=2)  # 平均四种特征
        
        # 残差连接
        output = encoded_features + fused_features
        
        return output


class AudioFeatureEncoder(nn.Module):
    """音频特征编码器"""
    def __init__(self, audio_dim=384, feature_dim=256, dropout=0.1):
        super().__init__()
        
        # Whisper特征投影
        self.audio_proj = FeatureProjection(audio_dim, feature_dim, dropout)
        
        # 音频时序建模
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=8,
            dim_feedforward=2 * feature_dim,
            dropout=dropout,
            batch_first=True
        )
        self.audio_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        
    def forward(self, audio_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            audio_features: Whisper音频特征 (B, T, 384)
        Returns:
            编码后的音频特征 (B, T, feature_dim)
        """
        # 投影到目标维度
        audio_feat = self.audio_proj(audio_features)
        
        # 时序编码
        encoded_audio = self.audio_encoder(audio_feat)
        
        return encoded_audio


class MultiModalFusionEncoder(nn.Module):
    """多模态融合编码器"""
    def __init__(self, feature_dim=256, period=25, max_seq_len=750, device='cpu'):
        super().__init__()
        self.feature_dim = feature_dim
        self.device = device
        self.period = period
        
        # 位置编码
        self.PPE = PeriodicPositionalEncoding(feature_dim, period=period, max_seq_len=max_seq_len)
        self.PE = PositionalEncoding(feature_dim)
        
        # 交叉注意力层
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=feature_dim,
            nhead=8,
            dim_feedforward=2 * feature_dim,
            batch_first=True
        )
        
        # 多层融合
        self.face_self_attention = nn.TransformerDecoder(decoder_layer, num_layers=1)
        self.audio_face_fusion = nn.TransformerDecoder(decoder_layer, num_layers=1)
        self.temporal_modeling = nn.TransformerDecoder(decoder_layer, num_layers=2)
        
        # 偏置掩码
        self.biased_mask = init_biased_mask(n_head=8, max_seq_len=max_seq_len, period=period)
        
    def forward(self, face_features: torch.Tensor, audio_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            face_features: 面部特征 (B, T, feature_dim)
            audio_features: 音频特征 (B, T, feature_dim)
        Returns:
            融合后的多模态特征 (B, T, feature_dim)
        """
        batch_size, seq_len, _ = face_features.shape
        
        # 添加位置编码
        face_with_pe = self.PPE(face_features)
        audio_with_pe = self.PE(audio_features)
        
        # 生成注意力掩码
        tgt_mask = self.biased_mask[:, :seq_len, :seq_len].clone().detach().to(
            device=self.device
        ).repeat(batch_size, 1, 1)
        
        memory_mask = init_biased_mask2(
            n_head=8,
            window_size=seq_len,
            max_seq_len=seq_len,
            period=self.period
        ).clone().detach().to(device=self.device).repeat(batch_size, 1, 1)
        
        # 1. 面部特征自注意力
        face_self_attended = self.face_self_attention(
            tgt=face_with_pe,
            memory=face_with_pe,
            tgt_mask=tgt_mask
        )
        
        # 2. 音频-面部交叉注意力
        audio_face_fused = self.audio_face_fusion(
            tgt=face_self_attended,
            memory=audio_with_pe,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask
        )
        
        # 3. 时序建模
        temporal_modeled = self.temporal_modeling(
            tgt=audio_face_fused,
            memory=audio_face_fused,
            tgt_mask=tgt_mask
        )
        
        return temporal_modeled


class SpeakerEncoder(nn.Module):
    """Speaker特征编码器 - 整合面部和音频特征"""
    def __init__(self, feature_dim=256, audio_dim=384, period=25, max_seq_len=750, device='cpu'):
        super().__init__()
        
        # 子编码器
        self.facial_encoder = FacialFeatureEncoder(feature_dim)
        self.audio_encoder = AudioFeatureEncoder(audio_dim, feature_dim)
        self.fusion_encoder = MultiModalFusionEncoder(feature_dim, period, max_seq_len, device)
        
    def forward(self, speaker_data: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            speaker_data: 包含面部特征和音频特征的字典
        Returns:
            tuple: (speaker_motion, speaker_audio, speaker_vector)
                - speaker_motion: 融合后的运动特征
                - speaker_audio: 编码后的音频特征  
                - speaker_vector: 整体特征表示
        """
        # 提取面部特征
        facial_features = {
            'landmarks': speaker_data['landmarks'],
            'au': speaker_data['au'], 
            'pose': speaker_data['pose'],
            'gaze': speaker_data['gaze']
        }
        
        # 编码各模态特征
        face_encoded = self.facial_encoder(facial_features)  # (B, T, feature_dim)
        audio_encoded = self.audio_encoder(speaker_data['audio'])  # (B, T, feature_dim)
        
        # 多模态融合
        fused_features = self.fusion_encoder(face_encoded, audio_encoded)  # (B, T, feature_dim)
        
        # 返回三种特征表示（模仿ReactFace的接口）
        speaker_motion = fused_features  # 用于运动生成
        speaker_audio = audio_encoded    # 用于音频条件
        speaker_vector = fused_features  # 用于整体表示
        
        return speaker_motion, speaker_audio, speaker_vector