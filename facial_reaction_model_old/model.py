"""
面部反应生成主模型 - 整合所有组件
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Tuple, Optional

from .encoder import SpeakerEncoder
from .vae import MotionSampleGenerator
from .decoder import ListenerReactionDecoder, SpeakerReconstructionDecoder
from .components import PeriodicPositionalEncoding


class FacialReactionModel(nn.Module):
    """
    面部反应生成主模型
    
    Args:
        feature_dim: 特征维度
        audio_dim: 音频特征维度  
        period: 周期性位置编码周期
        max_seq_len: 最大序列长度
        device: 设备
        window_size: 窗口大小
        momentum: 动量系数
    """
    
    def __init__(
        self,
        feature_dim: int = 256,
        audio_dim: int = 384,
        period: int = 25,
        max_seq_len: int = 750,
        device: str = 'cpu',
        window_size: int = 16,
        momentum: float = 0.9
    ):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.audio_dim = audio_dim
        self.window_size = window_size
        self.momentum = momentum
        self.device = device
        self.period = period
        self.max_seq_len = max_seq_len
        
        # 核心组件
        self.speaker_encoder = SpeakerEncoder(
            feature_dim=feature_dim,
            audio_dim=audio_dim,
            period=period,
            max_seq_len=max_seq_len,
            device=device
        )
        
        self.motion_generator = MotionSampleGenerator(
            feature_dim=feature_dim,
            period=period,
            device=device
        )
        
        self.listener_decoder = ListenerReactionDecoder(
            feature_dim=feature_dim,
            period=period,
            max_seq_len=max_seq_len,
            device=device,
            window_size=window_size
        )
        
        self.speaker_decoder = SpeakerReconstructionDecoder(feature_dim=feature_dim)
        
        # 历史状态处理
        self.past_motion_linear = nn.Linear(158, feature_dim)  # 136+18+4+2 -> feature_dim
        self.ppe = PeriodicPositionalEncoding(feature_dim, period=period, max_seq_len=max_seq_len)
    
    def forward(
        self,
        speaker_data: Dict[str, torch.Tensor],
        speaker_out: bool = False
    ) -> Tuple[Dict[str, torch.Tensor], List[torch.distributions.Normal], Optional[Dict[str, torch.Tensor]]]:
        """
        前向传播
        
        Args:
            speaker_data: 包含speaker的landmarks, au, pose, gaze, audio
            speaker_out: 是否输出speaker重建结果
            
        Returns:
            tuple: (listener_predictions, distributions, speaker_reconstruction)
        """
        # 1. Speaker特征编码
        speaker_motion, speaker_audio, speaker_vector = self.speaker_encoder(speaker_data)
        
        frame_num = speaker_motion.shape[1]
        batch_size = speaker_motion.shape[0]
        
        # 2. 窗口化处理
        # 初始化历史状态
        past_reaction_features = torch.zeros(
            (batch_size, self.window_size, 158),  # 136+18+4+2
            device=speaker_motion.device
        )
        
        distributions = []
        past_motion_sample = None
        iterations = math.ceil(frame_num / self.window_size)
        
        all_predictions = {
            'landmarks': [],
            'au': [],
            'pose': [],
            'gaze': []
        }
        
        # 3. 逐窗口处理
        for i in range(iterations):
            # 累积上下文
            speaker_motion_cumulative = speaker_motion[:, :(i + 1) * self.window_size]
            speaker_audio_cumulative = speaker_audio[:, :2 * (i + 1) * self.window_size]
            
            # 处理历史状态
            past_listener_motion = self.past_motion_linear(past_reaction_features)
            past_listener_motion = past_listener_motion + self.ppe(past_listener_motion)
            
            # 生成运动样本
            motion_sample, distribution = self.motion_generator(
                speaker_motion_cumulative,
                speaker_audio_cumulative,
                past_listener_motion
            )
            distributions.append(distribution)
            
            # 动量平滑
            if past_motion_sample is not None:
                motion_sample = self.momentum * past_motion_sample + (1 - self.momentum) * motion_sample
                
                # 插值生成motion_sample_input
                motion_sample_input = F.interpolate(
                    torch.cat((past_motion_sample.unsqueeze(-1), motion_sample.unsqueeze(-1)), dim=-1),
                    self.window_size,
                    mode='linear'
                )
                motion_sample_input = motion_sample_input.transpose(1, 2)  # (B, window_size, feature_dim)
            else:
                motion_sample_input = motion_sample.unsqueeze(1).expand(-1, self.window_size, -1)
            
            past_motion_sample = motion_sample
            
            # 解码当前窗口的listener反应
            window_predictions = self.listener_decoder(
                motion_sample_input,
                speaker_motion_cumulative,
                speaker_audio_cumulative
            )
            
            # 收集预测结果
            for feature_name in all_predictions:
                all_predictions[feature_name].append(window_predictions[feature_name])
            
            # 更新历史状态
            current_features = torch.cat([
                window_predictions['landmarks'],  # (B, window_size, 136)
                window_predictions['au'],         # (B, window_size, 17)
                window_predictions['pose'],       # (B, window_size, 3)
                window_predictions['gaze']        # (B, window_size, 2)
            ], dim=-1)  # (B, window_size, 158)
            
            if i == 0:
                past_reaction_features = current_features
            else:
                past_reaction_features = torch.cat([past_reaction_features, current_features], dim=1)
                # 保持window_size的长度
                if past_reaction_features.shape[1] > self.window_size:
                    past_reaction_features = past_reaction_features[:, -self.window_size:]
        
        # 4. 拼接所有窗口的预测结果
        final_predictions = {}
        for feature_name in all_predictions:
            concatenated = torch.cat(all_predictions[feature_name], dim=1)  # (B, total_frames, feature_dim)
            # 截断到输入序列长度
            final_predictions[feature_name] = concatenated[:, :frame_num]
        
        # 5. Speaker重建（可选）
        speaker_reconstruction = None
        if speaker_out:
            speaker_reconstruction = self.speaker_decoder(speaker_vector)
        
        return final_predictions, distributions, speaker_reconstruction
    
    @torch.no_grad()
    def inference_step(
        self,
        speaker_data: Dict[str, torch.Tensor],
        past_reaction_features: torch.Tensor,  # (B, window_size, 158)
        past_motion_sample: Optional[torch.Tensor] = None
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        单步在线推理
        
        Args:
            speaker_data: 当前窗口的speaker数据
            past_reaction_features: 历史listener特征
            past_motion_sample: 历史运动样本
            
        Returns:
            tuple: (当前窗口预测, 更新的历史特征, 新的运动样本)
        """
        # 1. 编码当前窗口的speaker特征
        speaker_motion, speaker_audio, _ = self.speaker_encoder(speaker_data)
        
        # 只取最后window_size的特征
        speaker_motion = speaker_motion[:, -self.window_size:]
        speaker_audio = speaker_audio[:, -2 * self.window_size:]
        
        # 2. 处理历史状态
        past_listener_motion = self.past_motion_linear(past_reaction_features)
        past_listener_motion = past_listener_motion + self.ppe(past_listener_motion)
        
        # 3. 生成运动样本
        motion_sample, _ = self.motion_generator(
            speaker_motion,
            speaker_audio,
            past_listener_motion
        )
        
        # 4. 动量平滑
        if past_motion_sample is not None:
            motion_sample = self.momentum * past_motion_sample + (1 - self.momentum) * motion_sample
            motion_sample_input = F.interpolate(
                torch.cat((past_motion_sample.unsqueeze(-1), motion_sample.unsqueeze(-1)), dim=-1),
                self.window_size,
                mode='linear'
            )
            motion_sample_input = motion_sample_input.transpose(1, 2)
        else:
            motion_sample_input = motion_sample.unsqueeze(1).expand(-1, self.window_size, -1)
        
        # 5. 解码当前窗口预测
        current_predictions = self.listener_decoder(
            motion_sample_input,
            speaker_motion,
            speaker_audio
        )
        
        # 6. 更新历史状态
        current_features = torch.cat([
            current_predictions['landmarks'],
            current_predictions['au'],
            current_predictions['pose'],
            current_predictions['gaze']
        ], dim=-1)  # (B, window_size, 158)
        
        return current_predictions, current_features, motion_sample
    
    def reset_window_size(self, window_size: int) -> None:
        """重置窗口大小"""
        self.window_size = window_size
        self.listener_decoder.reset_window_size(window_size)
    
    def get_model_info(self) -> Dict[str, int]:
        """获取模型信息"""
        return {
            'feature_dim': self.feature_dim,
            'audio_dim': self.audio_dim,
            'window_size': self.window_size,
            'max_seq_len': self.max_seq_len,
            'total_parameters': sum(p.numel() for p in self.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }


class OnlineInferenceManager:
    """在线推理管理器"""
    
    def __init__(self, model: FacialReactionModel, device: str = 'cpu'):
        self.model = model
        self.device = device
        self.model.eval()
        
        # 在线状态
        self.past_reaction_features = None
        self.past_motion_sample = None
        self.is_initialized = False
    
    def initialize(self, batch_size: int = 1):
        """初始化在线状态"""
        self.past_reaction_features = torch.zeros(
            (batch_size, self.model.window_size, 158),
            device=self.device
        )
        self.past_motion_sample = None
        self.is_initialized = True
    
    def process_window(
        self, 
        speaker_data: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """处理一个窗口的数据"""
        if not self.is_initialized:
            batch_size = speaker_data['landmarks'].shape[0]
            self.initialize(batch_size)
        
        with torch.no_grad():
            predictions, new_features, new_motion_sample = self.model.inference_step(
                speaker_data,
                self.past_reaction_features,
                self.past_motion_sample
            )
            
            # 更新状态
            self.past_reaction_features = new_features
            self.past_motion_sample = new_motion_sample
            
            return predictions
    
    def reset(self):
        """重置在线状态"""
        self.past_reaction_features = None
        self.past_motion_sample = None
        self.is_initialized = False