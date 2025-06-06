"""
面部反应生成模型使用示例
"""

import torch
from facial_reaction_model.model import FacialReactionModel, OnlineInferenceManager
from facial_reaction_model.losses import (
    FacialVAELoss, FacialContrastiveLoss, FacialDiversityLoss, 
    FacialSmoothLoss, FacialConsistencyLoss
)


def create_model(device='cuda'):
    """创建模型实例"""
    model = FacialReactionModel(
        feature_dim=256,
        audio_dim=384,
        period=25,
        max_seq_len=750,
        device=device,
        window_size=16,
        momentum=0.9
    )
    
    return model.to(device)


def create_loss_functions():
    """创建损失函数"""
    return {
        'vae_loss': FacialVAELoss(kl_weight=0.0002),
        'contrastive_loss': FacialContrastiveLoss(temperature=0.1, margin=0.7),
        'diversity_loss': FacialDiversityLoss(),
        'smooth_loss': FacialSmoothLoss(),
        'consistency_loss': FacialConsistencyLoss()
    }


def example_training_step():
    """训练步骤示例"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 创建模型和损失
    model = create_model(device)
    loss_functions = create_loss_functions()
    
    # 模拟输入数据 (batch_size=2, seq_len=750)
    batch_size, seq_len = 2, 750
    speaker_data = {
        'landmarks': torch.randn(batch_size, seq_len, 136, device=device),  # 68*2
        'au': torch.randn(batch_size, seq_len, 18, device=device),
        'pose': torch.randn(batch_size, seq_len, 4, device=device),
        'gaze': torch.randn(batch_size, seq_len, 2, device=device),
        'audio': torch.randn(batch_size, seq_len, 384, device=device)  # Whisper特征
    }
    
    # 模拟真实listener数据
    gt_listener = {
        'landmarks': torch.randn(batch_size, seq_len, 136, device=device),
        'au': torch.randn(batch_size, seq_len, 18, device=device),
        'pose': torch.randn(batch_size, seq_len, 4, device=device),
        'gaze': torch.randn(batch_size, seq_len, 2, device=device)
    }
    
    # 前向传播
    model.train()
    predictions, distributions, speaker_reconstruction = model(
        speaker_data, 
        speaker_out=True
    )
    
    # 计算损失
    vae_loss, rec_loss, kld_loss, feature_losses = loss_functions['vae_loss'](
        gt_listener, predictions, distributions
    )
    
    smooth_loss = loss_functions['smooth_loss'](predictions)
    consistency_loss = loss_functions['consistency_loss'](predictions)
    
    # Speaker重建损失
    speaker_rec_loss = 0.0
    if speaker_reconstruction:
        speaker_rec_loss, _, _,_ = loss_functions['vae_loss'](
            speaker_data, speaker_reconstruction, []
        )
    
    # 多样性损失和对比损失（需要多次前向传播）
    predictions_2, _, _ = model(speaker_data)
    predictions_3, _, _ = model(speaker_data)
    
    # 多样性损失
    div_loss = (
        loss_functions['diversity_loss'](predictions, predictions_2) +
        loss_functions['diversity_loss'](predictions, predictions_3) +
        loss_functions['diversity_loss'](predictions_2, predictions_3)
    )
    
    # 对比损失（替代邻居损失）
    contrastive_loss = loss_functions['contrastive_loss'](
        predictions, predictions_2, gt_listener
    )
    
    # 总损失（调整权重以补偿没有邻居损失）
    total_loss = (
        vae_loss + 
        15.0 * smooth_loss +          # 增加平滑损失权重
        100.0 * div_loss + 
        speaker_rec_loss +
        2.0 * consistency_loss +      # 增加一致性损失权重  
        5.0 * contrastive_loss        # 新增对比损失
    )
    
    print(f"Total Loss: {total_loss.item():.4f}")
    print(f"  VAE Loss: {vae_loss.item():.4f}")
    print(f"  Reconstruction Loss: {rec_loss.item():.4f}")
    print(f"  KLD Loss: {kld_loss.item():.4f}")
    print(f"  Smooth Loss: {smooth_loss.item():.4f}")
    print(f"  Diversity Loss: {div_loss.item():.4f}")
    print(f"  Speaker Rec Loss: {speaker_rec_loss.item() if isinstance(speaker_rec_loss, torch.Tensor) else speaker_rec_loss:.4f}")
    print(f"  Consistency Loss: {consistency_loss.item():.4f}")
    print(f"  Contrastive Loss: {contrastive_loss.item():.4f}")
    
    return total_loss


def example_online_inference():
    """在线推理示例"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 创建模型
    model = create_model(device)
    model.eval()
    
    # 创建在线推理管理器
    online_manager = OnlineInferenceManager(model, device)
    
    # 模拟实时数据流
    window_size = 16
    num_windows = 5
    
    print("开始在线推理...")
    
    for i in range(num_windows):
        # 模拟当前窗口的speaker数据
        current_speaker_data = {
            'landmarks': torch.randn(1, window_size, 136, device=device),
            'au': torch.randn(1, window_size, 18, device=device), 
            'pose': torch.randn(1, window_size, 4, device=device),
            'gaze': torch.randn(1, window_size, 2, device=device),
            'audio': torch.randn(1, window_size, 384, device=device)
        }
        
        # 在线推理
        listener_predictions = online_manager.process_window(current_speaker_data)
        
        print(f"Window {i+1}:")
        print(f"  Landmarks: {listener_predictions['landmarks'].shape}")
        print(f"  AU: {listener_predictions['au'].shape}")
        print(f"  Pose: {listener_predictions['pose'].shape}")
        print(f"  Gaze: {listener_predictions['gaze'].shape}")
    
    print("在线推理完成!")


def example_model_info():
    """模型信息示例"""
    model = create_model('cpu')
    info = model.get_model_info()
    
    print("模型信息:")
    for key, value in info.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    print("=== 面部反应生成模型示例 ===\n")
    
    print("1. 模型信息:")
    example_model_info()
    
    print("\n2. 训练步骤示例:")
    training_loss = example_training_step()
    
    print("\n3. 在线推理示例:")
    example_online_inference()
    
    print("\n=== 示例完成 ===")


# 兼容训练脚本的接口
class ReactFaceWrapper:
    """为了兼容现有训练脚本的包装器"""
    
    def __init__(self, **kwargs):
        # 映射参数名
        self.model = FacialReactionModel(
            feature_dim=kwargs.get('feature_dim', 256),
            audio_dim=kwargs.get('audio_dim', 384),
            period=kwargs.get('period', 25),
            max_seq_len=kwargs.get('max_seq_len', 750),
            device=kwargs.get('device', 'cpu'),
            window_size=kwargs.get('window_size', 16),
            momentum=kwargs.get('momentum', 0.9)
        )
    
    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)
    
    def __getattr__(self, name):
        return getattr(self.model, name)


# 用于替换原始模型的函数
def ReactFace(**kwargs):
    """创建面部反应模型的工厂函数"""
    return ReactFaceWrapper(**kwargs).model