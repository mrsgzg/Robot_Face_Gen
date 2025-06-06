"""
简单测试脚本 - 验证基本功能
"""

import torch
import traceback

def test_components():
    """测试基础组件"""
    print("=== 测试基础组件 ===")
    
    try:
        from facial_reaction_model.components import PositionalEncoding
        
        # 测试PositionalEncoding
        pe = PositionalEncoding(d_model=256, dropout=0.1)
        test_input = torch.randn(2, 16, 256)  # (batch, seq, features)
        output = pe(test_input)
        print(f"✅ PositionalEncoding成功 - 输入: {test_input.shape}, 输出: {output.shape}")
        
    except Exception as e:
        print(f"❌ 基础组件测试失败: {e}")
        traceback.print_exc()
        return False
    
    return True


def test_vae_simple():
    """简单VAE测试"""
    print("\n=== 测试VAE模块 ===")
    
    try:
        from facial_reaction_model.vae import FacialVAE
        
        # 创建VAE
        vae = FacialVAE(feature_dim=256, latent_dim=256)
        print("✅ VAE创建成功")
        
        # 简单前向传播测试
        test_input = torch.randn(2, 8, 256)  # 更小的输入
        motion_sample, distribution = vae(test_input)
        
        print(f"✅ VAE前向传播成功")
        print(f"输入形状: {test_input.shape}")
        print(f"Motion sample形状: {motion_sample.shape}")
        print(f"Distribution mean形状: {distribution.mean.shape}")
        
    except Exception as e:
        print(f"❌ VAE测试失败: {e}")
        traceback.print_exc()
        return False
    
    return True


def test_encoder():
    """测试编码器"""
    print("\n=== 测试编码器 ===")
    
    try:
        from facial_reaction_model.encoder import SpeakerEncoder
        
        # 创建编码器
        encoder = SpeakerEncoder(feature_dim=256, audio_dim=384, device='cpu')
        print("✅ SpeakerEncoder创建成功")
        
        # 测试数据
        batch_size, seq_len = 2, 16
        speaker_data = {
            'landmarks': torch.randn(batch_size, seq_len, 136),
            'au': torch.randn(batch_size, seq_len, 18),
            'pose': torch.randn(batch_size, seq_len, 4),
            'gaze': torch.randn(batch_size, seq_len, 2),
            'audio': torch.randn(batch_size, seq_len, 384)
        }
        
        speaker_motion, speaker_audio, speaker_vector = encoder(speaker_data)
        
        print(f"✅ SpeakerEncoder前向传播成功")
        print(f"Speaker motion形状: {speaker_motion.shape}")
        print(f"Speaker audio形状: {speaker_audio.shape}")
        print(f"Speaker vector形状: {speaker_vector.shape}")
        
    except Exception as e:
        print(f"❌ 编码器测试失败: {e}")
        traceback.print_exc()
        return False
    
    return True


def test_decoder():
    """测试解码器"""
    print("\n=== 测试解码器 ===")
    
    try:
        from facial_reaction_model.decoder import ListenerReactionDecoder
        
        # 创建解码器
        decoder = ListenerReactionDecoder(feature_dim=256, window_size=8, device='cpu')
        print("✅ ListenerReactionDecoder创建成功")
        
        # 测试数据
        batch_size, window_size = 2, 8
        motion_sample = torch.randn(batch_size, 1, 256)
        speaker_motion = torch.randn(batch_size, window_size, 256)
        speaker_audio = torch.randn(batch_size, window_size * 2, 256)
        
        predictions = decoder(motion_sample, speaker_motion, speaker_audio)
        
        print(f"✅ ListenerReactionDecoder前向传播成功")
        print("预测结果形状:")
        for key, value in predictions.items():
            print(f"  {key}: {value.shape}")
        
    except Exception as e:
        print(f"❌ 解码器测试失败: {e}")
        traceback.print_exc()
        return False
    
    return True


def test_minimal_model():
    """测试最小模型"""
    print("\n=== 测试最小模型 ===")
    
    try:
        from facial_reaction_model.model import FacialReactionModel
        
        # 创建较小的模型
        model = FacialReactionModel(
            feature_dim=128,  # 减小特征维度
            audio_dim=384,
            window_size=8,    # 减小窗口大小
            device='cpu'
        )
        print("✅ FacialReactionModel创建成功")
        
        # 使用很小的测试数据
        batch_size, seq_len = 1, 16  # 单batch，短序列
        speaker_data = {
            'landmarks': torch.randn(batch_size, seq_len, 136),
            'au': torch.randn(batch_size, seq_len, 18),
            'pose': torch.randn(batch_size, seq_len, 4),
            'gaze': torch.randn(batch_size, seq_len, 2),
            'audio': torch.randn(batch_size, seq_len, 384)
        }
        
        print("开始前向传播...")
        predictions, distributions, _ = model(speaker_data, speaker_out=False)
        
        print(f"✅ 最小模型前向传播成功")
        print("预测结果形状:")
        for key, value in predictions.items():
            print(f"  {key}: {value.shape}")
        print(f"分布数量: {len(distributions)}")
        
    except Exception as e:
        print(f"❌ 最小模型测试失败: {e}")
        traceback.print_exc()
        return False
    
    return True


def main():
    """主函数"""
    print("开始简单测试...")
    
    tests = [
        ("基础组件", test_components),
        ("VAE模块", test_vae_simple),
        ("编码器", test_encoder),
        ("解码器", test_decoder),
        ("最小模型", test_minimal_model)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"测试 {test_name}")
        print('='*50)
        
        if test_func():
            passed += 1
            print(f"✅ {test_name} 测试通过")
        else:
            print(f"❌ {test_name} 测试失败")
    
    print(f"\n{'='*50}")
    print(f"测试总结: {passed}/{total} 通过")
    print('='*50)
    
    if passed == total:
        print("🎉 所有测试通过！")
        return True
    else:
        print("❌ 仍有失败的测试")
        return False


if __name__ == "__main__":
    # 设置torch不使用多线程，避免一些潜在问题
    torch.set_num_threads(1)
    main()