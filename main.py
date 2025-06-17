"""
面部反应生成模型训练主程序
通过命令行参数配置，自动保存运行配置
"""

import os
import sys
import argparse
import random
import numpy as np
import torch
import json
from datetime import datetime

from train import FacialReactionTrainer


def set_seed(seed: int = 42):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"🎲 设置随机种子: {seed}")


def check_environment():
    """检查运行环境"""
    print("🔍 检查运行环境...")
    
    # CUDA检查
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"🎮 GPU: {gpu_name} x{gpu_count} ({gpu_memory:.1f}GB)")
    else:
        print("⚠️ CUDA不可用，将使用CPU训练")
    
    # 依赖检查
    try:
        import transformers
        import torchaudio
        print("✅ 依赖库检查通过")
    except ImportError as e:
        print(f"❌ 缺少依赖库: {e}")
        sys.exit(1)


def create_config_from_args(args) -> dict:
    """根据命令行参数创建配置"""
    
    # 如果没有指定实验名，自动生成
    if not args.exp_name:
        args.exp_name = f"facial_reaction_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    config = {
        'exp_name': args.exp_name,
        'device': args.device,
        'seed': args.seed,
        
        # 模型配置
        'model': {
            'feature_dim': args.feature_dim,
            'audio_dim': args.audio_dim,
            'period': args.period,
            'max_seq_len': args.max_seq_len,
            'window_size': args.window_size,
            'momentum': args.momentum
        },
        
        # 数据配置
        'data': {
            'train_csv': args.train_csv,
            'val_csv': args.val_csv,
            'batch_size': args.batch_size,
            'num_workers': args.num_workers,
            'whisper_model': args.whisper_model
        },
        
        # 训练配置
        'training': {
            'num_epochs': args.epochs,
            'num_samples': args.num_samples,
            'accumulate_grad_batches': args.accumulate_grad,
            'max_grad_norm': args.max_grad_norm,
            'log_interval': args.log_interval,
            'save_interval': args.save_interval,
            'early_stopping': args.early_stopping,
            'patience': args.patience
        },
        
        # 优化器配置
        'optimizer': {
            'type': args.optimizer,
            'lr': args.lr,
            'betas': [0.9, 0.999],
            'weight_decay': args.weight_decay
        },
        
        # 学习率调度器配置
        'scheduler': {
            'type': args.scheduler,
            'min_lr': args.min_lr
        },
        
        # 损失函数配置
        'loss': {
            'kl_weight': args.kl_weight,
            'contrastive_temperature': args.contrastive_temp,
            'contrastive_margin': args.contrastive_margin,
            'smooth_motion_weight': args.smooth_motion_weight,
            'smooth_expression_weight': args.smooth_expression_weight,
            'weights': {
                'vae': args.vae_weight,
                'smooth': args.smooth_weight,
                'diversity': args.diversity_weight,
                'contrastive': args.contrastive_weight,
                'consistency': args.consistency_weight,
                'speaker_rec': args.speaker_rec_weight
            }
        }
    }
    
    # 添加恢复检查点路径
    if args.resume:
        config['resume_from_checkpoint'] = args.resume
    
    return config


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="面部反应生成模型训练")
    
    # === 必需参数 ===
    parser.add_argument('--train-csv', type=str, default='/mnt/iusers01/fatpou01/compsci01/k09562zs/scratch/Face_data/train.csv',
                       help='训练数据CSV路径')
    
    # === 基本训练参数 ===
    parser.add_argument('--val-csv', type=str, default='/mnt/iusers01/fatpou01/compsci01/k09562zs/scratch/Face_data/val.csv',
                       help='验证数据CSV路径')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='批次大小 (默认: 2)')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='学习率 (默认: 1e-4)')
    parser.add_argument('--epochs', type=int, default=200,
                       help='训练轮数 (默认: 100)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='训练设备 (默认: cuda)')
    
    # === 模型参数 ===
    parser.add_argument('--feature-dim', type=int, default=256,
                       help='特征维度 (默认: 256)')
    parser.add_argument('--audio-dim', type=int, default=384,
                       help='音频特征维度 (默认: 384)')
    parser.add_argument('--window-size', type=int, default=8,
                       help='窗口大小 (默认: 16)')
    parser.add_argument('--period', type=int, default=25,
                       help='周期性位置编码周期 (默认: 25)')
    parser.add_argument('--max-seq-len', type=int, default=750,
                       help='最大序列长度 (默认: 750)')
    parser.add_argument('--momentum', type=float, default=0.9,
                       help='动量系数 (默认: 0.9)')
    
    # === 训练策略参数 ===
    parser.add_argument('--num-samples', type=int, default=3,
                       help='每batch采样次数 (默认: 3)')
    parser.add_argument('--accumulate-grad', type=int, default=4,
                       help='梯度累积步数 (默认: 4)')
    parser.add_argument('--max-grad-norm', type=float, default=1.0,
                       help='梯度裁剪阈值 (默认: 1.0)')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='数据加载器工作进程数 (默认: 4)')
    
    # === 优化器参数 ===
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['adam', 'adamw'],
                       help='优化器类型 (默认: adamw)')
    parser.add_argument('--weight-decay', type=float, default=0.01,
                       help='权重衰减 (默认: 0.01)')
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['cosine', 'step', 'none'],
                       help='学习率调度器 (默认: cosine)')
    parser.add_argument('--min-lr', type=float, default=1e-6,
                       help='最小学习率 (默认: 1e-6)')
    
    # === 损失函数权重 ===
    parser.add_argument('--vae-weight', type=float, default=5.0,
                       help='VAE损失权重 (默认: 1.0)')
    parser.add_argument('--smooth-weight', type=float, default=1.0,
                       help='平滑损失权重 (默认: 15.0)')
    parser.add_argument('--diversity-weight', type=float, default=1.0,
                       help='多样性损失权重 (默认: 100.0)')
    parser.add_argument('--contrastive-weight', type=float, default=1.0,
                       help='对比损失权重 (默认: 5.0)')
    parser.add_argument('--consistency-weight', type=float, default=1.0,
                       help='一致性损失权重 (默认: 2.0)')
    parser.add_argument('--speaker-rec-weight', type=float, default=1.0,
                       help='Speaker重建损失权重 (默认: 1.0)')
    
    # === 损失函数超参数 ===
    parser.add_argument('--kl-weight', type=float, default=0.0002,
                       help='KL散度权重 (默认: 0.0002)')
    parser.add_argument('--contrastive-temp', type=float, default=0.1,
                       help='对比损失温度 (默认: 0.1)')
    parser.add_argument('--contrastive-margin', type=float, default=0.7,
                       help='对比损失边距 (默认: 0.7)')
    parser.add_argument('--smooth-motion-weight', type=float, default=1.0,
                       help='运动平滑权重 (默认: 1.0)')
    parser.add_argument('--smooth-expression-weight', type=float, default=0.1,
                       help='表情平滑权重 (默认: 0.1)')
    
    # === 其他参数 ===
    parser.add_argument('--whisper-model', type=str, default='openai/whisper-tiny',
                       help='Whisper模型名称 (默认: openai/whisper-tiny)')
    parser.add_argument('--exp-name', type=str, default=None,
                       help='实验名称 (默认: 自动生成)')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子 (默认: 42)')
    parser.add_argument('--resume', type=str, default=None,
                       help='从检查点恢复训练路径')
    
    # === 训练控制参数 ===
    parser.add_argument('--log-interval', type=int, default=10,
                       help='日志记录间隔 (默认: 10)')
    parser.add_argument('--save-interval', type=int, default=10,
                       help='模型保存间隔 (默认: 10)')
    parser.add_argument('--early-stopping', action='store_true',
                       help='启用早停')
    parser.add_argument('--patience', type=int, default=20,
                       help='早停耐心值 (默认: 20)')
    
    # === 工具功能 ===
    parser.add_argument('--check-env', action='store_true',
                       help='只检查环境，不训练')
    
    args = parser.parse_args()
    
    # 检查环境
    check_environment()
    if args.check_env:
        return
    
    # 检查必需的数据路径
    if not os.path.exists(args.train_csv):
        print(f"❌ 训练数据不存在: {args.train_csv}")
        sys.exit(1)
    
    if args.val_csv and not os.path.exists(args.val_csv):
        print(f"❌ 验证数据不存在: {args.val_csv}")
        sys.exit(1)
    
    # 根据参数创建配置
    config = create_config_from_args(args)
    
    # 设置随机种子
    set_seed(config['seed'])
    
    # 显示配置信息
    print("\n" + "="*60)
    print("🚀 训练配置信息")
    print("="*60)
    print(f"实验名称: {config['exp_name']}")
    print(f"训练数据: {config['data']['train_csv']}")
    print(f"验证数据: {config['data'].get('val_csv', 'None')}")
    print(f"批次大小: {config['data']['batch_size']} (累积: {config['training']['accumulate_grad_batches']})")
    print(f"学习率: {config['optimizer']['lr']}")
    print(f"训练轮数: {config['training']['num_epochs']}")
    print(f"设备: {config['device']}")
    print(f"窗口大小: {config['model']['window_size']}")
    print(f"特征维度: {config['model']['feature_dim']}")
    print("="*60 + "\n")
    
    try:
        # 创建训练器
        trainer = FacialReactionTrainer(config)
        
        # 开始训练
        trainer.train()
        
    except KeyboardInterrupt:
        print("\n⚠️ 训练被用户中断")
    except Exception as e:
        print(f"\n❌ 训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    # 设置环境变量
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    
    # 运行主程序
    main()