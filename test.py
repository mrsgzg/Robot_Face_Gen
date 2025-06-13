#!/usr/bin/env python3
"""
简易模型测试脚本
加载checkpoint，生成预测landmarks，保存结果并可视化
"""

import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm

# 导入你的模型和数据集
from facial_reaction_model.model import FacialReactionModel
from Data_Set import SpeakerListenerDataset


class SimpleTester:
    """简易测试器"""
    
    def __init__(self, checkpoint_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.checkpoint_path = checkpoint_path
        
        # 加载模型
        self.load_model()
        
    def load_model(self):
        """加载训练好的模型"""
        print(f"🔄 加载模型从: {self.checkpoint_path}")
        
        # 加载checkpoint
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        
        # 从checkpoint获取模型配置
        config = checkpoint.get('config', {})
        model_config = config.get('model', {})
        
        # 创建模型实例
        self.model = FacialReactionModel(
            feature_dim=model_config.get('feature_dim', 256),
            audio_dim=model_config.get('audio_dim', 384),
            period=model_config.get('period', 25),
            max_seq_len=model_config.get('max_seq_len', 750),
            device=str(self.device),
            window_size=model_config.get('window_size', 16),
            momentum=model_config.get('momentum', 0.9)
        )
        
        # 加载模型权重
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print(f"✅ 模型加载成功，epoch: {checkpoint.get('epoch', 'unknown')}")
        
    def load_test_sample(self, data_csv, sample_idx=0):
        """加载单个测试样本"""
        print(f"📊 加载测试数据: {data_csv}, 样本索引: {sample_idx}")
        
        # 创建数据集
        dataset = SpeakerListenerDataset(data_csv)
        
        # 检查索引有效性
        if sample_idx >= len(dataset):
            sample_idx = 0
            print(f"⚠️ 样本索引超出范围，使用索引 0")
        
        # 获取样本
        sample = dataset[sample_idx]
        if sample is None:
            raise ValueError(f"无法加载样本 {sample_idx}")
        
        # 转换为batch格式并移动到设备
        batch_data = {}
        for role in ['speaker', 'listener']:
            batch_data[role] = {}
            for key, value in sample[role].items():
                batch_data[role][key] = value.unsqueeze(0).to(self.device)
        
        print(f"✅ 测试样本加载成功")
        print(f"   Speaker landmarks: {batch_data['speaker']['landmarks'].shape}")
        print(f"   Speaker AU: {batch_data['speaker']['au'].shape}")
        print(f"   Listener landmarks: {batch_data['listener']['landmarks'].shape}")
        print(f"   Listener AU: {batch_data['listener']['au'].shape}")
        
        return batch_data
    
    def generate_predictions(self, batch_data, num_samples=3):
        """生成预测结果"""
        print(f"🎯 开始生成预测 (采样次数: {num_samples})")
        
        speaker_data = batch_data['speaker']
        
        predictions_list = []
        
        with torch.no_grad():
            for i in range(num_samples):
                predictions, distributions, _ = self.model(speaker_data, speaker_out=False)
                
                # 验证预测结果包含所有必要的特征
                expected_features = ['landmarks', 'au', 'pose', 'gaze']
                for feature in expected_features:
                    if feature not in predictions:
                        raise ValueError(f"预测结果缺少特征: {feature}")
                
                predictions_list.append(predictions)
        
        print(f"✅ 预测生成完成")
        
        # 返回第一次预测作为主要结果，以及所有预测用于多样性分析
        return predictions_list[0], predictions_list
    
    def save_results(self, speaker_data, gt_listener, predictions, output_dir='test_results'):
        """保存预测结果为CSV"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 转换为numpy并去掉batch维度
        def to_numpy(tensor_dict):
            return {k: v.cpu().numpy().squeeze(0) for k, v in tensor_dict.items()}
        
        speaker_np = to_numpy(speaker_data)
        gt_listener_np = to_numpy(gt_listener)
        pred_listener_np = to_numpy(predictions)
        
        # 反归一化所有特征 (根据Data_Set.py中的归一化方式)
        def denormalize_features(data_dict):
            result = data_dict.copy()
            result['landmarks'] = data_dict['landmarks'] * 256.0  # landmarks除以256归一化
            result['au'] = data_dict['au'] * 5.0                  # AU除以5归一化
            # pose和gaze保持原始值，无需反归一化
            return result
        
        speaker_denorm = denormalize_features(speaker_np)
        gt_denorm = denormalize_features(gt_listener_np)
        pred_denorm = denormalize_features(pred_listener_np)
        
        # 保存所有特征为CSV
        def save_all_features_csv(features_dict, prefix, output_dir):
            """保存所有特征到CSV文件"""
            for feature_name, feature_data in features_dict.items():
                if feature_name == 'audio':  # 音频特征太大，跳过
                    continue
                    
                filename = os.path.join(output_dir, f'{prefix}_{feature_name}.csv')
                
                if feature_name == 'landmarks':
                    # landmarks特殊处理：分离x,y坐标
                    x_coords = feature_data[:, :68]
                    y_coords = feature_data[:, 68:]
                    columns = [f'x_{i}' for i in range(68)] + [f'y_{i}' for i in range(68)]
                    df = pd.DataFrame(feature_data, columns=columns)
                else:
                    # 其他特征直接保存
                    if feature_name == 'au':
                        columns = [f'AU_{i}' for i in range(feature_data.shape[1])]
                    elif feature_name == 'pose':
                        columns = ['pose_Rx', 'pose_Ry', 'pose_Rz']
                    elif feature_name == 'gaze':
                        columns = ['gaze_angle_x', 'gaze_angle_y']
                    else:
                        columns = [f'{feature_name}_{i}' for i in range(feature_data.shape[1])]
                    
                    df = pd.DataFrame(feature_data, columns=columns)
                
                df.insert(0, 'frame', range(len(df)))
                df.to_csv(filename, index=False)
        
        # 保存所有特征
        save_all_features_csv(speaker_denorm, 'speaker', output_dir)
        save_all_features_csv(gt_denorm, 'gt_listener', output_dir)
        save_all_features_csv(pred_denorm, 'predicted_listener', output_dir)
        
        # 为向后兼容，仍然返回landmarks的DataFrame
        def load_landmarks_df(features_dict):
            landmarks = features_dict['landmarks']
            x_coords = landmarks[:, :68]
            y_coords = landmarks[:, 68:]
            columns = [f'x_{i}' for i in range(68)] + [f'y_{i}' for i in range(68)]
            df = pd.DataFrame(landmarks, columns=columns)
            df.insert(0, 'frame', range(len(df)))
            return df
        
        speaker_df = load_landmarks_df(speaker_denorm)
        gt_df = load_landmarks_df(gt_denorm)
        pred_df = load_landmarks_df(pred_denorm)
        
        print(f"💾 结果已保存到 {output_dir}:")
        print(f"   Speaker特征: speaker_landmarks.csv, speaker_au.csv, speaker_pose.csv, speaker_gaze.csv")
        print(f"   GT Listener特征: gt_listener_landmarks.csv, gt_listener_au.csv, gt_listener_pose.csv, gt_listener_gaze.csv")
        print(f"   预测Listener特征: predicted_listener_landmarks.csv, predicted_listener_au.csv, predicted_listener_pose.csv, predicted_listener_gaze.csv")
        
        return speaker_df, gt_df, pred_df
    
    def create_comparison_animation(self, speaker_df, gt_df, pred_df, 
                                  output_path='comparison_animation.gif', max_frames=300):
        """创建三路对比动画"""
        print(f"🎬 创建对比动画: {output_path}")
        
        # 限制帧数
        n_frames = min(len(speaker_df), len(gt_df), len(pred_df), max_frames)
        
        # 提取坐标数据
        def extract_coords(df):
            x_cols = [f'x_{i}' for i in range(68)]
            y_cols = [f'y_{i}' for i in range(68)]
            x_coords = df[x_cols].values[:n_frames]
            y_coords = df[y_cols].values[:n_frames]
            return x_coords, y_coords
        
        speaker_x, speaker_y = extract_coords(speaker_df)
        gt_x, gt_y = extract_coords(gt_df)
        pred_x, pred_y = extract_coords(pred_df)
        
        # 计算坐标范围
        all_x = np.concatenate([speaker_x.flatten(), gt_x.flatten(), pred_x.flatten()])
        all_y = np.concatenate([speaker_y.flatten(), gt_y.flatten(), pred_y.flatten()])
        x_min, x_max = np.min(all_x), np.max(all_x)
        y_min, y_max = np.min(all_y), np.max(all_y)
        margin = 0.1 * max(x_max - x_min, y_max - y_min)
        
        # 创建三列子图
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        
        def animate(frame):
            # 清除所有子图
            for ax in [ax1, ax2, ax3]:
                ax.clear()
                ax.set_xlim(x_min - margin, x_max + margin)
                ax.set_ylim(y_max + margin, y_min - margin)  # 翻转y轴
                ax.set_aspect('equal')
            
            # 设置标题
            ax1.set_title(f'Speaker - 帧 {frame+1}/{n_frames}', fontsize=12)
            ax2.set_title(f'GT Listener - 帧 {frame+1}/{n_frames}', fontsize=12)
            ax3.set_title(f'Predicted Listener - 帧 {frame+1}/{n_frames}', fontsize=12)
            
            # 绘制特征点
            ax1.scatter(speaker_x[frame], speaker_y[frame], c='blue', s=15, alpha=0.7)
            ax2.scatter(gt_x[frame], gt_y[frame], c='green', s=15, alpha=0.7)
            ax3.scatter(pred_x[frame], pred_y[frame], c='red', s=15, alpha=0.7)
            
            # 绘制基本连接线
            for ax, x_coords, y_coords in [(ax1, speaker_x[frame], speaker_y[frame]),
                                         (ax2, gt_x[frame], gt_y[frame]),
                                         (ax3, pred_x[frame], pred_y[frame])]:
                self.draw_face_connections(ax, x_coords, y_coords)
        
        # 创建动画
        anim = animation.FuncAnimation(fig, animate, frames=n_frames, 
                                     interval=100, repeat=True)
        
        # 保存
        anim.save(output_path, writer='pillow', fps=10, dpi=80)
        plt.close()
        
        print(f"✅ 动画已保存到: {output_path}")
    
    def draw_face_connections(self, ax, x_coords, y_coords):
        """绘制面部连接线"""
        # 简化的面部连接线
        connections = {
            'jaw': list(range(0, 17)),           # 下颌线
            'right_eyebrow': list(range(17, 22)), # 右眉毛
            'left_eyebrow': list(range(22, 27)),  # 左眉毛
            'nose_bridge': list(range(27, 31)),   # 鼻梁
            'nose_tip': list(range(31, 36)),      # 鼻翼
            'right_eye': list(range(36, 42)) + [36], # 右眼
            'left_eye': list(range(42, 48)) + [42],  # 左眼
            'outer_lip': list(range(48, 60)) + [48], # 外嘴唇
            'inner_lip': list(range(60, 68)) + [60]  # 内嘴唇
        }
        
        colors = ['brown', 'orange', 'orange', 'red', 'red', 'purple', 'purple', 'pink', 'pink']
        
        for i, (name, indices) in enumerate(connections.items()):
            if len(indices) > 1:
                x_line = [x_coords[idx] for idx in indices]
                y_line = [y_coords[idx] for idx in indices]
                ax.plot(x_line, y_line, color=colors[i % len(colors)], 
                       linewidth=1, alpha=0.6)
    
    def calculate_metrics(self, gt_listener, predictions):
        """计算评估指标"""
        print("📊 计算评估指标...")
        
        metrics = {}
        
        # 对每种特征计算指标
        for feature_name in ['landmarks', 'au', 'pose', 'gaze']:
            gt_data = gt_listener[feature_name].cpu().numpy().squeeze(0)
            pred_data = predictions[feature_name].cpu().numpy().squeeze(0)
            
            # 计算基本指标
            mse = np.mean((gt_data - pred_data) ** 2)
            mae = np.mean(np.abs(gt_data - pred_data))
            rmse = np.sqrt(mse)
            
            metrics[feature_name] = {
                'mse': mse,
                'mae': mae,
                'rmse': rmse
            }
            
            print(f"  {feature_name.upper()}:")
            print(f"    MSE: {mse:.6f}")
            print(f"    MAE: {mae:.6f}")
            print(f"    RMSE: {rmse:.6f}")
        
        # 对landmarks计算更详细的指标
        if 'landmarks' in metrics:
            gt_landmarks = gt_listener['landmarks'].cpu().numpy().squeeze(0)
            pred_landmarks = predictions['landmarks'].cpu().numpy().squeeze(0)
            
            # 分离x和y坐标的误差
            x_errors = np.abs(gt_landmarks[:, :68] - pred_landmarks[:, :68])
            y_errors = np.abs(gt_landmarks[:, 68:] - pred_landmarks[:, 68:])
            
            metrics['landmarks']['x_mae'] = np.mean(x_errors)
            metrics['landmarks']['y_mae'] = np.mean(y_errors)
            metrics['landmarks']['max_point_error'] = np.max(np.sqrt(x_errors**2 + y_errors**2))
            
            print(f"    X坐标MAE: {metrics['landmarks']['x_mae']:.6f}")
            print(f"    Y坐标MAE: {metrics['landmarks']['y_mae']:.6f}")
            print(f"    最大点误差: {metrics['landmarks']['max_point_error']:.6f}")
        
        return metrics
    
    def run_test(self, data_csv, sample_idx=0, output_dir='test_results'):
        """运行完整测试"""
        print("🚀 开始模型测试...")
        
        # 1. 加载测试样本
        batch_data = self.load_test_sample(data_csv, sample_idx)
        
        # 2. 生成预测
        predictions, all_predictions = self.generate_predictions(batch_data)
        
        # 3. 计算指标
        metrics = self.calculate_metrics(batch_data['listener'], predictions)
        
        # 4. 保存结果
        speaker_df, gt_df, pred_df = self.save_results(
            batch_data['speaker'], 
            batch_data['listener'], 
            predictions, 
            output_dir
        )
        
        # 5. 创建可视化
        animation_path = os.path.join(output_dir, 'comparison_animation.gif')
        self.create_comparison_animation(speaker_df, gt_df, pred_df, animation_path)
        
        print("🎉 测试完成！")
        return metrics, animation_path


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="简易模型测试")
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='模型checkpoint路径')
    parser.add_argument('--data-csv', type=str, 
                       default='/mnt/iusers01/fatpou01/compsci01/k09562zs/scratch/Face_data/val.csv',
                       help='测试数据CSV路径')
    parser.add_argument('--sample-idx', type=int, default=0,
                       help='测试样本索引')
    parser.add_argument('--output-dir', type=str, default='test_results',
                       help='输出目录')
    parser.add_argument('--device', type=str, default='cuda',
                       help='设备 (cuda/cpu)')
    
    args = parser.parse_args()
    
    # 检查checkpoint文件
    if not os.path.exists(args.checkpoint):
        print(f"❌ Checkpoint文件不存在: {args.checkpoint}")
        return
    
    # 检查数据文件
    if not os.path.exists(args.data_csv):
        print(f"❌ 数据文件不存在: {args.data_csv}")
        return
    
    try:
        # 创建测试器
        tester = SimpleTester(args.checkpoint, args.device)
        
        # 运行测试
        metrics, animation_path = tester.run_test(
            args.data_csv, 
            args.sample_idx, 
            args.output_dir
        )
        
        print("\n" + "="*50)
        print("🎯 测试结果总结")
        print("="*50)
        print(f"输出目录: {args.output_dir}")
        print(f"动画文件: {animation_path}")
        print("="*50)
        
    except Exception as e:
        print(f"❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()