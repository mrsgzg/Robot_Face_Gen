#!/usr/bin/env python3
"""
重叠窗口（滑动窗口）推理测试
实现逐帧输出的流式推理
"""

import os
import time
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import argparse
from tqdm import tqdm
from collections import deque

from facial_reaction_model_old.model import FacialReactionModel
from Data_Set_old import SpeakerListenerDataset


class SlidingWindowTester:
    """重叠窗口推理测试器"""
    
    def __init__(self, checkpoint_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.checkpoint_path = checkpoint_path
        
        # 滑动窗口状态
        self.window_size = None
        self.past_reaction_features = None
        self.past_motion_sample = None
        
        # 性能统计
        self.inference_times = []
        self.frame_predictions = []
        
        # 加载模型
        self.load_model()
        
    def load_model(self):
        """加载模型"""
        print(f"🔄 加载模型从: {self.checkpoint_path}")
        
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        config = checkpoint.get('config', {})
        model_config = config.get('model', {})
        
        self.model = FacialReactionModel(
            feature_dim=model_config.get('feature_dim', 256),
            audio_dim=model_config.get('audio_dim', 384),
            period=model_config.get('period', 25),
            max_seq_len=model_config.get('max_seq_len', 750),
            device=str(self.device),
            window_size=model_config.get('window_size', 16),
            momentum=model_config.get('momentum', 0.9)
        )
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        self.window_size = self.model.window_size
        print(f"✅ 模型加载成功，窗口大小: {self.window_size}")
        
    def reset_states(self, batch_size=1):
        """重置滑动窗口状态"""
        self.past_reaction_features = torch.zeros(
            (batch_size, self.window_size, 158),  # 136+17+3+2
            device=self.device
        )
        self.past_motion_sample = None
        
    def sliding_window_inference(self, sample_data, stride=1, max_frames=None):
        """
        执行滑动窗口推理
        Args:
            sample_data: 测试样本数据
            stride: 滑动步长（1=逐帧，4=每4帧，16=非重叠）
            max_frames: 最大处理帧数
        """
        print(f"🚀 开始滑动窗口推理 (步长: {stride})")
        
        speaker_data = sample_data['speaker']
        listener_data = sample_data['listener']
        total_frames = speaker_data['landmarks'].shape[0]
        
        if max_frames:
            total_frames = min(total_frames, max_frames)
        
        print(f"  总帧数: {total_frames}")
        print(f"  窗口大小: {self.window_size}")
        print(f"  预期输出帧数: {max(0, total_frames - self.window_size + 1)}")
        
        # 重置状态
        self.reset_states()
        self.inference_times.clear()
        self.frame_predictions.clear()
        
        # 滑动窗口推理
        progress_bar = tqdm(
            range(self.window_size, total_frames + 1, stride),
            desc="滑动窗口推理"
        )
        
        for end_frame in progress_bar:
            start_frame = end_frame - self.window_size
            
            # 准备当前窗口数据（包含历史context）
            window_speaker = {}
            for feature_name in ['landmarks', 'au', 'pose', 'gaze', 'audio']:
                if feature_name in speaker_data:
                    # 提供从开始到当前的所有数据，模型内部会截取最后window_size
                    feature_data = speaker_data[feature_name][:end_frame]
                    window_speaker[feature_name] = feature_data.unsqueeze(0).to(self.device)
            
            # 推理
            start_time = time.time()
            
            with torch.no_grad():
                current_predictions, new_reaction_features, new_motion_sample = self.model.inference_step(
                    window_speaker,
                    self.past_reaction_features,
                    self.past_motion_sample
                )
            
            inference_time = time.time() - start_time
            self.inference_times.append(inference_time)
            
            # 更新状态
            self.past_reaction_features = new_reaction_features
            self.past_motion_sample = new_motion_sample
            
            # 保存预测结果（只保存最后一帧或全部，取决于需求）
            # 这里保存整个窗口的预测，但标记为对应的结束帧
            frame_result = {
                'end_frame': end_frame,
                'start_frame': start_frame,
                'predictions': {k: v.cpu().numpy().squeeze(0) for k, v in current_predictions.items()},
                'inference_time': inference_time
            }
            
            self.frame_predictions.append(frame_result)
            
            # 更新进度条
            progress_bar.set_postfix({
                'inf_time': f"{inference_time*1000:.1f}ms",
                'avg_time': f"{np.mean(self.inference_times[-10:])*1000:.1f}ms"
            })
        
        print(f"✅ 滑动窗口推理完成")
        print(f"  处理了 {len(self.frame_predictions)} 个位置")
        print(f"  平均推理时间: {np.mean(self.inference_times)*1000:.2f}ms")
        print(f"  总推理时间: {sum(self.inference_times):.2f}s")
        
        return self.frame_predictions
    
    def compare_stride_effects(self, sample_data, strides=[1, 4, 8, 16], max_frames=100):
        """比较不同步长的效果"""
        print("🔄 比较不同步长的推理效果...")
        
        results = {}
        
        for stride in strides:
            print(f"\n📊 测试步长: {stride}")
            
            start_time = time.time()
            predictions = self.sliding_window_inference(sample_data, stride, max_frames)
            total_time = time.time() - start_time
            
            results[stride] = {
                'predictions': predictions,
                'total_time': total_time,
                'avg_inference_time': np.mean(self.inference_times),
                'num_inferences': len(predictions),
                'throughput': len(predictions) / total_time if total_time > 0 else 0
            }
            
            print(f"  推理次数: {len(predictions)}")
            print(f"  总时间: {total_time:.2f}s")
            print(f"  吞吐量: {results[stride]['throughput']:.1f} inferences/sec")
        
        # 打印对比表格
        print(f"\n📈 步长对比总结:")
        print(f"{'步长':<6} {'推理次数':<8} {'总时间(s)':<10} {'平均时间(ms)':<12} {'吞吐量':<10}")
        print("-" * 60)
        
        for stride, data in results.items():
            print(f"{stride:<6} {data['num_inferences']:<8} {data['total_time']:<10.2f} "
                  f"{data['avg_inference_time']*1000:<12.1f} {data['throughput']:<10.1f}")
        
        return results
    
    def denormalize_features(self, data_dict):
        """反归一化特征"""
        result = {}
        for key, value in data_dict.items():
            if key == 'landmarks':
                result[key] = value * 256.0
            elif key == 'au':
                result[key] = value * 5.0
            else:  # pose, gaze保持原值
                result[key] = value
        return result
    
    def save_complete_results(self, predictions, sample_data, output_dir='sliding_results'):
        """保存完整的滑动窗口结果"""
        print("💾 保存完整滑动窗口结果...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 提取所有特征的预测序列
        pred_sequences = {
            'landmarks': [],
            'au': [],
            'pose': [],
            'gaze': []
        }
        
        frame_indices = []
        
        for pred in predictions:
            frame_indices.append(pred['end_frame'] - 1)  # 转换为0-based索引
            
            for feature_name in pred_sequences.keys():
                if feature_name in pred['predictions']:
                    feature_data = pred['predictions'][feature_name]
                    # 取每个窗口预测的最后一帧
                    if len(feature_data.shape) > 1:
                        feature_data = feature_data[-1]
                    pred_sequences[feature_name].append(feature_data)
        
        # 转换为numpy数组并反归一化
        pred_denorm = {}
        for feature_name, feature_list in pred_sequences.items():
            if feature_list:
                feature_array = np.array(feature_list)
                pred_denorm[feature_name] = feature_array
        
        pred_denorm = self.denormalize_features(pred_denorm)
        
        # 获取对应的GT数据
        gt_denorm = {}
        speaker_denorm = {}
        
        for feature_name in ['landmarks', 'au', 'pose', 'gaze']:
            if feature_name in sample_data['listener']:
                gt_data = sample_data['listener'][feature_name].numpy()
                gt_aligned = gt_data[frame_indices]  # 只取有预测的帧
                gt_denorm[feature_name] = gt_aligned
            
            if feature_name in sample_data['speaker']:
                speaker_data = sample_data['speaker'][feature_name].numpy()
                speaker_aligned = speaker_data[frame_indices]  # 只取有预测的帧
                speaker_denorm[feature_name] = speaker_aligned
        
        gt_denorm = self.denormalize_features(gt_denorm)
        speaker_denorm = self.denormalize_features(speaker_denorm)
        
        # 保存所有特征为CSV
        def save_all_features_csv(features_dict, prefix, output_dir):
            for feature_name, feature_data in features_dict.items():
                filename = os.path.join(output_dir, f'sliding_{prefix}_{feature_name}.csv')
                
                if feature_name == 'landmarks':
                    columns = [f'x_{i}' for i in range(68)] + [f'y_{i}' for i in range(68)]
                elif feature_name == 'au':
                    columns = [f'AU_{i}' for i in range(feature_data.shape[1])]
                elif feature_name == 'pose':
                    columns = ['pose_Rx', 'pose_Ry', 'pose_Rz']
                elif feature_name == 'gaze':
                    columns = ['gaze_angle_x', 'gaze_angle_y']
                
                df = pd.DataFrame(feature_data, columns=columns)
                df.insert(0, 'frame', frame_indices)
                df.to_csv(filename, index=False)
        
        # 保存所有特征
        save_all_features_csv(speaker_denorm, 'speaker', output_dir)
        save_all_features_csv(gt_denorm, 'gt_listener', output_dir)
        save_all_features_csv(pred_denorm, 'predicted_listener', output_dir)
        
        print(f"💾 完整结果已保存到 {output_dir}:")
        print(f"   Speaker特征: sliding_speaker_*.csv")
        print(f"   GT Listener特征: sliding_gt_listener_*.csv")
        print(f"   预测Listener特征: sliding_predicted_listener_*.csv")
        
        return pred_denorm, gt_denorm, speaker_denorm, frame_indices
    
    def calculate_metrics(self, pred_denorm, gt_denorm):
        """计算详细的评估指标"""
        print("📊 计算评估指标...")
        
        metrics = {}
        
        for feature_name in ['landmarks', 'au', 'pose', 'gaze']:
            if feature_name in pred_denorm and feature_name in gt_denorm:
                pred_data = pred_denorm[feature_name]
                gt_data = gt_denorm[feature_name]
                
                # 确保长度一致
                min_len = min(len(pred_data), len(gt_data))
                pred_data = pred_data[:min_len]
                gt_data = gt_data[:min_len]
                
                # 计算基本指标
                mse = np.mean((pred_data - gt_data) ** 2)
                mae = np.mean(np.abs(pred_data - gt_data))
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
                
                # landmarks的详细分析
                if feature_name == 'landmarks':
                    x_errors = np.abs(pred_data[:, :68] - gt_data[:, :68])
                    y_errors = np.abs(pred_data[:, 68:] - gt_data[:, 68:])
                    
                    metrics[feature_name]['x_mae'] = np.mean(x_errors)
                    metrics[feature_name]['y_mae'] = np.mean(y_errors)
                    metrics[feature_name]['max_point_error'] = np.max(np.sqrt(x_errors**2 + y_errors**2))
                    
                    print(f"    X坐标MAE: {metrics[feature_name]['x_mae']:.6f}")
                    print(f"    Y坐标MAE: {metrics[feature_name]['y_mae']:.6f}")
                    print(f"    最大点误差: {metrics[feature_name]['max_point_error']:.6f}")
        
        return metrics
    
    def analyze_sliding_window_output(self, predictions, sample_data, output_dir='sliding_results'):
        """分析滑动窗口输出（增强版）"""
        print("📊 分析滑动窗口输出...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存完整结果
        pred_denorm, gt_denorm, speaker_denorm, frame_indices = self.save_complete_results(
            predictions, sample_data, output_dir
        )
        
        # 计算评估指标
        metrics = self.calculate_metrics(pred_denorm, gt_denorm)
        
        # 分析平滑性（使用landmarks）
        if 'landmarks' in pred_denorm:
            landmarks_sequence = pred_denorm['landmarks']
            
            if len(landmarks_sequence) > 1:
                frame_diffs = np.diff(landmarks_sequence, axis=0)
                smoothness = np.mean(np.abs(frame_diffs))
                
                print(f"\n🎯 平滑性分析:")
                print(f"  输出帧数: {len(landmarks_sequence)}")
                print(f"  帧间平滑性: {smoothness:.6f}")
                
                # 保存平滑性分析图
                self.create_smoothness_plots(frame_diffs, output_dir)
                
                # 创建调试静态图
                self.create_debug_static_plot(
                    speaker_denorm, gt_denorm, pred_denorm, frame_indices, output_dir
                )
                
                return {
                    'smoothness': smoothness,
                    'output_frames': len(landmarks_sequence),
                    'metrics': metrics,
                    'frame_indices': frame_indices
                }
        
        return {
            'smoothness': 0,
            'output_frames': 0,
            'metrics': metrics,
            'frame_indices': frame_indices
        }
    
    def create_smoothness_plots(self, frame_diffs, output_dir):
        """创建平滑性分析图"""
        plt.figure(figsize=(15, 10))
        
        # 帧间差异趋势
        plt.subplot(2, 3, 1)
        plt.plot(np.abs(frame_diffs).mean(axis=1))
        plt.title('帧间差异趋势')
        plt.xlabel('帧索引')
        plt.ylabel('平均绝对差异')
        plt.grid(True, alpha=0.3)
        
        # 帧间差异分布
        plt.subplot(2, 3, 2)
        plt.hist(np.abs(frame_diffs).flatten(), bins=50, alpha=0.7)
        plt.title('帧间差异分布')
        plt.xlabel('绝对差异')
        plt.ylabel('频次')
        plt.grid(True, alpha=0.3)
        
        # X坐标差异
        plt.subplot(2, 3, 3)
        x_diffs = frame_diffs[:, :68]
        plt.plot(np.abs(x_diffs).mean(axis=1), color='blue', alpha=0.7)
        plt.title('X坐标帧间差异')
        plt.xlabel('帧索引')
        plt.ylabel('平均绝对差异')
        plt.grid(True, alpha=0.3)
        
        # Y坐标差异
        plt.subplot(2, 3, 4)
        y_diffs = frame_diffs[:, 68:]
        plt.plot(np.abs(y_diffs).mean(axis=1), color='red', alpha=0.7)
        plt.title('Y坐标帧间差异')
        plt.xlabel('帧索引')
        plt.ylabel('平均绝对差异')
        plt.grid(True, alpha=0.3)
        
        # 运动速度分析
        plt.subplot(2, 3, 5)
        motion_speed = np.sqrt(np.sum(frame_diffs**2, axis=1))
        plt.plot(motion_speed, color='purple', alpha=0.7)
        plt.title('运动速度趋势')
        plt.xlabel('帧索引')
        plt.ylabel('运动速度')
        plt.grid(True, alpha=0.3)
        
        # 关键点运动热图
        plt.subplot(2, 3, 6)
        point_motion = np.sqrt(frame_diffs[:, :68]**2 + frame_diffs[:, 68:]**2)
        plt.imshow(point_motion.T, aspect='auto', cmap='hot', interpolation='nearest')
        plt.title('关键点运动热图')
        plt.xlabel('帧索引')
        plt.ylabel('关键点索引')
        plt.colorbar(label='运动幅度')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'smoothness_analysis.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 平滑性分析图已保存")
    
    def create_debug_static_plot(self, speaker_denorm, gt_denorm, pred_denorm, frame_indices, output_dir):
        """创建调试静态图"""
        print("🔍 创建调试静态图...")
        
        # 选择第一帧进行分析
        if len(frame_indices) == 0:
            return
            
        first_idx = 0
        
        speaker_x = speaker_denorm['landmarks'][first_idx, :68]
        speaker_y = speaker_denorm['landmarks'][first_idx, 68:]
        gt_x = gt_denorm['landmarks'][first_idx, :68]
        gt_y = gt_denorm['landmarks'][first_idx, 68:]
        pred_x = pred_denorm['landmarks'][first_idx, :68]
        pred_y = pred_denorm['landmarks'][first_idx, 68:]
        
        # 创建2x2子图
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 计算坐标范围
        all_x = np.concatenate([speaker_x, gt_x, pred_x])
        all_y = np.concatenate([speaker_y, gt_y, pred_y])
        x_range = np.max(all_x) - np.min(all_x)
        y_range = np.max(all_y) - np.min(all_y)
        margin = 0.1 * max(x_range, y_range)
        
        # 左上：三路对比
        ax1 = axes[0, 0]
        ax1.set_title(f'第一帧对比 - 帧 {frame_indices[first_idx]}', fontsize=14, fontweight='bold')
        ax1.scatter(speaker_x, speaker_y, c='blue', s=30, alpha=0.7, label='Speaker')
        ax1.scatter(gt_x, gt_y, c='green', s=30, alpha=0.7, label='GT Listener')
        ax1.scatter(pred_x, pred_y, c='red', s=30, alpha=0.7, label='Predicted')
        
        self.draw_face_connections(ax1, speaker_x, speaker_y, 'blue')
        self.draw_face_connections(ax1, gt_x, gt_y, 'green')
        self.draw_face_connections(ax1, pred_x, pred_y, 'red')
        
        ax1.set_xlim(np.min(all_x) - margin, np.max(all_x) + margin)
        ax1.set_ylim(np.max(all_y) + margin, np.min(all_y) - margin)
        ax1.set_aspect('equal')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 右上：误差分析
        ax2 = axes[0, 1]
        ax2.set_title('预测误差热图', fontsize=14, fontweight='bold')
        
        errors = np.sqrt((gt_x - pred_x)**2 + (gt_y - pred_y)**2)
        scatter = ax2.scatter(gt_x, gt_y, c=errors, s=50, cmap='Reds', alpha=0.8)
        plt.colorbar(scatter, ax=ax2, label='预测误差')
        
        ax2.set_xlim(gt_x.min() - 10, gt_x.max() + 10)
        ax2.set_ylim(gt_y.max() + 10, gt_y.min() - 10)
        ax2.set_aspect('equal')
        ax2.grid(True, alpha=0.3)
        
        # 添加误差统计
        error_stats = f"平均误差: {errors.mean():.2f}\n最大误差: {errors.max():.2f}\n误差标准差: {errors.std():.2f}"
        ax2.text(0.02, 0.98, error_stats, transform=ax2.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        # 左下：数据统计
        ax3 = axes[1, 0]
        ax3.axis('off')
        ax3.set_title('滑动窗口统计', fontsize=14, fontweight='bold')
        
        stats_text = f"""
滑动窗口推理统计:

总处理帧数: {len(frame_indices)}
平均推理时间: {np.mean(self.inference_times)*1000:.2f}ms
最大推理时间: {np.max(self.inference_times)*1000:.2f}ms
最小推理时间: {np.min(self.inference_times)*1000:.2f}ms

坐标范围分析:
Speaker: X[{speaker_x.min():.1f}, {speaker_x.max():.1f}], Y[{speaker_y.min():.1f}, {speaker_y.max():.1f}]
GT:      X[{gt_x.min():.1f}, {gt_x.max():.1f}], Y[{gt_y.min():.1f}, {gt_y.max():.1f}]
Pred:    X[{pred_x.min():.1f}, {pred_x.max():.1f}], Y[{pred_y.min():.1f}, {pred_y.max():.1f}]
        """
        
        ax3.text(0.05, 0.95, stats_text, transform=ax3.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace')
        
        # 右下：推理时间趋势
        ax4 = axes[1, 1]
        ax4.set_title('推理时间趋势', fontsize=14, fontweight='bold')
        
        times_ms = np.array(self.inference_times) * 1000
        ax4.plot(times_ms, 'b-', alpha=0.7)
        ax4.axhline(y=np.mean(times_ms), color='r', linestyle='--', 
                   label=f'平均: {np.mean(times_ms):.1f}ms')
        ax4.set_xlabel('窗口索引')
        ax4.set_ylabel('推理时间 (毫秒)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        debug_path = os.path.join(output_dir, 'sliding_debug_first_frame.png')
        plt.savefig(debug_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 调试静态图已保存到: {debug_path}")
    
    def create_comparison_animation(self, predictions, sample_data, output_path='sliding_results/sliding_comparison_300.gif'):
        """创建三路对比动画（Speaker + GT + Predicted）"""
        print("🎬 创建滑动窗口三路对比动画...")
        
        # 提取预测landmarks序列
        pred_landmarks = []
        frame_indices = []
        
        for pred in predictions:
            landmarks = pred['predictions']['landmarks']
            if len(landmarks.shape) > 1:
                landmarks = landmarks[-1]  # 取最后一帧
            pred_landmarks.append(landmarks * 256.0)  # 反归一化
            frame_indices.append(pred['end_frame'] - 1)  # 转换为0-based索引
        
        pred_landmarks = np.array(pred_landmarks)
        
        # 获取对应的真实landmarks和speaker landmarks
        gt_landmarks = sample_data['listener']['landmarks'].numpy() * 256.0
        speaker_landmarks = sample_data['speaker']['landmarks'].numpy() * 256.0
        
        # 只保留有预测的帧
        gt_landmarks_aligned = gt_landmarks[frame_indices]
        speaker_landmarks_aligned = speaker_landmarks[frame_indices]
        
        n_frames = min(len(pred_landmarks), 600)  # 限制动画长度
        
        # 提取坐标
        pred_x = pred_landmarks[:n_frames, :68]
        pred_y = pred_landmarks[:n_frames, 68:]
        gt_x = gt_landmarks_aligned[:n_frames, :68]
        gt_y = gt_landmarks_aligned[:n_frames, 68:]
        speaker_x = speaker_landmarks_aligned[:n_frames, :68]
        speaker_y = speaker_landmarks_aligned[:n_frames, 68:]
        
        # 计算坐标范围
        all_x = np.concatenate([pred_x.flatten(), gt_x.flatten(), speaker_x.flatten()])
        all_y = np.concatenate([pred_y.flatten(), gt_y.flatten(), speaker_y.flatten()])
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
                ax.set_ylim(y_max + margin, y_min - margin)
                ax.set_aspect('equal')
                ax.grid(True, alpha=0.3)
            
            # 设置标题
            ax1.set_title(f'Speaker - 帧 {frame_indices[frame]}', fontsize=12)
            ax2.set_title(f'GT Listener - 帧 {frame_indices[frame]}', fontsize=12)
            ax3.set_title(f'滑动窗口预测 - 帧 {frame_indices[frame]}', fontsize=12)
            
            # 绘制特征点
            ax1.scatter(speaker_x[frame], speaker_y[frame], c='blue', s=15, alpha=0.7)
            ax2.scatter(gt_x[frame], gt_y[frame], c='green', s=15, alpha=0.7)
            ax3.scatter(pred_x[frame], pred_y[frame], c='red', s=15, alpha=0.7)
            
            # 绘制面部连接线
            self.draw_face_connections(ax1, speaker_x[frame], speaker_y[frame], 'blue')
            self.draw_face_connections(ax2, gt_x[frame], gt_y[frame], 'green')
            self.draw_face_connections(ax3, pred_x[frame], pred_y[frame], 'red')
        
        anim = animation.FuncAnimation(fig, animate, frames=n_frames, interval=200, repeat=True)
        anim.save(output_path, writer='pillow', fps=25, dpi=80)
        plt.close()
        
        print(f"✅ 滑动窗口三路对比动画已保存到: {output_path}")
        return output_path
    
    def create_performance_analysis(self, output_dir):
        """创建性能分析图表"""
        print("📈 创建性能分析图表...")
        
        if not self.inference_times:
            print("⚠️ 没有推理时间数据")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 推理时间趋势
        ax1 = axes[0, 0]
        times_ms = np.array(self.inference_times) * 1000
        ax1.plot(times_ms, 'b-', alpha=0.7)
        ax1.axhline(y=np.mean(times_ms), color='r', linestyle='--', 
                   label=f'平均值: {np.mean(times_ms):.2f}ms')
        ax1.set_title('滑动窗口推理时间趋势')
        ax1.set_xlabel('窗口索引')
        ax1.set_ylabel('推理时间 (毫秒)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 推理时间分布
        ax2 = axes[0, 1]
        ax2.hist(times_ms, bins=20, alpha=0.7, edgecolor='black')
        ax2.axvline(x=np.mean(times_ms), color='r', linestyle='--',
                   label=f'平均值: {np.mean(times_ms):.2f}ms')
        ax2.set_title('推理时间分布')
        ax2.set_xlabel('推理时间 (毫秒)')
        ax2.set_ylabel('频次')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 累积推理时间
        ax3 = axes[1, 0]
        cumulative_time = np.cumsum(self.inference_times)
        ax3.plot(cumulative_time, 'g-', alpha=0.7)
        ax3.set_title('累积推理时间')
        ax3.set_xlabel('窗口索引')
        ax3.set_ylabel('累积时间 (秒)')
        ax3.grid(True, alpha=0.3)
        
        # 性能统计表
        ax4 = axes[1, 1]
        ax4.axis('off')
        ax4.set_title('性能统计总结', fontsize=14, fontweight='bold')
        
        stats_text = f"""
滑动窗口推理性能统计:

总窗口数: {len(self.inference_times)}
总推理时间: {sum(self.inference_times):.2f}s
平均推理时间: {np.mean(self.inference_times)*1000:.2f}ms
标准差: {np.std(self.inference_times)*1000:.2f}ms
最小推理时间: {np.min(self.inference_times)*1000:.2f}ms
最大推理时间: {np.max(self.inference_times)*1000:.2f}ms

吞吐量: {len(self.inference_times)/sum(self.inference_times):.1f} windows/sec
理论延迟: {self.window_size/25*1000:.0f}ms ({self.window_size}帧@25fps)
        """
        
        ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        performance_path = os.path.join(output_dir, 'sliding_performance_analysis.png')
        plt.savefig(performance_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 性能分析图表已保存到: {performance_path}")
        return performance_path
    
    def draw_face_connections(self, ax, x_coords, y_coords, color='blue'):
        """绘制面部连接线"""
        connections = {
            'jaw': list(range(0, 17)),
            'right_eyebrow': list(range(17, 22)),
            'left_eyebrow': list(range(22, 27)),
            'nose_bridge': list(range(27, 31)),
            'nose_tip': list(range(31, 36)),
            'right_eye': list(range(36, 42)) + [36],
            'left_eye': list(range(42, 48)) + [42],
            'outer_lip': list(range(48, 60)) + [48],
            'inner_lip': list(range(60, 68)) + [60]
        }
        
        for name, indices in connections.items():
            if len(indices) > 1:
                try:
                    x_line = [x_coords[idx] for idx in indices]
                    y_line = [y_coords[idx] for idx in indices]
                    ax.plot(x_line, y_line, color=color, linewidth=1, alpha=0.6)
                except IndexError:
                    continue


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="重叠窗口推理测试")
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='模型checkpoint路径')
    parser.add_argument('--data-csv', type=str,
                       default='/mnt/iusers01/fatpou01/compsci01/k09562zs/scratch/Face_data/val.csv',
                       help='测试数据CSV路径')
    parser.add_argument('--sample-idx', type=int, default=0,
                       help='测试样本索引')
    parser.add_argument('--stride', type=int, default=1,
                       help='滑动步长 (1=逐帧, 4=每4帧, 16=非重叠)')
    parser.add_argument('--max-frames', type=int, default=700,
                       help='最大处理帧数')
    parser.add_argument('--compare-strides', action='store_true',
                       help='比较不同步长的效果')
    parser.add_argument('--device', type=str, default='cuda',
                       help='设备')
    
    args = parser.parse_args()
    
    # 检查文件
    if not os.path.exists(args.checkpoint):
        print(f"❌ Checkpoint文件不存在: {args.checkpoint}")
        return
    
    if not os.path.exists(args.data_csv):
        print(f"❌ 数据文件不存在: {args.data_csv}")
        return
    
    try:
        # 创建测试器
        tester = SlidingWindowTester(args.checkpoint, args.device)
        
        # 加载测试数据
        print("📊 加载测试数据...")
        dataset = SpeakerListenerDataset(args.data_csv)
        sample = dataset[args.sample_idx]
        
        if sample is None:
            print("❌ 无法加载测试样本")
            return
        
        print(f"✅ 测试样本加载成功 (总帧数: {sample['speaker']['landmarks'].shape[0]})")
        
        if args.compare_strides:
            # 比较不同步长
            results = tester.compare_stride_effects(sample)
            
            print(f"\n🎯 不同步长对比完成！")
            
        else:
            # 单一步长测试
            predictions = tester.sliding_window_inference(sample, args.stride, args.max_frames)
            
            # 完整分析结果
            analysis = tester.analyze_sliding_window_output(predictions, sample)
            
            # 创建性能分析
            performance_path = tester.create_performance_analysis('sliding_results')
            
            # 创建三路对比动画
            animation_path = tester.create_comparison_animation(predictions, sample)
            
            print(f"\n🎯 滑动窗口测试结果总结:")
            print("="*60)
            print(f"🚀 测试配置:")
            print(f"  滑动步长: {args.stride}")
            print(f"  窗口大小: {tester.window_size}")
            print(f"  处理帧数: {args.max_frames}")
            
            print(f"\n📊 输出统计:")
            print(f"  输出帧数: {analysis['output_frames']}")
            print(f"  帧间平滑性: {analysis['smoothness']:.6f}")
            print(f"  处理窗口数: {len(predictions)}")
            
            print(f"\n⚡ 性能统计:")
            if tester.inference_times:
                print(f"  平均推理时间: {np.mean(tester.inference_times)*1000:.2f}ms")
                print(f"  最大推理时间: {np.max(tester.inference_times)*1000:.2f}ms")
                print(f"  最小推理时间: {np.min(tester.inference_times)*1000:.2f}ms")
                print(f"  总推理时间: {sum(tester.inference_times):.2f}s")
                print(f"  推理吞吐量: {len(tester.inference_times)/sum(tester.inference_times):.1f} windows/sec")
            
            print(f"\n📈 评估指标:")
            if 'metrics' in analysis:
                for feature_name, metrics in analysis['metrics'].items():
                    print(f"  {feature_name.upper()}:")
                    print(f"    MAE: {metrics['mae']:.6f}")
                    print(f"    RMSE: {metrics['rmse']:.6f}")
            
            print(f"\n📁 生成文件:")
            print(f"  结果目录: sliding_results/")
            print(f"  三路对比动画: {animation_path}")
            print(f"  性能分析图: {performance_path}")
            print(f"  调试静态图: sliding_results/sliding_debug_first_frame.png")
            print(f"  平滑性分析: sliding_results/smoothness_analysis.png")
            print(f"  CSV结果文件: sliding_results/sliding_*_*.csv")
            
            print("="*60)
            
            # 保存测试总结
            summary = {
                'config': {
                    'stride': args.stride,
                    'window_size': tester.window_size,
                    'max_frames': args.max_frames,
                    'sample_idx': args.sample_idx
                },
                'performance': {
                    'avg_inference_time_ms': float(np.mean(tester.inference_times) * 1000) if tester.inference_times else 0,
                    'total_inference_time_s': float(sum(tester.inference_times)) if tester.inference_times else 0,
                    'throughput_windows_per_sec': float(len(tester.inference_times)/sum(tester.inference_times)) if tester.inference_times and sum(tester.inference_times) > 0 else 0
                },
                'results': {
                    'output_frames': analysis['output_frames'],
                    'smoothness': float(analysis['smoothness']),
                    'num_windows_processed': len(predictions)
                },
                'metrics': {k: {mk: float(mv) for mk, mv in v.items()} for k, v in analysis.get('metrics', {}).items()},
                'files': {
                    'animation': animation_path,
                    'performance_plot': performance_path,
                    'debug_plot': 'sliding_results/sliding_debug_first_frame.png',
                    'smoothness_plot': 'sliding_results/smoothness_analysis.png'
                }
            }
            
            # 保存测试总结JSON
            import json
            summary_path = 'sliding_results/test_summary.json'
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            
            print(f"📋 测试总结已保存: {summary_path}")
            
            return summary
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()