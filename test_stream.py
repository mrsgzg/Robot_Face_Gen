#!/usr/bin/env python3
"""
流式推理测试脚本 - 模拟实时数据流处理
支持窗口级流式推理、性能分析、实时可视化
"""

import os
import time
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
import argparse
from tqdm import tqdm
from collections import deque
from typing import Dict, List, Tuple, Optional
import threading
import queue

from facial_reaction_model.model import FacialReactionModel, OnlineInferenceManager
from scratch.Robot_Face_Gen.Data_Set_old import SpeakerListenerDataset


class StreamingTester:
    """流式推理测试器"""
    
    def __init__(self, checkpoint_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.checkpoint_path = checkpoint_path
        
        # 性能统计
        self.inference_times = []
        self.memory_usage = []
        self.window_delays = []
        
        # 流式数据缓存
        self.streaming_results = {
            'timestamps': [],
            'predictions': {
                'landmarks': [],
                'au': [],
                'pose': [],
                'gaze': []
            },
            'gt_data': {
                'landmarks': [],
                'au': [],
                'pose': [],
                'gaze': []
            },
            'speaker_data': {
                'landmarks': [],
                'au': [],
                'pose': [],
                'gaze': []
            }
        }
        
        # 加载模型
        self.load_model()
        
    def load_model(self):
        """加载训练好的模型"""
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
        
    def load_test_sample(self, data_csv, sample_idx=0):
        """加载测试样本"""
        print(f"📊 加载测试数据: {data_csv}, 样本索引: {sample_idx}")
        
        dataset = SpeakerListenerDataset(data_csv)
        
        if sample_idx >= len(dataset):
            sample_idx = 0
            print(f"⚠️ 样本索引超出范围，使用索引 0")
        
        sample = dataset[sample_idx]
        if sample is None:
            raise ValueError(f"无法加载样本 {sample_idx}")
        
        # 不转换为batch格式，保持原始形状用于流式处理
        return sample
    
    def simulate_data_stream(self, sample_data, stream_fps=25, real_time=False):
        """
        模拟数据流
        Args:
            sample_data: 完整的样本数据
            stream_fps: 流的帧率
            real_time: 是否按真实时间流式处理
        Returns:
            生成器，产出每个窗口的数据
        """
        speaker_data = sample_data['speaker']
        listener_data = sample_data['listener']
        
        total_frames = speaker_data['landmarks'].shape[0]
        frame_interval = 1.0 / stream_fps if real_time else 0
        
        print(f"🎬 开始模拟数据流: {total_frames}帧, {stream_fps}fps")
        
        # 按窗口生成数据
        for start_idx in range(0, total_frames, self.window_size):
            end_idx = min(start_idx + self.window_size, total_frames)
            window_size = end_idx - start_idx
            
            # 准备当前窗口的数据
            window_speaker = {}
            window_listener = {}
            
            for feature_name in ['landmarks', 'au', 'pose', 'gaze', 'audio']:
                if feature_name in speaker_data:
                    window_speaker[feature_name] = speaker_data[feature_name][start_idx:end_idx]
                if feature_name in listener_data:
                    window_listener[feature_name] = listener_data[feature_name][start_idx:end_idx]
            
            # 转换为batch格式供模型使用
            batch_speaker = {}
            for key, value in window_speaker.items():
                batch_speaker[key] = value.unsqueeze(0).to(self.device)
            
            yield {
                'window_idx': start_idx // self.window_size,
                'start_frame': start_idx,
                'end_frame': end_idx,
                'window_size': window_size,
                'speaker_data': batch_speaker,
                'gt_listener': window_listener,
                'timestamp': time.time()
            }
            
            # 实时模式下按帧率等待
            if real_time and window_size == self.window_size:
                time.sleep(frame_interval * self.window_size)
    
    def streaming_inference(self, sample_data, output_dir='streaming_results', 
                          stream_fps=25, real_time=False, max_windows=None):
        """
        执行流式推理
        Args:
            sample_data: 测试样本数据
            output_dir: 输出目录
            stream_fps: 流帧率
            real_time: 是否实时处理
            max_windows: 最大处理窗口数（None为全部）
        """
        print(f"🚀 开始流式推理 (实时模式: {real_time})")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 创建在线推理管理器
        online_manager = OnlineInferenceManager(self.model, self.device)
        
        # 重置统计信息
        self.inference_times.clear()
        self.memory_usage.clear()
        self.window_delays.clear()
        
        # 重置流式结果
        for key in self.streaming_results:
            if key == 'predictions' or key == 'gt_data' or key == 'speaker_data':
                for feature in self.streaming_results[key]:
                    self.streaming_results[key][feature].clear()
            else:
                self.streaming_results[key].clear()
        
        processed_windows = 0
        
        # 开始流式处理
        for window_data in self.simulate_data_stream(sample_data, stream_fps, real_time):
            if max_windows and processed_windows >= max_windows:
                break
                
            window_idx = window_data['window_idx']
            start_frame = window_data['start_frame']
            
            print(f"🎯 处理窗口 {window_idx}: 帧 {start_frame}-{window_data['end_frame']}")
            
            # 记录推理开始时间
            inference_start = time.time()
            
            # 记录内存使用
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                memory_before = torch.cuda.memory_allocated() / 1024**2
            
            # 执行推理
            with torch.no_grad():
                window_predictions = online_manager.process_window(window_data['speaker_data'])
            
            # 记录推理时间
            inference_time = time.time() - inference_start
            self.inference_times.append(inference_time)
            
            # 记录内存使用
            if torch.cuda.is_available():
                memory_after = torch.cuda.memory_allocated() / 1024**2
                self.memory_usage.append(memory_after - memory_before)
            
            # 计算窗口延迟
            window_delay = window_data['timestamp'] - inference_start
            self.window_delays.append(window_delay)
            
            # 保存预测结果
            self.save_window_results(
                window_data, window_predictions, 
                window_idx, start_frame
            )
            
            processed_windows += 1
            
            # 实时显示进度
            if processed_windows % 5 == 0:
                avg_inference_time = np.mean(self.inference_times[-5:])
                print(f"    📊 最近5窗口平均推理时间: {avg_inference_time*1000:.2f}ms")
        
        print(f"✅ 流式推理完成，处理了 {processed_windows} 个窗口")
        
        # 保存性能统计
        self.save_performance_stats(output_dir)
        
        # 保存完整结果
        self.save_complete_results(output_dir)
        
        return processed_windows
    
    def save_window_results(self, window_data, predictions, window_idx, start_frame):
        """保存单个窗口的结果"""
        # 反归一化预测结果
        pred_denorm = self.denormalize_features({
            k: v.cpu().numpy().squeeze(0) for k, v in predictions.items()
        })
        
        # 反归一化真实值
        gt_denorm = self.denormalize_features(window_data['gt_listener'])
        
        # 反归一化speaker数据
        speaker_denorm = self.denormalize_features({
            k: v.cpu().numpy().squeeze(0) if torch.is_tensor(v) else v.numpy()
            for k, v in window_data['speaker_data'].items() if k != 'audio'
        })
        
        # 保存到流式结果中
        self.streaming_results['timestamps'].append(window_data['timestamp'])
        
        for feature_name in ['landmarks', 'au', 'pose', 'gaze']:
            if feature_name in pred_denorm:
                self.streaming_results['predictions'][feature_name].append(pred_denorm[feature_name])
            if feature_name in gt_denorm:
                self.streaming_results['gt_data'][feature_name].append(gt_denorm[feature_name])
            if feature_name in speaker_denorm:
                self.streaming_results['speaker_data'][feature_name].append(speaker_denorm[feature_name])
    
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
    
    def save_performance_stats(self, output_dir):
        """保存性能统计"""
        stats = {
            'total_windows': len(self.inference_times),
            'total_inference_time': sum(self.inference_times),
            'avg_inference_time': np.mean(self.inference_times),
            'std_inference_time': np.std(self.inference_times),
            'min_inference_time': np.min(self.inference_times),
            'max_inference_time': np.max(self.inference_times),
            'window_size': self.window_size,
            'theoretical_delay': self.window_size / 25.0,  # 假设25fps
        }
        
        if self.memory_usage:
            stats.update({
                'avg_memory_usage': np.mean(self.memory_usage),
                'max_memory_usage': np.max(self.memory_usage),
            })
        
        # 保存统计JSON
        import json
        stats_path = os.path.join(output_dir, 'performance_stats.json')
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        # 打印统计信息
        print(f"\n📊 性能统计:")
        print(f"  总窗口数: {stats['total_windows']}")
        print(f"  平均推理时间: {stats['avg_inference_time']*1000:.2f}ms")
        print(f"  推理时间标准差: {stats['std_inference_time']*1000:.2f}ms")
        print(f"  最小推理时间: {stats['min_inference_time']*1000:.2f}ms")
        print(f"  最大推理时间: {stats['max_inference_time']*1000:.2f}ms")
        print(f"  理论延迟: {stats['theoretical_delay']*1000:.0f}ms ({self.window_size}帧@25fps)")
        
        if self.memory_usage:
            print(f"  平均内存使用: {stats['avg_memory_usage']:.2f}MB")
            print(f"  最大内存使用: {stats['max_memory_usage']:.2f}MB")
        
        # 创建性能图表
        self.create_performance_plots(output_dir)
    
    def save_complete_results(self, output_dir):
        """保存完整的流式结果"""
        print("💾 保存完整流式结果...")
        
        # 将窗口级结果拼接成完整序列
        complete_results = {}
        
        for role in ['predictions', 'gt_data', 'speaker_data']:
            complete_results[role] = {}
            for feature_name in ['landmarks', 'au', 'pose', 'gaze']:
                if self.streaming_results[role][feature_name]:
                    # 拼接所有窗口的结果
                    feature_data = np.concatenate(
                        self.streaming_results[role][feature_name], axis=0
                    )
                    complete_results[role][feature_name] = feature_data
        
        # 保存为CSV文件
        for role in complete_results:
            for feature_name, feature_data in complete_results[role].items():
                filename = os.path.join(output_dir, f'streaming_{role}_{feature_name}.csv')
                
                if feature_name == 'landmarks':
                    columns = [f'x_{i}' for i in range(68)] + [f'y_{i}' for i in range(68)]
                elif feature_name == 'au':
                    columns = [f'AU_{i}' for i in range(feature_data.shape[1])]
                elif feature_name == 'pose':
                    columns = ['pose_Rx', 'pose_Ry', 'pose_Rz']
                elif feature_name == 'gaze':
                    columns = ['gaze_angle_x', 'gaze_angle_y']
                
                df = pd.DataFrame(feature_data, columns=columns)
                df.insert(0, 'frame', range(len(df)))
                df.to_csv(filename, index=False)
        
        return complete_results
    
    def create_performance_plots(self, output_dir):
        """创建性能分析图表"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 推理时间趋势
        ax1 = axes[0, 0]
        ax1.plot(self.inference_times, 'b-', alpha=0.7)
        ax1.axhline(y=np.mean(self.inference_times), color='r', linestyle='--', 
                   label=f'平均值: {np.mean(self.inference_times)*1000:.2f}ms')
        ax1.set_title('推理时间趋势')
        ax1.set_xlabel('窗口索引')
        ax1.set_ylabel('推理时间 (秒)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 推理时间分布
        ax2 = axes[0, 1]
        ax2.hist(np.array(self.inference_times) * 1000, bins=20, alpha=0.7, edgecolor='black')
        ax2.axvline(x=np.mean(self.inference_times) * 1000, color='r', linestyle='--',
                   label=f'平均值: {np.mean(self.inference_times)*1000:.2f}ms')
        ax2.set_title('推理时间分布')
        ax2.set_xlabel('推理时间 (毫秒)')
        ax2.set_ylabel('频次')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 内存使用（如果有的话）
        ax3 = axes[1, 0]
        if self.memory_usage:
            ax3.plot(self.memory_usage, 'g-', alpha=0.7)
            ax3.axhline(y=np.mean(self.memory_usage), color='r', linestyle='--',
                       label=f'平均值: {np.mean(self.memory_usage):.2f}MB')
            ax3.set_title('内存使用趋势')
            ax3.set_xlabel('窗口索引')
            ax3.set_ylabel('内存使用 (MB)')
            ax3.legend()
        else:
            ax3.text(0.5, 0.5, '无内存使用数据', ha='center', va='center',
                    transform=ax3.transAxes, fontsize=14)
            ax3.set_title('内存使用趋势')
        ax3.grid(True, alpha=0.3)
        
        # 延迟分析
        ax4 = axes[1, 1]
        theoretical_delay = self.window_size / 25.0 * 1000  # 转换为毫秒
        actual_delays = np.array(self.inference_times) * 1000
        
        ax4.plot(actual_delays, 'b-', alpha=0.7, label='实际推理时间')
        ax4.axhline(y=theoretical_delay, color='r', linestyle='--',
                   label=f'理论延迟: {theoretical_delay:.0f}ms')
        ax4.axhline(y=np.mean(actual_delays), color='g', linestyle='--',
                   label=f'平均推理时间: {np.mean(actual_delays):.2f}ms')
        ax4.set_title('延迟分析')
        ax4.set_xlabel('窗口索引')
        ax4.set_ylabel('时间 (毫秒)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'performance_analysis.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 性能分析图表已保存到: {os.path.join(output_dir, 'performance_analysis.png')}")
    
    def create_streaming_animation(self, output_dir, max_frames=300):
        """创建流式处理动画"""
        print("🎬 创建流式处理动画...")
        
        # 获取完整结果
        pred_landmarks = np.concatenate(self.streaming_results['predictions']['landmarks'], axis=0)
        gt_landmarks = np.concatenate(self.streaming_results['gt_data']['landmarks'], axis=0)
        speaker_landmarks = np.concatenate(self.streaming_results['speaker_data']['landmarks'], axis=0)
        
        n_frames = min(len(pred_landmarks), max_frames)
        
        # 提取坐标
        def extract_coords(landmarks):
            x_coords = landmarks[:n_frames, :68]
            y_coords = landmarks[:n_frames, 68:]
            return x_coords, y_coords
        
        pred_x, pred_y = extract_coords(pred_landmarks)
        gt_x, gt_y = extract_coords(gt_landmarks)
        speaker_x, speaker_y = extract_coords(speaker_landmarks)
        
        # 计算坐标范围
        all_x = np.concatenate([pred_x.flatten(), gt_x.flatten(), speaker_x.flatten()])
        all_y = np.concatenate([pred_y.flatten(), gt_y.flatten(), speaker_y.flatten()])
        x_min, x_max = np.min(all_x), np.max(all_x)
        y_min, y_max = np.min(all_y), np.max(all_y)
        margin = 0.1 * max(x_max - x_min, y_max - y_min)
        
        # 创建动画
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        def animate(frame):
            # 清除所有子图
            for ax in axes.flat:
                ax.clear()
            
            # 当前窗口信息
            current_window = frame // self.window_size
            frame_in_window = frame % self.window_size
            
            # 子图1: Speaker
            ax1 = axes[0, 0]
            ax1.scatter(speaker_x[frame], speaker_y[frame], c='blue', s=15, alpha=0.7)
            ax1.set_title(f'Speaker - 帧 {frame+1}', fontsize=12)
            ax1.set_xlim(x_min - margin, x_max + margin)
            ax1.set_ylim(y_max + margin, y_min - margin)
            ax1.set_aspect('equal')
            self.draw_face_connections(ax1, speaker_x[frame], speaker_y[frame], 'blue')
            
            # 子图2: GT Listener
            ax2 = axes[0, 1]
            ax2.scatter(gt_x[frame], gt_y[frame], c='green', s=15, alpha=0.7)
            ax2.set_title(f'GT Listener - 帧 {frame+1}', fontsize=12)
            ax2.set_xlim(x_min - margin, x_max + margin)
            ax2.set_ylim(y_max + margin, y_min - margin)
            ax2.set_aspect('equal')
            self.draw_face_connections(ax2, gt_x[frame], gt_y[frame], 'green')
            
            # 子图3: Predicted Listener
            ax3 = axes[1, 0]
            ax3.scatter(pred_x[frame], pred_y[frame], c='red', s=15, alpha=0.7)
            ax3.set_title(f'Predicted Listener - 帧 {frame+1}', fontsize=12)
            ax3.set_xlim(x_min - margin, x_max + margin)
            ax3.set_ylim(y_max + margin, y_min - margin)
            ax3.set_aspect('equal')
            self.draw_face_connections(ax3, pred_x[frame], pred_y[frame], 'red')
            
            # 子图4: 流式处理状态
            ax4 = axes[1, 1]
            ax4.set_xlim(0, 10)
            ax4.set_ylim(0, 10)
            ax4.set_title(f'流式处理状态', fontsize=12)
            
            # 显示当前窗口信息
            status_text = f"""
窗口 {current_window + 1}
帧 {frame_in_window + 1}/{self.window_size}
总帧数: {frame + 1}/{n_frames}

窗口大小: {self.window_size}
延迟: ~{self.window_size/25*1000:.0f}ms
            """
            ax4.text(0.1, 0.7, status_text, fontsize=10, verticalalignment='top',
                    fontfamily='monospace')
            
            # 绘制窗口进度条
            window_progress = frame_in_window / self.window_size
            progress_bar = Rectangle((0.5, 2), 8, 1, facecolor='lightblue', edgecolor='black')
            ax4.add_patch(progress_bar)
            
            filled_bar = Rectangle((0.5, 2), 8 * window_progress, 1, 
                                 facecolor='blue', edgecolor='black')
            ax4.add_patch(filled_bar)
            
            ax4.text(4.5, 1.5, f'窗口进度: {window_progress*100:.1f}%', 
                    ha='center', fontsize=10)
            
            ax4.set_xticks([])
            ax4.set_yticks([])
        
        # 创建动画
        anim = animation.FuncAnimation(fig, animate, frames=n_frames, 
                                     interval=100, repeat=True)
        
        # 保存动画
        animation_path = os.path.join(output_dir, 'streaming_animation.gif')
        anim.save(animation_path, writer='pillow', fps=10, dpi=80)
        plt.close()
        
        print(f"✅ 流式处理动画已保存到: {animation_path}")
        return animation_path
    
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
    
    def compare_streaming_vs_batch(self, sample_data, output_dir):
        """对比流式推理和批量推理的结果"""
        print("🔄 对比流式推理 vs 批量推理...")
        
        # 批量推理
        batch_start = time.time()
        
        batch_speaker = {}
        for key, value in sample_data['speaker'].items():
            batch_speaker[key] = value.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            batch_predictions, _, _ = self.model(batch_speaker, speaker_out=False)
        
        batch_time = time.time() - batch_start
        
        # 反归一化批量结果
        batch_denorm = self.denormalize_features({
            k: v.cpu().numpy().squeeze(0) for k, v in batch_predictions.items()
        })
        
        # 获取流式结果
        streaming_denorm = {}
        for feature_name in ['landmarks', 'au', 'pose', 'gaze']:
            if self.streaming_results['predictions'][feature_name]:
                streaming_denorm[feature_name] = np.concatenate(
                    self.streaming_results['predictions'][feature_name], axis=0
                )
        
        # 计算差异
        comparison_stats = {}
        for feature_name in streaming_denorm:
            if feature_name in batch_denorm:
                # 确保长度一致
                min_len = min(len(streaming_denorm[feature_name]), 
                             len(batch_denorm[feature_name]))
                
                stream_data = streaming_denorm[feature_name][:min_len]
                batch_data = batch_denorm[feature_name][:min_len]
                
                # 计算差异
                diff = np.abs(stream_data - batch_data)
                comparison_stats[feature_name] = {
                    'max_diff': np.max(diff),
                    'mean_diff': np.mean(diff),
                    'std_diff': np.std(diff)
                }
        
        # 时间对比
        streaming_total_time = sum(self.inference_times)
        
        print(f"\n⚖️ 流式 vs 批量推理对比:")
        print(f"  批量推理时间: {batch_time*1000:.2f}ms")
        print(f"  流式推理总时间: {streaming_total_time*1000:.2f}ms")
        print(f"  时间比率: {streaming_total_time/batch_time:.2f}x")
        
        print(f"\n📊 预测差异分析:")
        for feature_name, stats in comparison_stats.items():
            print(f"  {feature_name.upper()}:")
            print(f"    最大差异: {stats['max_diff']:.6f}")
            print(f"    平均差异: {stats['mean_diff']:.6f}")
            print(f"    差异标准差: {stats['std_diff']:.6f}")
        
        # 保存对比结果
        comparison_data = {
            'batch_inference_time': str(batch_time),
            'streaming_total_time': str(streaming_total_time),
            'time_ratio': str(streaming_total_time / batch_time),
            'feature_differences': str(comparison_stats)
        }
        
        import json
        comparison_path = os.path.join(output_dir, 'streaming_vs_batch_comparison.json')
        with open(comparison_path, 'w') as f:
            json.dump(comparison_data, f, indent=2)
        
        return comparison_data
    
    def run_streaming_test(self, data_csv, sample_idx=0, output_dir='streaming_results',
                          stream_fps=25, real_time=False, max_windows=None):
        """运行完整的流式测试"""
        print("🚀 开始流式推理测试...")
        
        # 1. 加载测试样本
        sample_data = self.load_test_sample(data_csv, sample_idx)
        
        # 2. 执行流式推理
        processed_windows = self.streaming_inference(
            sample_data, output_dir, stream_fps, real_time, max_windows
        )
        
        # 3. 对比流式和批量推理
        comparison_data = self.compare_streaming_vs_batch(sample_data, output_dir)
        
        # 4. 创建可视化
        animation_path = self.create_streaming_animation(output_dir)
        
        print("🎉 流式测试完成！")
        
        return {
            'processed_windows': processed_windows,
            'animation_path': animation_path,
            'comparison_data': comparison_data,
            'output_dir': output_dir
        }


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="流式推理测试")
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='模型checkpoint路径')
    parser.add_argument('--data-csv', type=str,
                       default='/mnt/iusers01/fatpou01/compsci01/k09562zs/scratch/Face_data/val.csv',
                       help='测试数据CSV路径')
    parser.add_argument('--sample-idx', type=int, default=0,
                       help='测试样本索引')
    parser.add_argument('--output-dir', type=str, default='streaming_results',
                       help='输出目录')
    parser.add_argument('--device', type=str, default='cuda',
                       help='设备 (cuda/cpu)')
    parser.add_argument('--stream-fps', type=int, default=25,
                       help='流帧率 (默认: 25)')
    parser.add_argument('--real-time', action='store_true',
                       help='启用实时处理模式')
    parser.add_argument('--max-windows', type=int, default=None,
                       help='最大处理窗口数')
    
    args = parser.parse_args()
    
    # 检查文件
    if not os.path.exists(args.checkpoint):
        print(f"❌ Checkpoint文件不存在: {args.checkpoint}")
        return
    
    if not os.path.exists(args.data_csv):
        print(f"❌ 数据文件不存在: {args.data_csv}")
        return
    
    try:
        # 创建流式测试器
        tester = StreamingTester(args.checkpoint, args.device)
        
        # 运行流式测试
        results = tester.run_streaming_test(
            args.data_csv,
            args.sample_idx,
            args.output_dir,
            args.stream_fps,
            args.real_time,
            args.max_windows
        )
        
        print("\n" + "="*60)
        print("🎯 流式测试结果总结")
        print("="*60)
        print(f"输出目录: {results['output_dir']}")
        print(f"处理窗口数: {results['processed_windows']}")
        print(f"动画文件: {results['animation_path']}")
        print(f"时间比率 (流式/批量): {float(results['comparison_data']['time_ratio']):.2f}x")

        print("="*60)
        
    except Exception as e:
        print(f"❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()