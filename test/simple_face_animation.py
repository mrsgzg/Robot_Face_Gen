#!/usr/bin/env python3
"""
简化版面部特征点动画生成器
使用方法：python face_animation.py your_file.parquet
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sys
import os

def create_face_animations(parquet_file, max_frames=300):
    """
    从parquet文件创建面部动画
    """
    # 读取数据
    print(f"读取文件: {parquet_file}")
    df = pd.read_parquet(parquet_file)
    df = df.head(max_frames)
    
    # 分离坐标
    x_coords = df.iloc[:, 1:69].values
    y_coords = df.iloc[:, 69:137].values
    
    print(f"数据形状: {x_coords.shape}, 帧数: {len(x_coords)}")
    
    # 1. 原始坐标动画
    print("创建原始坐标动画...")
    create_original_animation(x_coords, y_coords)
    
    # 2. 相对坐标动画
    print("创建相对坐标动画...")
    create_relative_animation(x_coords, y_coords)
    
    print("完成！")

def create_original_animation(x_coords, y_coords):
    """创建原始坐标动画"""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # 计算坐标范围
    all_x, all_y = x_coords.flatten(), y_coords.flatten()
    x_range = np.max(all_x) - np.min(all_x)
    y_range = np.max(all_y) - np.min(all_y)
    margin = 0.1 * max(x_range, y_range)
    
    ax.set_xlim(np.min(all_x) - margin, np.max(all_x) + margin)
    ax.set_ylim(np.max(all_y) + margin, np.min(all_y) - margin)
    
    def animate(frame):
        ax.clear()
        ax.set_xlim(np.min(all_x) - margin, np.max(all_x) + margin)
        ax.set_ylim(np.max(all_y) + margin, np.min(all_y) - margin)
        ax.set_title(f'原始坐标面部动画 - 帧 {frame+1}', fontsize=14)
        
        # 绘制特征点
        ax.scatter(x_coords[frame], y_coords[frame], c='red', s=30, alpha=0.8)
        
        # 绘制基本连接线
        draw_face_outline(ax, x_coords[frame], y_coords[frame])
    
    anim = animation.FuncAnimation(fig, animate, frames=len(x_coords), 
                                 interval=100, repeat=True)
    anim.save('original_face.gif', writer='pillow', fps=10)
    plt.close()

def create_relative_animation(x_coords, y_coords):
    """创建相对坐标动画"""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # 计算相对坐标
    rel_coords = []
    for frame in range(len(x_coords)):
        center_x = np.mean(x_coords[frame])
        center_y = np.mean(y_coords[frame])
        rel_x = x_coords[frame] - center_x
        rel_y = y_coords[frame] - center_y
        rel_coords.append((rel_x, rel_y))
    
    # 计算范围
    all_rel = np.array(rel_coords)
    max_range = np.max(np.abs(all_rel))
    margin = 0.1 * max_range
    
    def animate(frame):
        ax.clear()
        ax.set_xlim(-max_range - margin, max_range + margin)
        ax.set_ylim(max_range + margin, -max_range - margin)
        ax.set_title(f'相对坐标面部动画 - 帧 {frame+1}', fontsize=14)
        
        # 坐标轴
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        ax.grid(True, alpha=0.2)
        
        rel_x, rel_y = rel_coords[frame]
        
        # 绘制特征点
        ax.scatter(rel_x, rel_y, c='blue', s=30, alpha=0.8)
        ax.scatter(0, 0, c='black', s=80, marker='+', linewidth=3)
        
        # 绘制基本连接线
        draw_face_outline(ax, rel_x, rel_y)
    
    anim = animation.FuncAnimation(fig, animate, frames=len(x_coords), 
                                 interval=100, repeat=True)
    anim.save('relative_face.gif', writer='pillow', fps=10)
    plt.close()

def draw_face_outline(ax, x, y):
    """绘制面部轮廓线"""
    # 面部边界 (0-16)
    ax.plot(x[0:17], y[0:17], 'g-', linewidth=2, alpha=0.7)
    
    # 眉毛
    ax.plot(x[17:22], y[17:22], 'orange', linewidth=2, alpha=0.7)  # 右眉
    ax.plot(x[22:27], y[22:27], 'orange', linewidth=2, alpha=0.7)  # 左眉
    
    # 鼻子
    ax.plot(x[27:31], y[27:31], 'brown', linewidth=2, alpha=0.7)  # 鼻梁
    ax.plot(x[31:36], y[31:36], 'brown', linewidth=2, alpha=0.7)  # 鼻翼
    
    # 眼睛
    eye_r = list(range(36, 42)) + [36]
    eye_l = list(range(42, 48)) + [42]
    ax.plot([x[i] for i in eye_r], [y[i] for i in eye_r], 'purple', linewidth=2, alpha=0.7)
    ax.plot([x[i] for i in eye_l], [y[i] for i in eye_l], 'purple', linewidth=2, alpha=0.7)
    
    # 嘴巴
    mouth_outer = list(range(48, 60)) + [48]
    ax.plot([x[i] for i in mouth_outer], [y[i] for i in mouth_outer], 'red', linewidth=2, alpha=0.7)

def main():
    """主函数"""
    if len(sys.argv) > 1:
        parquet_file = sys.argv[1]
    else:
        # 自动查找parquet文件
        parquet_files = [f for f in os.listdir('.') if f.endswith('.parquet')]
        if not parquet_files:
            print("错误：当前目录没有找到.parquet文件")
            print("使用方法：python face_animation.py your_file.parquet")
            return
        parquet_file = parquet_files[0]
        print(f"自动选择文件: {parquet_file}")
    
    if not os.path.exists(parquet_file):
        print(f"错误：文件 {parquet_file} 不存在")
        return
    
    try:
        create_face_animations(parquet_file)
        print("\n生成的文件:")
        print("- original_face.gif: 原始坐标动画")
        print("- relative_face.gif: 相对坐标动画")
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()