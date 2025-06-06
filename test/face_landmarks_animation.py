import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Polygon
import os
import imageio
from PIL import Image, ImageDraw
import warnings
warnings.filterwarnings('ignore')

def load_landmark_data(parquet_file, max_frames=300):
    """
    加载面部特征点数据
    前68列：x坐标，后68列：y坐标
    """
    df = pd.read_parquet(parquet_file)
    
    # 取前300帧
    df = df.head(max_frames)
    
    # 分离x和y坐标
    x_coords = df.iloc[:, 1:69].values  # 前68列是x坐标
    y_coords = df.iloc[:, 69:137].values  # 后68列是y坐标
    
    return x_coords, y_coords

def get_face_connections():
    """
    定义68个面部特征点的连接关系，用于绘制面部轮廓
    """
    # 面部轮廓 (0-16)
    jaw_line = list(range(0, 17))
    
    # 右眉毛 (17-21)
    right_eyebrow = list(range(17, 22))
    
    # 左眉毛 (22-26)
    left_eyebrow = list(range(22, 27))
    
    # 鼻梁 (27-30)
    nose_bridge = list(range(27, 31))
    
    # 鼻翼 (31-35)
    nose_tip = list(range(31, 36))
    
    # 右眼 (36-41)
    right_eye = list(range(36, 42)) + [36]  # 闭合
    
    # 左眼 (42-47)
    left_eye = list(range(42, 48)) + [42]  # 闭合
    
    # 外嘴唇 (48-59)
    outer_lip = list(range(48, 60)) + [48]  # 闭合
    
    # 内嘴唇 (60-67)
    inner_lip = list(range(60, 68)) + [60]  # 闭合
    
    return {
        'jaw_line': jaw_line,
        'right_eyebrow': right_eyebrow,
        'left_eyebrow': left_eyebrow,
        'nose_bridge': nose_bridge,
        'nose_tip': nose_tip,
        'right_eye': right_eye,
        'left_eye': left_eye,
        'outer_lip': outer_lip,
        'inner_lip': inner_lip
    }

def create_original_coordinates_gif(x_coords, y_coords, output_path='original_face_animation.gif'):
    """
    创建原始坐标的面部动画
    """
    n_frames = len(x_coords)
    connections = get_face_connections()
    
    # 设置图形
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # 计算坐标范围用于设置轴限制
    all_x = x_coords.flatten()
    all_y = y_coords.flatten()
    x_min, x_max = np.min(all_x), np.max(all_x)
    y_min, y_max = np.min(all_y), np.max(all_y)
    
    # 添加边距
    margin = 0.1 * max(x_max - x_min, y_max - y_min)
    ax.set_xlim(x_min - margin, x_max + margin)
    ax.set_ylim(y_max + margin, y_min - margin)  # 翻转y轴，因为图像坐标系
    
    def animate(frame):
        ax.clear()
        ax.set_xlim(x_min - margin, x_max + margin)
        ax.set_ylim(y_max + margin, y_min - margin)
        ax.set_title(f'原始坐标面部动画 - 帧 {frame+1}/{n_frames}', fontsize=14)
        ax.set_aspect('equal')
        
        x_frame = x_coords[frame]
        y_frame = y_coords[frame]
        
        # 绘制特征点
        ax.scatter(x_frame, y_frame, c='red', s=20, alpha=0.7)
        
        # 绘制连接线
        colors = ['blue', 'green', 'green', 'orange', 'orange', 'purple', 'purple', 'red', 'red']
        for i, (name, indices) in enumerate(connections.items()):
            if len(indices) > 1:
                x_line = [x_frame[idx] for idx in indices]
                y_line = [y_frame[idx] for idx in indices]
                ax.plot(x_line, y_line, color=colors[i % len(colors)], linewidth=1.5, alpha=0.8)
        
        # 添加特征点标号（可选，可能会让图像显得拥挤）
        # for i in range(len(x_frame)):
        #     ax.annotate(str(i), (x_frame[i], y_frame[i]), fontsize=6, alpha=0.5)
    
    # 创建动画
    anim = animation.FuncAnimation(fig, animate, frames=n_frames, interval=100, repeat=True)
    
    # 保存为GIF
    print(f"正在保存原始坐标动画到 {output_path}...")
    anim.save(output_path, writer='pillow', fps=10, dpi=80)
    plt.close()
    print(f"原始坐标动画已保存到 {output_path}")

def create_relative_coordinates_gif(x_coords, y_coords, output_path='relative_face_animation.gif'):
    """
    创建相对坐标的面部动画（以脸部中心为原点）
    """
    n_frames = len(x_coords)
    connections = get_face_connections()
    
    # 计算每帧的相对坐标
    relative_x_coords = []
    relative_y_coords = []
    
    for frame in range(n_frames):
        x_frame = x_coords[frame]
        y_frame = y_coords[frame]
        
        # 计算脸部中心（所有特征点的平均值）
        center_x = np.mean(x_frame)
        center_y = np.mean(y_frame)
        
        # 转换为相对坐标
        rel_x = x_frame - center_x
        rel_y = y_frame - center_y
        
        relative_x_coords.append(rel_x)
        relative_y_coords.append(rel_y)
    
    relative_x_coords = np.array(relative_x_coords)
    relative_y_coords = np.array(relative_y_coords)
    
    # 设置图形
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # 计算相对坐标的范围
    all_rel_x = relative_x_coords.flatten()
    all_rel_y = relative_y_coords.flatten()
    max_range = max(np.max(np.abs(all_rel_x)), np.max(np.abs(all_rel_y)))
    
    # 设置对称的坐标轴
    margin = 0.1 * max_range
    ax.set_xlim(-max_range - margin, max_range + margin)
    ax.set_ylim(max_range + margin, -max_range - margin)  # 翻转y轴
    
    def animate(frame):
        ax.clear()
        ax.set_xlim(-max_range - margin, max_range + margin)
        ax.set_ylim(max_range + margin, -max_range - margin)
        ax.set_title(f'相对坐标面部动画 - 帧 {frame+1}/{n_frames}', fontsize=14)
        ax.set_aspect('equal')
        
        # 添加坐标轴
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        ax.grid(True, alpha=0.3)
        
        x_frame = relative_x_coords[frame]
        y_frame = relative_y_coords[frame]
        
        # 绘制特征点
        ax.scatter(x_frame, y_frame, c='red', s=20, alpha=0.7)
        
        # 绘制连接线
        colors = ['blue', 'green', 'green', 'orange', 'orange', 'purple', 'purple', 'red', 'red']
        for i, (name, indices) in enumerate(connections.items()):
            if len(indices) > 1:
                x_line = [x_frame[idx] for idx in indices]
                y_line = [y_frame[idx] for idx in indices]
                ax.plot(x_line, y_line, color=colors[i % len(colors)], linewidth=1.5, alpha=0.8)
        
        # 标记中心点
        ax.scatter(0, 0, c='black', s=50, marker='+', linewidth=2)
    
    # 创建动画
    anim = animation.FuncAnimation(fig, animate, frames=n_frames, interval=100, repeat=True)
    
    # 保存为GIF
    print(f"正在保存相对坐标动画到 {output_path}...")
    anim.save(output_path, writer='pillow', fps=10, dpi=80)
    plt.close()
    print(f"相对坐标动画已保存到 {output_path}")

def create_comparison_gif(x_coords, y_coords, output_path='comparison_face_animation.gif'):
    """
    创建并排比较的动画
    """
    n_frames = len(x_coords)
    connections = get_face_connections()
    
    # 计算相对坐标
    relative_x_coords = []
    relative_y_coords = []
    
    for frame in range(n_frames):
        x_frame = x_coords[frame]
        y_frame = y_coords[frame]
        center_x = np.mean(x_frame)
        center_y = np.mean(y_frame)
        rel_x = x_frame - center_x
        rel_y = y_frame - center_y
        relative_x_coords.append(rel_x)
        relative_y_coords.append(rel_y)
    
    relative_x_coords = np.array(relative_x_coords)
    relative_y_coords = np.array(relative_y_coords)
    
    # 设置双子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # 原始坐标范围
    all_x = x_coords.flatten()
    all_y = y_coords.flatten()
    x_min, x_max = np.min(all_x), np.max(all_x)
    y_min, y_max = np.min(all_y), np.max(all_y)
    margin1 = 0.1 * max(x_max - x_min, y_max - y_min)
    
    # 相对坐标范围
    all_rel_x = relative_x_coords.flatten()
    all_rel_y = relative_y_coords.flatten()
    max_range = max(np.max(np.abs(all_rel_x)), np.max(np.abs(all_rel_y)))
    margin2 = 0.1 * max_range
    
    def animate(frame):
        ax1.clear()
        ax2.clear()
        
        # 原始坐标子图
        ax1.set_xlim(x_min - margin1, x_max + margin1)
        ax1.set_ylim(y_max + margin1, y_min - margin1)
        ax1.set_title(f'原始坐标 - 帧 {frame+1}/{n_frames}', fontsize=12)
        ax1.set_aspect('equal')
        
        # 相对坐标子图
        ax2.set_xlim(-max_range - margin2, max_range + margin2)
        ax2.set_ylim(max_range + margin2, -max_range - margin2)
        ax2.set_title(f'相对坐标 - 帧 {frame+1}/{n_frames}', fontsize=12)
        ax2.set_aspect('equal')
        ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax2.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        ax2.grid(True, alpha=0.3)
        
        # 绘制原始坐标
        x_frame = x_coords[frame]
        y_frame = y_coords[frame]
        ax1.scatter(x_frame, y_frame, c='red', s=15, alpha=0.7)
        
        # 绘制相对坐标
        rel_x_frame = relative_x_coords[frame]
        rel_y_frame = relative_y_coords[frame]
        ax2.scatter(rel_x_frame, rel_y_frame, c='red', s=15, alpha=0.7)
        ax2.scatter(0, 0, c='black', s=40, marker='+', linewidth=2)
        
        # 绘制连接线
        colors = ['blue', 'green', 'green', 'orange', 'orange', 'purple', 'purple', 'red', 'red']
        for i, (name, indices) in enumerate(connections.items()):
            if len(indices) > 1:
                # 原始坐标连接线
                x_line = [x_frame[idx] for idx in indices]
                y_line = [y_frame[idx] for idx in indices]
                ax1.plot(x_line, y_line, color=colors[i % len(colors)], linewidth=1, alpha=0.8)
                
                # 相对坐标连接线
                rel_x_line = [rel_x_frame[idx] for idx in indices]
                rel_y_line = [rel_y_frame[idx] for idx in indices]
                ax2.plot(rel_x_line, rel_y_line, color=colors[i % len(colors)], linewidth=1, alpha=0.8)
    
    # 创建动画
    anim = animation.FuncAnimation(fig, animate, frames=n_frames, interval=100, repeat=True)
    
    # 保存为GIF
    print(f"正在保存对比动画到 {output_path}...")
    anim.save(output_path, writer='pillow', fps=10, dpi=80)
    plt.close()
    print(f"对比动画已保存到 {output_path}")

def main():
    """
    主函数
    """
    # 请修改为你的parquet文件路径
    parquet_file = "/mnt/iusers01/fatpou01/compsci01/k09562zs/scratch/Face_data/Face/NoXI/002_2016-03-17_Paris/Expert_video/10.parquet"  # 请替换为实际文件名
    
    # 检查文件是否存在
    if not os.path.exists(parquet_file):
        print(f"错误：找不到文件 {parquet_file}")
        print("当前目录下的parquet文件：")
        for file in os.listdir('.'):
            if file.endswith('.parquet'):
                print(f"  {file}")
        return
    
    try:
        # 加载数据
        print("正在加载面部特征点数据...")
        x_coords, y_coords = load_landmark_data(parquet_file, max_frames=300)
        print(f"已加载 {len(x_coords)} 帧数据，每帧包含 68 个特征点")
        
        # 创建原始坐标动画
        print("\n创建原始坐标动画...")
        create_original_coordinates_gif(x_coords, y_coords, 'original_face_animation.gif')
        
        # 创建相对坐标动画
        print("\n创建相对坐标动画...")
        create_relative_coordinates_gif(x_coords, y_coords, 'relative_face_animation.gif')
        
        # 创建对比动画
        print("\n创建对比动画...")
        create_comparison_gif(x_coords, y_coords, 'comparison_face_animation.gif')
        
        print("\n所有动画已创建完成！")
        print("生成的文件：")
        print("  - original_face_animation.gif: 原始坐标动画")
        print("  - relative_face_animation.gif: 相对坐标动画")
        print("  - comparison_face_animation.gif: 并排对比动画")
        
    except Exception as e:
        print(f"错误：{e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()