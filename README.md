# Robot_Face_Gen
# README.md

# Facial Reaction Generation Model

基于OpenFace特征和Whisper音频特征的面部反应生成模型，支持在线实时推理。

## 特性

- ✅ **多模态输入**: 支持面部landmarks、AU、pose、gaze和Whisper音频特征
- ✅ **多分支输出**: 独立预测四种面部特征，精细控制
- ✅ **在线推理**: 窗口化处理，支持实时应用
- ✅ **VAE生成**: 潜在空间采样，生成多样化反应
- ✅ **无邻居依赖**: 不需要邻居数据，简化训练流程

## 安装

### 1. 克隆仓库
```bash
git clone https://github.com/your-username/facial-reaction-generation.git
cd facial-reaction-generation
```

### 2. 创建环境
```bash
conda create -n facial_reaction python=3.11
conda activate facial_reaction
```

### 3. 安装依赖
```bash
pip install -r requirements.txt
# 或者
pip install -e .
```

## 快速开始

### 1. 基本使用
```python
from facial_reaction_model import create_model
import torch

# 创建模型
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = create_model(device=device)

# 准备数据
speaker_data = {
    'landmarks': torch.randn(1, 100, 136),  # (B, T, 136)
    'au': torch.randn(1, 100, 18),          # (B, T, 18)
    'pose': torch.randn(1, 100, 4),         # (B, T, 4)
    'gaze': torch.randn(1, 100, 2),         # (B, T, 2)
    'audio': torch.randn(1, 100, 384)       # (B, T, 384) Whisper特征
}

# 推理
predictions, distributions, _ = model(speaker_data)
```

### 2. 在线推理
```python
from facial_reaction_model import OnlineInferenceManager

# 创建在线管理器
online_manager = OnlineInferenceManager(model, device)

# 逐窗口处理
for window_data in real_time_stream:
    listener_predictions = online_manager.process_window(window_data)
```

### 3. 训练模型
```bash
python training/train.py --config configs/training_config.yaml
```

## 项目结构

```
FacialReactionGeneration/
├── facial_reaction_model/     # 主模型包
│   ├── components.py         # 基础组件
│   ├── encoder.py           # 多模态编码器
│   ├── vae.py              # VAE模块
│   ├── decoder.py          # 多分支解码器
│   ├── losses.py           # 损失函数
│   └── model.py            # 主模型
├── data/                   # 数据处理
├── training/               # 训练脚本
├── configs/               # 配置文件
├── examples/              # 使用示例
└── utils/                 # 工具函数
```

## 模型架构

### 数据流
```
Speaker Input (Landmarks+AU+Pose+Gaze+Audio)
    ↓
Multi-modal Encoder (Face + Audio Fusion)
    ↓
VAE Motion Sample Generator
    ↓
Multi-branch Decoder
    ↓
Listener Output (Landmarks+AU+Pose+Gaze)
```

### 核心组件
1. **FacialFeatureEncoder**: 融合四种面部特征
2. **AudioFeatureEncoder**: 处理Whisper音频特征
3. **FacialVAE**: 潜在空间采样
4. **ListenerReactionDecoder**: 四分支独立解码

## 配置

### 模型配置 (configs/model_config.yaml)
```yaml
model:
  feature_dim: 256
  audio_dim: 384
  window_size: 16
  max_seq_len: 750
```

### 训练配置 (configs/training_config.yaml)
```yaml
training:
  batch_size: 4
  learning_rate: 1e-4
  num_epochs: 100

loss_weights:
  vae_weight: 1.0
  smooth_weight: 15.0
  diversity_weight: 100.0
```

## 数据格式

### 输入数据结构
```python
speaker_data = {
    'landmarks': torch.Tensor,  # (B, T, 136) - 68个点的x,y坐标
    'au': torch.Tensor,         # (B, T, 18)  - Action Units
    'pose': torch.Tensor,       # (B, T, 4)   - 头部姿态
    'gaze': torch.Tensor,       # (B, T, 2)   - 视线方向
    'audio': torch.Tensor       # (B, T, 384) - Whisper特征
}
```

### 数据预处理
- **Landmarks**: 除以224归一化到[0,1]
- **AU/Pose/Gaze**: 保持原始值
- **Audio**: 使用Whisper预训练特征，无需额外处理

## 训练策略

```

### 2. 损失函数组合
- **VAE Loss**: 重建损失 + KL散度
- **Smooth Loss**: 时序平滑约束
- **Diversity Loss**: 生成多样性
- **Contrastive Loss**: 多次采样一致性
- **Consistency Loss**: 特征间协调性

### 3. 数据增强
- 时序dropout (0.1)
- 特征噪声 (0.02)
- 序列裁剪
- 多次采样 (3次)

## 评估指标

- **重建精度**: MSE, MAE, RMSE
- **时序一致性**: 帧间差异，加速度
- **特征相关性**: 皮尔逊相关系数
- **生成多样性**: 多次采样差异度
