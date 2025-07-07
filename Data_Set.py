import os
import numpy as np
import pandas as pd
import torch
import torchaudio
import soundfile as sf
from torch.utils.data import Dataset, DataLoader
from transformers import WhisperFeatureExtractor, WhisperModel
import warnings
warnings.filterwarnings('ignore')

class SpeakerListenerDataset(Dataset):
    def __init__(self, mapping_csv, whisper_model_name="openai/whisper-tiny"):
        self.target_length = 750  # ✅ 统一长度为 750 帧
        self.mapping_df = pd.read_csv(mapping_csv, engine="c")  # ✅ 使用 C 解析器加速
        
        # **🔹 定义OpenFace特征列索引**
        self.landmark_x_cols = list(range(18, 69))        # 点17-67的x坐标 (列18-68)
        self.landmark_y_cols = list(range(86, 137)) 
        #self.landmark_cols = list(range(1, 137))      # Face landmarks (68x + 68y)
        self.au_cols = list(range(137, 154))          # Face AU 
        self.pose_cols = list(range(157, 160))        # Head pose
        self.gaze_cols = list(range(160, 162))        # Gaze angle
        
        # **🔹 加载Whisper模型用于音频特征提取**
        print(f"🎵 Loading Whisper model: {whisper_model_name}")
        self.whisper_feature_extractor = WhisperFeatureExtractor.from_pretrained(whisper_model_name)
        self.whisper_model = WhisperModel.from_pretrained(whisper_model_name)
        self.whisper_model.eval()  # 设置为评估模式
        
        # 获取Whisper模型的采样率
        self.whisper_sampling_rate = self.whisper_feature_extractor.sampling_rate
        print(f"✅ Whisper model loaded, sampling rate: {self.whisper_sampling_rate}")

    def _convert_path(self, path, ext):
        """自动适配 Windows 和 Ubuntu 路径，同时替换扩展名"""
        path = "/mnt/iusers01/fatpou01/compsci01/k09562zs/scratch/Face_data/Face/"+path+"."+ext
        return path
    
    def _convert_audio_path(self, path, ext):
        path = "/mnt/iusers01/fatpou01/compsci01/k09562zs/scratch/Face_data/Audio_files/"+path+"."+ext
        return path
    #/mnt/iusers01/fatpou01/compsci01/k09562zs/scratch/Face_data
    def _interpolate_missing_frames(self, df):
        """填充缺失帧，确保 750 帧"""
        if "frame" not in df.columns:
            return df  # 若无 frame 列，则直接返回
        if len(df) == self.target_length:
            return df

        full_frame_range = np.arange(0, self.target_length)  # 生成 0~749 帧
        df = df.set_index("frame").reindex(full_frame_range)  # 重新索引
        df.interpolate(method="linear", inplace=True)  # 线性插值
        df.ffill(inplace=True)  # 向前填充
        df.bfill(inplace=True)  # 向后填充

        return df.reset_index()

    def _load_parquet(self, parquet_path):
        """读取 Parquet 并按特征类型分离，只对landmarks进行归一化"""
        if not os.path.exists(parquet_path) or os.stat(parquet_path).st_size == 0:
            print(f"❌ 文件不存在或为空: {parquet_path}")
            return None, None, None, None

        try:
            df = pd.read_parquet(parquet_path)  # 读取 Parquet
            #print("load_successfully")
            df = df.loc[:self.target_length - 1]  # ✅ 限制为 750 帧
            df = self._interpolate_missing_frames(df)  # **✅ 插值补帧**
            
            landmarks_x = df.iloc[:, self.landmark_x_cols].values  # (750, 51) - 51个点的x坐标
            landmarks_y = df.iloc[:, self.landmark_y_cols].values  # (750, 51) - 51个点的y坐标
            
            # **🔹 新增：landmarks中心化处理**
            landmarks_x_centered, landmarks_y_centered = self._center_landmarks(landmarks_x, landmarks_y)
            
            # 合并为一个数组 (750, 102) - 51个点的x,y坐标
            landmarks = np.concatenate([landmarks_x_centered, landmarks_y_centered], axis=1)
            
            # 2. Face AU (137-154列，索引136-153) - 保持原始值
            au_features = df.iloc[:, self.au_cols].values / 5.0
            
            # 3. Head pose (157-160列，索引156-159) - 保持原始值
            pose_features = df.iloc[:, self.pose_cols].values / np.pi
            
            # 4. Gaze angle (161-162列，索引160-161) - 保持原始值
            gaze_features = df.iloc[:, self.gaze_cols].values  / np.pi

            return landmarks, au_features, pose_features, gaze_features
            
        except Exception as e:
            print(f"⚠️  加载失败: {parquet_path} - {e}")
            return None, None, None, None
    def _center_landmarks(self, landmarks_x, landmarks_y):
        """
        对landmarks进行中心化处理
        Args:
            landmarks_x: (T, 51) - 51个点的x坐标  
            landmarks_y: (T, 51) - 51个点的y坐标
        Returns:
            tuple: (centered_x, centered_y) - 中心化后的坐标
        """
        centered_x = np.zeros_like(landmarks_x)
        centered_y = np.zeros_like(landmarks_y)
        
        for frame_idx in range(landmarks_x.shape[0]):
            # 计算当前帧所有点的质心
            center_x = np.mean(landmarks_x[frame_idx])
            center_y = np.mean(landmarks_y[frame_idx])
            
            # 中心化：减去质心
            centered_x[frame_idx] = landmarks_x[frame_idx] - center_x
            centered_y[frame_idx] = landmarks_y[frame_idx] - center_y
        
        # 归一化：除以一个尺度因子来控制数值范围
        # 可以使用标准差或者固定的尺度因子
        scale_factor = 224  # 可以根据实际情况调整
        centered_x = centered_x / scale_factor
        centered_y = centered_y / scale_factor
        
        return centered_x, centered_y
    
    def _load_audio_with_whisper(self, audio_path):
        """使用Whisper提取音频特征"""
        if not os.path.exists(audio_path):
            print(f"❌ 音频文件不存在: {audio_path}")
            return None

        try:
            # 1️⃣ **读取音频文件**
            waveform, sr = torchaudio.load(audio_path)
            
            # 转换为单声道
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            
            # 转换为numpy并重采样到Whisper所需的采样率
            audio_array = waveform.squeeze().numpy()
            
            # 重采样到Whisper的采样率 (16kHz)
            if sr != self.whisper_sampling_rate:
                resampler = torchaudio.transforms.Resample(sr, self.whisper_sampling_rate)
                audio_tensor = torch.from_numpy(audio_array).unsqueeze(0)
                audio_array = resampler(audio_tensor).squeeze().numpy()
            
            # 2️⃣ **使用Whisper特征提取器处理音频**
            # Whisper期望音频长度为30秒，我们需要确保音频长度适合
            max_length = 30 * self.whisper_sampling_rate  # 30秒的样本数
            if len(audio_array) > max_length:
                # 如果音频超过30秒，截断到30秒
                audio_array = audio_array[:max_length]
            
            # 使用特征提取器处理音频
            inputs = self.whisper_feature_extractor(
                audio_array, 
                sampling_rate=self.whisper_sampling_rate,
                return_tensors="pt"
            )
            
            # 3️⃣ **通过Whisper编码器提取特征**
            with torch.no_grad():
                # 获取编码器输出
                encoder_outputs = self.whisper_model.encoder(
                    inputs.input_features,
                    return_dict=True
                )
                
                # 获取最后一层的隐藏状态
                audio_features = encoder_outputs.last_hidden_state  # shape: (1, time_steps, hidden_size)
                audio_features = audio_features.squeeze(0)  # 移除batch维度: (time_steps, hidden_size)
                audio_features = audio_features.numpy()
            
            # 4️⃣ **调整特征长度以匹配视频帧数 (750帧)**
            current_frames = audio_features.shape[0]
            target_frames = self.target_length
            
            if current_frames < target_frames:
                # 如果特征帧数不足，使用重复填充
                repeat_factor = target_frames // current_frames + 1
                audio_features = np.tile(audio_features, (repeat_factor, 1))
                audio_features = audio_features[:target_frames]
            elif current_frames > target_frames:
                # 如果特征帧数过多，使用插值下采样
                indices = np.linspace(0, current_frames-1, target_frames).astype(int)
                audio_features = audio_features[indices] /21
            
            return audio_features  # shape: (750, hidden_size)
            
        except Exception as e:
            print(f"⚠️  Whisper音频特征提取失败: {audio_path} - {e}")
            return None

    def __len__(self):
        """返回数据集大小"""
        return len(self.mapping_df)

    def __getitem__(self, idx):
        """按索引读取数据（Lazy Loading）- 直接返回完整750帧样本"""
        row = self.mapping_df.iloc[idx]
        speaker_parquet = self._convert_path(row.iloc[1], "parquet")
        listener_parquet = self._convert_path(row.iloc[2], "parquet")
        speaker_audio = self._convert_audio_path(row.iloc[1], "wav")
        listener_audio = self._convert_audio_path(row.iloc[2], "wav")

        # 加载面部特征数据
        speaker_landmarks, speaker_au, speaker_pose, speaker_gaze = self._load_parquet(speaker_parquet)
        listener_landmarks, listener_au, listener_pose, listener_gaze = self._load_parquet(listener_parquet)
        
        # 加载音频特征数据 (使用Whisper)
        speaker_audio_features = self._load_audio_with_whisper(speaker_audio)
        listener_audio_features = self._load_audio_with_whisper(listener_audio)

        # 检查数据完整性
        face_data_valid = all(x is not None for x in [speaker_landmarks, speaker_au, speaker_pose, speaker_gaze,
                                                     listener_landmarks, listener_au, listener_pose, listener_gaze])
        audio_data_valid = all(x is not None for x in [speaker_audio_features, listener_audio_features])
        
        if not (face_data_valid and audio_data_valid):
            print(f"⚠️  跳过索引 {idx}: 数据加载失败")
            return None

        # 🔹 直接返回完整的750帧数据，不进行序列切片
        return {
            'speaker': {
                'landmarks': torch.FloatTensor(speaker_landmarks),     # (750, 136)
                'au': torch.FloatTensor(speaker_au),                  # (750, 18)
                'pose': torch.FloatTensor(speaker_pose),              # (750, 4)
                'gaze': torch.FloatTensor(speaker_gaze),              # (750, 2)
                'audio': torch.FloatTensor(speaker_audio_features)    # (750, 384)
            },
            'listener': {
                'landmarks': torch.FloatTensor(listener_landmarks),   # (750, 136)
                'au': torch.FloatTensor(listener_au),                # (750, 18)
                'pose': torch.FloatTensor(listener_pose),            # (750, 4)
                'gaze': torch.FloatTensor(listener_gaze),            # (750, 2)
                'audio': torch.FloatTensor(listener_audio_features)  # (750, 384)
            }
        }


def get_dataloader(mapping_csv, batch_size=2, num_workers=2, whisper_model_name="openai/whisper-tiny"):
    dataset = SpeakerListenerDataset(mapping_csv, whisper_model_name=whisper_model_name)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)


if __name__ == '__main__':
    # ✅ 指定测试路径
    mapping_csv = "/mnt/iusers01/fatpou01/compsci01/k09562zs/scratch/Face_data/val.csv"
    
    # ✅ 创建 DataLoader
    dataloader = get_dataloader(mapping_csv, batch_size=32, num_workers=0, whisper_model_name="openai/whisper-tiny")

    # ✅ 取出一个 batch 并检查形状
    for i, batch_data in enumerate(dataloader):
        if batch_data is None:
            continue
            
        print(f"🟢 Batch {i} Loaded Successfully!")

        # **检查Speaker面部特征数据 Shape**
        print(f"\n📊 Speaker Face Features:")
        print(f"  🗿 Landmarks Shape: {batch_data['speaker']['landmarks'].shape}")  # (batch, 750, 136)
        print(f"  😊 AU Features Shape: {batch_data['speaker']['au'].shape}")      # (batch, 750, 18)
        print(f"  📐 Pose Features Shape: {batch_data['speaker']['pose'].shape}")  # (batch, 750, 4)
        print(f"  👁️ Gaze Features Shape: {batch_data['speaker']['gaze'].shape}")   # (batch, 750, 2)

        # **检查Listener面部特征数据 Shape**
        print(f"\n📊 Listener Face Features:")
        print(f"  🗿 Landmarks Shape: {batch_data['listener']['landmarks'].shape}")
        print(f"  😊 AU Features Shape: {batch_data['listener']['au'].shape}")
        print(f"  📐 Pose Features Shape: {batch_data['listener']['pose'].shape}")
        print(f"  👁️ Gaze Features Shape: {batch_data['listener']['gaze'].shape}")

        # **检查Whisper音频特征数据 Shape**
        print(f"\n🎵 Audio Features (Whisper):")
        print(f"  🎵 Speaker Audio Shape: {batch_data['speaker']['audio'].shape}")    # (batch, 750, whisper_hidden_size)
        print(f"  🎵 Listener Audio Shape: {batch_data['listener']['audio'].shape}")

        # **数值范围检查**
        print(f"\n📈 Data Range Check:")
        print(f"  🗿 Landmarks range: [{batch_data['speaker']['landmarks'].min():.4f}, {batch_data['speaker']['landmarks'].max():.4f}]")
        print(f"  😊 AU range: [{batch_data['speaker']['au'].min():.4f}, {batch_data['speaker']['au'].max():.4f}]")
        print(f"  📐 Pose range: [{batch_data['speaker']['pose'].min():.4f}, {batch_data['speaker']['pose'].max():.4f}]")
        print(f"  👁️ Gaze range: [{batch_data['speaker']['gaze'].min():.4f}, {batch_data['speaker']['gaze'].max():.4f}]")
        print(f"  🎵 Audio range: [{batch_data['speaker']['audio'].min():.4f}, {batch_data['speaker']['audio'].max():.4f}]")

        # **检查序列长度**
        print(f"\n📏 Sequence Length Check:")
        print(f"  All sequences are 750 frames long: {batch_data['speaker']['landmarks'].shape[1]} frames")

        # **只测试一个 batch**
        break