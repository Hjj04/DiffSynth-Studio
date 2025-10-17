#!/usr/bin/env python3
"""
eval_fvd.py - 计算Fréchet Video Distance (FVD)

核心修复：
1. 使用预训练的I3D模型（而非未训练的简单CNN）
2. 与真实视频比较（而非生成视频自己和自己比较）
3. 正确实现FVD计算流程

修复前的错误：
  使用随机初始化的简单3D CNN提取特征
  这些特征是无意义的噪声，无法反映真实的视频质量

修复后的正确做法：
  使用在Kinetics数据集上预训练的I3D模型
  该模型理解真实世界的运动模式和物体

作者: AI Assistant
日期: 2025-01-16
"""

import argparse
import json
import numpy as np
import torch
import cv2
from pathlib import Path
from glob import glob
from tqdm import tqdm
from scipy import linalg
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# I3D特征提取器
# ============================================================================

class I3DFeatureExtractor:
    """
    I3D特征提取器
    
    I3D (Inflated 3D ConvNet) 是一个在大规模视频动作识别数据集
    （Kinetics）上预训练的3D卷积网络，专门用于理解视频中的运动模式
    """
    
    def __init__(self, device='cuda'):
        """
        初始化I3D模型
        
        Args:
            device: 计算设备
        """
        self.device = device
        self.input_size = (224, 224)
        self.num_frames = 16
        
        print("=" * 80)
        print("初始化I3D特征提取器")
        print("=" * 80)
        print(f"设备: {device}")
        print()
        
        # 尝试加载预训练I3D模型
        self._load_i3d_model()
        
        print("=" * 80)
        print()
    
    def _load_i3d_model(self):
        """
        加载预训练I3D模型
        
        尝试多种来源：
        1. PyTorch Hub (pytorchvideo)
        2. torchvision (如果有I3D)
        3. 备选：使用3D ResNet（次优但可用）
        """
        print("尝试加载预训练I3D模型...")
        
        # 方法1: 从PyTorch Hub加载（推荐）
        try:
            print("  [方法1] 从PyTorch Hub加载 pytorchvideo I3D...")
            
            # 需要安装: pip install pytorchvideo
            import torch.hub
            
            self.model = torch.hub.load(
                'facebookresearch/pytorchvideo',
                'i3d_r50',
                pretrained=True
            )
            
            # 移除最后的分类层
            if hasattr(self.model, 'head'):
                self.model.head = torch.nn.Identity()
            
            self.model = self.model.to(self.device)
            self.model.eval()
            
            print("  ✓ I3D模型加载成功（PyTorch Hub）")
            self.model_type = "i3d_pytorch_hub"
            
            return
            
        except Exception as e:
            print(f"  ✗ 从PyTorch Hub加载失败: {e}")
        
        # 方法2: 使用3D ResNet作为备选
        try:
            print("  [方法2] 使用3D ResNet-18作为备选...")
            
            from torchvision.models.video import r3d_18
            
            self.model = r3d_18(pretrained=True)
            
            # 移除分类层
            self.model.fc = torch.nn.Identity()
            
            self.model = self.model.to(self.device)
            self.model.eval()
            
            print("  ✓ 3D ResNet-18加载成功")
            print("  ⚠️  注意: 这不是标准I3D，但可用于FVD计算")
            self.model_type = "r3d_18"
            
            return
            
        except Exception as e:
            print(f"  ✗ 加载3D ResNet失败: {e}")
        
        # 方法3: 构建简单3D CNN（最后备选，不推荐）
        print("  [方法3] 使用简单3D CNN（不推荐，仅用于测试）...")
        print("  ⚠️  警告: 这不是预训练模型，FVD结果可能不准确")
        
        self.model = self._build_fallback_3dcnn()
        self.model_type = "simple_3dcnn"
    
    def _build_fallback_3dcnn(self):
        """
        构建简单3D CNN作为最后备选
        
        注意: 这不是预训练模型，结果不可靠！
        """
        import torch.nn as nn
        
        model = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(2),
            
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.MaxPool3d(2),
            
            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.MaxPool3d(2),
            
            nn.Conv3d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(),
            
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten()
        )
        
        return model.to(self.device).eval()
    
    def load_video_frames(self, video_path: Path, num_frames: int = None):
        """
        加载视频帧
        
        Args:
            video_path: 视频文件路径
            num_frames: 要加载的帧数（默认：self.num_frames）
            
        Returns:
            视频帧数组 [T, H, W, C]
        """
        if num_frames is None:
            num_frames = self.num_frames
        
        cap = cv2.VideoCapture(str(video_path))
        
        # 获取总帧数
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 均匀采样
        if total_frames > num_frames:
            indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        else:
            indices = list(range(total_frames))
        
        frames = []
        frame_idx = 0
        
        while len(frames) < num_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx in indices:
                # BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Resize
                frame = cv2.resize(frame, self.input_size)
                
                frames.append(frame)
            
            frame_idx += 1
        
        cap.release()
        
        # Pad if necessary
        while len(frames) < num_frames:
            if frames:
                frames.append(frames[-1])
            else:
                frames.append(np.zeros((self.input_size[1], self.input_size[0], 3), dtype=np.uint8))
        
        return np.array(frames[:num_frames])
    
    def preprocess_video(self, frames: np.ndarray):
        """
        预处理视频帧
        
        Args:
            frames: 视频帧 [T, H, W, C]
            
        Returns:
            预处理后的张量 [1, C, T, H, W]
        """
        # Normalize to [0, 1]
        frames = frames.astype(np.float32) / 255.0
        
        # Normalize using ImageNet statistics
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        
        frames = (frames - mean) / std
        
        # Convert to tensor: [T, H, W, C] -> [C, T, H, W] -> [1, C, T, H, W]
        frames = torch.from_numpy(frames).permute(3, 0, 1, 2).unsqueeze(0)
        
        return frames.to(self.device)
    
    def extract_features(self, video_path: Path):
        """
        提取单个视频的I3D特征
        
        Args:
            video_path: 视频文件路径
            
        Returns:
            特征向量 [D]
        """
        # 加载视频帧
        frames = self.load_video_frames(video_path)
        
        # 预处理
        video_tensor = self.preprocess_video(frames)
        
        # 提取特征
        with torch.no_grad():
            features = self.model(video_tensor)
        
        return features.cpu().numpy().flatten()
    
    def extract_features_batch(self, video_paths: list):
        """
        批量提取视频特征
        
        Args:
            video_paths: 视频路径列表
            
        Returns:
            特征矩阵 [N, D]
        """
        all_features = []
        
        for video_path in tqdm(video_paths, desc="提取I3D特征", leave=False):
            try:
                features = self.extract_features(Path(video_path))
                all_features.append(features)
            except Exception as e:
                print(f"\n⚠️  提取 {Path(video_path).name} 的特征失败: {e}")
                continue
        
        if not all_features:
            raise ValueError("未能提取任何视频特征")
        
        return np.vstack(all_features)

# ============================================================================
# FVD计算器
# ============================================================================

class FVDCalculator:
    """
    FVD（Fréchet Video Distance）计算器
    
    FVD原理：
    与FID类似，但使用视频特征而非图像特征
    1. 使用I3D模型提取真实视频和生成视频的特征
    2. 计算特征分布的均值和协方差
    3. 计算两个分布之间的Fréchet距离
    """
    
    def __init__(self, device='cuda'):
        """
        初始化FVD计算器
        
        Args:
            device: 计算设备
        """
        self.device = device
        self.feature_extractor = I3DFeatureExtractor(device=device)
    
    def calculate_frechet_distance(self, mu1, sigma1, mu2, sigma2):
        """
        计算Fréchet距离
        
        Args:
            mu1: 分布1的均值向量
            sigma1: 分布1的协方差矩阵
            mu2: 分布2的均值向量
            sigma2: 分布2的协方差矩阵
            
        Returns:
            Fréchet距离
        """
        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)
        
        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)
        
        # 计算均值差异
        diff = mu1 - mu2
        
        # 计算协方差平方根
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        
        # 处理数值误差
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError(f"协方差平方根包含显著虚部: {m}")
            covmean = covmean.real
        
        # FVD公式
        fvd = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
        
        return float(fvd)
    
    def calculate_statistics(self, features: np.ndarray):
        """
        计算特征的统计量
        
        Args:
            features: 特征数组 [N, D]
            
        Returns:
            (mu, sigma) 均值和协方差矩阵
        """
        mu = np.mean(features, axis=0)
        sigma = np.cov(features, rowvar=False)
        
        return mu, sigma
    
    def calculate_fvd(
        self,
        real_video_paths: list,
        generated_video_paths: list
    ):
        """
        计算FVD分数
        
        核心修复：真实视频 vs 生成视频
        
        Args:
            real_video_paths: 真实视频路径列表
            generated_video_paths: 生成视频路径列表
            
        Returns:
            FVD分数
        """
        print()
        print("=" * 80)
        print("计算FVD分数")
        print("=" * 80)
        print(f"真实视频数量: {len(real_video_paths)}")
        print(f"生成视频数量: {len(generated_video_paths)}")
        print(f"模型类型: {self.feature_extractor.model_type}")
        print("=" * 80)
        print()
        
        # 提取真实视频特征
        print("[1/4] 提取真实视频特征...")
        real_features = self.feature_extractor.extract_features_batch(real_video_paths)
        print(f"  ✓ 真实视频特征形状: {real_features.shape}")
        
        # 提取生成视频特征
        print()
        print("[2/4] 提取生成视频特征...")
        gen_features = self.feature_extractor.extract_features_batch(generated_video_paths)
        print(f"  ✓ 生成视频特征形状: {gen_features.shape}")
        
        # 计算真实视频的统计量
        print()
        print("[3/4] 计算真实视频统计量...")
        mu_real, sigma_real = self.calculate_statistics(real_features)
        print(f"  ✓ 真实视频均值形状: {mu_real.shape}")
        print(f"  ✓ 真实视频协方差形状: {sigma_real.shape}")
        
        # 计算生成视频的统计量
        print()
        print("[4/4] 计算生成视频统计量...")
        mu_gen, sigma_gen = self.calculate_statistics(gen_features)
        print(f"  ✓ 生成视频均值形状: {mu_gen.shape}")
        print(f"  ✓ 生成视频协方差形状: {sigma_gen.shape}")
        
        # 计算Fréchet距离
        print()
        print("计算Fréchet距离...")
        fvd_score = self.calculate_frechet_distance(
            mu_real, sigma_real,
            mu_gen, sigma_gen
        )
        
        print()
        print("=" * 80)
        print(f"FVD Score: {fvd_score:.6f}")
        print("=" * 80)
        print()
        
        return fvd_score

# ============================================================================
# 主函数
# ============================================================================

def main():
    """主函数"""
    
    parser = argparse.ArgumentParser(
        description="计算FVD分数（修正版）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python eval_fvd.py \\
      --videos_dir ../../evaluation_outputs/videos \\
      --real_videos_dir /path/to/real/videos \\
      --level level_0 \\
      --output_file ../../evaluation_outputs/metrics/fvd_level_0.json

核心修复说明:
  此版本使用预训练I3D模型提取特征，并与真实视频比较。
  如果I3D不可用，会使用3D ResNet作为备选。

依赖安装:
  pip install pytorchvideo  # 用于加载I3D模型
        """
    )
    
    parser.add_argument(
        "--videos_dir",
        type=str,
        required=True,
        help="生成的视频目录"
    )
    
    parser.add_argument(
        "--real_videos_dir",
        type=str,
        required=True,
        help="真实视频目录"
    )
    
    parser.add_argument(
        "--level",
        type=str,
        required=True,
        choices=["level_0", "level_1", "level_2", "level_3"],
        help="评估级别"
    )
    
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="输出JSON文件路径"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="计算设备（默认：cuda）"
    )
    
    args = parser.parse_args()
    
    # 查找生成视频
    level_num = args.level.split('_')[1]
    video_pattern = f"video_{level_num}_*.mp4"
    generated_video_paths = sorted(glob(str(Path(args.videos_dir) / video_pattern)))
    
    if not generated_video_paths:
        raise ValueError(f"未找到 {args.level} 的生成视频")
    
    # 查找真实视频
    real_video_paths = sorted(glob(str(Path(args.real_videos_dir) / "*.mp4")))
    
    if not real_video_paths:
        raise ValueError(f"未找到真实视频")
    
    print()
    print("=" * 80)
    print("FVD评估（修正版）")
    print("=" * 80)
    print(f"生成视频: {len(generated_video_paths)} 个")
    print(f"真实视频: {len(real_video_paths)} 个")
    print(f"级别: {args.level}")
    print("=" * 80)
    print()
    
    # 初始化FVD计算器
    calculator = FVDCalculator(device=args.device)
    
    # 计算FVD
    fvd_score = calculator.calculate_fvd(
        real_video_paths=real_video_paths,
        generated_video_paths=generated_video_paths
    )
    
    # 保存结果
    results = {
        "level": args.level,
        "fvd_score": fvd_score,
        "lower_is_better": True,
        "num_real_videos": len(real_video_paths),
        "num_generated_videos": len(generated_video_paths),
        "real_videos_dir": str(args.real_videos_dir),
        "generated_videos_dir": str(args.videos_dir),
        "model_type": calculator.feature_extractor.model_type
    }
    
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print()
    print("=" * 80)
    print("FVD评估完成")
    print("=" * 80)
    print(f"结果已保存到: {output_path}")
    print("=" * 80)
    print()

if __name__ == "__main__":
    main()
