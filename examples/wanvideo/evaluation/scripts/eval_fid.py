#!/usr/bin/env python3
"""
eval_fid.py - 计算Fréchet Inception Distance (FID)

核心修复：
1. 与真实数据比较（而非生成数据自己和自己比较）
2. 正确实现FID计算流程
3. 使用预训练Inception-v3模型

修复前的错误：
  计算 generated_features[:mid] 和 generated_features[mid:] 的FID
  这是错误的！这只是测量生成集内部的多样性

修复后的正确做法：
  计算 real_features 和 generated_features 的FID
  这才能测量生成图像对真实图像的保真度

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
from torchvision.models import inception_v3
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# FID计算器
# ============================================================================

class FIDCalculator:
    """
    FID（Fréchet Inception Distance）计算器
    
    FID原理：
    1. 使用Inception-v3提取真实图像和生成图像的特征
    2. 计算特征分布的均值和协方差
    3. 计算两个分布之间的Fréchet距离
    
    FID = ||μ_real - μ_gen||² + Tr(Σ_real + Σ_gen - 2√(Σ_real·Σ_gen))
    """
    
    def __init__(self, device='cuda', dims=2048):
        """
        初始化FID计算器
        
        Args:
            device: 计算设备
            dims: Inception特征维度（默认2048）
        """
        self.device = device
        self.dims = dims
        
        print("=" * 80)
        print("初始化FID计算器")
        print("=" * 80)
        print(f"设备: {device}")
        print(f"特征维度: {dims}")
        print()
        
        # 加载预训练的Inception-v3模型
        print("加载Inception-v3模型...")
        self.model = inception_v3(pretrained=True, transform_input=False)
        
        # 移除最后的分类层，只保留特征提取部分
        self.model.fc = torch.nn.Identity()
        
        self.model = self.model.to(device)
        self.model.eval()
        
        print("✓ Inception-v3模型加载完成")
        print("=" * 80)
        print()
    
    def extract_frames_from_video(self, video_path: Path, max_frames: int = 16):
        """
        从视频中提取帧
        
        Args:
            video_path: 视频文件路径
            max_frames: 最大提取帧数
            
        Returns:
            帧数组 [N, H, W, C]
        """
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        
        frame_count = 0
        while frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Resize to 299x299 (Inception-v3 input size)
            frame = cv2.resize(frame, (299, 299))
            
            frames.append(frame)
            frame_count += 1
        
        cap.release()
        
        return np.array(frames) if frames else np.array([])
    
    def load_images_from_directory(self, image_dir: Path):
        """
        从目录加载所有图像
        
        Args:
            image_dir: 图像目录
            
        Returns:
            图像数组 [N, H, W, C]
        """
        # 支持的图像格式
        image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp']
        image_paths = []
        
        for ext in image_extensions:
            image_paths.extend(list(image_dir.glob(ext)))
        
        if not image_paths:
            raise ValueError(f"目录 {image_dir} 中没有找到图像")
        
        images = []
        
        print(f"  从 {image_dir.name} 加载 {len(image_paths)} 张图像...")
        
        for img_path in tqdm(image_paths, desc="  加载图像", leave=False):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            
            # BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Resize to 299x299
            img = cv2.resize(img, (299, 299))
            
            images.append(img)
        
        return np.array(images)
    
    def load_frames_from_videos(self, video_dir: Path, level: str, max_frames_per_video: int = 16):
        """
        从视频目录提取所有帧
        
        Args:
            video_dir: 视频目录
            level: 级别（例如 "level_0" 或 "real"）
            max_frames_per_video: 每个视频最多提取的帧数
            
        Returns:
            帧数组 [N, H, W, C]
        """
        # ✅ 修复：处理真实数据的特殊情况
        if level == "real":
            # 真实数据：加载所有mp4视频
            video_pattern = "*.mp4"
            print(f"  加载真实数据视频（pattern: {video_pattern}）")
        else:
            # 生成数据：按级别过滤
            level_num = level.split('_')[1]
            video_pattern = f"video_{level_num}_*.mp4"
            print(f"  加载生成数据视频（pattern: {video_pattern}）")
        
        video_paths = sorted(glob(str(video_dir / video_pattern)))
        
        if not video_paths:
            raise ValueError(f"未找到 {level} 的视频文件（pattern: {video_pattern}）")
        
        all_frames = []
        
        print(f"  从 {len(video_paths)} 个视频提取帧...")
        
        for video_path in tqdm(video_paths, desc="  提取视频帧", leave=False):
            frames = self.extract_frames_from_video(Path(video_path), max_frames_per_video)
            if len(frames) > 0:
                all_frames.append(frames)
        
        # ✅ 修复：检查是否成功提取帧
        if not all_frames:
            raise ValueError(f"未能从任何视频中提取帧")
        
        # 合并所有帧
        all_frames = np.vstack(all_frames)
        
        print(f"  ✓ 提取了 {len(all_frames)} 帧")
        
        return all_frames
    
    def preprocess_images(self, images: np.ndarray):
        """
        预处理图像用于Inception-v3
        
        Args:
            images: 图像数组 [N, H, W, C]
            
        Returns:
            预处理后的张量 [N, C, H, W]
        """
        # Normalize to [-1, 1]
        images = images.astype(np.float32) / 255.0
        images = (images - 0.5) / 0.5
        
        # Convert to torch tensor [N, C, H, W]
        images = torch.from_numpy(images).permute(0, 3, 1, 2)
        
        return images
    
    def extract_features(self, images: np.ndarray, batch_size: int = 50):
        """
        提取Inception特征
        
        Args:
            images: 图像数组 [N, H, W, C]
            batch_size: 批处理大小
            
        Returns:
            特征数组 [N, dims]
        """
        num_images = len(images)
        
        print(f"  提取 {num_images} 张图像的Inception特征...")
        
        all_features = []
        
        for i in tqdm(range(0, num_images, batch_size), desc="  提取特征", leave=False):
            batch = images[i:i+batch_size]
            batch_tensor = self.preprocess_images(batch).to(self.device)
            
            with torch.no_grad():
                features = self.model(batch_tensor)
            
            all_features.append(features.cpu().numpy())
        
        # 合并所有特征
        all_features = np.vstack(all_features)
        
        print(f"  ✓ 特征提取完成，形状: {all_features.shape}")
        
        return all_features
    
    def calculate_frechet_distance(self, mu1, sigma1, mu2, sigma2, eps=1e-6):
        """
        计算Fréchet距离
        
        Args:
            mu1: 分布1的均值向量
            sigma1: 分布1的协方差矩阵
            mu2: 分布2的均值向量
            sigma2: 分布2的协方差矩阵
            eps: 数值稳定性常数
            
        Returns:
            Fréchet距离
        """
        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)
        
        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)
        
        assert mu1.shape == mu2.shape, "均值向量维度必须相同"
        assert sigma1.shape == sigma2.shape, "协方差矩阵维度必须相同"
        
        # 计算均值差异
        diff = mu1 - mu2
        
        # 计算协方差矩阵的平方根
        # 使用 scipy.linalg.sqrtm
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        
        # 处理数值误差导致的复数结果
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError(f"协方差平方根包含显著虚部: {m}")
            covmean = covmean.real
        
        # FID公式
        # FID = ||μ1 - μ2||² + Tr(Σ1 + Σ2 - 2√(Σ1·Σ2))
        fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
        
        return float(fid)
    
    def calculate_statistics(self, features: np.ndarray):
        """
        计算特征的统计量（均值和协方差）
        
        Args:
            features: 特征数组 [N, D]
            
        Returns:
            (mu, sigma) 均值和协方差矩阵
        """
        mu = np.mean(features, axis=0)
        sigma = np.cov(features, rowvar=False)
        
        return mu, sigma
    
    def calculate_fid(
        self,
        real_images: np.ndarray,
        generated_images: np.ndarray,
        batch_size: int = 50
    ):
        """
        计算FID分数
        
        核心修复：真实数据 vs 生成数据
        
        Args:
            real_images: 真实图像数组 [N, H, W, C]
            generated_images: 生成图像数组 [M, H, W, C]
            batch_size: 批处理大小
            
        Returns:
            FID分数
        """
        print()
        print("=" * 80)
        print("计算FID分数")
        print("=" * 80)
        print(f"真实图像数量: {len(real_images)}")
        print(f"生成图像数量: {len(generated_images)}")
        print("=" * 80)
        print()
        
        # 提取真实图像特征
        print("[1/4] 提取真实图像特征...")
        real_features = self.extract_features(real_images, batch_size)
        
        # 提取生成图像特征
        print()
        print("[2/4] 提取生成图像特征...")
        gen_features = self.extract_features(generated_images, batch_size)
        
        # 计算真实图像的统计量
        print()
        print("[3/4] 计算真实图像统计量...")
        mu_real, sigma_real = self.calculate_statistics(real_features)
        print(f"  ✓ 真实图像均值形状: {mu_real.shape}")
        print(f"  ✓ 真实图像协方差形状: {sigma_real.shape}")
        
        # 计算生成图像的统计量
        print()
        print("[4/4] 计算生成图像统计量...")
        mu_gen, sigma_gen = self.calculate_statistics(gen_features)
        print(f"  ✓ 生成图像均值形状: {mu_gen.shape}")
        print(f"  ✓ 生成图像协方差形状: {sigma_gen.shape}")
        
        # 计算Fréchet距离
        print()
        print("计算Fréchet距离...")
        fid_score = self.calculate_frechet_distance(
            mu_real, sigma_real,
            mu_gen, sigma_gen
        )
        
        print()
        print("=" * 80)
        print(f"FID Score: {fid_score:.6f}")
        print("=" * 80)
        print()
        
        return fid_score

# ============================================================================
# 主函数
# ============================================================================

def main():
    """主函数"""
    
    parser = argparse.ArgumentParser(
        description="计算FID分数（修正版）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 真实数据为图像目录
  python eval_fid.py \\
      --videos_dir ../../evaluation_outputs/videos \\
      --real_data_dir /path/to/real/frames \\
      --level level_0 \\
      --output_file ../../evaluation_outputs/metrics/fid_level_0.json

  # 真实数据为视频目录（会自动提取帧）
  python eval_fid.py \\
      --videos_dir ../../evaluation_outputs/videos \\
      --real_data_dir /path/to/real/videos \\
      --level level_0 \\
      --output_file ../../evaluation_outputs/metrics/fid_level_0.json

核心修复说明:
  此版本确保FID是通过比较真实数据和生成数据的分布来计算的，
  而不是错误地比较生成数据的两个子集。
        """
    )
    
    parser.add_argument(
        "--videos_dir",
        type=str,
        required=True,
        help="生成的视频目录"
    )
    
    parser.add_argument(
        "--real_data_dir",
        type=str,
        required=True,
        help="真实数据目录（图像或视频）"
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
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=50,
        help="批处理大小（默认：50）"
    )
    
    parser.add_argument(
        "--max_frames_per_video",
        type=int,
        default=16,
        help="每个视频最多提取的帧数（默认：16）"
    )
    
    args = parser.parse_args()
    
    # 初始化FID计算器
    calculator = FIDCalculator(device=args.device)
    
    # 加载生成的视频帧
    print()
    print("=" * 80)
    print("步骤1: 加载生成的视频帧")
    print("=" * 80)
    
    generated_images = calculator.load_frames_from_videos(
        video_dir=Path(args.videos_dir),
        level=args.level,
        max_frames_per_video=args.max_frames_per_video
    )
    
    # 加载真实数据
    print()
    print("=" * 80)
    print("步骤2: 加载真实数据")
    print("=" * 80)
    
    real_data_path = Path(args.real_data_dir)
    
    # 检查真实数据是图像目录还是视频目录
    has_images = any(real_data_path.glob('*.png')) or any(real_data_path.glob('*.jpg'))
    has_videos = any(real_data_path.glob('*.mp4'))
    
    if has_images:
        print("  检测到真实数据为图像目录")
        real_images = calculator.load_images_from_directory(real_data_path)
    elif has_videos:
        print("  检测到真实数据为视频目录，将提取帧")
        real_images = calculator.load_frames_from_videos(
            video_dir=real_data_path,
            level="real",  # 使用特殊标识
            max_frames_per_video=args.max_frames_per_video
        )
    else:
        raise ValueError(f"真实数据目录 {real_data_path} 中未找到图像或视频")
    
    # 计算FID
    print()
    print("=" * 80)
    print("步骤3: 计算FID")
    print("=" * 80)
    
    fid_score = calculator.calculate_fid(
        real_images=real_images,
        generated_images=generated_images,
        batch_size=args.batch_size
    )
    
    # 保存结果
    results = {
        "level": args.level,
        "fid_score": fid_score,
        "lower_is_better": True,
        "num_real_images": len(real_images),
        "num_generated_images": len(generated_images),
        "real_data_dir": str(args.real_data_dir),
        "generated_videos_dir": str(args.videos_dir),
        "batch_size": args.batch_size,
        "max_frames_per_video": args.max_frames_per_video
    }
    
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print()
    print("=" * 80)
    print("FID评估完成")
    print("=" * 80)
    print(f"结果已保存到: {output_path}")
    print("=" * 80)
    print()

if __name__ == "__main__":
    main()
