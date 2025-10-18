#!/usr/bin/env python3
"""
eval_fvd.py - 计算FVD（使用项目本地I3D模型）

模型位置: ../models/i3d/I3D_8x8_R50.pyth
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
# 配置
# ============================================================================

# 项目根目录下的模型路径
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # evaluation目录
DEFAULT_I3D_PATH = PROJECT_ROOT / "models" / "i3d" / "I3D_8x8_R50.pyth"

# ============================================================================
# I3D特征提取器
# ============================================================================

class I3DFeatureExtractor:
    """I3D特征提取器（优先使用本地模型）"""
    
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
        print(f"项目根目录: {PROJECT_ROOT}")
        print(f"默认I3D路径: {DEFAULT_I3D_PATH}")
        print()
        
        # 加载I3D模型
        self._load_i3d_model()
        
        print("=" * 80)
        print()
    
    def _load_i3d_model(self):
        """加载I3D模型（优先级：本地 > 在线 > R3D备选）"""
        
        # ========== 优先级1: 项目本地模型 ==========
        if DEFAULT_I3D_PATH.exists():
            print(f"[优先级1] 发现本地I3D模型: {DEFAULT_I3D_PATH}")
            print(f"  文件大小: {DEFAULT_I3D_PATH.stat().st_size / 1024 / 1024:.1f} MB")
            
            try:
                # 加载模型结构（不下载权重）
                import torch.hub
                
                print("  加载I3D模型结构...")
                self.model = torch.hub.load(
                    'facebookresearch/pytorchvideo',
                    'i3d_r50',
                    pretrained=False  # 不下载预训练权重
                )
                
                # 加载本地权重
                print("  加载本地权重...")
                checkpoint = torch.load(DEFAULT_I3D_PATH, map_location='cpu')
                
                # 提取state_dict
                if isinstance(checkpoint, dict):
                    if 'model_state' in checkpoint:
                        state_dict = checkpoint['model_state']
                    elif 'state_dict' in checkpoint:
                        state_dict = checkpoint['state_dict']
                    else:
                        state_dict = checkpoint
                else:
                    state_dict = checkpoint
                
                self.model.load_state_dict(state_dict, strict=False)
                
                # 移除分类层
                if hasattr(self.model, 'head'):
                    self.model.head = torch.nn.Identity()
                
                self.model = self.model.to(self.device).float().eval()
                
                print("  ✓ 本地I3D模型加载成功")
                self.model_type = "i3d_local"
                return
                
            except Exception as e:
                print(f"  ✗ 加载本地模型失败: {e}")
                print(f"  将尝试其他方法...")
                import traceback
                traceback.print_exc()
        else:
            print(f"[优先级1] 本地I3D模型不存在: {DEFAULT_I3D_PATH}")
            print(f"  请按照README说明下载并上传模型")
            print()
        
        # ========== 优先级2: 在线下载（到本地缓存） ==========
        try:
            print("[优先级2] 尝试在线下载I3D模型...")
            print("  ⚠️  这可能需要较长时间（约214MB）")
            
            import torch.hub
            
            # 设置缓存目录到项目内
            cache_dir = PROJECT_ROOT / "models" / ".cache"
            cache_dir.mkdir(parents=True, exist_ok=True)
            torch.hub.set_dir(str(cache_dir))
            
            print(f"  缓存目录: {cache_dir}")
            
            self.model = torch.hub.load(
                'facebookresearch/pytorchvideo',
                'i3d_r50',
                pretrained=True
            )
            
            if hasattr(self.model, 'head'):
                self.model.head = torch.nn.Identity()
            
            self.model = self.model.to(self.device).float().eval()
            
            print("  ✓ I3D模型在线下载成功")
            self.model_type = "i3d_online"
            return
            
        except Exception as e:
            print(f"  ✗ 在线下载失败: {e}")
        
        # ========== 优先级3: 使用3D ResNet备选 ==========
        try:
            print("[优先级3] 使用3D ResNet-18作为备选...")
            print("  ⚠️  注意: 这不是标准I3D，结果可能略有不同")
            
            from torchvision.models.video import r3d_18
            
            self.model = r3d_18(pretrained=True)
            self.model.fc = torch.nn.Identity()
            self.model = self.model.to(self.device).float().eval()
            
            print("  ✓ 3D ResNet-18加载成功")
            self.model_type = "r3d_18"
            return
            
        except Exception as e:
            print(f"  ✗ 3D ResNet加载失败: {e}")
            raise RuntimeError("无法加载任何视频特征提取模型")
    
    def load_video_frames(self, video_path: Path, num_frames: int = None):
        """加载视频帧"""
        if num_frames is None:
            num_frames = self.num_frames
        
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
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
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, self.input_size)
                frames.append(frame)
            
            frame_idx += 1
        
        cap.release()
        
        while len(frames) < num_frames:
            if frames:
                frames.append(frames[-1])
            else:
                frames.append(np.zeros((self.input_size[1], self.input_size[0], 3), dtype=np.uint8))
        
        return np.array(frames[:num_frames], dtype=np.float32)
    
    def preprocess_video(self, frames: np.ndarray):
        """预处理视频帧"""
        if frames.dtype != np.float32:
            frames = frames.astype(np.float32)
        
        frames = frames / 255.0
        
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        
        frames = (frames - mean) / std
        
        frames_tensor = torch.from_numpy(frames).permute(3, 0, 1, 2).unsqueeze(0)
        frames_tensor = frames_tensor.to(device=self.device, dtype=torch.float32)
        
        return frames_tensor
    
    def extract_features(self, video_path: Path):
        """提取单个视频特征"""
        frames = self.load_video_frames(video_path)
        video_tensor = self.preprocess_video(frames)
        
        with torch.no_grad():
            features = self.model(video_tensor)
        
        return features.cpu().numpy().astype(np.float32).flatten()
    
    def extract_features_batch(self, video_paths: list):
        """批量提取视频特征"""
        all_features = []
        
        for video_path in tqdm(video_paths, desc="提取I3D特征"):
            try:
                features = self.extract_features(Path(video_path))
                all_features.append(features)
            except Exception as e:
                print(f"\n⚠️  提取 {Path(video_path).name} 失败: {e}")
                continue
        
        if not all_features:
            raise ValueError("未能提取任何视频特征")
        
        return np.vstack(all_features).astype(np.float32)

# ============================================================================
# FVD计算器
# ============================================================================

class FVDCalculator:
    """FVD计算器"""
    
    def __init__(self, device='cuda'):
        self.device = device
        self.feature_extractor = I3DFeatureExtractor(device=device)
    
    def calculate_frechet_distance(self, mu1, sigma1, mu2, sigma2):
        """计算Fréchet距离"""
        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)
        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)
        
        diff = mu1 - mu2
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError(f"协方差平方根包含显著虚部: {m}")
            covmean = covmean.real
        
        fvd = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
        return float(fvd)
    
    def calculate_statistics(self, features: np.ndarray):
        """计算统计量"""
        mu = np.mean(features, axis=0)
        sigma = np.cov(features, rowvar=False)
        return mu, sigma
    
    def calculate_fvd(self, real_video_paths: list, generated_video_paths: list):
        """计算FVD分数"""
        print()
        print("=" * 80)
        print("计算FVD分数")
        print("=" * 80)
        print(f"真实视频: {len(real_video_paths)} 个")
        print(f"生成视频: {len(generated_video_paths)} 个")
        print(f"模型类型: {self.feature_extractor.model_type}")
        print("=" * 80)
        print()
        
        print("[1/4] 提取真实视频特征...")
        real_features = self.feature_extractor.extract_features_batch(real_video_paths)
        print(f"  ✓ 真实视频特征: {real_features.shape}")
        
        print()
        print("[2/4] 提取生成视频特征...")
        gen_features = self.feature_extractor.extract_features_batch(generated_video_paths)
        print(f"  ✓ 生成视频特征: {gen_features.shape}")
        
        print()
        print("[3/4] 计算真实视频统计量...")
        mu_real, sigma_real = self.calculate_statistics(real_features)
        
        print()
        print("[4/4] 计算生成视频统计量...")
        mu_gen, sigma_gen = self.calculate_statistics(gen_features)
        
        print()
        print("计算Fréchet距离...")
        fvd_score = self.calculate_frechet_distance(mu_real, sigma_real, mu_gen, sigma_gen)
        
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
    parser = argparse.ArgumentParser(
        description="计算FVD（使用项目本地I3D模型）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
I3D模型位置:
  {DEFAULT_I3D_PATH}

如果模型不存在，请：
  1. 在本地下载: https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/kinetics/I3D_8x8_R50.pyth
  2. 上传到服务器: {DEFAULT_I3D_PATH}

或者脚本会自动使用3D ResNet作为备选
        """
    )
    
    parser.add_argument("--videos_dir", type=str, required=True, help="生成视频目录")
    parser.add_argument("--real_videos_dir", type=str, required=True, help="真实视频目录")
    parser.add_argument("--level", type=str, required=True, choices=["level_0", "level_1", "level_2", "level_3"])
    parser.add_argument("--output_file", type=str, required=True, help="输出JSON文件")
    parser.add_argument("--device", type=str, default="cuda")
    
    args = parser.parse_args()
    
    # 查找视频
    level_num = args.level.split('_')[1]
    video_pattern = f"video_{level_num}_*.mp4"
    generated_video_paths = sorted(glob(str(Path(args.videos_dir) / video_pattern)))
    real_video_paths = sorted(glob(str(Path(args.real_videos_dir) / "*.mp4")))
    
    if not generated_video_paths:
        raise ValueError(f"未找到生成视频: {args.videos_dir}/{video_pattern}")
    if not real_video_paths:
        raise ValueError(f"未找到真实视频: {args.real_videos_dir}")
    
    print()
    print("=" * 80)
    print("FVD评估")
    print("=" * 80)
    print(f"生成视频: {len(generated_video_paths)} 个")
    print(f"真实视频: {len(real_video_paths)} 个")
    print(f"级别: {args.level}")
    print("=" * 80)
    print()
    
    # 计算FVD
    calculator = FVDCalculator(device=args.device)
    fvd_score = calculator.calculate_fvd(real_video_paths, generated_video_paths)
    
    # 保存结果
    results = {
        "level": args.level,
        "fvd_score": fvd_score,
        "lower_is_better": True,
        "num_real_videos": len(real_video_paths),
        "num_generated_videos": len(generated_video_paths),
        "model_type": calculator.feature_extractor.model_type,
        "model_path": str(DEFAULT_I3D_PATH) if DEFAULT_I3D_PATH.exists() else "fallback"
    }
    
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print()
    print("=" * 80)
    print(f"结果已保存: {output_path}")
    print("=" * 80)

if __name__ == "__main__":
    main()
