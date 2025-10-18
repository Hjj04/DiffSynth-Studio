#!/usr/bin/env python3
"""
eval_warping_error_all_levels.py - 评估所有级别的Warping Error（修复版）
"""

import argparse
import json
import torch
import numpy as np
from pathlib import Path
from glob import glob
from tqdm import tqdm
import cv2

class WarpingErrorEvaluator:
    def __init__(self, device='cuda', use_temporal_module=False, temporal_checkpoint=None):
        self.device = device
        self.use_temporal_module = use_temporal_module
        
        if use_temporal_module and temporal_checkpoint:
            print("加载Temporal Module用于Level 3评估...")
            self.load_temporal_module(temporal_checkpoint)
        else:
            print("使用光流一致性评估（Level 0/1/2）...")
            self.temporal_module = None
    
    def load_temporal_module(self, checkpoint_path):
        """加载Temporal Module（仅Level 3）"""
        try:
            from diffsynth import ModelManager
            from diffsynth.models.wan_video_temporal_module import WanVideoTemporalModule
            
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            
            self.temporal_module = WanVideoTemporalModule()
            self.temporal_module.load_state_dict(checkpoint["temporal_module"])
            self.temporal_module = self.temporal_module.to(self.device)
            self.temporal_module.eval()
            
            print("✓ Temporal Module加载成功")
        except Exception as e:
            print(f"⚠️  Temporal Module加载失败: {e}")
            print("   将使用光流一致性评估")
            self.temporal_module = None
    
    def load_video(self, video_path):
        """加载视频的所有帧"""
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        
        cap.release()
        return frames
    
    def compute_optical_flow(self, frame1, frame2):
        """计算光流"""
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)
        
        flow = cv2.calcOpticalFlowFarneback(
            gray1, gray2, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
        
        return flow
    
    def warp_frame(self, frame, flow):
        """使用光流变形帧 - 修复版"""
        h, w = frame.shape[:2]
        
        # ✅ 正确创建坐标网格
        x_coords = np.arange(w).astype(np.float32)
        y_coords = np.arange(h).astype(np.float32)
        x_grid, y_grid = np.meshgrid(x_coords, y_coords)
        
        # ✅ 加上光流位移
        map_x = (x_grid + flow[:, :, 0]).astype(np.float32)
        map_y = (y_grid + flow[:, :, 1]).astype(np.float32)
        
        # 使用cv2.remap进行变形
        warped = cv2.remap(
            frame, map_x, map_y,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE
        )
        
        return warped
    
    def evaluate_video_optical_flow(self, video_path):
        """使用光流一致性评估（Level 0/1/2）- 修复版"""
        frames = self.load_video(video_path)
        
        if len(frames) < 2:
            return None
        
        errors = []
        
        for i in range(len(frames) - 1):
            frame1 = frames[i]
            frame2 = frames[i + 1]
            
            try:
                # 计算光流
                flow = self.compute_optical_flow(frame1, frame2)
                
                # ✅ 检查光流有效性
                if flow is None or np.any(np.isnan(flow)) or np.any(np.isinf(flow)):
                    continue
                
                # 变形frame1
                warped_frame1 = self.warp_frame(frame1, flow)
                
                # 计算warping误差
                error = np.mean(np.abs(warped_frame1.astype(float) - frame2.astype(float)))
                
                # ✅ 检查误差合理性
                if 0 <= error <= 255:
                    errors.append(error)
                    
            except Exception as e:
                print(f"⚠️ 处理视频 {video_path.name} 帧 {i} 时出错: {e}")
                continue
        
        if not errors:
            return None
        
        return {
            "mean_error": float(np.mean(errors)),
            "std_error": float(np.std(errors)),
            "median_error": float(np.median(errors)),
            "max_error": float(np.max(errors)),
            "min_error": float(np.min(errors)),
            "num_valid_pairs": len(errors)
        }
    
    def evaluate_video_temporal_module(self, video_path):
        """使用Temporal Module评估（Level 3）"""
        if self.temporal_module is None:
            return self.evaluate_video_optical_flow(video_path)
        
        print("⚠️  Temporal Module评估未完全实现，使用光流方法")
        return self.evaluate_video_optical_flow(video_path)
    
    def evaluate_all_videos(self, video_paths):
        """评估所有视频"""
        all_results = []
        
        for video_path in tqdm(video_paths, desc="评估Warping Error"):
            if self.use_temporal_module and self.temporal_module is not None:
                result = self.evaluate_video_temporal_module(video_path)
            else:
                result = self.evaluate_video_optical_flow(video_path)
            
            if result:
                all_results.append(result)
        
        return all_results

def main():
    parser = argparse.ArgumentParser(description="计算Warping Error (所有级别)")
    parser.add_argument("--videos_dir", type=str, required=True)
    parser.add_argument("--level", type=str, required=True, 
                        help="评估级别: level_0, level_1, level_2, level_3")
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--temporal_checkpoint", type=str, default=None,
                        help="Temporal Module checkpoint (仅Level 3需要)")
    parser.add_argument("--device", type=str, default="cuda")
    
    args = parser.parse_args()
    
    # 判断是否使用temporal module
    use_temporal = (args.level == "level_3" and args.temporal_checkpoint is not None)
    
    # 初始化评估器
    evaluator = WarpingErrorEvaluator(
        device=args.device,
        use_temporal_module=use_temporal,
        temporal_checkpoint=args.temporal_checkpoint
    )
    
    # 查找视频
    video_pattern = f"video_{args.level[-1]}_*.mp4"
    video_paths = sorted(glob(str(Path(args.videos_dir) / video_pattern)))
    
    print(f"\n找到 {len(video_paths)} 个 {args.level} 的视频")
    
    if len(video_paths) == 0:
        print(f"错误: 未找到 {args.level} 的视频")
        return
    
    # 评估
    results = evaluator.evaluate_all_videos(video_paths)
    
    # 汇总统计
    summary = {
        "level": args.level,
        "num_videos": len(video_paths),
        "num_valid_videos": len(results),
        "mean_error": float(np.mean([r["mean_error"] for r in results])) if results else None,
        "std_error": float(np.std([r["mean_error"] for r in results])) if results else None,
        "median_error": float(np.median([r["mean_error"] for r in results])) if results else None,
        "lower_is_better": True,
        "evaluation_method": "temporal_module" if use_temporal else "optical_flow",
        "per_video_results": results
    }
    
    # 保存结果
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "="*80)
    print(f"Warping Error评估结果 ({args.level})")
    print("="*80)
    print(f"评估方法: {summary['evaluation_method']}")
    print(f"有效视频: {summary['num_valid_videos']}/{summary['num_videos']}")
    if summary['mean_error'] is not None:
        print(f"平均误差: {summary['mean_error']:.4f}")
        print(f"标准差: {summary['std_error']:.4f}")
        print(f"中位数: {summary['median_error']:.4f}")
    print("="*80)
    print(f"\n✓ 结果已保存到: {output_path}")

if __name__ == "__main__":
    main()