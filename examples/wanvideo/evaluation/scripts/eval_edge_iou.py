#!/usr/bin/env python3
"""
eval_edge_iou.py - 计算Edge IoU
"""

import argparse
import json
import numpy as np
from pathlib import Path
from glob import glob
from tqdm import tqdm
import cv2

class EdgeIoUEvaluator:
    def __init__(self, canny_threshold1=50, canny_threshold2=150):
        self.canny_threshold1 = canny_threshold1
        self.canny_threshold2 = canny_threshold2
    
    def extract_edges(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, self.canny_threshold1, self.canny_threshold2)
        return edges > 0
    
    def calculate_iou(self, edges1, edges2):
        intersection = np.logical_and(edges1, edges2).sum()
        union = np.logical_or(edges1, edges2).sum()
        
        if union == 0:
            return 1.0
        
        return intersection / union
    
    def evaluate_video(self, video_path):
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        
        cap.release()
        
        if len(frames) < 2:
            return None
        
        ious = []
        for i in range(len(frames) - 1):
            edges1 = self.extract_edges(frames[i])
            edges2 = self.extract_edges(frames[i + 1])
            iou = self.calculate_iou(edges1, edges2)
            ious.append(iou)
        
        return {
            "mean_iou": np.mean(ious),
            "std_iou": np.std(ious),
            "min_iou": np.min(ious),
            "max_iou": np.max(ious)
        }

def main():
    parser = argparse.ArgumentParser(description="计算Edge IoU")
    parser.add_argument("--videos_dir", type=str, required=True)
    parser.add_argument("--level", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--canny_threshold1", type=int, default=50)
    parser.add_argument("--canny_threshold2", type=int, default=150)
    parser.add_argument("--device", type=str, default="cuda")
    
    args = parser.parse_args()
    
    evaluator = EdgeIoUEvaluator(args.canny_threshold1, args.canny_threshold2)
    
    video_pattern = f"video_{args.level[-1]}_*.mp4"
    video_paths = sorted(glob(str(Path(args.videos_dir) / video_pattern)))
    
    print(f"\n找到 {len(video_paths)} 个 {args.level} 的视频")
    
    if len(video_paths) == 0:
        print(f"错误: 未找到 {args.level} 的视频")
        return
    
    all_results = []
    
    for video_path in tqdm(video_paths, desc="计算Edge IoU"):
        result = evaluator.evaluate_video(video_path)
        if result:
            all_results.append(result)
    
    summary = {
        "level": args.level,
        "num_videos": len(all_results),
        "mean_edge_iou": np.mean([r["mean_iou"] for r in all_results]),
        "std_edge_iou": np.std([r["mean_iou"] for r in all_results]),
        "median_edge_iou": np.median([r["mean_iou"] for r in all_results]),
        "higher_is_better": True,
        "per_video_results": all_results
    }
    
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "="*80)
    print(f"Edge IoU评估结果 ({args.level})")
    print("="*80)
    print(f"平均: {summary['mean_edge_iou']:.4f}")
    print(f"标准差: {summary['std_edge_iou']:.4f}")
    print("="*80)

if __name__ == "__main__":
    main()
