#!/usr/bin/env python3
"""
eval_tlpips.py

计算Temporal LPIPS (tLPIPS) - 时序感知的感知相似度
衡量相邻帧之间的感知跳跃程度
"""

import sys
import argparse
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List
import lpips
from torchvision.transforms import functional as TF

# 添加项目路径
utils_dir = Path(__file__).resolve().parent.parent / "utils"
sys.path.append(str(utils_dir))
from video_utils import load_video_frames, parse_video_filename

class TLPIPSEvaluator:
    """Temporal LPIPS评估器"""
    
    def __init__(self, net: str = 'alex', device: str = 'cuda'):
        """
        初始化LPIPS模型
        
        Args:
            net: 特征提取网络 ('alex', 'vgg', 'squeeze')
            device: 计算设备
        """
        print(f"正在加载LPIPS模型 (网络: {net})...")
        self.device = torch.device(device)
        
        # spatial=False 表示输出单一的全局距离得分
        self.lpips_model = lpips.LPIPS(net=net, spatial=False).to(self.device)
        self.lpips_model.eval()
        
        print("✓ LPIPS模型加载完成")
    
    @torch.no_grad()
    def calculate_video_tlpips(self, video_path: Path) -> float:
        """
        计算单个视频的平均tLPIPS
        
        Args:
            video_path: 视频文件路径
            
        Returns:
            float: 平均tLPIPS值
        """
        try:
            # 加载视频帧 (T, C, H, W)
            frames = load_video_frames(video_path)
            
            if frames.shape[0] < 2:
                return 0.0  # 单帧视频
            
            tlpips_scores = []
            
            # 计算相邻帧的LPIPS
            for t in range(frames.shape[0] - 1):
                frame_t = frames[t].unsqueeze(0).to(self.device)  # (1, C, H, W)
                frame_t_plus_1 = frames[t + 1].unsqueeze(0).to(self.device)
                
                # LPIPS要求输入在[-1, 1]范围
                frame_t_norm = (frame_t * 2.0) - 1.0
                frame_t_plus_1_norm = (frame_t_plus_1 * 2.0) - 1.0
                
                # 计算LPIPS距离
                score = self.lpips_model(frame_t_norm, frame_t_plus_1_norm)
                tlpips_scores.append(score.item())
            
            return np.mean(tlpips_scores)
        
        except Exception as e:
            print(f"\n警告: 处理 {video_path.name} 失败: {e}")
            return np.nan
    
    def evaluate_directory(
        self,
        video_dir: Path,
        output_file: Path = None
    ) -> Dict[str, Dict[str, float]]:
        """
        评估目录中所有视频的tLPIPS
        
        Args:
            video_dir: 视频目录
            output_file: 结果保存路径（可选）
            
        Returns:
            Dict: 按级别组织的结果
                {
                    "level_0": {"mean": 0.123, "std": 0.045, "videos": 33},
                    ...
                }
        """
        video_files = sorted(video_dir.glob("video_*.mp4"))
        
        if not video_files:
            print(f"错误: 在 {video_dir} 中未找到视频文件")
            return {}
        
        print(f"\n找到 {len(video_files)} 个视频文件")
        print("开始计算tLPIPS...\n")
        
        # 按级别分组结果
        results_by_level = {
            "level_0": [],
            "level_1": [],
            "level_2": [],
            "level_3": []
        }
        
        # 计算每个视频的tLPIPS
        for video_path in tqdm(video_files, desc="计算tLPIPS"):
            level, prompt_id, seed = parse_video_filename(video_path.name)
            
            tlpips_value = self.calculate_video_tlpips(video_path)
            
            if not np.isnan(tlpips_value):
                results_by_level[f"level_{level}"].append(tlpips_value)
        
        # 汇总统计
        summary = {}
        for level, scores in results_by_level.items():
            if scores:
                summary[level] = {
                    "mean": float(np.mean(scores)),
                    "std": float(np.std(scores)),
                    "median": float(np.median(scores)),
                    "min": float(np.min(scores)),
                    "max": float(np.max(scores)),
                    "videos": len(scores)
                }
            else:
                summary[level] = {
                    "mean": np.nan,
                    "std": np.nan,
                    "median": np.nan,
                    "min": np.nan,
                    "max": np.nan,
                    "videos": 0
                }
        
        # 打印结果
        self._print_results(summary)
        
        # 保存结果
        if output_file:
            self._save_results(summary, output_file)
        
        return summary
    
    def _print_results(self, summary: Dict):
        """打印格式化的结果"""
        print("\n" + "="*80)
        print("Temporal LPIPS (tLPIPS) 评估结果")
        print("="*80)
        print(f"{'级别':<25} {'均值':<12} {'标准差':<12} {'中位数':<12} {'视频数'}")
        print("-"*80)
        
        for level in ["level_0", "level_1", "level_2", "level_3"]:
            stats = summary[level]
            level_name = {
                "level_0": "Level 0 (Abs. Baseline)",
                "level_1": "Level 1 (Style Baseline)",
                "level_2": "Level 2 (Joint Style)",
                "level_3": "Level 3 (Fully Enhanced)"
            }[level]
            
            if stats["videos"] > 0:
                print(f"{level_name:<25} {stats['mean']:<12.6f} {stats['std']:<12.6f} "
                      f"{stats['median']:<12.6f} {stats['videos']}")
            else:
                print(f"{level_name:<25} {'N/A':<12} {'N/A':<12} {'N/A':<12} {stats['videos']}")
        
        print("="*80)
        print("\n💡 解读: tLPIPS越低越好，表示帧间感知跳跃越小")
        print("   预期: Level 3 < Level 2 < Level 1 ≈ Level 0")
        print("="*80 + "\n")
    
    def _save_results(self, summary: Dict, output_file: Path):
        """保存结果到JSON文件"""
        import json
        
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"✓ 结果已保存到: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="计算视频的Temporal LPIPS")
    parser.add_argument(
        "--video_dir",
        type=str,
        required=True,
        help="包含所有评估视频的目录"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="./evaluation_outputs/metrics/tlpips_results.json",
        help="结果保存路径"
    )
    parser.add_argument(
        "--net",
        type=str,
        choices=['alex', 'vgg', 'squeeze'],
        default='alex',
        help="LPIPS特征提取网络"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="计算设备"
    )
    
    args = parser.parse_args()
    
    # 创建评估器
    evaluator = TLPIPSEvaluator(net=args.net, device=args.device)
    
    # 执行评估
    evaluator.evaluate_directory(
        video_dir=Path(args.video_dir),
        output_file=Path(args.output_file)
    )

if __name__ == "__main__":
    main()