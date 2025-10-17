#!/usr/bin/env python3
"""
prepare_real_data.py

从训练数据集提取帧用于FID/FVD评估
"""

import sys
from pathlib import Path
from tqdm import tqdm
import argparse

# 添加项目路径
repo_root = Path(__file__).resolve().parents[4]
sys.path.append(str(repo_root))
sys.path.append(str(Path(__file__).resolve().parent.parent / "utils"))

from video_utils import extract_frames_to_directory


def prepare_real_dataset(
    videos_dir: Path,
    output_dir: Path,
    max_videos: int = None,
    max_frames_per_video: int = 100
):
    """
    从真实数据集提取帧
    
    Args:
        videos_dir: 真实视频目录
        output_dir: 输出帧目录
        max_videos: 最多处理的视频数量
        max_frames_per_video: 每个视频最多提取的帧数
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    video_files = sorted(videos_dir.glob("*.mp4"))
    
    if max_videos:
        video_files = video_files[:max_videos]
    
    print(f"找到 {len(video_files)} 个真实视频")
    print(f"开始提取帧到: {output_dir}")
    
    total_frames = 0
    for video_path in tqdm(video_files, desc="提取真实数据帧"):
        try:
            count = extract_frames_to_directory(
                video_path,
                output_dir / video_path.stem,
                max_frames=max_frames_per_video
            )
            total_frames += count
        except Exception as e:
            print(f"\n警告: 处理 {video_path.name} 失败: {e}")
    
    print(f"\n完成! 总共提取 {total_frames} 帧")
    print(f"保存位置: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="准备真实数据集用于评估")
    parser.add_argument(
        "--videos_dir",
        type=str,
        required=True,
        help="真实视频目录路径"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./evaluation_outputs/real_data_frames",
        help="输出帧目录"
    )
    parser.add_argument(
        "--max_videos",
        type=int,
        default=None,
        help="最多处理的视频数量"
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=100,
        help="每个视频最多提取的帧数"
    )
    
    args = parser.parse_args()
    
    prepare_real_dataset(
        Path(args.videos_dir),
        Path(args.output_dir),
        max_videos=args.max_videos,
        max_frames_per_video=args.max_frames
    )


if __name__ == "__main__":
    main()