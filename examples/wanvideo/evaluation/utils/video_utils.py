#!/usr/bin/env python3
"""
video_utils.py

视频处理的通用工具函数
"""

import cv2
import torch
import numpy as np
from pathlib import Path
from typing import List, Tuple
from PIL import Image


def load_video_frames(video_path: Path) -> torch.Tensor:
    """
    从视频文件加载帧
    
    Args:
        video_path: 视频文件路径
        
    Returns:
        torch.Tensor: (T, C, H, W) 格式的帧，值在[0, 1]范围
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"无法打开视频: {video_path}")
    
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # BGR -> RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # (H, W, C) -> (C, H, W)
        frame_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0
        frames.append(frame_tensor)
    
    cap.release()
    
    if not frames:
        raise ValueError(f"视频为空: {video_path}")
    
    # Stack to (T, C, H, W)
    return torch.stack(frames, dim=0)


def save_video_frames(frames: torch.Tensor, output_path: Path, fps: int = 8):
    """
    保存帧为视频文件
    
    Args:
        frames: (T, C, H, W) 格式的帧，值在[0, 1]范围
        output_path: 输出视频路径
        fps: 帧率
    """
    from torchvision.io import write_video
    
    # (T, C, H, W) -> (T, H, W, C)
    frames_np = (frames * 255).clamp(0, 255).byte().permute(0, 2, 3, 1).cpu().numpy()
    write_video(str(output_path), frames_np, fps=fps)


def extract_frames_to_directory(
    video_path: Path,
    output_dir: Path,
    max_frames: int = None
) -> int:
    """
    提取视频帧到目录（用于FID/FVD评估）
    
    Args:
        video_path: 视频路径
        output_dir: 输出目录
        max_frames: 最大帧数限制
        
    Returns:
        int: 提取的帧数
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"无法打开视频: {video_path}")
    
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret or (max_frames and count >= max_frames):
            break
        
        frame_path = output_dir / f"frame_{count:05d}.png"
        cv2.imwrite(str(frame_path), frame)
        count += 1
    
    cap.release()
    return count


def compute_frame_differences(frames: torch.Tensor) -> List[float]:
    """
    计算相邻帧的L1差异
    
    Args:
        frames: (T, C, H, W) 格式的帧
        
    Returns:
        List[float]: 每对相邻帧的差异值
    """
    if frames.shape[0] < 2:
        return []
    
    diffs = []
    for t in range(frames.shape[0] - 1):
        diff = torch.abs(frames[t + 1] - frames[t]).mean().item()
        diffs.append(diff)
    
    return diffs


def parse_video_filename(filename: str) -> Tuple[str, str, int]:
    """
    解析视频文件名
    
    格式: video_{level}_{prompt_id}_seed{seed}.mp4
    
    Returns:
        (level, prompt_id, seed)
    """
    parts = filename.replace('.mp4', '').split('_')
    # video_0_01_slow_calligraphy_seed42
    level = parts[1]  # "0"
    prompt_id = '_'.join(parts[2:-1])  # "01_slow_calligraphy"
    seed = int(parts[-1].replace('seed', ''))  # 42
    
    return level, prompt_id, seed


if __name__ == "__main__":
    # 测试代码
    print("视频工具模块已加载")