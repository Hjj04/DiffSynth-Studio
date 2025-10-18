#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_evaluation_suite.py (Corrected & Robust Version)

此脚本基于四级验证逻辑，实现了大规模、全自动的视频生成套件。
它会遍历一组预定义的11个Prompts和3个随机种子，为每个组合生成四个
评估级别的视频，用于全面的模型效果对比。

The 4 Levels of Evaluation:
1.  **Level 0: Absolute Baseline**
    -   使用纯净的 Wan2.1 基础模型。
2.  **Level 1: Style Baseline**
    -   在基础模型上应用外部的水墨风格 LoRA。
3.  **Level 2: Jointly Trained Style**
    -   使用在时序训练中被微调过的 DiT 模型（风格已内化）。
4.  **Level 3: Fully Enhanced Model**
    -   使用微调后的 DiT 模型，并加载时序模块（Temporal Module + Flow Predictor）。
"""
import sys
import torch
from pathlib import Path
from tqdm import tqdm
import traceback

# ============================================================================
# 1. 设置路径与导入依赖
# ============================================================================
try:
    import diffsynth
except ImportError:
    repo_root = Path(__file__).resolve().parent
    sys.path.append(str(repo_root))
    print(f"Added {repo_root} to sys.path")

from diffsynth.pipelines.wan_video_new import WanVideoPipeline
from diffsynth.data.video import save_video
from diffsynth.modules.temporal_module import TemporalModule
from diffsynth.modules.latent_flow_predictor import LatentFlowPredictor
from diffsynth.utils import ModelConfig

# ============================================================================
# 2. 评估数据与配置
# ============================================================================

# --- 完整的11个评估用Prompts列表 ---
EVALUATION_PROMPTS = [
    {
        "id": "prompt_01_detailed_calligraphy",
        "english": "In a high-contrast black and white ink wash style, a woman in hanfu paints on an ancient bridge. The camera slowly pans horizontally, transitioning from a close-up to a half-body view. She makes continuous long brush strokes, with the wet ink naturally flowing and diffusing. Requires frame-to-frame coherence and strong negative space. 8 frames.",
        "num_frames": 17 # 8帧视频通常使用17帧输入以获得更好效果
    },
    {
        "id": "prompt_02_slow_calligraphy",
        "english": "Black-and-white ink-wash style, a woman in hanfu slowly painting on an ancient bridge, long continuous brush strokes with ink flow and natural bleeding, slow lateral camera pan, 8 frames. Require frame-to-frame continuity, continuous strokes, no flicker or broken strokes, strong negative space.",
        "num_frames": 17
    },
    {
        "id": "prompt_03_fast_action",
        "english": "Ink wash style, a young warrior swiftly swings a sword, brush strokes and ink droplets are rapidly flung out, emphasizing the ink trajectory during high-speed motion, 8 frames. Forcing no breaks or jitter.",
        "num_frames": 17
    },
    {
        "id": "prompt_04_negative_space",
        "english": "On a blank rice paper with significant negative space, a brush tip slowly drags across. The negative space subtly changes with the stroke, showing a smooth temporal variation of the white area, 8 frames. Low jitter required.",
        "num_frames": 17
    },
    {
        "id": "prompt_05_wet_to_dry",
        "english": "A single long brush stroke. Wet ink begins to bleed outward and then gradually dries, emphasizing the continuity of the bleeding and edge deformation, 8 frames. Frame-to-frame coherence is required.",
        "num_frames": 17
    },
    {
        "id": "prompt_06_occlusion_recovery",
        "english": "In the frame, a hand briefly occludes the brush tip and then moves away. The ink stroke extends continuously at the point of occlusion, testing the quality of recovery from occlusion-induced breaks, 8 frames.",
        "num_frames": 17
    },
    {
        "id": "prompt_07_edge_consistency",
        "english": "Complex tree branch line art, outlined with a pen-style ink wash. Emphasizes the preservation and coherence of the edge contour, 8 frames. High edge consistency (no broken edges) required.",
        "num_frames": 17
    },
    {
        "id": "prompt_08_multi_brush",
        "english": "Two brushes alternately draw interwoven strokes on paper, with the ink overlapping and flowing together. Tests the coherence at the intersections of the strokes, 8 frames.",
        "num_frames": 17
    },
    {
        "id": "prompt_09_long_zoom",
        "english": "A slow-panning long shot. The view gradually zooms out from a close-up of brush stroke details to a full view of the rice paper, observing the coherence of strokes during the zoom, 16 frames, low speed.",
        "num_frames": 17 # 16帧视频同样建议使用17帧输入
    },
    {
        "id": "prompt_10_particle_merge",
        "english": "Ink droplets splash on paper, forming petal shapes. Then, a gentle wind causes the ink petals to move slowly, 8 frames. Tests the merging and coherence of particle-like ink droplets.",
        "num_frames": 17
    },
    {
        "id": "prompt_11_calligraphy_stroke",
        "english": "A stroke of running-cursive style calligraphy, with continuous momentum and a natural finish, emphasizing the continuity of the brush tip's path, 8 frames.",
        "num_frames": 17
    }
]

# --- 随机种子列表 ---
RANDOM_SEEDS = [42, 100, 400]

# --- 路径配置 ---
OUTPUT_DIR = Path("./evaluation_videos_suite")
ORIGINAL_LORA_PATH = Path("/share/project/chengweiwu/code/Chinese_ink/hanzhe/ink_wash/lora_outputs/inkwash_style_v1/epoch-18.safetensors")
FINETUNED_DIT_PATH = Path("/share/project/chengweiwu/code/Chinese_ink/hanzhe/code/DiffSynth-Studio/runs/staged_training_final_oom_fix/checkpoints/dit_finetuned_step_final.pth")
TEMPORAL_MODULE_PATH = Path("/share/project/chengweiwu/code/Chinese_ink/hanzhe/code/DiffSynth-Studio/runs/staged_training_final_oom_fix/checkpoints/temporal_module_step_final.pth")
FLOW_PREDICTOR_PATH = Path("/share/project/chengweiwu/code/Chinese_ink/hanzhe/code/DiffSynth-Studio/runs/staged_training_final_oom_fix/checkpoints/flow_predictor_step_final.pth")
BASE_MODEL_ID = "Wan-AI/Wan2.1-T2V-1.3B"

# --- 模型与推理参数 ---
device = torch.device("cuda")
dtype = torch.float16
height, width = 512, 512
fps = 8
num_inference_steps = 50
cfg_scale = 7.0

# ============================================================================
# 3. 初始化模型与环境 (函数已修正)
# ============================================================================

def initialize_pipeline():
    """
    加载基础模型并返回pipeline实例和原始DiT权重。(已修正)
    
    修复说明：
    直接通过 `WanVideoPipeline.from_pretrained` 加载基础模型，并
    使用 `ModelConfig` 显式指定所需的权重文件，确保在本地缓存
    已存在的情况下能够稳定复现训练脚本的初始化流程。
    """
    print("--- 正在加载基础 WanVideoPipeline (from_pretrained) ---")

    model_configs = [
        ModelConfig(model_id=BASE_MODEL_ID, origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth"),
        ModelConfig(model_id=BASE_MODEL_ID, origin_file_pattern="diffusion_pytorch_model.safetensors"),
        ModelConfig(model_id=BASE_MODEL_ID, origin_file_pattern="Wan2.1_VAE.pth"),
    ]

    pipe = WanVideoPipeline.from_pretrained(
        device=device,
        torch_dtype=dtype,
        model_configs=model_configs,
    )
    
    # 保存一份干净的原始DiT权重，用于在各级别间重置状态
    original_dit_state_dict = {k: v.clone().cpu() for k, v in pipe.dit.state_dict().items()}
    print("--- 基础模型加载完毕，原始DiT状态已保存 ---")
    return pipe, original_dit_state_dict

def main():
    """主执行函数"""
    
    # --- 创建输出目录 ---
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"视频将保存至: {OUTPUT_DIR.resolve()}")
    
    # --- 初始化模型 ---
    pipe, original_dit_state_dict = initialize_pipeline()

    # 加载一次微调后的DiT权重，以备后用
    if FINETUNED_DIT_PATH.exists():
        print(f"--- 预加载微调DiT权重来源: {FINETUNED_DIT_PATH.name} ---")
        finetuned_dit_state_dict = torch.load(FINETUNED_DIT_PATH, map_location="cpu")
    else:
        print(f"警告: 未找到微调DiT权重文件: {FINETUNED_DIT_PATH}")
        finetuned_dit_state_dict = None

    # --- 计算任务总量并初始化进度条 ---
    total_videos = len(EVALUATION_PROMPTS) * len(RANDOM_SEEDS) * 4
    pbar = tqdm(total=total_videos, desc="全局生成进度", unit="video")

    # --- 开始批量生成循环 ---
    for prompt_config in EVALUATION_PROMPTS:
        for seed in RANDOM_SEEDS:
            
            prompt_id = prompt_config["id"]
            prompt_text = prompt_config["english"]
            num_frames = prompt_config["num_frames"]
            
            # --- Level 0: Absolute Baseline ---
            level_id = 0
            video_path = OUTPUT_DIR / f"{prompt_id}_level{level_id}_seed{seed}.mp4"
            pbar.set_description(f"Processing Level {level_id} (Seed {seed}, {prompt_id})")
            try:
                # 状态重置: 恢复原始DiT，移除时序模块
                pipe.dit.load_state_dict({k: v.to(device) for k, v in original_dit_state_dict.items()})
                pipe.temporal_module = None
                pipe.flow_predictor = None
                
                # 生成视频
                video_frames = pipe(prompt_text, seed=seed, num_frames=num_frames, height=height, width=width, num_inference_steps=num_inference_steps, cfg_scale=cfg_scale)
                save_video(video_frames, str(video_path), fps=fps)
                # print(f"✅ Level {level_id} 视频已保存: {video_path.name}")
            except Exception as e:
                print(f"\n❌ Level {level_id} ({prompt_id}, seed {seed}) 生成失败: {e}")
                traceback.print_exc()
            pbar.update(1)

            # --- Level 1: Style Baseline (LoRA) ---
            level_id = 1
            video_path = OUTPUT_DIR / f"{prompt_id}_level{level_id}_seed{seed}.mp4"
            pbar.set_description(f"Processing Level {level_id} (Seed {seed}, {prompt_id})")
            try:
                # 状态重置: 恢复原始DiT，然后加载LoRA
                pipe.dit.load_state_dict({k: v.to(device) for k, v in original_dit_state_dict.items()})
                pipe.load_lora(pipe.dit, str(ORIGINAL_LORA_PATH), alpha=1.0)
                pipe.temporal_module = None
                pipe.flow_predictor = None

                # 生成视频
                video_frames = pipe(prompt_text, seed=seed, num_frames=num_frames, height=height, width=width, num_inference_steps=num_inference_steps, cfg_scale=cfg_scale)
                save_video(video_frames, str(video_path), fps=fps)
                # print(f"✅ Level {level_id} 视频已保存: {video_path.name}")
            except Exception as e:
                print(f"\n❌ Level {level_id} ({prompt_id}, seed {seed}) 生成失败: {e}")
                traceback.print_exc()
            pbar.update(1)
            
            # --- Level 2: Jointly Trained Style (Fine-tuned DiT) ---
            level_id = 2
            video_path = OUTPUT_DIR / f"{prompt_id}_level{level_id}_seed{seed}.mp4"
            pbar.set_description(f"Processing Level {level_id} (Seed {seed}, {prompt_id})")
            if finetuned_dit_state_dict:
                try:
                    # 状态重置: 恢复基础DiT，再加载微调后的权重，移除时序模块
                    pipe.dit.load_state_dict({k: v.to(device) for k, v in original_dit_state_dict.items()})
                    pipe.dit.load_state_dict({k: v.to(device) for k, v in finetuned_dit_state_dict.items()}, strict=False)
                    pipe.temporal_module = None
                    pipe.flow_predictor = None

                    # 生成视频
                    video_frames = pipe(prompt_text, seed=seed, num_frames=num_frames, height=height, width=width, num_inference_steps=num_inference_steps, cfg_scale=cfg_scale)
                    save_video(video_frames, str(video_path), fps=fps)
                    # print(f"✅ Level {level_id} 视频已保存: {video_path.name}")
                except Exception as e:
                    print(f"\n❌ Level {level_id} ({prompt_id}, seed {seed}) 生成失败: {e}")
                    traceback.print_exc()
            else:
                print(f"\n⏭️  跳过 Level {level_id} ({prompt_id}, seed {seed})，因为缺少微调DiT权重。")
            pbar.update(1)

            # --- Level 3: Fully Enhanced Model ---
            level_id = 3
            video_path = OUTPUT_DIR / f"{prompt_id}_level{level_id}_seed{seed}.mp4"
            pbar.set_description(f"Processing Level {level_id} (Seed {seed}, {prompt_id})")
            if finetuned_dit_state_dict and TEMPORAL_MODULE_PATH.exists() and FLOW_PREDICTOR_PATH.exists():
                try:
                    # 状态设置: 先恢复基础DiT，再加载微调版本，然后加载时序模块
                    pipe.dit.load_state_dict({k: v.to(device) for k, v in original_dit_state_dict.items()})
                    pipe.dit.load_state_dict({k: v.to(device) for k, v in finetuned_dit_state_dict.items()}, strict=False)
                    
                    # 加载并挂载Temporal Module
                    temporal_module = TemporalModule(latent_channels=16, style_dim=None).to(device, dtype=dtype).eval()
                    temporal_module.load_state_dict(torch.load(TEMPORAL_MODULE_PATH, map_location=device))
                    pipe.temporal_module = temporal_module

                    # 加载并挂载Flow Predictor
                    flow_predictor = LatentFlowPredictor(in_channels=16).to(device, dtype=dtype).eval()
                    flow_predictor.load_state_dict(torch.load(FLOW_PREDICTOR_PATH, map_location=device))
                    pipe.flow_predictor = flow_predictor

                    # 生成视频
                    video_frames = pipe(prompt_text, seed=seed, num_frames=num_frames, height=height, width=width, num_inference_steps=num_inference_steps, cfg_scale=cfg_scale)
                    save_video(video_frames, str(video_path), fps=fps)
                    # print(f"✅ Level {level_id} 视频已保存: {video_path.name}")
                except Exception as e:
                    print(f"\n❌ Level {level_id} ({prompt_id}, seed {seed}) 生成失败: {e}")
                    traceback.print_exc()
            else:
                 print(f"\n⏭️  跳过 Level {level_id} ({prompt_id}, seed {seed})，因为缺少必要的模型权重。")
            pbar.update(1)

    pbar.close()
    print("\n🎉 所有视频生成任务已完成! 🎉")

if __name__ == "__main__":
    main()
