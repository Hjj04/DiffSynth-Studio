#!/usr/bin/env python3
"""
validate_all_levels.py (4-Way Comparison, Robust Checkpoint Loading)

This script performs a comprehensive four-way comparison to evaluate training results.
It has been made more robust to find the correct fine-tuned weights, whether they
were saved as a full DiT model state_dict or a LoRA-only file.

The 4 Experiments:
1.  **Absolute Baseline:** Base Wan Model without any LoRA.
2.  **Style Baseline:** Base Wan Model + Your original Inkwash LoRA.
3.  **Jointly Trained Style:** Base Model + DiT fine-tuned during temporal training.
4.  **Fully Enhanced:** Base Model + Fine-tuned DiT + Temporal Modules.
"""
import sys
import torch
from pathlib import Path

# --- Setup Paths and Imports ---
try:
    import diffsynth
except ImportError:
    # Adjust parents count if your script is in a different location
    repo_root = Path(__file__).resolve().parents[1]
    if "diffsynth" not in str(repo_root):
         repo_root = Path(__file__).resolve().parent
    sys.path.append(str(repo_root))
    print(f"Added {repo_root} to sys.path")

from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
from diffsynth.data.video import save_video
from diffsynth.modules.temporal_module import TemporalModule
from diffsynth.modules.latent_flow_predictor import LatentFlowPredictor
from diffsynth.models.utils import load_state_dict as load_state_dict_from_file

def find_latest_checkpoint_step(checkpoint_dir: Path) -> str:
    """Finds the checkpoint file with the highest step number, falling back to 'final'."""
    if not checkpoint_dir.exists():
        print(f"Warning: Checkpoint directory not found at {checkpoint_dir}")
        return "final"
    steps = [int(f.stem.split('_')[-1]) for f in checkpoint_dir.glob("*.pth") if f.stem.split('_')[-1].isdigit()]
    if not steps:
        # Check for 'final' if no numbered steps are found
        if (checkpoint_dir / "temporal_module_step_final.pth").exists():
            return "final"
        else:
            raise FileNotFoundError(f"No valid checkpoints ('*_step_NUM.pth' or '*_step_final.pth') found in {checkpoint_dir}")
    return str(max(steps))

# --- 1. Configuration ---
ORIGINAL_LORA_PATH = "/share/project/chengweiwu/code/Chinese_ink/hanzhe/ink_wash/lora_outputs/inkwash_style_v1/epoch-18.safetensors"
MODEL_ID = "Wan-AI/Wan2.1-T2V-1.3B"

# **IMPORTANT**: Point this to the specific run directory from your last training session
# Example: Path("./runs/staged_training_final_oom_fix/staged_lr0.0001_warmup2000_1760519461")
TRAINING_OUTPUT_DIR = Path("./runs/staged_training_final_oom_fix")
# --- Auto-discover checkpoint paths (Robust Version)1 ---
CHECKPOINT_DIR = TRAINING_OUTPUT_DIR / "checkpoints"
LATEST_STEP = find_latest_checkpoint_step(CHECKPOINT_DIR)
print(f"Found latest checkpoint step: '{LATEST_STEP}' in '{CHECKPOINT_DIR}'")

TEMPORAL_MODULE_PATH = CHECKPOINT_DIR / f"temporal_module_step_{LATEST_STEP}.pth"
FLOW_PREDICTOR_PATH = CHECKPOINT_DIR / f"flow_predictor_step_{LATEST_STEP}.pth"
FINETUNED_DIT_PATH = CHECKPOINT_DIR / f"dit_finetuned_step_{LATEST_STEP}.pth"
TRAINED_LORA_PATH = CHECKPOINT_DIR / f"lora_step_{LATEST_STEP}.pth"

# Determine which file to use for the fine-tuned model
use_full_dit = FINETUNED_DIT_PATH.exists()
use_lora_only = TRAINED_LORA_PATH.exists()

print(f"Full DiT checkpoint found: {use_full_dit}")
print(f"LoRA-only checkpoint found: {use_lora_only}")

# --- Inference Parameters ---
device = torch.device("cuda")
dtype = torch.float16
prompt = "a majestic dragon flying through the clouds, fast and graceful movement, inkwash style, fluid lines."
seed = 42
num_frames = 64
height, width = 512, 512
fps = 16
num_inference_steps = 50
cfg_scale = 7.0

# --- 2. Load Base Pipeline ---
print("--- Loading Base WanVideoPipeline (this might take a moment) ---")
model_configs = [
    ModelConfig(model_id=MODEL_ID, origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth"),
    ModelConfig(model_id=MODEL_ID, origin_file_pattern="diffusion_pytorch_model.safetensors"),
    ModelConfig(model_id=MODEL_ID, origin_file_pattern="Wan2.1_VAE.pth"),
]
pipe = WanVideoPipeline.from_pretrained(model_configs=model_configs, torch_dtype=dtype)
pipe.to(device)

# Save a clean copy of the original DiT weights for resetting
original_dit_state_dict = pipe.dit.state_dict()

# --- 3. Run Inference for all Comparisons ---

# === [0/3] Absolute Baseline ===
print("\n--- [0/3] Generating ABSOLUTE BASELINE (Base Model only) ---")
pipe.dit.load_state_dict(original_dit_state_dict)
pipe.temporal_module = None
pipe.flow_predictor = None
output_path_abs_baseline = "video_0_absolute_baseline.mp4"
video_abs_baseline = pipe(prompt, seed=seed, num_frames=num_frames, height=height, width=width, num_inference_steps=num_inference_steps, cfg_scale=cfg_scale)
save_video(video_abs_baseline, output_path_abs_baseline, fps=fps)
print(f"Success! Absolute Baseline video saved as: {output_path_abs_baseline}")

# === [1/3] Style Baseline ===
print("\n--- [1/3] Generating STYLE BASELINE (Base Model + Your Inkwash LoRA) ---")
pipe.dit.load_state_dict(original_dit_state_dict)
pipe.load_lora(pipe.dit, ORIGINAL_LORA_PATH, alpha=1.0)
pipe.temporal_module = None
pipe.flow_predictor = None
output_path_style_baseline = "video_1_style_baseline.mp4"
video_style_baseline = pipe(prompt, seed=seed, num_frames=num_frames, height=height, width=width, num_inference_steps=num_inference_steps, cfg_scale=cfg_scale)
save_video(video_style_baseline, output_path_style_baseline, fps=fps)
print(f"Success! Style Baseline video saved as: {output_path_style_baseline}")

# === [2/3] & [3/3] Load the fine-tuned model for the final two comparisons ===
print("\n--- Loading Fine-Tuned Models for Final Comparisons ---")

fine_tuned_model_loaded = False
if use_full_dit:
    # Method 1: Load the entire fine-tuned DiT state dict
    print(f"Loading FULL fine-tuned DiT from {FINETUNED_DIT_PATH}")
    pipe.dit.load_state_dict(torch.load(FINETUNED_DIT_PATH, map_location=device))
    fine_tuned_model_loaded = True
elif use_lora_only:
    # Method 2 (Fallback): Load the original DiT and apply the trained LoRA on top
    print(f"Loading fine-tuned LoRA from {TRAINED_LORA_PATH}")
    pipe.dit.load_state_dict(original_dit_state_dict) # Reset to original DiT
    pipe.load_lora(pipe.dit, str(TRAINED_LORA_PATH), alpha=1.0) # Apply trained LoRA
    fine_tuned_model_loaded = True

if fine_tuned_model_loaded:
    # === [2/3] Jointly Trained Style (No Temporal Modules) ===
    print("\n--- [2/3] Generating JOINTLY TRAINED STYLE (Fine-tuned DiT/LoRA only) ---")
    pipe.temporal_module = None
    pipe.flow_predictor = None
    output_path_joint_style = "video_2_jointly_trained_style.mp4"
    video_joint_style = pipe(prompt, seed=seed, num_frames=num_frames, height=height, width=width, num_inference_steps=num_inference_steps, cfg_scale=cfg_scale)
    save_video(video_joint_style, output_path_joint_style, fps=fps)
    print(f"Success! Jointly Trained Style video saved as: {output_path_joint_style}")
    
    # === [3/3] Fully Enhanced Model ===
    print("\n--- [3/3] Generating FULLY ENHANCED video (Fine-tuned DiT/LoRA + Temporal Modules) ---")
    if TEMPORAL_MODULE_PATH.exists():
        temporal_module = TemporalModule(latent_channels=16, style_dim=None).to(device, dtype=dtype).eval()
        temporal_module.load_state_dict(torch.load(TEMPORAL_MODULE_PATH, map_location=device))
        pipe.temporal_module = temporal_module
        print("Loaded and attached Temporal Module.")

        if FLOW_PREDICTOR_PATH.exists():
            flow_predictor = LatentFlowPredictor(in_channels=16).to(device, dtype=dtype).eval()
            flow_predictor.load_state_dict(torch.load(FLOW_PREDICTOR_PATH, map_location=device))
            pipe.flow_predictor = flow_predictor
            print("Loaded and attached Flow Predictor.")
        else:
            pipe.flow_predictor = None

        output_path_enhanced = "video_3_fully_enhanced.mp4"
        video_enhanced = pipe(prompt, seed=seed, num_frames=num_frames, height=height, width=width, num_inference_steps=num_inference_steps, cfg_scale=cfg_scale)
        save_video(video_enhanced, output_path_enhanced, fps=fps)
        print(f"Success! Fully Enhanced video saved as: {output_path_enhanced}")
    else:
        print("Skipping Fully Enhanced Model: Temporal Module checkpoint not found.")
else:
    print("Skipping final two comparisons: Neither a full DiT nor a LoRA checkpoint was found.")

print("\nAll comparisons finished.")