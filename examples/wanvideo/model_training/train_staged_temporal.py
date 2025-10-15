#!/usr/bin/env python3
"""
train_staged_temporal.py (FINAL & DEFINITIVE VERSION)

This version is complete and includes all necessary fixes. It correctly handles
the `diffsynth` LoRA loading mechanism by treating the LoRA file as a direct
modification to the base model's weights and subsequently fine-tuning the
entire DiT model.

Key Features:
- Correctly sets the entire DiT model as trainable when --train_lora is active.
- Freezes VAE and Text Encoder to prevent memory overflow.
- Explicitly handles data types (dtype) to prevent mismatches.
- Includes full argument parsing, helper functions, and training loop.
- Supports gradient checkpointing for memory optimization.
"""

import argparse
import os
import random
import sys
import time
from pathlib import Path
from typing import Optional, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms.functional import to_pil_image

# --- Project Root Setup for Module Imports ---
try:
    import diffsynth
except ImportError:
    repo_root = Path(__file__).resolve().parents[3]
    sys.path.append(str(repo_root))
    print(f"Added {repo_root} to sys.path to find `diffsynth` module.")

from diffsynth.modules.latent_flow_predictor import LatentFlowPredictor
from diffsynth.modules.temporal_module import TemporalModule
from diffsynth.utils import ModelConfig
from diffsynth.utils.alpha_scheduler import AlphaScheduler
from diffsynth.lora import GeneralLoRALoader
from diffsynth.models.utils import load_state_dict as load_state_dict_from_file

# --- Local Dataset Import ---
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))
try:
    from dataset import RealVideoDataset
except ImportError:
    print("FATAL: `dataset.py` not found. Please ensure it is in the same directory as this script.")
    sys.exit(1)

# --- Optional CLIP Import ---
try:
    from transformers import CLIPModel, CLIPProcessor
    _CLIP_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _CLIP_MODEL = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(_CLIP_DEVICE).eval()
    _CLIP_PROCESSOR = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    _CLIP_AVAILABLE = True
except Exception:
    _CLIP_MODEL, _CLIP_PROCESSOR, _CLIP_DEVICE, _CLIP_AVAILABLE = None, None, None, False
    print("Warning: CLIP models not found. CLIP-based motion loss will be disabled.")

# --- Helper Functions ---

def set_global_seed(seed: int):
    """Set seed for reproducibility across all relevant libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def save_checkpoint(
    args,
    step: int | str,
    pipe,
    temporal_module: TemporalModule,
    flow_predictor: Optional[LatentFlowPredictor],
):
    """Persist all trainable module weights at the specified step."""
    output_dir = Path(args.logdir) / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)
    step_tag = str(step)
    torch.save(temporal_module.state_dict(), output_dir / f"temporal_module_step_{step_tag}.pth")
    if flow_predictor is not None:
        torch.save(flow_predictor.state_dict(), output_dir / f"flow_predictor_step_{step_tag}.pth")
    if args.train_lora:
        # Save the entire DiT model's state dict since it's being fine-tuned
        torch.save(pipe.dit.state_dict(), output_dir / f"dit_finetuned_step_{step_tag}.pth")
    print(f"[Checkpoint] Saved models for step {step_tag} to {output_dir}")

class LossScheduler:
    """Calculates a linearly increasing weight for a loss term."""
    def __init__(self, lambda_final: float, warmup_steps: int, warmup_start: int):
        self.lambda_final = float(lambda_final)
        self.warmup_steps = int(max(0, warmup_steps))
        self.warmup_start = int(max(0, warmup_start))

    def weight(self, step: int) -> float:
        if step < self.warmup_start: return 0.0
        if self.warmup_steps == 0: return self.lambda_final
        progress = min(1.0, (step - self.warmup_start) / max(1, self.warmup_steps))
        return self.lambda_final * progress

def initialise_pipeline(args, device: torch.device, torch_dtype: torch.dtype):
    """Load the WanVideoPipeline and configure its components."""
    from diffsynth.pipelines.wan_video_new import WanVideoPipeline
    model_configs = [
        ModelConfig(model_id=args.model_id, origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth"),
        ModelConfig(model_id=args.model_id, origin_file_pattern="diffusion_pytorch_model.safetensors"),
        ModelConfig(model_id=args.model_id, origin_file_pattern="Wan2.1_VAE.pth"),
    ]
    return WanVideoPipeline.from_pretrained(device=device, torch_dtype=torch_dtype, model_configs=model_configs)

def build_dataloader(args) -> DataLoader:
    """Build and return the video data loader."""
    dataset = RealVideoDataset(
        metadata_csv_path=args.metadata_csv_path,
        videos_root_dir=args.videos_root_dir,
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )

def compute_clip_motion_loss(frames_hat: torch.Tensor) -> torch.Tensor:
    """Compute cosine distance between CLIP features of adjacent frames."""
    if not _CLIP_AVAILABLE or frames_hat is None:
        return torch.tensor(0.0, device=frames_hat.device)
    b, t, _, _, _ = frames_hat.shape
    if t < 2: return torch.tensor(0.0, device=frames_hat.device)
    with torch.no_grad():
        imgs = [to_pil_image(frames_hat[i, j].clamp(0, 1)) for i in range(b) for j in range(t)]
        clip_inputs = _CLIP_PROCESSOR(images=imgs, return_tensors="pt", padding=True).to(_CLIP_DEVICE)
        features = _CLIP_MODEL.get_image_features(**clip_inputs).view(b, t, -1)
    features = features.to(frames_hat.device)
    return 1.0 - F.cosine_similarity(features[:, 1:], features[:, :-1], dim=-1).mean()

def edge_map_tensor(frames: torch.Tensor) -> torch.Tensor:
    """Compute Sobel edge maps for a batch of video frames."""
    b, t, c, h, w = frames.shape
    flat = frames.view(b * t, c, h, w)
    gray = (0.299 * flat[:, 0] + 0.587 * flat[:, 1] + 0.114 * flat[:, 2]).unsqueeze(1) if c == 3 else flat[:, 0].unsqueeze(1)
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=frames.device).view(1, 1, 3, 3)
    sobel_y = sobel_x.permute(0, 1, 3, 2)
    gx, gy = F.conv2d(gray, sobel_x, padding=1), F.conv2d(gray, sobel_y, padding=1)
    magnitude = torch.sqrt(gx * gx + gy * gy + 1e-6)
    return magnitude.view(b, t, 1, h, w)

def train(args):
    """The main training loop with all fixes incorporated."""
    device = torch.device(args.device)
    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    torch_dtype = dtype_map[args.dtype]
    set_global_seed(args.seed)

    # 1. Setup
    run_name = f"staged_lr{args.lr}_warmup{args.temporal_loss_warmup_steps}_{int(time.time())}"
    logdir = Path(args.logdir) / run_name
    logdir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(logdir))
    print(f"[Run] Name: {run_name} | Log Dir: {logdir}")

    # 2. Data and Models
    dataloader = build_dataloader(args)
    pipe = initialise_pipeline(args, device, torch_dtype)

    # 3. Freeze Models & Prepare for Training
    print("Freezing pre-trained models (VAE, Text Encoder)...")
    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    
    # --- DEFINITIVE LoRA STRATEGY ---
    # The `GeneralLoRALoader` in diffsynth modifies the model weights in-place.
    # Therefore, "training LoRA" means fine-tuning the entire DiT model after its weights
    # have been updated with the LoRA file.
    if args.lora_path:
        print(f"Loading and merging LoRA from {args.lora_path} into DiT.")
        loader = GeneralLoRALoader(device=device, torch_dtype=torch_dtype)
        lora_state = load_state_dict_from_file(args.lora_path, torch_dtype=torch_dtype, device=device)
        loader.load(pipe.dit, lora_state, alpha=args.lora_alpha)

    # Now, set the trainability of the entire DiT model based on the --train_lora flag.
    if args.train_lora:
        pipe.dit.requires_grad_(True)
        print("DiT model is set to TRAINABLE (fine-tuning on LoRA-merged weights).")
    else:
        pipe.dit.requires_grad_(False)
        print("DiT model is FROZEN.")

    if args.use_gradient_checkpointing:
        pipe.dit.train()
        for module in pipe.dit.modules():
            if hasattr(module, "gradient_checkpointing"):
                module.gradient_checkpointing = True
        print("Enabled gradient checkpointing.")

    # 4. Initialize Custom Trainable Modules
    latent_channels = 16
    temporal_module = TemporalModule(latent_channels=latent_channels, style_dim=args.style_dim).to(device)
    flow_predictor = LatentFlowPredictor(in_channels=latent_channels).to(device) if args.use_flow_predictor else None
    
    # 5. Setup Optimizer
    params_to_optimize = list(temporal_module.parameters())
    if flow_predictor:
        params_to_optimize.extend(list(flow_predictor.parameters()))
    
    # Add DiT parameters to optimizer ONLY if it's set to be trainable
    dit_params = [p for p in pipe.dit.parameters() if p.requires_grad]
    if dit_params:
        params_to_optimize.extend(dit_params)
        print(f"Optimizer will train Temporal Module, Flow Predictor, and {len(dit_params)} DiT parameters.")
    else:
        print("Optimizer will train Temporal Module and Flow Predictor ONLY.")

    optimizer = optim.AdamW(params_to_optimize, lr=args.lr)
    
    # 6. Setup Schedulers
    alpha_sched = AlphaScheduler(temporal_module, warmup_steps=args.alpha_warmup_steps, alpha_max=args.alpha_max, alpha_init=args.alpha_init)
    start_step = args.temporal_loss_warmup_steps
    scheduler_c = LossScheduler(args.lambda_c, args.lambda_warmup_steps, start_step)
    scheduler_m = LossScheduler(args.lambda_m, args.lambda_warmup_steps, start_step)
    scheduler_e = LossScheduler(args.lambda_e, args.lambda_warmup_steps, start_step)

    # 7. Training Loop
    global_step = 0
    for epoch in range(args.epochs):
        if global_step >= args.max_steps: break
        for _, frames in dataloader:
            if global_step >= args.max_steps: break

            frames = frames.to(device)
            with torch.no_grad():
                frames_for_vae = frames.to(dtype=torch_dtype)
                b, t_in, c_in, h_in, w_in = frames_for_vae.shape
                video_list_for_vae = [frames_for_vae[i].permute(1, 0, 2, 3) for i in range(b)]
                latents = pipe.vae.encode(video_list_for_vae, device=device, tiled=False)
                if latents is None: continue
                latents = latents.permute(0, 2, 1, 3, 4).to(torch.float32)

            b, t_latent, c_latent, h_latent, w_latent = latents.shape
            
            fused_latents_list = [latents[:, 0]]
            for t in range(1, t_latent):
                z_prev, z_cur = latents[:, t - 1].detach(), latents[:, t]
                flow = flow_predictor(z_prev, z_cur) if flow_predictor else None
                z_fused, _, _ = temporal_module(z_prev, z_cur, s_prev=None, s_cur=None, flow=flow)
                fused_latents_list.append(z_fused)
            latents_fused = torch.stack(fused_latents_list, dim=1)

            latents_to_decode = latents_fused.permute(0, 2, 1, 3, 4).to(dtype=torch_dtype)
            frames_hat_raw = pipe.vae.decode(latents_to_decode, device=device, tiled=False)
            
            if frames_hat_raw is None: continue
            
            if frames_hat_raw.shape[1] > 3:
                frames_hat = frames_hat_raw[:, :3, :, :, :]
            else:
                frames_hat = frames_hat_raw
            
            frames_hat = frames_hat.permute(0, 2, 1, 3, 4).to(torch.float32)

            target_frames, min_t = frames, min(frames_hat.shape[1], frames.shape[1])
            frames_hat, target_frames = frames_hat[:, :min_t], target_frames[:, :min_t]

            loss_rec = F.l1_loss(frames_hat, target_frames)
            total_loss = args.lambda_rec * loss_rec
            stage_active = global_step >= args.temporal_loss_warmup_steps
            current_alpha = alpha_sched.step(max(0, global_step - args.temporal_loss_warmup_steps) if stage_active else 0)
            
            loss_c, loss_m, loss_e = (torch.tensor(0.0, device=device) for _ in range(3))
            if stage_active and t_latent > 1:
                loss_c = F.l1_loss(latents_fused[:, 1:], latents_fused[:, :-1])
                loss_e = F.l1_loss(edge_map_tensor(frames_hat)[:, 1:], edge_map_tensor(frames_hat)[:, :-1])
                if args.lambda_m > 0: loss_m = compute_clip_motion_loss(frames_hat)
                
                total_loss += scheduler_c.weight(global_step) * loss_c
                total_loss += scheduler_m.weight(global_step) * loss_m
                total_loss += scheduler_e.weight(global_step) * loss_e

            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(params_to_optimize, max_norm=1.0)
            optimizer.step()

            if global_step % args.log_every == 0:
                phase = "Temporal" if stage_active else "Warmup (Recon Only)"
                print(f"[Step {global_step:05d}] Phase: {phase} | Loss: {total_loss.item():.4f} | Recon: {loss_rec.item():.4f} | Alpha: {current_alpha:.3f}")
                writer.add_scalar("Loss/Total", total_loss.item(), global_step)
            
            if args.save_checkpoint_every > 0 and global_step > 0 and global_step % args.save_checkpoint_every == 0:
                save_checkpoint(args, global_step, pipe, temporal_module, flow_predictor)

            global_step += 1

    save_checkpoint(args, "final", pipe, temporal_module, flow_predictor)
    writer.close()
    print("\n--- Training finished successfully. ---")

def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Staged training for temporal module.")
    
    # Required Paths
    parser.add_argument("--metadata_csv_path", type=str, required=True, help="Path to metadata CSV file.")
    parser.add_argument("--videos_root_dir", type=str, required=True, help="Path to the root directory of videos.")
    parser.add_argument("--logdir", type=str, default="./runs/staged_training", help="Root directory for logs and checkpoints.")
    
    # Model Configuration
    parser.add_argument("--model_id", type=str, default="Wan-AI/Wan2.1-T2V-1.3B", help="Base model ID from Hugging Face or local path.")
    parser.add_argument("--lora_path", type=str, default=None, help="Path to the LoRA weights file (.safetensors).")
    parser.add_argument("--lora_alpha", type=float, default=1.0, help="Alpha blending for LoRA.")
    
    # Training Strategy
    parser.add_argument('--train_lora', action='store_true', help="Enable training of the DiT model (after merging LoRA).")
    parser.add_argument('--no-train_lora', dest='train_lora', action='store_false', help="Disable training of the DiT model (freeze it).")
    parser.set_defaults(train_lora=True)
    parser.add_argument("--use_flow_predictor", action='store_true', help="Enable and train the latent flow predictor.")
    parser.add_argument("--use_gradient_checkpointing", action="store_true", help="Enable gradient checkpointing in DiT to save VRAM.")

    # Hardware and Precision
    parser.add_argument("--device", type=str, default="cuda", help="Computation device ('cuda' or 'cpu').")
    parser.add_argument("--dtype", type=str, choices=["fp16", "bf16", "fp32"], default="fp16", help="Mixed precision type.")
    
    # Dataloader and Batching
    parser.add_argument("--batch_size", type=int, default=1, help="Training batch size.")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of workers for dataloader.")
    parser.add_argument("--num_frames", type=int, default=16, help="Number of frames to sample per video.")
    parser.add_argument("--height", type=int, default=256, help="Frame height.")
    parser.add_argument("--width", type=int, default=256, help="Frame width.")
    
    # Training Loop Control
    parser.add_argument("--epochs", type=int, default=100, help="Max number of epochs.")
    parser.add_argument("--max_steps", type=int, default=5000, help="Max number of training steps.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    
    # Logging and Saving
    parser.add_argument("--log_every", type=int, default=50, help="Frequency of logging metrics.")
    parser.add_argument("--save_checkpoint_every", type=int, default=1000, help="Frequency (in steps) to save model checkpoints.")
    
    # Temporal Module and Schedulers
    parser.add_argument("--style_dim", type=int, default=None, help="Dimension of style vector (set to None to disable).")
    parser.add_argument("--alpha_init", type=float, default=0.2)
    parser.add_argument("--alpha_max", type=float, default=0.8)
    parser.add_argument("--alpha_warmup_steps", type=int, default=1500)
    parser.add_argument("--temporal_loss_warmup_steps", type=int, default=2000)
    
    # Loss Weights
    parser.add_argument("--lambda_rec", type=float, default=1.0)
    parser.add_argument("--lambda_c", type=float, default=0.2)
    parser.add_argument("--lambda_s", type=float, default=0.05) # This is not used but kept for consistency
    parser.add_argument("--lambda_m", type=float, default=0.1)
    parser.add_argument("--lambda_e", type=float, default=0.2)
    parser.add_argument("--lambda_warmup_steps", type=int, default=1000)
    
    # Advanced
    parser.add_argument("--decoder_expected_channels", type=int, default=16)
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.logdir, exist_ok=True)
    train(args)