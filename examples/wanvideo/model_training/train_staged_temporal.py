#!/usr/bin/env python3
"""
train_staged_temporal.py

Dedicated staged-training loop for the temporal module with WanVideoPipeline.
Stage 1 focuses on pure reconstruction. After `temporal_loss_warmup_steps`,
temporal consistency losses are enabled with scheduled weights.
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

try:
    import diffsynth  # noqa: F401
except ImportError:
    repo_root = Path(__file__).resolve().parents[3]
    sys.path.append(str(repo_root))

from diffsynth.modules.latent_flow_predictor import LatentFlowPredictor
from diffsynth.modules.temporal_module import TemporalModule
from diffsynth.utils import ModelConfig
from diffsynth.utils.alpha_scheduler import AlphaScheduler
from diffsynth.utils.pipeline_adapter import decode_latents_auto, encode_frames_auto
from diffsynth.lora import GeneralLoRALoader
from diffsynth.models.utils import load_state_dict as load_state_dict_file

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))

from dataset import RealVideoDataset  # noqa: E402

try:
    from transformers import CLIPModel, CLIPProcessor

    _CLIP_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _CLIP_MODEL = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(_CLIP_DEVICE).eval()
    _CLIP_PROCESSOR = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    _CLIP_AVAILABLE = True
except Exception:
    _CLIP_MODEL, _CLIP_PROCESSOR, _CLIP_DEVICE, _CLIP_AVAILABLE = None, None, None, False


def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_lora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Extract LoRA-specific weights from a model's state dict."""
    return {k: v.detach().cpu() for k, v in model.state_dict().items() if "lora_" in k}


def save_checkpoint(
    args,
    step: int | str,
    pipe,
    temporal_module: TemporalModule,
    flow_predictor: Optional[LatentFlowPredictor],
    latent_bridge: Optional[nn.Module],
):
    """Persist all trainable module weights at the specified step."""
    output_dir = Path(args.logdir) / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)
    step_tag = str(step)

    torch.save(temporal_module.state_dict(), output_dir / f"temporal_module_step_{step_tag}.pth")

    if flow_predictor is not None:
        torch.save(flow_predictor.state_dict(), output_dir / f"flow_predictor_step_{step_tag}.pth")

    if args.train_lora:
        lora_state = get_lora_state_dict(pipe.dit)
        if lora_state:
            torch.save(lora_state, output_dir / f"lora_step_{step_tag}.pth")
        else:
            print(
                f"Warning: LoRA training enabled but no LoRA tensors found on pipe.dit at step {step_tag}."
            )

    if latent_bridge is not None:
        torch.save(latent_bridge.state_dict(), output_dir / f"latent_bridge_step_{step_tag}.pth")

    print(f"[Checkpoint] Saved models for step {step_tag} to {output_dir}")


class LossScheduler:
    def __init__(self, lambda_final: float, warmup_steps: int, warmup_start: int):
        self.lambda_final = float(lambda_final)
        self.warmup_steps = int(max(0, warmup_steps))
        self.warmup_start = int(max(0, warmup_start))

    def weight(self, step: int) -> float:
        if step < self.warmup_start:
            return 0.0
        if self.warmup_steps == 0:
            return self.lambda_final
        progress = min(1.0, (step - self.warmup_start) / max(1, self.warmup_steps))
        return self.lambda_final * progress


def edge_map_tensor(frames: torch.Tensor) -> torch.Tensor:
    b, t, c, h, w = frames.shape
    flat = frames.view(b * t, c, h, w)
    if c == 3:
        gray = (0.299 * flat[:, 0] + 0.587 * flat[:, 1] + 0.114 * flat[:, 2]).unsqueeze(1)
    else:
        gray = flat[:, 0].unsqueeze(1)
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=frames.device).view(
        1, 1, 3, 3
    )
    sobel_y = sobel_x.permute(0, 1, 3, 2)
    gx = F.conv2d(gray, sobel_x, padding=1)
    gy = F.conv2d(gray, sobel_y, padding=1)
    magnitude = torch.sqrt(gx * gx + gy * gy + 1e-6)
    return magnitude.view(b, t, 1, h, w)


def build_dataloader(args) -> DataLoader:
    if RealVideoDataset is None:
        raise ImportError("RealVideoDataset is unavailable. Ensure dataset.py is located alongside this script.")
    if not os.path.isfile(args.metadata_csv_path):
        raise FileNotFoundError(f"Metadata CSV not found: {args.metadata_csv_path}")
    if not os.path.isdir(args.videos_root_dir):
        raise FileNotFoundError(f"Videos directory not found: {args.videos_root_dir}")
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
        drop_last=False,
    )


def initialise_pipeline(args, device: torch.device, torch_dtype: torch.dtype):
    from diffsynth.pipelines.wan_video_new import WanVideoPipeline

    model_configs = [
        ModelConfig(model_id=args.model_id, origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth"),
        ModelConfig(model_id=args.model_id, origin_file_pattern="diffusion_pytorch_model.safetensors"),
        ModelConfig(model_id=args.model_id, origin_file_pattern="Wan2.1_VAE.pth"),
    ]
    pipe = WanVideoPipeline.from_pretrained(device=device, torch_dtype=torch_dtype, model_configs=model_configs)
    if args.lora_path:
        print(f"Loading LoRA from {args.lora_path} (alpha={args.lora_alpha})")
        if args.train_lora:
            if not os.path.isfile(args.lora_path):
                raise FileNotFoundError(f"LoRA checkpoint not found: {args.lora_path}")
            lora_state = load_state_dict_file(args.lora_path, torch_dtype=torch_dtype, device=device)
            loader = GeneralLoRALoader(device=device, torch_dtype=torch_dtype)
            loader.load(pipe.dit, lora_state, alpha=args.lora_alpha)
        else:
            pipe.load_lora(pipe.dit, args.lora_path, alpha=args.lora_alpha)
    return pipe


def detect_latent_channels(pipe, dataloader, torch_dtype: torch.dtype, device: torch.device) -> Optional[int]:
    for _, frames in dataloader:
        frames = frames.to(device, dtype=torch_dtype)
        with torch.no_grad():
            latents = encode_frames_auto(pipe, frames)
        if latents is not None:
            return latents.shape[2]
    return None


def compute_clip_motion_loss(frames_hat: torch.Tensor) -> torch.Tensor:
    if not _CLIP_AVAILABLE:
        return torch.tensor(0.0, device=frames_hat.device)
    b, t, c, h, w = frames_hat.shape
    if t < 2:
        return torch.tensor(0.0, device=frames_hat.device)
    with torch.no_grad():
        imgs = [to_pil_image(frames_hat[i, j].clamp(0, 1)) for i in range(b) for j in range(t)]
        clip_inputs = _CLIP_PROCESSOR(images=imgs, return_tensors="pt", padding=True).to(_CLIP_DEVICE)
        features = _CLIP_MODEL.get_image_features(**clip_inputs).view(b, t, -1)
    features = features.to(frames_hat.device)
    return 1.0 - F.cosine_similarity(features[:, 1:], features[:, :-1], dim=-1).mean()


def train(args):
    device = torch.device(args.device)
    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    torch_dtype = dtype_map[args.dtype]

    set_global_seed(args.seed)

    run_name = f"staged_lr{args.lr}_warmup{args.temporal_loss_warmup_steps}_{int(time.time())}"
    logdir = Path(args.logdir) / run_name
    samples_dir = logdir / "samples"
    logdir.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(logdir))
    print(f"[Run] {run_name}")

    dataloader = build_dataloader(args)
    pipe = initialise_pipeline(args, device, torch_dtype)

    latent_channels = detect_latent_channels(pipe, dataloader, torch_dtype, device)
    if latent_channels is None:
        raise RuntimeError("Failed to determine latent channels from the pipeline encoder.")
    print(f"Detected latent channels: {latent_channels}")

    temporal_module = TemporalModule(latent_channels=latent_channels, style_dim=args.style_dim).to(device)
    flow_predictor = None
    if args.use_flow_predictor:
        flow_predictor = LatentFlowPredictor(in_channels=latent_channels).to(device)

    latent_bridge = None
    if latent_channels != args.decoder_expected_channels:
        print(f"Using 1x1 conv bridge: {latent_channels} -> {args.decoder_expected_channels}")
        latent_bridge = nn.Conv2d(latent_channels, args.decoder_expected_channels, kernel_size=1).to(device)

    params_to_optimize = list(temporal_module.parameters())
    if flow_predictor:
        params_to_optimize += list(flow_predictor.parameters())
    if args.train_lora:
        params_to_optimize += [p for p in pipe.dit.parameters() if p.requires_grad]
    if latent_bridge:
        params_to_optimize += list(latent_bridge.parameters())
    optimizer = optim.AdamW(params_to_optimize, lr=args.lr)

    alpha_sched = AlphaScheduler(
        temporal_module,
        warmup_steps=args.alpha_warmup_steps,
        alpha_max=args.alpha_max,
        alpha_init=args.alpha_init,
    )

    start_step = args.temporal_loss_warmup_steps
    scheduler_c = LossScheduler(args.lambda_c, args.lambda_warmup_steps, start_step)
    scheduler_s = LossScheduler(args.lambda_s, args.lambda_warmup_steps, start_step)
    scheduler_m = LossScheduler(args.lambda_m, args.lambda_warmup_steps, start_step)
    scheduler_e = LossScheduler(args.lambda_e, args.lambda_warmup_steps, start_step)

    global_step = 0
    for epoch in range(args.epochs):
        for _, frames in dataloader:
            if global_step >= args.max_steps:
                break

            frames = frames.to(device)
            with torch.no_grad():
                latents = encode_frames_auto(pipe, frames.to(dtype=torch_dtype))
                if latents is None:
                    continue
                latents = latents.to(torch.float32)

            b, t_latent, c_latent, h_latent, w_latent = latents.shape
            style_seq = None
            if args.style_dim and args.style_dim > 0:
                style_seq = torch.randn(b, t_latent, args.style_dim, device=device)

            fused_latents = [latents[:, 0]]
            for t in range(1, t_latent):
                z_prev, z_cur = latents[:, t - 1], latents[:, t]
                s_prev = style_seq[:, t - 1] if style_seq is not None else None
                s_cur = style_seq[:, t] if style_seq is not None else None
                flow = flow_predictor(z_prev, z_cur) if flow_predictor else None
                z_fused, _, _ = temporal_module(z_prev, z_cur, s_prev=s_prev, s_cur=s_cur, flow=flow)
                fused_latents.append(z_fused)
            latents_fused = torch.stack(fused_latents, dim=1)

            decode_input = latents_fused
            if latent_bridge:
                b_d, t_d, c_d, h_d, w_d = decode_input.shape
                decode_input = latent_bridge(decode_input.view(b_d * t_d, c_d, h_d, w_d)).view(
                    b_d, t_d, args.decoder_expected_channels, h_d, w_d
                )

            frames_hat = decode_latents_auto(pipe, decode_input.to(dtype=torch_dtype))
            if frames_hat is None:
                continue
            frames_hat = frames_hat.to(dtype=torch.float32)

            target_frames = frames
            if frames_hat.shape != target_frames.shape:
                min_t = min(frames_hat.shape[1], target_frames.shape[1])
                min_h = min(frames_hat.shape[3], target_frames.shape[3])
                min_w = min(frames_hat.shape[4], target_frames.shape[4])
                frames_hat = frames_hat[:, :min_t, :, :min_h, :min_w]
                target_frames = target_frames[:, :min_t, :, :min_h, :min_w]

            loss_rec = F.l1_loss(frames_hat, target_frames)
            total_loss = args.lambda_rec * loss_rec

            stage_active = global_step >= args.temporal_loss_warmup_steps
            current_alpha = alpha_sched.step(
                max(0, global_step - args.temporal_loss_warmup_steps) if stage_active else 0
            )

            loss_c = torch.tensor(0.0, device=device)
            loss_s = torch.tensor(0.0, device=device)
            loss_m = torch.tensor(0.0, device=device)
            loss_e = torch.tensor(0.0, device=device)

            if stage_active:
                if t_latent > 1:
                    loss_c = F.l1_loss(latents_fused[:, 1:], latents_fused[:, :-1])
                    if style_seq is not None:
                        loss_s = F.l1_loss(style_seq[:, 1:], style_seq[:, :-1])
                    edges_hat = edge_map_tensor(frames_hat)
                    loss_e = F.l1_loss(edges_hat[:, 1:], edges_hat[:, :-1])
                    if args.lambda_m > 0:
                        loss_m = compute_clip_motion_loss(frames_hat)

                total_loss = total_loss + scheduler_c.weight(global_step) * loss_c
                total_loss = total_loss + scheduler_s.weight(global_step) * loss_s
                total_loss = total_loss + scheduler_m.weight(global_step) * loss_m
                total_loss = total_loss + scheduler_e.weight(global_step) * loss_e

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(params_to_optimize, max_norm=1.0)
            optimizer.step()

            if global_step % args.log_every == 0:
                writer.add_scalar("loss/total", total_loss.item(), global_step)
                writer.add_scalar("loss/reconstruction", loss_rec.item(), global_step)
                writer.add_scalar("scheduler/alpha", current_alpha, global_step)
                if stage_active:
                    writer.add_scalar("loss/latent_consistency", loss_c.item(), global_step)
                    writer.add_scalar("loss/style_smooth", loss_s.item(), global_step)
                    writer.add_scalar("loss/clip_motion", loss_m.item(), global_step)
                    writer.add_scalar("loss/edge", loss_e.item(), global_step)
                phase = "temporal" if stage_active else "warmup"
                print(
                    f"[Step {global_step:05d}] phase={phase} "
                    f"loss={total_loss.item():.4f} rec={loss_rec.item():.4f} alpha={current_alpha:.3f}"
                )

            if args.save_samples_every > 0 and global_step % args.save_samples_every == 0:
                with torch.no_grad():
                    num_frames_to_save = min(target_frames.shape[1], args.sample_frames_to_save)
                    for idx in range(num_frames_to_save):
                        gt_path = samples_dir / f"step_{global_step}_gt_{idx}.png"
                        recon_path = samples_dir / f"step_{global_step}_recon_{idx}.png"
                        to_pil_image(target_frames[0, idx].clamp(0, 1)).save(gt_path)
                        to_pil_image(frames_hat[0, idx].clamp(0, 1)).save(recon_path)

            if args.save_checkpoint_every > 0 and global_step > 0 and global_step % args.save_checkpoint_every == 0:
                save_checkpoint(args, global_step, pipe, temporal_module, flow_predictor, latent_bridge)

            global_step += 1

        if global_step >= args.max_steps:
            break

    save_checkpoint(args, "final", pipe, temporal_module, flow_predictor, latent_bridge)
    writer.close()
    print("Training finished.")


def parse_args():
    parser = argparse.ArgumentParser(description="Staged training for temporal module.")
    parser.add_argument("--metadata_csv_path", type=str, default="/share/project/chengweiwu/code/Chinese_ink/hanzhe/ink_wash/final_inkwash_dataset/metadata.csv")
    parser.add_argument("--videos_root_dir", type=str, default="/share/project/chengweiwu/code/Chinese_ink/hanzhe/ink_wash/final_inkwash_dataset/videos")
    parser.add_argument("--logdir", type=str, default="./runs/staged_training")
    parser.add_argument("--model_id", type=str, default="Wan-AI/Wan2.1-T2V-1.3B")
    parser.add_argument("--lora_path", type=str, default="/share/project/chengweiwu/code/Chinese_ink/hanzhe/ink_wash/lora_outputs/inkwash_style_v1/epoch-18.safetensors")
    parser.add_argument("--lora_alpha", type=float, default=1.0)
    parser.add_argument("--train_lora", dest="train_lora", action="store_true", help="Enable LoRA fine-tuning.")
    parser.add_argument("--no_train_lora", dest="train_lora", action="store_false", help="Disable LoRA fine-tuning.")
    parser.add_argument("--use_flow_predictor", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--num_frames", type=int, default=8)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--max_steps", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--save_samples_every", type=int, default=1000)
    parser.add_argument("--save_checkpoint_every", type=int, default=1000, help="Frequency (in steps) to save checkpoints. Set to 0 to disable periodic saving.")
    parser.add_argument("--sample_frames_to_save", type=int, default=4)
    parser.add_argument("--style_dim", type=int, default=128)
    parser.add_argument("--alpha_init", type=float, default=0.2)
    parser.add_argument("--alpha_max", type=float, default=0.8)
    parser.add_argument("--alpha_warmup_steps", type=int, default=1500)
    parser.add_argument("--temporal_loss_warmup_steps", type=int, default=2000)
    parser.add_argument("--lambda_rec", type=float, default=1.0)
    parser.add_argument("--lambda_c", type=float, default=0.2)
    parser.add_argument("--lambda_s", type=float, default=0.05)
    parser.add_argument("--lambda_m", type=float, default=0.1)
    parser.add_argument("--lambda_e", type=float, default=0.2)
    parser.add_argument("--lambda_warmup_steps", type=int, default=3000)
    parser.add_argument("--decoder_expected_channels", type=int, default=16)
    parser.set_defaults(train_lora=True)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
