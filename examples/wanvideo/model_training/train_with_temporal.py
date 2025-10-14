"""Minimal training loop integrating the TemporalModule."""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import functional as TF
from torchvision.utils import make_grid
from PIL import Image

try:  # Pillow>=10
    Resample = Image.Resampling
except AttributeError:  # pragma: no cover - Pillow<10 fallback
    Resample = Image

try:  # Optional dependency; only required when CLIP loss is enabled.
    from transformers import CLIPModel, CLIPProcessor
except ImportError:  # pragma: no cover - optional dependency
    CLIPModel = None
    CLIPProcessor = None

try:  # Optional heavy dependency for real pipeline integration.
    from diffsynth.pipelines.wan_video_new import WanVideoPipeline
except ImportError:  # pragma: no cover - fallback if pipeline is not available.
    WanVideoPipeline = None

try:
    from diffsynth.utils import ModelConfig
except ImportError:  # pragma: no cover - allow running with minimal deps
    ModelConfig = None

from diffsynth.modules.latent_flow_predictor import LatentFlowPredictor
from diffsynth.modules.temporal_module import TemporalModule
from diffsynth.utils.alpha_scheduler import AlphaScheduleConfig, AlphaScheduler


@dataclass
class TrainingConfig:
    """Hyper-parameters for the temporal smoke test."""

    device: str = "cuda"
    batch_size: int = 1
    num_frames: int = 8
    height: int = 64
    width: int = 64
    latent_channels: int = 64
    style_dim: int = 0
    lr: float = 1e-4
    steps: int = 200
    log_interval: int = 20
    clip_weight: float = 0.1
    rec_weight: float = 1.0
    content_weight: float = 0.2
    style_weight: float = 0.05
    edge_weight: float = 0.2
    warmup_steps: int = 500
    alpha_init: float = 0.2
    alpha_max: float = 0.8
    latent_downsample: int = 4
    use_clip: bool = True
    logdir: str = "./runs/temporal_smoke"
    use_pipeline: bool = False
    model_paths: Optional[Sequence[str]] = None
    data_root: Optional[str] = None


class VideoFrameFolderDataset(Dataset):
    """Dataset that reads videos from ``root/clip_xxx/frame_yyy.png`` folders."""

    def __init__(
        self,
        root: Path,
        num_frames: int,
        size: Tuple[int, int],
    ) -> None:
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root '{self.root}' does not exist")
        self.sequence_dirs = sorted(p for p in self.root.iterdir() if p.is_dir())
        if not self.sequence_dirs:
            raise RuntimeError(f"No sub-folders found under '{self.root}'")
        self.num_frames = num_frames
        self.size = size

    def __len__(self) -> int:
        return len(self.sequence_dirs)

    def __getitem__(self, idx: int) -> Tuple[str, torch.Tensor]:
        seq_dir = self.sequence_dirs[idx]
        frame_paths = sorted(
            [p for p in seq_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}]
        )
        if len(frame_paths) < self.num_frames:
            raise RuntimeError(f"Sequence '{seq_dir}' has fewer than {self.num_frames} frames")
        frames = []
        for path in frame_paths[: self.num_frames]:
            img = Image.open(path).convert("RGB")
            if self.size is not None:
                img = img.resize((self.size[1], self.size[0]), resample=Resample.BILINEAR)
            frames.append(TF.to_tensor(img))
        stacked = torch.stack(frames, dim=0)  # [T, C, H, W]
        return seq_dir.name, stacked


class DummyVideoDataset(Dataset):
    """Fallback dataset that synthesises random clips for smoke testing."""

    def __init__(self, length: int, num_frames: int, height: int, width: int) -> None:
        self.length = length
        self.num_frames = num_frames
        self.height = height
        self.width = width

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Tuple[str, torch.Tensor]:
        frames = torch.rand(self.num_frames, 3, self.height, self.width)
        return f"dummy_{idx}", frames


class SimpleLatentDecoder(nn.Module):
    """Small convolutional decoder used when a VAE is unavailable."""

    def __init__(self, latent_channels: int, out_channels: int, upsample_factor: int) -> None:
        super().__init__()
        self.upsample_factor = upsample_factor
        self.net = nn.Sequential(
            nn.Conv2d(latent_channels, latent_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(latent_channels, out_channels, kernel_size=1),
        )

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = latents.shape
        x = latents.view(b * t, c, h, w)
        if self.upsample_factor != 1:
            x = F.interpolate(
                x,
                scale_factor=float(self.upsample_factor),
                mode="bilinear",
                align_corners=False,
            )
        x = self.net(x)
        x = torch.sigmoid(x)
        x = x.view(b, t, -1, x.shape[-2], x.shape[-1])
        return x


class SimpleStyleEncoder(nn.Module):
    """Average-pool style encoder used when style embeddings are desired."""

    def __init__(self, in_channels: int, style_dim: int) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(in_channels, style_dim)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = frames.shape
        x = frames.view(b * t, c, h, w)
        pooled = self.pool(x).view(b, t, c)
        style = self.proj(pooled)
        return style


def build_dataloader(cfg: TrainingConfig) -> DataLoader:
    if cfg.data_root is not None:
        dataset = VideoFrameFolderDataset(Path(cfg.data_root), cfg.num_frames, (cfg.height, cfg.width))
    else:
        dataset = DummyVideoDataset(length=400, num_frames=cfg.num_frames, height=cfg.height, width=cfg.width)
    return DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=True)


def edge_map_tensor(img_tensor: torch.Tensor) -> torch.Tensor:
    if img_tensor.size(1) == 3:
        gray = 0.299 * img_tensor[:, 0] + 0.587 * img_tensor[:, 1] + 0.114 * img_tensor[:, 2]
    else:
        gray = img_tensor[:, 0]
    gray = gray.unsqueeze(1)
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=img_tensor.dtype, device=img_tensor.device).view(1, 1, 3, 3)
    sobel_y = sobel_x.permute(0, 1, 3, 2)
    gx = F.conv2d(gray, sobel_x, padding=1)
    gy = F.conv2d(gray, sobel_y, padding=1)
    mag = torch.sqrt(gx * gx + gy * gy + 1e-6)
    mag = mag / (mag.amax(dim=[2, 3], keepdim=True) + 1e-6)
    return torch.sigmoid((mag - 0.1) * 10.0)


def encode_latents(
    frames: torch.Tensor,
    cfg: TrainingConfig,
    pipeline: Optional[WanVideoPipeline],
    device: torch.device,
) -> torch.Tensor:
    b, t, c, h, w = frames.shape
    if pipeline is not None:
        if not hasattr(pipeline, "vae"):
            raise RuntimeError("Provided pipeline lacks a VAE; cannot encode latents.")
        flat = frames.view(b * t, c, h, w).to(device=pipeline.device, dtype=pipeline.torch_dtype)
        with torch.no_grad():  # VAE encoder is frozen for smoke tests
            encoded = pipeline.vae.encode(flat, device=pipeline.device)
        if isinstance(encoded, (list, tuple)):
            encoded = encoded[0]
        encoded = encoded.view(b, t, *encoded.shape[1:]).to(device)
        encoded.requires_grad_(True)
        return encoded

    latent_h = h // cfg.latent_downsample
    latent_w = w // cfg.latent_downsample
    latents = torch.randn(b, t, cfg.latent_channels, latent_h, latent_w, device=device, requires_grad=True)
    return latents


def decode_latents(
    latents: torch.Tensor,
    decoder: Optional[SimpleLatentDecoder],
    pipeline: Optional[WanVideoPipeline],
    frames_ref: torch.Tensor,
) -> torch.Tensor:
    if pipeline is not None:
        if not hasattr(pipeline, "vae"):
            raise RuntimeError("Provided pipeline lacks a VAE; cannot decode latents.")
        b, t, c, h, w = latents.shape
        flat = latents.view(b * t, c, h, w).to(device=pipeline.device, dtype=pipeline.torch_dtype)
        video = pipeline.vae.decode(flat, device=pipeline.device)
        if isinstance(video, (list, tuple)):
            video = video[0]
        video = video.view(b, t, *video.shape[1:])
        return video.to(latents.device)

    if decoder is None:
        raise ValueError("decoder must be provided when pipeline is None")
    return decoder(latents)


def load_pipeline(cfg: TrainingConfig) -> Optional[WanVideoPipeline]:
    if not cfg.use_pipeline:
        return None
    if WanVideoPipeline is None:
        raise RuntimeError("WanVideoPipeline is not available. Install the required dependencies first.")
    if not cfg.model_paths:
        raise ValueError("--model-paths must be provided when --use-pipeline is set")

    if ModelConfig is None:
        raise RuntimeError("ModelConfig is unavailable; ensure diffsynth.utils is importable.")

    model_configs = [ModelConfig(path=path) for path in cfg.model_paths]

    pipeline = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.float32,
        device=cfg.device,
        model_configs=model_configs,
    )
    return pipeline


def prepare_clip(cfg: TrainingConfig, device: torch.device) -> Tuple[Optional[CLIPModel], Optional[CLIPProcessor]]:
    if not cfg.use_clip:
        return None, None
    if CLIPModel is None or CLIPProcessor is None:
        raise RuntimeError("transformers is required for CLIP-based losses. Install `transformers` to enable it.")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()
    return model, processor


def compute_clip_temporal_loss(
    frames_hat: torch.Tensor,
    clip_model: CLIPModel,
    clip_processor: CLIPProcessor,
    device: torch.device,
) -> torch.Tensor:
    b, t, c, h, w = frames_hat.shape
    imgs = frames_hat.view(b * t, c, h, w).clamp(0, 1)
    pil_images = [TF.to_pil_image(img.cpu()) for img in imgs]
    inputs = clip_processor(images=pil_images, return_tensors="pt").to(device)
    with torch.no_grad():
        clip_embs = clip_model.get_image_features(**inputs)
    clip_embs = clip_embs.view(b, t, -1)
    loss = 0.0
    for idx in range(1, t):
        cos = F.cosine_similarity(clip_embs[:, idx, :], clip_embs[:, idx - 1, :], dim=-1)
        loss = loss + torch.mean(1.0 - cos)
    return loss / max(1, t - 1)


def parse_args() -> TrainingConfig:
    parser = argparse.ArgumentParser(description="TemporalModule smoke training")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--latent-channels", type=int, default=64)
    parser.add_argument("--style-dim", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--clip-weight", type=float, default=0.1)
    parser.add_argument("--rec-weight", type=float, default=1.0)
    parser.add_argument("--content-weight", type=float, default=0.2)
    parser.add_argument("--style-weight", type=float, default=0.05)
    parser.add_argument("--edge-weight", type=float, default=0.2)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--alpha-init", type=float, default=0.2)
    parser.add_argument("--alpha-max", type=float, default=0.8)
    parser.add_argument("--latent-downsample", type=int, default=4)
    parser.add_argument("--logdir", type=str, default="./runs/temporal_smoke")
    parser.add_argument("--use-pipeline", action="store_true")
    parser.add_argument("--model-paths", type=str, nargs="*")
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--disable-clip", action="store_true")

    args = parser.parse_args()

    cfg = TrainingConfig(
        device=args.device,
        batch_size=args.batch_size,
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
        latent_channels=args.latent_channels,
        style_dim=args.style_dim,
        lr=args.lr,
        steps=args.steps,
        log_interval=args.log_interval,
        clip_weight=args.clip_weight,
        rec_weight=args.rec_weight,
        content_weight=args.content_weight,
        style_weight=args.style_weight,
        edge_weight=args.edge_weight,
        warmup_steps=args.warmup_steps,
        alpha_init=args.alpha_init,
        alpha_max=args.alpha_max,
        latent_downsample=args.latent_downsample,
        use_clip=not args.disable_clip,
        logdir=args.logdir,
        use_pipeline=args.use_pipeline,
        model_paths=args.model_paths,
        data_root=args.data_root,
    )
    return cfg


def main() -> None:
    cfg = parse_args()
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    loader = build_dataloader(cfg)
    pipeline = load_pipeline(cfg)
    clip_model, clip_processor = prepare_clip(cfg, device) if cfg.use_clip else (None, None)

    temporal_module = TemporalModule(
        latent_channels=cfg.latent_channels,
        style_dim=cfg.style_dim if cfg.style_dim > 0 else None,
        learnable_alpha=True,
        alpha_init=cfg.alpha_init,
    ).to(device)
    flow_predictor = LatentFlowPredictor(cfg.latent_channels).to(device)
    decoder = None
    if pipeline is None:
        decoder = SimpleLatentDecoder(cfg.latent_channels, out_channels=3, upsample_factor=cfg.latent_downsample).to(device)
    style_encoder = None
    if cfg.style_dim > 0:
        style_encoder = SimpleStyleEncoder(3, cfg.style_dim).to(device)

    alpha_sched = AlphaScheduler(
        temporal_module,
        AlphaScheduleConfig(
            warmup_steps=cfg.warmup_steps,
            alpha_init=cfg.alpha_init,
            alpha_max=cfg.alpha_max,
        ),
    )

    modules: List[nn.Module] = [temporal_module, flow_predictor]
    if decoder is not None:
        modules.append(decoder)
    if style_encoder is not None:
        modules.append(style_encoder)

    params: List[nn.Parameter] = []
    for module in modules:
        params.extend(list(module.parameters()))

    optimizer = optim.AdamW(params, lr=cfg.lr)
    writer = SummaryWriter(log_dir=cfg.logdir)

    global_step = 0
    for epoch in range(10_000):  # loop until steps consumed
        for prompt, frames in loader:
            frames = frames.to(device)
            if frames.dim() != 5:
                raise RuntimeError("Expected frames with shape [B, T, C, H, W]")
            batch_size, num_frames, channels, height, width = frames.shape

            latents = encode_latents(frames, cfg, pipeline, device)
            if style_encoder is not None:
                style_seq = style_encoder(frames)
            else:
                style_seq = None

            fused_latents: List[torch.Tensor] = [latents[:, 0]]
            fused_styles: List[torch.Tensor] = [style_seq[:, 0]] if style_seq is not None else []
            aux_history: List[dict] = []

            for t in range(1, num_frames):
                z_prev = fused_latents[-1]
                z_cur = latents[:, t]
                s_prev = fused_styles[-1] if fused_styles else None
                s_cur = style_seq[:, t] if style_seq is not None else None
                flow = flow_predictor(z_prev, z_cur)
                z_fused, s_fused, aux = temporal_module(z_prev, z_cur, s_prev=s_prev, s_cur=s_cur, flow=flow)
                fused_latents.append(z_fused)
                if s_fused is not None:
                    fused_styles.append(s_fused)
                aux_history.append(aux)

            latents_fused = torch.stack(fused_latents, dim=1)
            styles_fused = torch.stack(fused_styles, dim=1) if fused_styles else None

            frames_hat = decode_latents(latents_fused, decoder, pipeline, frames)

            loss_rec = F.l1_loss(frames_hat, frames)
            loss_content = torch.zeros((), device=device, dtype=frames_hat.dtype)
            for t in range(1, num_frames):
                loss_content = loss_content + torch.mean((latents_fused[:, t] - latents_fused[:, t - 1]) ** 2)
            loss_content = loss_content / max(1, num_frames - 1)

            if styles_fused is not None:
                loss_style = torch.zeros((), device=device, dtype=frames_hat.dtype)
                for t in range(1, num_frames):
                    loss_style = loss_style + torch.mean((styles_fused[:, t] - styles_fused[:, t - 1]) ** 2)
                loss_style = loss_style / max(1, num_frames - 1)
            else:
                loss_style = torch.zeros((), device=device, dtype=frames_hat.dtype)

            if cfg.use_clip and clip_model is not None and clip_processor is not None:
                loss_clip = compute_clip_temporal_loss(frames_hat, clip_model, clip_processor, device)
            else:
                loss_clip = torch.zeros((), device=device, dtype=frames_hat.dtype)

            edges_hat = edge_map_tensor(frames_hat.view(batch_size * num_frames, channels, height, width)).view(
                batch_size, num_frames, 1, height, width
            )
            edges_ref = edge_map_tensor(frames.view(batch_size * num_frames, channels, height, width)).view(
                batch_size, num_frames, 1, height, width
            )
            loss_edge = torch.zeros((), device=device, dtype=frames_hat.dtype)
            for t in range(1, num_frames):
                loss_edge = loss_edge + F.l1_loss(edges_hat[:, t], edges_ref[:, t - 1])
            loss_edge = loss_edge / max(1, num_frames - 1)

            loss = (
                cfg.rec_weight * loss_rec
                + cfg.content_weight * loss_content
                + cfg.style_weight * loss_style
                + cfg.clip_weight * loss_clip
                + cfg.edge_weight * loss_edge
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            alpha_val = alpha_sched.step(global_step)

            if global_step % cfg.log_interval == 0:
                print(
                    f"step {global_step:05d} loss={loss.item():.4f} "
                    f"L_rec={loss_rec.item():.4f} L_c={loss_content.item():.4f} "
                    f"alpha={alpha_val:.4f}"
                )
                writer.add_scalar("train/loss", loss.item(), global_step)
                writer.add_scalar("train/L_rec", loss_rec.item(), global_step)
                writer.add_scalar("train/L_c", loss_content.item(), global_step)
                writer.add_scalar("temporal/alpha", alpha_val, global_step)
                if cfg.use_clip:
                    writer.add_scalar("train/L_clip", loss_clip.item(), global_step)
                writer.add_scalar("train/L_edge", loss_edge.item(), global_step)

                grid = make_grid(frames_hat[0], nrow=num_frames)
                writer.add_image("samples/reconstructions", grid, global_step)
                if aux_history:
                    writer.add_scalar("temporal/z_warp_mean", aux_history[-1]["z_warp_mean"], global_step)

            global_step += 1
            if global_step >= cfg.steps:
                writer.close()
                print("Smoke training finished. total steps:", global_step)
                return

        # safety escape
        if global_step >= cfg.steps:
            break


if __name__ == "__main__":
    main()
