# examples/wanvideo/model_training/train_with_temporal.py (FINAL CORRECTED VERSION v4)
import os
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter
from transformers import CLIPProcessor, CLIPModel
from torchvision.transforms.functional import to_pil_image

# --- Local Module Imports ---
from diffsynth.modules.temporal_module import TemporalModule
from diffsynth.modules.latent_flow_predictor import LatentFlowPredictor
from diffsynth.utils.alpha_scheduler import AlphaScheduler
from diffsynth.utils.pipeline_adapter import encode_frames_auto, decode_latents_auto

# --- Try to import DiffSynth specific modules ---
try:
    from diffsynth.pipelines.wan_video_new import WanVideoPipeline
    from diffsynth.utils import ModelConfig
    DIFFSYNTH_AVAILABLE = True
except ImportError:
    print("Warning: `diffsynth` specific pipeline modules not found. `--use_pipe` will not be available.")
    WanVideoPipeline = None
    ModelConfig = None
    DIFFSYNTH_AVAILABLE = False
    
# ---- Helper Function: Sobel Edge Detector ----
def sobel_edge_map(img_tensor: torch.Tensor) -> torch.Tensor:
    if img_tensor.size(1) == 3:
        gray = 0.299 * img_tensor[:, 0] + 0.587 * img_tensor[:, 1] + 0.114 * img_tensor[:, 2]
    else:
        gray = img_tensor[:, 0]
    gray = gray.unsqueeze(1)
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=img_tensor.device).view(1, 1, 3, 3)
    sobel_y = sobel_x.permute(0, 1, 3, 2)
    gx = torch.nn.functional.conv2d(gray, sobel_x, padding=1)
    gy = torch.nn.functional.conv2d(gray, sobel_y, padding=1)
    magnitude = torch.sqrt(gx**2 + gy**2 + 1e-6)
    magnitude = magnitude / (magnitude.amax(dim=(2, 3), keepdim=True) + 1e-6)
    return torch.sigmoid((magnitude - 0.1) * 10.0)

# ---- Placeholder: Dummy Video Dataset ----
class DummyVideoDataset(torch.utils.data.Dataset):
    def __init__(self, num_samples=100, T=8, H=64, W=64):
        self.num_samples, self.T, self.H, self.W = num_samples, T, H, W
    def __len__(self): return self.num_samples
    def __getitem__(self, idx):
        text = "a beautiful landscape painting"
        frames = torch.rand(self.T, 3, self.H, self.W)
        return text, frames

# ---- Main Training Function ----
def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    print(f"Starting training run on device: {args.device} with seed: {args.seed}")

    # --- 1. Initialize Pipeline ---
    pipe = None
    if args.use_pipe:
        if not DIFFSYNTH_AVAILABLE: raise ImportError("`diffsynth` library is required for `--use_pipe`.")
        print("Initializing WanVideoPipeline...")
        model_configs = [
            ModelConfig(model_id=args.model_id, origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth"),
            ModelConfig(model_id=args.model_id, origin_file_pattern="diffusion_pytorch_model.safetensors"),
            ModelConfig(model_id=args.model_id, origin_file_pattern="Wan2.1_VAE.pth"),
        ]
        pipe = WanVideoPipeline.from_pretrained(
            device=args.device, torch_dtype=torch.float16, model_configs=model_configs
        )
        print("Pipeline created.")
        
    # --- Dynamically determine latent channels ---
    actual_latent_channels = args.latent_channels
    DECODER_EXPECTED_CHANNELS = 16
    if args.use_pipe:
        print("Probing VAE for actual latent channel count...")
        with torch.no_grad():
            dummy_frames = torch.zeros(1, args.num_frames, 3, args.height, args.width, device=args.device, dtype=torch.float16)
            dummy_latents = encode_frames_auto(pipe, dummy_frames)
            if dummy_latents is not None:
                actual_latent_channels = dummy_latents.shape[2]
                print(f"Detected actual encoder output channels: {actual_latent_channels}")
            else:
                print(f"Warning: Could not probe latent channels. Falling back to default: {args.latent_channels}")

    # --- 2. Initialize Modules ---
    temporal_module = TemporalModule(latent_channels=actual_latent_channels, style_dim=args.style_dim).to(args.device)
    flow_predictor = LatentFlowPredictor(in_channels=actual_latent_channels).to(args.device) if args.use_flow_predictor else None
    
    latent_upsampler = None
    if args.use_pipe and actual_latent_channels != DECODER_EXPECTED_CHANNELS:
        print(f"Creating a latent channel upsampler bridge: {actual_latent_channels} -> {DECODER_EXPECTED_CHANNELS}")
        latent_upsampler = nn.Conv2d(actual_latent_channels, DECODER_EXPECTED_CHANNELS, kernel_size=1).to(args.device)

    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(args.device).eval()
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    alpha_scheduler = AlphaScheduler(temporal_module, warmup_steps=args.warmup_steps, alpha_max=args.alpha_max, alpha_init=args.alpha_init)
    
    # --- 3. Configure Optimizer ---
    params_to_optimize = []
    if args.train_lora and pipe and args.lora_path:
        print(f"Loading LoRA from {args.lora_path} into pipeline DIT model...")
        # Correctly load LoRA into the DIT sub-model via the main pipe interface
        pipe.load_lora(pipe.dit, args.lora_path)
        # [FINAL LORA FIX] Correctly set the alpha on the main pipe object
        pipe.set_lora_alpha(args.lora_alpha)
        
        lora_params = [p for p in pipe.parameters() if p.requires_grad]
        params_to_optimize.extend(lora_params)
        print(f"Added {len(lora_params)} LoRA params to optimizer.")
    
    params_to_optimize.extend(list(temporal_module.parameters()))
    if flow_predictor:
        params_to_optimize.extend(list(flow_predictor.parameters()))
    if latent_upsampler:
        params_to_optimize.extend(list(latent_upsampler.parameters()))

    optimizer = optim.AdamW(params_to_optimize, lr=args.lr)

    # --- 4. Prepare Data and Logging ---
    dataset = DummyVideoDataset(num_samples=50, T=args.num_frames, H=args.height, W=args.width)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    writer = SummaryWriter(log_dir=args.logdir)

    # --- 5. Main Training Loop ---
    global_step = 0
    training_complete = False
    for epoch in range(args.epochs):
        if training_complete: break
        for text, frames in dataloader:
            if training_complete: break
            
            frames = frames.to(args.device)
            B, T_input, C, H, W = frames.shape
            
            with torch.no_grad():
                if args.use_pipe:
                    latents = encode_frames_auto(pipe, frames.to(torch.float16))
                    if latents is None: continue
                    latents = latents.to(torch.float32)
                else:
                    latents = torch.randn(B, T_input, actual_latent_channels, H // 8, W // 8, device=args.device)
            
            s_seq = torch.randn(B, T_input, args.style_dim, device=args.device) if args.style_dim else None
            _, T_latents, _, _, _ = latents.shape
            
            fused_latents_list = [latents[:, 0]]
            for t in range(1, T_latents):
                z_prev, z_cur = latents[:, t - 1].detach(), latents[:, t]
                s_prev = s_seq[:, t - 1] if s_seq is not None else None
                s_cur = s_seq[:, t] if s_seq is not None else None
                flow = flow_predictor(z_prev, z_cur) if flow_predictor else None
                z_fused, _, _ = temporal_module(z_prev, z_cur, s_prev, s_cur, flow=flow)
                fused_latents_list.append(z_fused)
            latents_fused = torch.stack(fused_latents_list, dim=1)

            frames_hat = None
            if args.use_pipe:
                latents_to_decode = latents_fused
                if latent_upsampler:
                    B_lt, T_lt, C_lt, H_lt, W_lt = latents_to_decode.shape
                    latents_reshaped = latents_to_decode.view(B_lt * T_lt, C_lt, H_lt, W_lt)
                    latents_upsampled = latent_upsampler(latents_reshaped)
                    latents_to_decode = latents_upsampled.view(B_lt, T_lt, DECODER_EXPECTED_CHANNELS, H_lt, W_lt)

                frames_hat = decode_latents_auto(pipe, latents_to_decode.to(torch.float16))
                if frames_hat is not None:
                    frames_hat = frames_hat.to(torch.float32)

            if frames_hat is None:
                frames_hat = latents_fused.mean(dim=2, keepdim=True).repeat(1, 1, 3, 1, 1)
                frames_hat = nn.functional.interpolate(frames_hat.view(-1, 3, *frames_hat.shape[3:]), size=(H, W)).view(B, T_latents, 3, H, W)
            
            num_frames_for_loss = min(frames.shape[1], frames_hat.shape[1])
            frames_for_loss = frames[:, :num_frames_for_loss]
            frames_hat_for_loss = frames_hat[:, :num_frames_for_loss]

            loss_rec = nn.functional.l1_loss(frames_hat_for_loss, frames_for_loss)
            loss_c = torch.mean((latents_fused[:, 1:] - latents_fused[:, :-1])**2)
            loss_s = torch.mean((s_seq[:, 1:T_latents] - s_seq[:, :T_latents-1])**2) if s_seq is not None else torch.tensor(0.0, device=args.device)
            
            with torch.no_grad():
                pil_images = [to_pil_image(frame.clamp(0, 1)) for frame_seq in frames_hat_for_loss for frame in frame_seq]
                clip_inputs = clip_processor(images=pil_images, return_tensors="pt", padding=True).to(args.device)
                image_features = clip_model.get_image_features(**clip_inputs).view(B, num_frames_for_loss, -1)
            loss_m = 1.0 - torch.nn.functional.cosine_similarity(image_features[:, 1:], image_features[:, :-1], dim=-1).mean()

            edges_hat = sobel_edge_map(frames_hat_for_loss.reshape(-1, C, H, W)).view(B, num_frames_for_loss, 1, H, W)
            edges_gt = sobel_edge_map(frames_for_loss.reshape(-1, C, H, W)).view(B, num_frames_for_loss, 1, H, W)
            loss_e = torch.nn.functional.l1_loss(edges_hat[:, 1:], edges_gt[:, :-1])
            
            total_loss = (args.lambda_rec * loss_rec + args.lambda_c * loss_c + 
                          args.lambda_s * loss_s + args.lambda_m * loss_m + args.lambda_e * loss_e)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            current_alpha = alpha_scheduler.step(global_step)
            
            if global_step % args.log_every == 0:
                print(f"Step: {global_step:04d} | Loss: {total_loss.item():.4f} | Alpha: {current_alpha:.3f}")

            global_step += 1
            if global_step >= args.max_steps:
                training_complete = True
    
    final_num_frames = min(frames.shape[1], frames_hat.shape[1])
    torch.save(frames[:, :final_num_frames].cpu(), os.path.join(args.logdir, "ground_truth.pt"))
    torch.save(frames_hat[:, :final_num_frames].cpu(), os.path.join(args.logdir, "predicted_frames.pt"))
    print(f"Saved final outputs to {args.logdir} for evaluation.")
    writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Temporal Module Training and Evaluation Script")
    parser.add_argument("--use_pipe", action="store_true")
    parser.add_argument("--model_id", type=str, default="Wan-AI/Wan2.1-T2V-1.3B")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--logdir", type=str, default="./runs/temporal_default")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--use_flow_predictor", action="store_true")
    parser.add_argument("--warmup_steps", type=int, default=400)
    parser.add_argument("--alpha_init", type=float, default=0.2)
    parser.add_argument("--alpha_max", type=float, default=0.8)
    parser.add_argument("--lora_path", type=str, default=None)
    parser.add_argument("--lora_alpha", type=float, default=1.0)
    parser.add_argument("--train_lora", action="store_true")
    parser.add_argument("--num_frames", type=int, default=8)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--latent_channels", type=int, default=16)
    parser.add_argument("--style_dim", type=int, default=None)
    parser.add_argument("--lambda_rec", type=float, default=1.0)
    parser.add_argument("--lambda_c", type=float, default=0.2)
    parser.add_argument("--lambda_s", type=float, default=0.05)
    parser.add_argument("--lambda_m", type=float, default=0.1)
    parser.add_argument("--lambda_e", type=float, default=0.2)
    parser.add_argument("--log_every", type=int, default=20)
    
    args = parser.parse_args()
    os.makedirs(args.logdir, exist_ok=True)
    main(args)