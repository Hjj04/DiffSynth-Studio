# tools/decode_checkpoint_to_video.py
#!/usr/bin/env python3
import os
import argparse
import torch
import numpy as np
import imageio
from diffsynth.utils.pipeline_adapter import decode_latents_auto
from diffsynth.pipelines.wan_video_new import WanVideoPipeline
from diffsynth.utils import ModelConfig
from tools.pick_best_checkpoint import find_metrics_csv, pick_best_checkpoint

def tensors_to_video(frames_tensor: torch.Tensor, output_path: str, fps: int = 8):
    """Converts a tensor of shape [T, C, H, W] to a video file."""
    if frames_tensor.dim() == 5: # [B, T, C, H, W]
        frames_tensor = frames_tensor[0]
    
    # Convert from [T, C, H, W] float [0,1] to list of [H, W, C] uint8
    images = []
    for frame in frames_tensor:
        img_np = frame.permute(1, 2, 0).cpu().numpy()
        img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
        images.append(img_np)
        
    # Save video
    imageio.mimsave(output_path, images, fps=fps)
    print(f"Video saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Decode a checkpoint (.pt file) containing latents or frames into a video.")
    parser.add_argument("--checkpoint", type=str, help="Path to the .pt checkpoint file.")
    parser.add_argument("--auto_best_from", type=str, help="Path to an experiment run directory to automatically find the best checkpoint.")
    parser.add_argument("--model_id", type=str, default="Wan-AI/Wan2.1-T2V-1.3B", help="Model ID for the WanVideoPipeline used for decoding.")
    parser.add_argument("--out", type=str, default="output.mp4", help="Output video file path.")
    parser.add_argument("--fps", type=int, default=8, help="Frames per second for the output video.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run decoding on.")
    args = parser.parse_args()

    checkpoint_path = args.checkpoint
    if args.auto_best_from:
        print(f"Automatically selecting best checkpoint from '{args.auto_best_from}'...")
        metrics_file = find_metrics_csv(args.auto_best_from)
        # We assume the best model is the one from the final saved state
        checkpoint_path = os.path.join(args.auto_best_from, "predicted_frames.pt")

    if not checkpoint_path or not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    print(f"Loading data from: {checkpoint_path}")
    data = torch.load(checkpoint_path, map_location=args.device)

    # Check if the loaded data is already image frames
    # Our training script saves frames as [B, T, C, H, W]
    if data.dim() == 5 and data.shape[2] == 3:
        print("Checkpoint contains decoded frames. Converting directly to video.")
        tensors_to_video(data, args.out, args.fps)
        return

    # If not frames, assume it's latents and try to decode
    print("Checkpoint contains latents. Initializing pipeline for decoding...")
    model_configs = [ModelConfig(model_id=args.model_id)]
    pipe = WanVideoPipeline.from_pretrained(
        device=args.device, torch_dtype=torch.float16, model_configs=model_configs
    )
    
    frames_hat = decode_latents_auto(pipe, data)
    if frames_hat is None:
        raise RuntimeError("Failed to decode latents using the pipeline adapter.")
    
    tensors_to_video(frames_hat.to(torch.float32), args.out, args.fps)

if __name__ == "__main__":
    main()