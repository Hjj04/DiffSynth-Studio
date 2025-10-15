# diffsynth/utils/pipeline_adapter.py
"""
Auto-adapter to find encode/decode helpers in Wan/DiffSynth pipelines.

It tries multiple common API names and returns the first callable that works.
If an uninitialized WanVideoPipeline is detected, it will attempt to
auto-load the default models to ensure functionality.
"""
import torch
import torch.nn as nn
import torchvision
from typing import Optional, Callable, List, Tuple
import warnings

# --- Lazy Imports for Auto-Initialization ---
_WanVideoPipeline = None
_ModelConfig = None

_ENCODE_CANDIDATES = [
    "encode_frames", "encode", "encode_latents", "encode_batch",
    "vae.encode", "vae.encode_frames", "model.encode"
]
_DECODE_CANDIDATES = [
    "decode_latents", "decode", "decode_frames", "decode_batch",
    "vae.decode", "vae.decode_latents", "model.decode", "to_image"
]

def _try_call(obj: object, name: str, *args, **kwargs) -> Optional[torch.Tensor]:
    """Tries to call a method by its name string on an object."""
    try:
        # Handle dotted paths like 'vae.encode'
        parts = name.split('.')
        func_obj = obj
        for part in parts:
            if not hasattr(func_obj, part):
                return None
            func_obj = getattr(func_obj, part)
            
        if callable(func_obj):
            return func_obj(*args, **kwargs)
    except Exception as e:
        warnings.warn(f"Adapter tried '{name}' but failed with: {e}")
    return None

def _ensure_pipe_models_loaded(pipe):
    """
    Internal helper. If pipe is an uninitialized WanVideoPipeline,
    it attempts to load the default models and returns a new, valid pipe.
    """
    global _WanVideoPipeline, _ModelConfig
    # Check if it looks like a WanVideoPipeline and is uninitialized
    if pipe.__class__.__name__ == 'WanVideoPipeline' and getattr(pipe, "vae", None) is None:
        warnings.warn(
            "Detected uninitialized WanVideoPipeline (pipe.vae is None). "
            "Adapter will attempt to auto-load default 'Wan-AI/Wan2.1-T2V-1.3B' models. "
            "For other models, please initialize the pipeline with correct `model_configs` first."
        )
        try:
            # Lazy import to avoid circular dependencies
            if _WanVideoPipeline is None:
                from diffsynth.pipelines.wan_video_new import WanVideoPipeline as _WanVideoPipeline
            if _ModelConfig is None:
                from diffsynth.utils import ModelConfig as _ModelConfig

            model_id = "Wan-AI/Wan2.1-T2V-1.3B"
            model_configs = [
                _ModelConfig(model_id=model_id, origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth"),
                _ModelConfig(model_id=model_id, origin_file_pattern="diffusion_pytorch_model*.safensors"),
                _ModelConfig(model_id=model_id, origin_file_pattern="Wan2.1_VAE.pth"),
            ]
            
            new_pipe = _WanVideoPipeline.from_pretrained(
                device=pipe.device, 
                torch_dtype=pipe.torch_dtype,
                model_configs=model_configs
            )
            return new_pipe
        except Exception as e:
            warnings.warn(f"Adapter auto-initialization failed: {e}")
            return pipe # Return original pipe on failure
    return pipe # Return original pipe if already initialized or not a WanVideoPipeline

def encode_frames_auto(pipe, frames: torch.Tensor) -> Optional[torch.Tensor]:
    """
    Try multiple likely encode function names on `pipe` to convert frames->[B,T,C_lat,H',W'].
    Frames are expected in shape: [B, T, C, H, W], values 0..1.
    Returns latents tensor or None if no method is found or succeeds.
    """
    pipe = _ensure_pipe_models_loaded(pipe)

    # Special handling for WanVideoPipeline's VAE which expects a list of videos
    if pipe.__class__.__name__ == 'WanVideoPipeline':
        vae = getattr(pipe, "vae", None)
        if vae and hasattr(vae, "encode"):
            try:
                B, T, C, H, W = frames.shape
                # VAE expects [C, T, H, W], so we create a list of such tensors
                video_list = [frames[b].permute(1, 0, 2, 3) for b in range(B)]
                # Pass device explicitly as it's a required argument
                latents = vae.encode(video_list, tiled=False, device=pipe.device)
                if isinstance(latents, torch.Tensor):
                    # Convert latents from [B, C, T, H', W'] back to [B, T, C, H', W']
                    return latents.permute(0, 2, 1, 3, 4).contiguous()
            except Exception as e:
                warnings.warn(f"Special handling for WanVideoPipeline VAE encode failed: {e}")
    
    # Fallback to general candidate search
    for name in _ENCODE_CANDIDATES:
        out = _try_call(pipe, name, frames)
        if isinstance(out, torch.Tensor):
            # Ensure output is [B, T, C, H, W]
            if out.dim() == 5 and out.shape[2] != frames.shape[2]: # A common mistake is C and T swapped
                 return out.permute(0, 2, 1, 3, 4).contiguous()
            return out
            
    return None

def decode_latents_auto(pipe, latents: torch.Tensor) -> Optional[torch.Tensor]:
    """
    Try to decode latents [B,T,C,H',W'] -> frames [B,T,C,H,W] (pixel 0..1).
    """
    pipe = _ensure_pipe_models_loaded(pipe)

    # Special handling for WanVideoPipeline's VAE
    if pipe.__class__.__name__ == 'WanVideoPipeline':
        vae = getattr(pipe, "vae", None)
        if vae and hasattr(vae, "decode"):
            try:
                # VAE expects latents in [B, C, T, H', W'] format
                latents_c_first = latents.permute(0, 2, 1, 3, 4).contiguous()
                video_frames = vae.decode(latents_c_first, tiled=False, device=pipe.device)
                if isinstance(video_frames, torch.Tensor):
                    # Convert decoded frames from [B, C, T, H, W] back to [B, T, C, H, W]
                    if video_frames.dim() == 5 and video_frames.shape[1] == 3:
                        return video_frames.permute(0, 2, 1, 3, 4).contiguous()
                    return video_frames
            except Exception as e:
                warnings.warn(f"Special handling for WanVideoPipeline VAE decode failed: {e}")

    # Fallback to general candidate search
    # Note: Most decoders expect channel-first latents
    latents_c_first = latents.permute(0, 2, 1, 3, 4).contiguous()
    for name in _DECODE_CANDIDATES:
        out = _try_call(pipe, name, latents_c_first)
        if isinstance(out, torch.Tensor):
            # Ensure output is [B, T, C, H, W]
            if out.dim() == 5 and out.shape[1] == 3: # Check if channels are in the second dimension
                return out.permute(0, 2, 1, 3, 4).contiguous()
            return out
            
    return None