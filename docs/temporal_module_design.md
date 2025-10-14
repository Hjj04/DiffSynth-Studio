# Temporal Module Design (Skeleton)

## Motivation

Temporal coherence in DiffSynth requires leveraging information from previous
frames while preserving the fidelity of the current frame. The TemporalModule
acts directly in latent space to warp prior features to the current timestep and
blend them via a learnable gate. This document captures the Day 2 skeleton so we
can ship warp/gating refinements on Day 3.

## Forward API

```python
from diffsynth.modules.temporal_module import TemporalModule

module = TemporalModule(latent_channels=C, style_dim=S)
z_fused, s_fused, aux = module(z_prev, z_cur, s_prev=None, s_cur=None, flow=None)
```

```python
def forward(self, z_prev, z_cur, s_prev=None, s_cur=None, flow=None):
    """
    Args:
      z_prev: Tensor, shape [B, C, H, W] (previous frame latent)
      z_cur:  Tensor, shape [B, C, H, W] (current frame latent)
      s_prev: Tensor or None, [B, Sdim] (previous style embedding)
      s_cur:  Tensor or None, [B, Sdim] (current style embedding)
      flow:   Tensor or None, [B, 2, H, W] in pixel offsets (dx, dy)
    Returns:
      z_fused: Tensor [B, C, H, W]
      s_fused: Tensor [B, Sdim] or None
      aux: dict (e.g., {'alpha': alpha, 'z_warp': z_warp_mean})
    """
```

## Flow Convention

* ``flow`` is expressed as pixel offsets ``(dx, dy)`` matching RAFT-style optical
  flow outputs.
* ``TemporalModule.warp_latent`` converts these offsets to normalized grid
  coordinates before calling ``torch.nn.functional.grid_sample`` with
  ``align_corners=True`` and ``padding_mode='border'``.
* ``flow=None`` falls back to an identity warp, i.e. ``z_prev`` is used as-is.

## Alpha Gate Strategy

* ``alpha`` is stored in logit form (`alpha_param`) and passed through a sigmoid
  at runtime to guarantee values in ``(0, 1)``.
* ``alpha_init`` controls the warm start; recommended range is ``0.2`` to ``0.5``
  to prevent the warped latent from overpowering the current frame early in
  training.
* For production runs we plan to implement a warmup schedule that linearly ramps
  ``alpha`` toward ``0.8`` during the first ~5k steps. The skeleton keeps it
  constant but exposes ``learnable_alpha`` for experimentation.

## Future Extensions

* **Flow prediction** – integrate a small CNN/Transformer head to predict flow
  jointly with the latent update when external flow is unavailable.
* **Multi-scale fusion** – feed latents from multiple resolutions and use
  per-scale alpha gates or cross-scale attention.
* **Warp regularization** – add penalties or edge-aware losses encouraging
  smooth flow fields and consistent warps.
* **Diagnostics** – log ``aux['alpha']`` and visualize ``z_warp`` by decoding it
  through the generator and saving to TensorBoard for qualitative checks.

## Debugging Tips

* Validate tensor devices (CPU vs CUDA) before calling ``grid_sample``.
* Keep flow magnitudes small in smoke tests to avoid sampling outside the
  normalized grid; ``padding_mode='border'`` helps but large offsets still hurt.
* To inspect gating behaviour run with ``alpha`` fixed and compare decoded
  samples of ``z_cur`` vs ``z_warp``.
