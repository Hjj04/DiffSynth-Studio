# Temporal Module Design (Day 3)

## Motivation

Temporal coherence in DiffSynth requires leveraging information from previous
frames while preserving the fidelity of the current frame. The TemporalModule
acts directly in latent space to warp prior features to the current timestep and
blend them via a learnable gate. The Day 3 milestone elevates the skeleton by
adding a latent-flow predictor, an alpha warm-up scheduler, and a minimal
integration recipe for the training loop.

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
  at runtime to guarantee values in ``(0, 1)``. ``TemporalModule.set_alpha``
  exposes a safe override for schedulers and debugging.
* ``alpha_init`` controls the warm start; recommended range is ``0.2`` to ``0.5``
  to prevent the warped latent from overpowering the current frame early in
  training.
* ``diffsynth.utils.alpha_scheduler.AlphaScheduler`` linearly warms ``alpha``
  from ``alpha_init`` to ``alpha_max`` during the first ``warmup_steps``. After
  warmup the value is clamped to ``alpha_max``.

## Latent Flow Predictor

* ``diffsynth.modules.latent_flow_predictor.LatentFlowPredictor`` provides a
  lightweight CNN that maps ``(z_prev, z_cur)`` to latent pixel offsets. It is
  designed as an optional plug-in for scenarios where external flow (RAFT,
  etc.) is unavailable or too expensive.
* The predictor is differentiable end-to-end and should be optimised jointly
  with the TemporalModule parameters during training.

## Training Loop Integration

* ``examples/wanvideo/model_training/train_with_temporal.py`` demonstrates a
  minimal temporal-aware training loop. It encodes frames to latent space,
  fuses them sequentially without in-place modifications, decodes back to pixel
  space, and applies a collection of temporal consistency losses (latent L2,
  style smoothness, CLIP cosine, and edge consistency).
* ``run_smoke.sh`` launches the script with conservative defaults for a 100–200
  step smoke test. The script defaults to dummy data but can be pointed to a
  real dataset and pipeline once available.
* ``AlphaScheduler`` is invoked after each optimisation step to keep ``alpha``
  on the warm-up trajectory and logs ``temporal/alpha`` into TensorBoard. The
  TemporalModule ``aux`` dict is also surfaced for debugging (e.g. warp mean).

## Future Extensions

* **Multi-scale fusion** – feed latents from multiple resolutions and use
  per-scale alpha gates or cross-scale attention.
* **Warp regularization** – add penalties or edge-aware losses encouraging
  smooth flow fields and consistent warps.
* **Diagnostics** – log ``aux['alpha']`` and visualise ``z_warp`` by decoding it
  through the generator and saving to TensorBoard for qualitative checks.

## Debugging Tips

* Validate tensor devices (CPU vs CUDA) before calling ``grid_sample``.
* Keep flow magnitudes small in smoke tests to avoid sampling outside the
  normalized grid; ``padding_mode='border'`` helps but large offsets still hurt.
* To inspect gating behaviour run with ``alpha`` fixed and compare decoded
  samples of ``z_cur`` vs ``z_warp``.
