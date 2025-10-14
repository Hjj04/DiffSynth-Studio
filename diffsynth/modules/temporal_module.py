import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalModule(nn.Module):
    """TemporalModule skeleton for latent-space warp & fusion.

    This module is intentionally lightweight so that future work can plug in
    more sophisticated warp predictors and gating strategies. The design covers
    three main responsibilities:

    1. Warp the previous latent representation to the current frame using an
       externally provided flow field (pixel offsets) via ``grid_sample``.
    2. Optionally refine the warped latent with a small convolutional block that
       can later be swapped for a residual stack or attention layer.
    3. Fuse the warped latent with the current latent through a learnable alpha
       gate (scalar for now, but expandable to channel/spatial gates).

    Args:
        latent_channels: Number of channels in the latent tensor ``z``.
        style_dim: Optional style embedding dimensionality. When provided, style
            embeddings are fused via a lightweight MLP.
        learnable_alpha: Whether ``alpha`` is a learnable parameter. If
            ``False`` the gate becomes fixed but still differentiable.
        alpha_init: Initial value for the alpha gate, expressed in the data
            domain (0, 1). Internally converted to a logit for stable training.
    """

    def __init__(
        self,
        latent_channels: int,
        style_dim: Optional[int] = None,
        learnable_alpha: bool = True,
        alpha_init: float = 0.5,
    ) -> None:
        super().__init__()
        self.latent_channels = latent_channels
        self.style_dim = style_dim

        init_logit = self._inv_sigmoid(alpha_init)
        if learnable_alpha:
            self.alpha_param = nn.Parameter(torch.tensor(init_logit, dtype=torch.float32))
        else:
            self.register_buffer("alpha_param", torch.tensor(init_logit, dtype=torch.float32))

        self.refiner = nn.Sequential(
            nn.Conv2d(latent_channels, latent_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(latent_channels, latent_channels, kernel_size=3, padding=1),
        )

        if style_dim is not None:
            self.style_fuse = nn.Sequential(
                nn.Linear(style_dim * 2, style_dim),
                nn.ReLU(inplace=True),
                nn.Linear(style_dim, style_dim),
            )
        else:
            self.style_fuse = None

    @staticmethod
    def _inv_sigmoid(x: float) -> float:
        x = float(max(min(x, 1.0 - 1e-6), 1e-6))
        return math.log(x / (1.0 - x))

    def _get_alpha(self) -> torch.Tensor:
        return torch.sigmoid(self.alpha_param)

    def warp_latent(self, z_prev: torch.Tensor, flow: Optional[torch.Tensor]) -> torch.Tensor:
        """Warp ``z_prev`` according to ``flow`` using ``grid_sample``.

        Args:
            z_prev: Tensor of shape ``[B, C, H, W]``.
            flow: Tensor of shape ``[B, 2, H, W]`` describing pixel offsets in
                the order ``(dx, dy)``. ``None`` falls back to identity warp.

        Returns:
            Tensor of the same shape as ``z_prev``.
        """

        if flow is None:
            return z_prev

        if z_prev.dim() != 4:
            raise ValueError("z_prev must be 4D [B, C, H, W]")

        B, _, H, W = z_prev.shape
        device = z_prev.device
        dtype = z_prev.dtype

        flow = flow.to(device=device, dtype=dtype)

        grid_y = torch.linspace(-1.0, 1.0, H, device=device, dtype=dtype)
        grid_x = torch.linspace(-1.0, 1.0, W, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(grid_y, grid_x, indexing="ij")
        base_grid = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0)
        base_grid = base_grid.expand(B, -1, -1, -1)

        dx = flow[:, 0:1, :, :]
        dy = flow[:, 1:2, :, :]
        dx_norm = dx / (W / 2.0)
        dy_norm = dy / (H / 2.0)
        offset = torch.cat((dx_norm, dy_norm), dim=1).permute(0, 2, 3, 1)

        sample_grid = base_grid + offset
        z_warp = F.grid_sample(
            z_prev,
            sample_grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        return z_warp

    def forward(
        self,
        z_prev: torch.Tensor,
        z_cur: torch.Tensor,
        s_prev: Optional[torch.Tensor] = None,
        s_cur: Optional[torch.Tensor] = None,
        flow: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, float]]:
        """Fuse current and warped previous latents with optional style fusion.

        Args:
            z_prev: Tensor ``[B, C, H, W]`` from the previous frame.
            z_cur: Tensor ``[B, C, H, W]`` for the current frame.
            s_prev: Optional style embedding ``[B, S]`` from the previous frame.
            s_cur: Optional style embedding ``[B, S]`` for the current frame.
            flow: Optional flow tensor ``[B, 2, H, W]`` expressed as pixel
                offsets. ``None`` falls back to identity warp.

        Returns:
            z_fused: Tensor ``[B, C, H, W]`` after alpha gating fusion.
            s_fused: Tensor ``[B, S]`` if style inputs are provided, otherwise
                ``None``.
            aux: Dictionary containing diagnostics such as the alpha value and
                warped latent statistics.
        """

        if z_prev.shape != z_cur.shape:
            raise ValueError("z_prev and z_cur must have the same shape")

        device = z_cur.device
        dtype = z_cur.dtype

        if flow is not None:
            flow = flow.to(device=device, dtype=dtype)

        z_warp = self.warp_latent(z_prev, flow)
        z_warp_refined = self.refiner(z_warp)

        alpha = self._get_alpha()
        alpha_exp = alpha.view(1, 1, 1, 1)

        z_fused = alpha_exp * z_warp_refined + (1.0 - alpha_exp) * z_cur

        s_fused = None
        if self.style_fuse is not None and s_prev is not None and s_cur is not None:
            s_prev = s_prev.to(device=device, dtype=dtype)
            s_cur = s_cur.to(device=device, dtype=dtype)
            s_cat = torch.cat([s_prev, s_cur], dim=-1)
            s_fused = self.style_fuse(s_cat)

        aux = {
            "alpha": float(alpha.detach().cpu().item()),
            "z_warp_mean": float(z_warp_refined.mean().detach().cpu().item()),
        }
        return z_fused, s_fused, aux
