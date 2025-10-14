"""Latent flow predictor module."""
from __future__ import annotations

import torch
import torch.nn as nn


class LatentFlowPredictor(nn.Module):
    """Lightweight latent-space flow predictor.

    The predictor consumes the previous and current latent feature maps,
    concatenates them channel-wise, and regresses a dense 2D flow field in the
    latent resolution. The predicted flow follows the TemporalModule convention
    of pixel offsets (``dx``, ``dy``) measured in latent-space pixels.
    """

    def __init__(self, in_channels: int, hidden_channels: int = 128) -> None:
        super().__init__()
        if in_channels <= 0:
            raise ValueError("in_channels must be positive")
        if hidden_channels <= 0:
            raise ValueError("hidden_channels must be positive")

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels

        mid_channels = max(1, hidden_channels // 2)

        self.net = nn.Sequential(
            nn.Conv2d(in_channels * 2, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, mid_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, 2, kernel_size=3, padding=1),
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialise prediction head for stable training."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, z_prev: torch.Tensor, z_cur: torch.Tensor) -> torch.Tensor:
        """Predict latent flow between two frames.

        Args:
            z_prev: Tensor ``[B, C, H, W]`` representing the previous latent.
            z_cur: Tensor ``[B, C, H, W]`` representing the current latent.

        Returns:
            Tensor ``[B, 2, H, W]`` with pixel offsets ``(dx, dy)``.
        """
        if z_prev.shape != z_cur.shape:
            raise ValueError("z_prev and z_cur must have identical shapes")
        if z_prev.dim() != 4:
            raise ValueError("Expected 4D latents [B, C, H, W]")

        x = torch.cat([z_prev, z_cur], dim=1)
        flow = self.net(x)
        return flow
