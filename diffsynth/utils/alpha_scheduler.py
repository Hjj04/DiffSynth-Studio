"""Utilities for managing TemporalModule alpha schedules."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch


def _inv_sigmoid(x: float) -> float:
    """Numerically stable logit used to write into ``alpha_param``."""
    x = float(max(min(x, 1.0 - 1e-6), 1e-6))
    return math.log(x / (1.0 - x))


@dataclass
class AlphaScheduleConfig:
    """Configuration for :class:`AlphaScheduler`.

    Attributes:
        warmup_steps: Number of steps over which to linearly ramp ``alpha``.
        alpha_max: Target value once warmup completes.
        alpha_init: Starting value for the warmup. Should be between 0 and 1.
    """

    warmup_steps: int = 5_000
    alpha_max: float = 0.8
    alpha_init: float = 0.2

    def __post_init__(self) -> None:
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        if not 0.0 < self.alpha_max < 1.0:
            raise ValueError("alpha_max must be in (0, 1)")
        if not 0.0 < self.alpha_init < 1.0:
            raise ValueError("alpha_init must be in (0, 1)")


class AlphaScheduler:
    """Linearly warm up the ``alpha`` gate of a :class:`TemporalModule`.

    The scheduler operates out-of-place: each call to :meth:`step` computes the
    desired ``alpha`` for the provided ``global_step`` and writes the logit value
    into ``module.alpha_param`` without disrupting gradient history.
    """

    def __init__(
        self,
        module: torch.nn.Module,
        config: Optional[AlphaScheduleConfig] = None,
        *,
        alpha_attr: str = "alpha_param",
    ) -> None:
        if config is None:
            config = AlphaScheduleConfig()
        self.module = module
        self.config = config
        self.alpha_attr = alpha_attr

        if not hasattr(module, alpha_attr):
            raise AttributeError(f"module is missing attribute '{alpha_attr}'")

        self.set_module_alpha(self.config.alpha_init)

    def compute_alpha(self, step: int) -> float:
        """Return the scheduled alpha for ``step``."""
        if step < 0:
            raise ValueError("step must be non-negative")
        if self.config.warmup_steps == 0:
            return self.config.alpha_max
        progress = min(step / float(self.config.warmup_steps), 1.0)
        delta = self.config.alpha_max - self.config.alpha_init
        return self.config.alpha_init + progress * delta

    def set_module_alpha(self, alpha_val: float) -> None:
        """Write ``alpha_val`` into the module's logit parameter."""
        logit = _inv_sigmoid(alpha_val)
        alpha_param = getattr(self.module, self.alpha_attr)

        if isinstance(alpha_param, torch.nn.Parameter):
            with torch.no_grad():
                alpha_param.data.fill_(logit)
        else:
            # registered buffer (non-trainable)
            alpha_param.copy_(torch.tensor(logit, dtype=alpha_param.dtype, device=alpha_param.device))

    def step(self, step: int) -> float:
        """Advance scheduler to ``step`` and update the module."""
        alpha = self.compute_alpha(step)
        self.set_module_alpha(alpha)
        return alpha

    def get_alpha(self) -> float:
        """Return the current alpha in data space."""
        alpha_param = getattr(self.module, self.alpha_attr)
        return float(torch.sigmoid(alpha_param.detach()).item())
