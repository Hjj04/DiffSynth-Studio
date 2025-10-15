# diffsynth/utils/alpha_scheduler.py
import math
import torch
import torch.nn as nn

def _inv_sigmoid(x: float) -> float:
    """Numerically stable inverse sigmoid (logit) function."""
    x = float(max(min(x, 1.0 - 1e-6), 1e-6))
    return math.log(x / (1.0 - x))

class AlphaScheduler:
    """
    Manages the alpha gating parameter for a TemporalModule during training.

    This helper class implements a linear warmup schedule for alpha. It directly
    modifies the `alpha_param` (which is in logit space) of a given
    TemporalModule instance at each training step.

    Usage:
        # In your training script setup
        temporal_module = TemporalModule(...)
        alpha_scheduler = AlphaScheduler(temporal_module, warmup_steps=5000, alpha_max=0.8)

        # In your training loop
        for step, batch in enumerate(dataloader):
            current_alpha = alpha_scheduler.step(step)
            # ... your training logic ...
            # Log current_alpha to TensorBoard
    """

    def __init__(self,
                 module: nn.Module,
                 warmup_steps: int = 5000,
                 alpha_max: float = 0.8,
                 alpha_init: float = 0.2):
        """
        Initializes the AlphaScheduler.

        Args:
            module (nn.Module): The TemporalModule instance whose alpha will be controlled.
            warmup_steps (int): The number of steps for the linear warmup.
            alpha_max (float): The target alpha value to reach after warmup.
            alpha_init (float): The initial alpha value at step 0.
        """
        if not hasattr(module, 'alpha_param'):
            raise ValueError("The provided module does not have an 'alpha_param' attribute.")

        self.module = module
        self.warmup_steps = int(warmup_steps)
        self.alpha_max = float(alpha_max)
        self.alpha_init = float(alpha_init)
        
        # Set the initial alpha value in the module at the beginning.
        self._set_module_alpha_from_value(self.alpha_init)

    def _compute_alpha_at_step(self, step: int) -> float:
        """Computes the target alpha value for a given training step."""
        if step >= self.warmup_steps:
            return self.alpha_max
        
        # Linear interpolation from alpha_init to alpha_max over warmup_steps
        progress = step / max(1, self.warmup_steps)
        current_alpha = self.alpha_init + (self.alpha_max - self.alpha_init) * progress
        return current_alpha

    def _set_module_alpha_from_value(self, alpha_value: float):
        """Converts an alpha value to logit and sets it on the module's parameter."""
        logit = _inv_sigmoid(alpha_value)
        
        param = self.module.alpha_param
        if isinstance(param, nn.Parameter):
            # Use torch.no_grad() to modify the parameter data without affecting gradients
            with torch.no_grad():
                param.data.fill_(logit)
        elif isinstance(param, torch.Tensor):
            # Handle the case where alpha_param is a non-trainable buffer
            param.fill_(logit)

    def step(self, current_step: int) -> float:
        """
        Calculates the new alpha based on the current step and updates the module.

        Args:
            current_step (int): The current global training step.

        Returns:
            float: The newly computed alpha value for logging purposes.
        """
        target_alpha = self._compute_alpha_at_step(current_step)
        self._set_module_alpha_from_value(target_alpha)
        return target_alpha