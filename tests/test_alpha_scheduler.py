import torch
import pytest

from diffsynth.modules.temporal_module import TemporalModule
from diffsynth.utils.alpha_scheduler import AlphaScheduleConfig, AlphaScheduler


def test_alpha_scheduler_warmup_progression():
    module = TemporalModule(latent_channels=16, learnable_alpha=True, alpha_init=0.1)
    sched = AlphaScheduler(module, AlphaScheduleConfig(warmup_steps=10, alpha_init=0.1, alpha_max=0.7))

    alphas = []
    for step in range(0, 15, 5):
        alpha = sched.step(step)
        alphas.append(alpha)
        module_alpha = module.alpha
        assert abs(module_alpha - alpha) < 1e-5

    assert alphas[0] == pytest.approx(0.1)
    assert alphas[-1] == pytest.approx(0.7)


def test_alpha_scheduler_rejects_missing_attr():
    class Dummy(torch.nn.Module):
        pass

    with pytest.raises(AttributeError):
        AlphaScheduler(Dummy())
