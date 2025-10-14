import pathlib
import sys

import torch

# Ensure repo root is on sys.path when running tests standalone
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diffsynth.modules.temporal_module import TemporalModule


def test_temporal_module_forward_backward():
    torch.manual_seed(42)
    B, C, H, W = 2, 64, 16, 16
    style_dim = 128

    z_prev = torch.randn(B, C, H, W, requires_grad=True)
    z_cur = torch.randn(B, C, H, W, requires_grad=True)

    s_prev = torch.randn(B, style_dim, requires_grad=True)
    s_cur = torch.randn(B, style_dim, requires_grad=True)

    flow = torch.randn(B, 2, H, W) * 0.5

    module = TemporalModule(latent_channels=C, style_dim=style_dim, learnable_alpha=True, alpha_init=0.3)
    module.train()

    z_fused, s_fused, aux = module(z_prev, z_cur, s_prev=s_prev, s_cur=s_cur, flow=flow)

    assert z_fused.shape == (B, C, H, W)
    assert s_fused is not None and s_fused.shape == (B, style_dim)
    assert isinstance(aux, dict)
    assert "alpha" in aux

    loss = z_fused.mean() + s_fused.mean()
    loss.backward()

    assert z_prev.grad is not None
    assert z_cur.grad is not None
    assert s_prev.grad is not None
    assert s_cur.grad is not None

    if isinstance(module.alpha_param, torch.nn.Parameter):
        assert module.alpha_param.grad is not None


if __name__ == "__main__":
    test_temporal_module_forward_backward()
