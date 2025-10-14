import torch

from diffsynth.modules.latent_flow_predictor import LatentFlowPredictor


def test_latent_flow_predictor_shape_and_grad():
    torch.manual_seed(0)
    predictor = LatentFlowPredictor(in_channels=32, hidden_channels=64)
    predictor.train()

    z_prev = torch.randn(2, 32, 8, 8, requires_grad=True)
    z_cur = torch.randn(2, 32, 8, 8, requires_grad=True)

    flow = predictor(z_prev, z_cur)
    assert flow.shape == (2, 2, 8, 8)

    loss = flow.mean()
    loss.backward()

    assert z_prev.grad is not None
    assert z_cur.grad is not None
    for param in predictor.parameters():
        assert param.grad is not None
