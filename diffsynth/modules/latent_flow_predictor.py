# diffsynth/modules/latent_flow_predictor.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class LatentFlowPredictor(nn.Module):
    """
    A lightweight convolutional network to predict optical flow in the latent space.

    This module takes the latent representations of two consecutive frames,
    concatenates them, and passes them through a simple U-Net like structure
    to predict a dense flow field (dx, dy) for each pixel in the latent map.

    The output flow represents pixel offsets and can be directly used by the
    TemporalModule's warping function.
    """

    def __init__(self, in_channels: int, hidden_channels: int = 128):
        """
        Initializes the LatentFlowPredictor.

        Args:
            in_channels (int): The number of channels in the input latent tensors (z_prev, z_cur).
            hidden_channels (int): The number of channels in the intermediate convolutional layers.
        """
        super().__init__()
        self.net = nn.Sequential(
            # Input is concatenated z_prev and z_cur, so in_channels * 2
            nn.Conv2d(in_channels * 2, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # Output 2 channels for dx and dy
            nn.Conv2d(64, 2, kernel_size=3, padding=1)
        )
        # Initialize the final layer's weights and biases to zero
        # to encourage the model to start with an identity warp (zero flow).
        self.net[-1].weight.data.zero_()
        self.net[-1].bias.data.zero_()


    def forward(self, z_prev: torch.Tensor, z_cur: torch.Tensor) -> torch.Tensor:
        """
        Predicts the flow from z_prev to z_cur.

        Args:
            z_prev (torch.Tensor): Latent of the previous frame, shape `[B, C, H, W]`.
            z_cur (torch.Tensor): Latent of the current frame, shape `[B, C, H, W]`.

        Returns:
            torch.Tensor: The predicted flow field as pixel offsets, shape `[B, 2, H, W]`.
        """
        x = torch.cat([z_prev, z_cur], dim=1)
        flow = self.net(x)
        return flow