from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv2d(nn.Module):
    """
    2D spectral convolution using low Fourier modes from both
    the positive and negative rows of the first frequency axis.
    """
    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes

        scale = 0.02
        self.weight1 = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes, modes, dtype=torch.cfloat)
        )
        self.weight2 = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes, modes, dtype=torch.cfloat)
        )

    def compl_mul2d(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        # x: (B, in_channels, m1, m2)
        # weight: (in_channels, out_channels, m1, m2)
        # out: (B, out_channels, m1, m2)
        return torch.einsum("bcij,coij->boij", x, weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        x_ft = torch.fft.rfft2(x)  # (B, C, H, W//2 + 1)

        out_ft = torch.zeros(
            B, self.out_channels, H, W // 2 + 1,
            dtype=torch.cfloat, device=x.device
        )

        m1 = min(self.modes, H)
        m2 = min(self.modes, W // 2 + 1)

        # positive low frequencies
        out_ft[:, :, :m1, :m2] = self.compl_mul2d(
            x_ft[:, :, :m1, :m2],
            self.weight1[:, :, :m1, :m2]
        )

        # negative low frequencies in the first spatial frequency dimension
        out_ft[:, :, -m1:, :m2] = self.compl_mul2d(
            x_ft[:, :, -m1:, :m2],
            self.weight2[:, :, :m1, :m2]
        )

        x = torch.fft.irfft2(out_ft, s=(H, W))
        return x


class FNO2d(nn.Module):
    """
    FNO for mapping a(x,y) -> u(x,y) on a regular grid.
    We concatenate coordinate channels to the input.
    """
    def __init__(self, width: int = 32, modes: int = 12, layers: int = 4):
        super().__init__()
        self.width = width
        self.modes = modes
        self.layers = layers

        # input channels: a(x,y), x-coordinate, y-coordinate
        self.lift = nn.Conv2d(3, width, kernel_size=1)
        self.spectral = nn.ModuleList([SpectralConv2d(width, width, modes) for _ in range(layers)])
        self.pointwise = nn.ModuleList([nn.Conv2d(width, width, kernel_size=1) for _ in range(layers)])
        self.proj1 = nn.Conv2d(width, width, kernel_size=1)
        self.proj2 = nn.Conv2d(width, 1, kernel_size=1)

    def get_grid(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape
        gridx = torch.linspace(0, 1, H, device=x.device).view(1, 1, H, 1).expand(B, 1, H, W)
        gridy = torch.linspace(0, 1, W, device=x.device).view(1, 1, 1, W).expand(B, 1, H, W)
        return torch.cat([gridx, gridy], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        grid = self.get_grid(x)
        x = torch.cat([x, grid], dim=1)   # (B,3,H,W)

        x = self.lift(x)
        for k in range(self.layers):
            x1 = self.spectral[k](x)
            x2 = self.pointwise[k](x)
            x = F.gelu(x1 + x2)

        x = F.gelu(self.proj1(x))
        x = self.proj2(x)
        return x