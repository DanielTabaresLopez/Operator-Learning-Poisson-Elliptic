from __future__ import annotations
import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, width: int, depth: int, act: str = "gelu"):
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")

        if act == "relu":
            activation = nn.ReLU()
        elif act == "tanh":
            activation = nn.Tanh()
        else:
            activation = nn.GELU()

        layers = []
        dims = [in_dim] + [width] * (depth - 1) + [out_dim]
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(activation)
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)