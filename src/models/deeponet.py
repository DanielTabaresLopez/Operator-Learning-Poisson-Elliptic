from __future__ import annotations
import torch
import torch.nn as nn
from src.models.mlp import MLP


class DeepONet(nn.Module):
    """
    DeepONet for scalar output u(x,y).

    Inputs:
      - branch_input: (B, n_sensors)   values of a(x) at sensors
      - coords:       (Nq, 2)          query coordinates (x,y) in [0,1]^2

    Output:
      - u_pred:       (B, Nq)
    """

    def __init__(self, n_sensors: int, p: int, branch_width: int, trunk_width: int, depth: int):
        super().__init__()
        self.p = p
        self.branch = MLP(in_dim=n_sensors, out_dim=p, width=branch_width, depth=depth, act="gelu")
        self.trunk = MLP(in_dim=2, out_dim=p, width=trunk_width, depth=depth, act="gelu")

    def forward(self, branch_input: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        # branch_input: (B, n_sensors)
        # coords: (Nq, 2)
        beta = self.branch(branch_input)        # (B, p)
        tau  = self.trunk(coords)               # (Nq, p)
        # broadcast and inner product over p
        u_pred = (beta[:, None, :] * tau[None, :, :]).sum(dim=-1)  # (B, Nq)
        return u_pred