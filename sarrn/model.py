"""sarrn.model — ResMLP used by SARRN (anchor + relative residual)."""
import torch
import torch.nn as nn


class ResidualMLP(nn.Module):
    """Same as train_and_plot_convergence.py (v36)."""
    def __init__(self, in_dim=210, width=512, depth=4, dropout=0.2, rel=False):
        super().__init__()
        self.rel = rel
        layers, d = [], in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.LayerNorm(width), nn.SiLU(inplace=True)]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = width
        layers.append(nn.Linear(d, 8))
        self.net = nn.Sequential(*layers)
        if rel:  # start exactly at the anchor
            nn.init.zeros_(layers[-1].weight); nn.init.zeros_(layers[-1].bias)

    def forward(self, x):
        a = x[:, -8:]
        d = self.net(x)
        if self.rel:                              # residual relative to the anchor norm
            d = d * torch.linalg.vector_norm(a, dim=1, keepdim=True)
        return a + d                              # anchor + residual
