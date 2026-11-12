"""Stage 2: the network — an MLP for regression on log1p(price)."""

import torch
from torch import nn


class MLP(nn.Module):
    """Linear -> [BatchNorm] -> ReLU -> [Dropout] per hidden layer, then one linear unit.

    The output is linear (no activation): the target log1p(price) is unbounded.
    """

    def __init__(self, n_in, hidden=(128, 64), dropout=0.0, batchnorm=False):
        super().__init__()
        layers = []
        width = n_in
        for units in hidden:
            layers.append(nn.Linear(width, units))
            if batchnorm:
                layers.append(nn.BatchNorm1d(units))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            width = units
        layers.append(nn.Linear(width, 1))
        self.net = nn.Sequential(*layers)
        self.apply(self._init)

    @staticmethod
    def _init(module):
        # He initialization: keeps the activation variance stable through ReLU layers.
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            nn.init.zeros_(module.bias)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = MLP(91)
    print(model)
    print("parameters:", count_parameters(model))
    print("output shape:", model(torch.zeros(4, 91)).shape)
