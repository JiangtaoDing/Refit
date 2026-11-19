"""MLP backbone used by ReFIT (unchanged model architecture)."""
import torch.nn as nn


class MLPModel(nn.Module):
    def __init__(self, n_features=35, n_classes=15, hidden_dims=None, dropout=0.2):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128]

        layers = []
        in_dim = n_features
        for h in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_dim = h
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(in_dim, n_classes)

    def forward(self, x):
        feats = self.backbone(x)
        out = self.head(feats)
        return out, [feats]
