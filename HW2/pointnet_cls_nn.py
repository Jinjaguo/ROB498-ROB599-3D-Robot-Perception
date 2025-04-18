import torch.nn as nn
import torch
import torch.nn.functional as F


class PointNetClassification(nn.Module):
    def __init__(self, num_classes=6):
        super(PointNetClassification, self).__init__()

        self.mlp1 = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )

        self.mlp2 = nn.Sequential(
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU()
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)  # [B, 3, N]
        x = self.mlp1(x)  # [B, 64, N]
        x = self.mlp2(x)  # [B, 1024, N]
        x = torch.max(x, 2)[0]  # [B, 1024]
        x = self.fc_layers(x)  # [B, num_classes]
        return x  # No softmax
