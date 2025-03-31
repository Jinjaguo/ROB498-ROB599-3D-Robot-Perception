import torch
import torch.nn as nn


class PointNetSegmentation(nn.Module):
    def __init__(self, num_classes=6):
        super(PointNetSegmentation, self).__init__()

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

        # segmentation 网络 MLP (512,256) -> (128, num_classes)
        self.seg_mlp1 = nn.Sequential(
            nn.Conv1d(1088, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )

        self.seg_mlp2 = nn.Sequential(
            nn.Conv1d(256, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, num_classes, 1)
        )

    def forward(self, x):
        # 输入维度: [B, N, 3]
        x = x.permute(0, 2, 1)  # -> [B, 3, N]

        local_features = self.mlp1(x)  # -> [B, 64, N]
        x = self.mlp2(local_features)  # -> [B, 1024, N]

        # max pool 得到全局特征 [B, 1024]
        global_feature = torch.max(x, dim=2, keepdim=True)[0]  # [B, 1024, 1]
        global_feature_expanded = global_feature.repeat(1, 1, local_features.size(2))  # [B, 1024, N]

        # 拼接 local + global => [B, 1088, N]
        seg_input = torch.cat([local_features, global_feature_expanded], dim=1)

        # 分割分支
        x = self.seg_mlp1(seg_input)  # -> [B, 256, N]
        x = self.seg_mlp2(x)  # -> [B, num_classes, N]
        x = x.permute(0, 2, 1)  # -> [B, N, num_classes]

        return x  # 每个点的分类分数（logits）

