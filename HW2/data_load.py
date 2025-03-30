import numpy as np
from torch.utils.data import Dataset


class PointCloudDataset(Dataset):
    def __init__(self, data_path, labels_path):
        self.data = np.load(data_path).astype(np.float32)  # [B, N, 3]
        self.labels = np.load(labels_path).astype(np.int64)  # [B, N]

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
