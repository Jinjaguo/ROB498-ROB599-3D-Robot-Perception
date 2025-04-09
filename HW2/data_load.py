import numpy as np
from torch.utils.data import Dataset


def random_rotate_point_cloud(pc):
    theta = np.random.uniform(0, 2 * np.pi)
    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ], dtype=np.float32)
    return pc @ rotation_matrix.T


def jitter_point_cloud(pc, sigma=0.01, clip=0.05):
    jitter = np.clip(sigma * np.random.randn(*pc.shape), -clip, clip)
    return pc + jitter


class PointCloudDataset(Dataset):
    def __init__(self, data_path, labels_path, train=True):
        self.data = np.load(data_path).astype(np.float32) # [B, N, 3]
        self.labels = np.load(labels_path).astype(np.int64)  # [B, N]
        self.train = train

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        pc = self.data[idx]
        label = self.labels[idx]

        if self.train:
            pc = random_rotate_point_cloud(pc)
            pc = jitter_point_cloud(pc)

        return pc.astype(np.float32), label


