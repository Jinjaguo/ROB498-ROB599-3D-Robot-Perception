from pointnet_seg_nn import *
from data_load import PointCloudDataset
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
import numpy as np

train_path = './data/seg'
data = np.load('./data/seg/train_data.npy', allow_pickle=True)
print("type:", type(data))
try:
    print("shape:", data.shape)
except Exception as e:
    print("can't access shape:", e)

if isinstance(data, np.ndarray):
    print("dtype:", data.dtype)
    if data.dtype == object:
        print("Looks like it's a list of arrays, not a regular ndarray.")


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = PointNetSegmentation(num_classes=6).to(device)
train_dataset = PointCloudDataset(train_path + '/train_data.npy', train_path + '/train_labels.npy')
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, drop_last=True)


# from collections import Counter
#
# label_counter = Counter()
#
# for _, labels in train_loader:
#     labels = labels.view(-1).tolist()
#     label_counter.update(labels)
#
# print("全训练集类别计数:")
# for cls, count in sorted(label_counter.items()):
#     print(f"Class {cls}: {count}")

counts = torch.tensor([208942, 58518, 14729889, 4057856, 8441224, 17303571], dtype=torch.float32)
freqs = counts / counts.sum()
weights = 1.0 / freqs
weights = weights / weights.sum()  # 归一化
class_weights = weights.to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=0.001)

EPOCHS = 50

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    total_correct = 0
    total_points = 0

    for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}"):
        inputs, labels = batch  # inputs: [B, N, 3], labels: [B, N]
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)  # outputs: [B, N, 6]
        outputs = outputs.view(-1, 6)
        labels = labels.view(-1)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        preds = outputs.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_points += labels.numel()

    avg_loss = running_loss / len(train_loader)
    acc = total_correct / total_points
    print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}, Accuracy: {acc:.4f}")

torch.save(model.state_dict(), train_path + '/pointnet_seg.pth')
