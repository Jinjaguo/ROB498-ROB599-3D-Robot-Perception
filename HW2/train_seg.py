from pointnet_seg_nn import *
from data_load import PointCloudDataset
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
import numpy as np

train_path = './data/seg'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = PointNetSegmentation(num_classes=6).to(device)
train_dataset = PointCloudDataset(train_path + '/train_data.npy', train_path + '/train_labels.npy')
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, drop_last=True)


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
        outputs = outputs.reshape(-1, 6)
        labels = labels.reshape(-1)

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

    # Save checkpoint every 10 epochs
    if (epoch + 1) % 10 == 0:
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss
        }, f'{train_path}/checkpoint_epoch{epoch + 1}.pth')

torch.save(model.state_dict(), train_path + '/pointnet_seg.pth')
