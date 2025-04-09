import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os

from pointnet_cls_nn import PointNetClassification
from data_load import PointCloudDataset

train_path = './data/cls'
os.makedirs('runs/pointnet_cls', exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = PointNetClassification(num_classes=6).to(device)

train_dataset = PointCloudDataset(train_path + '/train_data.npy',
                                  train_path + '/train_labels.npy',
                                  train=True)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, drop_last=True)

# Class weight balancing
counts = torch.tensor([208942, 58518, 14729889, 4057856, 8441224, 17303571], dtype=torch.float32)
freqs = counts / counts.sum()
weights = 1.0 / freqs
weights = weights / weights.sum()
class_weights = weights.to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=0.001)
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

writer = SummaryWriter(log_dir='runs/pointnet_cls')

EPOCHS = 50

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    total_correct = 0
    total_points = 0

    for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}"):
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)  # [B, num_classes]
        labels = labels

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        preds = outputs.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_points += labels.numel()

    avg_loss = running_loss / len(train_loader)
    acc = total_correct / total_points

    writer.add_scalar('Loss/train', avg_loss, epoch + 1)
    writer.add_scalar('Accuracy/train', acc, epoch + 1)

    print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}, Accuracy: {acc:.4f}")

    # Save checkpoint every 10 epochs
    if (epoch + 1) % 10 == 0:
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss
        }, f'{train_path}/checkpoint_epoch{epoch + 1}.pth')

writer.close()
torch.save(model.state_dict(), train_path + '/pointnet_cls_final.pth')
