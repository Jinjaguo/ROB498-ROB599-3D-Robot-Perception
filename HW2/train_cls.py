from pointnet_cls_nn import *
from data_load import PointCloudDataset
import torch.optim as optim
from tqdm import tqdm

train_path = './data/cls'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = PointNetClassification(num_classes=6).to(device)
train_dataset = PointCloudDataset(train_path + '/train_data.npy', train_path + '/train_labels.npy')
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, drop_last=True)

criterion = nn.CrossEntropyLoss()
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

torch.save(model.state_dict(), train_path + '/pointnet_cls.pth')
