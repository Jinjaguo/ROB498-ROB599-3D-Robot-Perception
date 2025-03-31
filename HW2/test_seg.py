from pointnet_seg_nn import *
from data_load import PointCloudDataset
import torch
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_path = './data/seg/pointnet_seg.pth'
test_path = './data/seg'

model = PointNetSegmentation(num_classes=6).to(device)
model.load_state_dict(torch.load(model_path))
model.eval()

test_dataset = PointCloudDataset(test_path + '/test_data.npy', test_path + '/test_labels.npy')
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, drop_last=False)

correct = 0
total = 0

with torch.no_grad():
    for data, labels in test_loader:
        data, labels = data.to(device), labels.to(device)  # [B, N, 3], [B]
        outputs = model(data)  # [B, 6]
        predicted = torch.argmax(outputs, dim=1)  # [B]

        correct += (predicted == labels).sum().item()
        total += labels.size(0)

accuracy = correct / total
print(f'Test Accuracy: {accuracy * 100:.2f}%')