from pointnet_cls_nn import *
from data_load import PointCloudDataset
import torch
from torch.utils.data import DataLoader


# 设备设置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = './data/cls/pointnet_cls_final.pth'
test_path = './data/cls/'

# 加载模型
model = PointNetClassification(num_classes=6).to(device)
model.load_state_dict(torch.load(model_path))
model.eval()

# 加载测试数据
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
