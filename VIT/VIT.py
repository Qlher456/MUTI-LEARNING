import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from transformers import ViTForImageClassification
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast  # 自动混合精度

# 数据路径
data_dir = "AIGC"
real_dir = os.path.join(data_dir, "Real")
fake_dir = os.path.join(data_dir, "Fake")

# 自定义数据集类
class CustomDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.images = []
        self.labels = []
        for label, folder in enumerate(["Real", "Fake"]):
            folder_path = os.path.join(data_dir, folder)
            for img_name in os.listdir(folder_path):
                self.images.append(os.path.join(folder_path, img_name))
                self.labels.append(label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        image = plt.imread(img_path)
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)

# 数据预处理
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# 数据加载器，设置多进程和固定内存
train_dataset = CustomDataset(data_dir, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)

# 加载ViT模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ViTForImageClassification.from_pretrained("./vit-base-patch16-224-in21k", num_labels=2).to(device)

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-6)
scaler = GradScaler()  # 自动混合精度

# 训练模型
epochs = 100
train_loss, train_acc = [], []
accumulation_steps = 2  # 梯度累积步数

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        with autocast():  # 自动混合精度训练
            outputs = model(images).logits
            loss = criterion(outputs, labels)
            loss = loss / accumulation_steps  # 梯度累积

        scaler.scale(loss).backward()

        if (i + 1) % accumulation_steps == 0:  # 每 accumulation_steps 更新一次梯度
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        running_loss += loss.item() * accumulation_steps  # 累积梯度还原为原始 loss
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = correct / total
    train_loss.append(epoch_loss)
    train_acc.append(epoch_acc)
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

# 绘制训练曲线
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(train_acc, label="Train Accuracy")
plt.title("Train Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_loss, label="Train Loss")
plt.title("Train Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.savefig("VIT_training_curves.png")
torch.save(model.state_dict(), "vit_model_pytorch.pth")
