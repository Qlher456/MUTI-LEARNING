import os
import random
import matplotlib.pyplot as plt
import numpy as np
from torchvision.transforms.functional import to_pil_image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, models
from torchvision.datasets import ImageFolder

# 超参数
data_dir = "JUST"
batch_size = 32
learning_rate = 0.00001
num_epochs = 100
split_ratio = 0.8  # 训练集与验证集的划分比例
device = torch.device("cuda")

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 加载数据
dataset = ImageFolder(data_dir, transform=transform)
class_names = dataset.classes

# 动态划分训练集和验证集
train_size = int(len(dataset) * split_ratio)
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# VGG16 模型加载与修改
model = models.vgg16(pretrained=True)
model.classifier[6] = nn.Linear(model.classifier[6].in_features, len(class_names))  # 修改分类器输出维度
model = model.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Grad-CAM 实现
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None

        # 注册钩子函数
        self.target_layer.register_forward_hook(self.forward_hook)
        self.target_layer.register_backward_hook(self.backward_hook)

    def forward_hook(self, module, input, output):
        self.activations = output

    def backward_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0]

    def generate_cam(self, class_idx):
        weights = torch.mean(self.gradients, dim=[2, 3])
        cam = torch.zeros(self.activations.shape[2:], dtype=torch.float32).to(self.activations.device)

        for i, w in enumerate(weights[class_idx]):
            cam += w * self.activations[0, i, :, :]

        cam = torch.relu(cam)  # ReLU激活
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-9)  # 归一化
        return cam

# 定义目标层（最后一个卷积层）
target_layer = model.features[-1]
grad_cam = GradCAM(model, target_layer)

# 训练和验证函数
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    for epoch in range(num_epochs):
        # 训练模式
        model.train()
        train_loss = 0.0
        train_correct = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 记录损失和准确率
            train_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            train_correct += torch.sum(preds == labels)

        # 验证模式
        model.eval()
        val_loss = 0.0
        val_correct = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                # 前向传播
                outputs = model(images)
                loss = criterion(outputs, labels)

                # 记录损失和准确率
                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += torch.sum(preds == labels)

        # 计算平均损失和准确率
        train_loss /= len(train_loader.dataset)
        val_loss /= len(val_loader.dataset)
        train_acc = train_correct.double() / len(train_loader.dataset)
        val_acc = val_correct.double() / len(val_loader.dataset)

        # 保存每轮的损失和准确率
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc.item())
        val_accuracies.append(val_acc.item())

        # 打印每轮结果
        print(f"Epoch [{epoch + 1}/{num_epochs}]: "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # 可视化 Grad-CAM（每 5 轮保存一次）
        if (epoch + 1) % 5 == 0:
            visualize_grad_cam(model, images, labels, grad_cam)

        # 保存损失和准确率的折线图
        plot_and_save_final(train_losses, val_losses, train_accuracies, val_accuracies, epoch + 1)

# Grad-CAM 可视化
def visualize_grad_cam(model, images, labels, grad_cam):
    model.eval()
    with torch.no_grad():
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

    for i in range(min(5, images.size(0))):  # 展示 5 张图片
        image = images[i].cpu()
        label = labels[i].item()
        pred = preds[i].item()

        # 计算 Grad-CAM
        model.zero_grad()
        class_idx = pred
        outputs[:, class_idx].backward(retain_graph=True)
        cam = grad_cam.generate_cam(class_idx)

        # 绘制原图与 Grad-CAM 热力图
        cam = cam.cpu().numpy()
        heatmap = np.uint8(255 * cam)
        heatmap = plt.cm.jet(heatmap)[:, :, :3]
        heatmap = np.transpose(heatmap, (2, 0, 1))
        overlay = (image.numpy() * 0.5 + heatmap * 0.5).clip(0, 1)

        plt.subplot(1, 2, 1)
        plt.imshow(to_pil_image(image))
        plt.title(f"Label: {class_names[label]}")

        plt.subplot(1, 2, 2)
        plt.imshow(to_pil_image(torch.tensor(overlay)))
        plt.title(f"Pred: {class_names[pred]}")

        plt.show()

# 绘制并保存总的损失和准确率曲线
def plot_and_save_final(train_losses, val_losses, train_accuracies, val_accuracies):
    plt.figure(figsize=(12, 6))

    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss")
    plt.plot(range(1, len(val_losses) + 1), val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()

    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, label="Train Accuracy")
    plt.plot(range(1, len(val_accuracies) + 1), val_accuracies, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()

    # 保存图像
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/training_progress.png")
    plt.close()

# 训练和验证函数
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    for epoch in range(num_epochs):
        # 训练模式
        model.train()
        train_loss = 0.0
        train_correct = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 记录损失和准确率
            train_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            train_correct += torch.sum(preds == labels)

        # 验证模式
        model.eval()
        val_loss = 0.0
        val_correct = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                # 前向传播
                outputs = model(images)
                loss = criterion(outputs, labels)

                # 记录损失和准确率
                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += torch.sum(preds == labels)

        # 计算平均损失和准确率
        train_loss /= len(train_loader.dataset)
        val_loss /= len(val_loader.dataset)
        train_acc = train_correct.double() / len(train_loader.dataset)
        val_acc = val_correct.double() / len(val_loader.dataset)

        # 保存每轮的损失和准确率
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc.item())
        val_accuracies.append(val_acc.item())

        # 打印每轮结果
        print(f"Epoch [{epoch + 1}/{num_epochs}]: "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    # 在训练结束后保存总的损失和准确率曲线
    plot_and_save_final(train_losses, val_losses, train_accuracies, val_accuracies)

    # 保存训练完成的模型
    os.makedirs("results", exist_ok=True)
    model_path = "results/VGG+CAM_model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

# 启动训练
train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs)
