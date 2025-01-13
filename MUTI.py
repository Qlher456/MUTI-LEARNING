import os
import torch
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from PIL import Image
from torchvision import datasets, transforms, models
import torch.optim as optim
import matplotlib.pyplot as plt

# 超参数定义
HYPERPARAMETERS = {
    "img_size": 224,
    "patch_size": 16,
    "num_classes": 2,
    "batch_size": 32,
    "num_epochs": 100,
    "learning_rate": 0.00001,
    "test_size": 0.2,
    "num_workers": 4
}

# JUST 数据集定义
class JUSTDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

# 数据集加载函数
def load_just_dataset(root_dir, test_size, num_workers, batch_size):
    classes = ['Fake', 'Real']
    file_paths, labels = [], []
    for idx, cls in enumerate(classes):
        cls_dir = os.path.join(root_dir, cls)
        for img_file in os.listdir(cls_dir):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                file_paths.append(os.path.join(cls_dir, img_file))
                labels.append(idx)

    train_paths, val_paths, train_labels, val_labels = train_test_split(
        file_paths, labels, test_size=test_size, stratify=labels, random_state=42
    )

    transform_train = transforms.Compose([
        transforms.Resize((HYPERPARAMETERS["img_size"], HYPERPARAMETERS["img_size"])),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    transform_val = transforms.Compose([
        transforms.Resize((HYPERPARAMETERS["img_size"], HYPERPARAMETERS["img_size"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    train_dataset = JUSTDataset(train_paths, train_labels, transform=transform_train)
    val_dataset = JUSTDataset(val_paths, val_labels, transform=transform_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader

# 手工实现的 ViT 模块
class ManualViT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768, num_heads=12, num_layers=6, num_classes=10):
        super(ManualViT, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim

        # Patch Embedding
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

        # Class Token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Positional Embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.dropout = nn.Dropout(0.1)

        # Transformer Encoder
        self.encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(embed_dim, num_heads, dim_feedforward=embed_dim * 4, dropout=0.1)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # Classification Head
        self.mlp_head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        # Step 1: Patch Embedding
        x = self.patch_embed(x)  # [B, embed_dim, H/patch_size, W/patch_size]
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]

        # Step 2: Add Class Token and Positional Embedding
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)  # [B, 1, embed_dim]
        x = torch.cat([cls_tokens, x], dim=1)  # [B, num_patches+1, embed_dim]
        x = x + self.pos_embed  # Add positional embedding
        x = self.dropout(x)

        # Step 3: Transformer Encoder
        for layer in self.encoder_layers:
            x = layer(x)

        x = self.norm(x)

        # Step 4: Classification using the CLS token
        cls_token_final = x[:, 0]  # [B, embed_dim]
        return cls_token_final

# VGG16 + CAM 模块
class VGG16CAM(nn.Module):
    def __init__(self, num_classes=10, pretrained=True):
        super(VGG16CAM, self).__init__()
        vgg16 = models.vgg16(pretrained=pretrained)
        self.features = vgg16.features
        self.gap = nn.AdaptiveAvgPool2d(1)  # 全局平均池化
        self.classifier = nn.Linear(512, num_classes)  # 用于分类的全连接层

    def forward(self, x):
        feature_maps = self.features(x)  # [B, 512, H, W]
        global_features = self.gap(feature_maps).view(feature_maps.size(0), -1)  # [B, 512]
        cam_weights = self.classifier.weight  # [num_classes, 512]
        cam = torch.einsum('bchw,oc->bohw', feature_maps, cam_weights)  # 计算CAM
        return global_features, cam

# 融合模块
class MultiModalFusion(nn.Module):
    def __init__(self, vit_dim, vgg_dim, num_classes):
        super(MultiModalFusion, self).__init__()
        self.fc_vit = nn.Linear(vit_dim, 512)
        self.fc_vgg = nn.Linear(vgg_dim, 512)
        self.fusion_fc = nn.Linear(512 * 2, num_classes)

    def forward(self, vit_features, vgg_features):
        vit_out = self.fc_vit(vit_features)
        vgg_out = self.fc_vgg(vgg_features)
        fused_features = torch.cat([vit_out, vgg_out], dim=1)
        logits = self.fusion_fc(fused_features)
        return logits

# 多模态融合整体模型
class MultiModalModel(nn.Module):
    def __init__(self, img_size=224, patch_size=16, num_classes=10):
        super(MultiModalModel, self).__init__()
        self.vit = ManualViT(img_size=img_size, patch_size=patch_size, num_classes=num_classes)
        self.vgg_cam = VGG16CAM(pretrained=True)
        self.fusion = MultiModalFusion(vit_dim=768, vgg_dim=512, num_classes=num_classes)

    def forward(self, x):
        # ViT特征
        vit_features = self.vit(x)
        # VGG16 + CAM特征
        vgg_features, cam = self.vgg_cam(x)
        # 特征融合并分类
        logits = self.fusion(vit_features, vgg_features)
        return logits, cam

# 可视化 CAM
def visualize_cam(input_tensor, cam, index, class_idx=None):
    input_img = input_tensor[index].cpu().permute(1, 2, 0).numpy()  # [H, W, C]
    input_img = (input_img - input_img.min()) / (input_img.max() - input_img.min())  # 归一化

    cam_sample = cam[index]  # [num_classes, 7, 7]
    if class_idx is None:
        class_idx = cam_sample.mean(dim=(1, 2)).argmax().item()  # 激活最大类别

    cam_heatmap = cam_sample[class_idx].detach().cpu().numpy()  # [7, 7]
    cam_heatmap = np.maximum(cam_heatmap, 0)  # ReLU
    cam_heatmap = cam_heatmap / cam_heatmap.max()  # 归一化

    # 插值到输入图像大小
    cam_heatmap = Image.fromarray(cam_heatmap).resize(input_img.shape[:2][::-1], Image.BILINEAR)
    cam_heatmap = np.array(cam_heatmap)

    ## 显示热力图叠加效果
    # plt.figure(figsize=(6, 6))
    # plt.imshow(input_img)
    # plt.imshow(cam_heatmap, cmap='jet', alpha=0.5)
    # plt.title(f"Class {class_idx} Activation Map")
    # plt.axis('off')
    # plt.show()

# 训练与验证
def train_and_validate(model, train_loader, val_loader, device, num_epochs, lr):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss, correct, total = 0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs, _ = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_acc = correct / total
        train_losses.append(train_loss / len(train_loader))
        train_accuracies.append(train_acc)

        # 验证阶段
        model.eval()
        val_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                outputs, _ = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        val_acc = correct / total
        val_losses.append(val_loss / len(val_loader))
        val_accuracies.append(val_acc)

        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_accuracies[-1]:.4f} "
              f"Val Loss: {val_losses[-1]:.4f}, Val Acc: {val_accuracies[-1]:.4f}")

    # 保存训练曲线
    save_training_curves(train_losses, val_losses, train_accuracies, val_accuracies)

def save_training_curves(train_losses, val_losses, train_accuracies, val_accuracies):
    plt.figure(figsize=(10, 5))

    # 绘制 Loss 曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()

    # 绘制 Accuracy 曲线
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curve')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_curves.png')  # 保存为图片

# 主程序
if __name__ == "__main__":
    # 数据集路径
    root_dir = "./JUST"  # 替换为实际路径
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载数据集
    train_loader, val_loader = load_just_dataset(root_dir,
                                                 test_size=HYPERPARAMETERS["test_size"],
                                                 num_workers=HYPERPARAMETERS["num_workers"],
                                                 batch_size=HYPERPARAMETERS["batch_size"])

    # 初始化多模态模型
    model = MultiModalModel(img_size=HYPERPARAMETERS["img_size"],
                            patch_size=HYPERPARAMETERS["patch_size"],
                            num_classes=HYPERPARAMETERS["num_classes"]).to(device)

    # 开始训练与验证
    train_and_validate(model, train_loader, val_loader, device,
                       num_epochs=HYPERPARAMETERS["num_epochs"],
                       lr=HYPERPARAMETERS["learning_rate"])
