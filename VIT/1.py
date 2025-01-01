import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast  # 自动混合精度

# =========================
# 自定义 Vision Transformer
# =========================
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.fc_out = nn.Linear(embed_dim, embed_dim)

        self.scale = torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))

    def forward(self, x):
        N, seq_len, embed_dim = x.shape
        q = self.query(x).view(N, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(N, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(N, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        attn_probs = torch.nn.functional.softmax(attn_weights, dim=-1)

        out = torch.matmul(attn_probs, v).transpose(1, 2).contiguous()
        out = out.view(N, seq_len, embed_dim)
        return self.fc_out(out)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, forward_expansion, dropout_rate):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadSelfAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, forward_expansion * embed_dim),
            nn.GELU(),
            nn.Linear(forward_expansion * embed_dim, embed_dim),
        )
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        attn_out = self.attention(x)
        x = self.norm1(x + self.dropout(attn_out))
        feedforward_out = self.feed_forward(x)
        x = self.norm2(x + self.dropout(feedforward_out))
        return x


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, patch_size, embed_dim, img_size):
        super(PatchEmbedding, self).__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_dim = in_channels * patch_size * patch_size
        self.projection = nn.Linear(self.patch_dim, embed_dim)
        self.patch_size = patch_size

    def forward(self, x):
        B, C, H, W = x.shape
        patches = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
        patches = patches.view(B, self.num_patches, -1)
        return self.projection(patches)


class VisionTransformer(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_channels=3,
        num_classes=2,
        embed_dim=768,
        num_heads=12,
        num_layers=12,
        forward_expansion=4,
        dropout_rate=0.01,
    ):
        super(VisionTransformer, self).__init__()
        self.patch_embed = PatchEmbedding(in_channels, patch_size, embed_dim, img_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.positional_encoding = nn.Parameter(
            torch.zeros(1, self.patch_embed.num_patches + 1, embed_dim)
        )
        self.dropout = nn.Dropout(dropout_rate)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(embed_dim, num_heads, forward_expansion, dropout_rate)
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.positional_encoding
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        return self.head(x[:, 0])


# ===================================
# 数据集和训练代码
# ===================================
# 数据路径
data_dir = "JUST"
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

# 数据加载器
train_dataset = CustomDataset(data_dir, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=48, shuffle=True, num_workers=4, pin_memory=True)

# 加载自定义 ViT 模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VisionTransformer().to(device)

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-5)
scaler = GradScaler()

# 训练模型
epochs = 100
train_loss, train_acc = [], []
accumulation_steps = 2

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss = loss / accumulation_steps

        scaler.scale(loss).backward()

        if (i + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        running_loss += loss.item() * accumulation_steps
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
torch.save(model.state_dict(), "vit_model_custom.pth")
