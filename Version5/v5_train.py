"""
基于毫米波雷达与边缘计算的隐私保护分布式智能中控系统
——手势分类卷积神经网络训练模块

核心特征：
- 三通道伪彩图训练：直接使用 128x128x3 轨迹图输入 CNN，不丢失三视图信息
- 现代优化策略：AdamW + Cosine 学习率调度 + Label Smoothing
- 高效训练：CUDA 场景下启用 AMP 混合精度与 GradScaler
- 泛化增强：几何增强 + MixUp + 类不平衡加权
- 模型持久化：基于验证精度保存最佳权重并支持早停
"""

import os
import random
import numpy as np
import torch as t
from torch import nn, optim
from torchvision import transforms as tf, datasets as ds
from torch.utils.data import DataLoader, Subset

# ============================================================================
# 1. 超参数配置（与数据采集模块保持一致）
# ============================================================================
DATASET_ROOT = 'dataset_v4'     # 数据集根目录
EPOCHS = 40                      # 训练轮数
BATCH_SIZE = 32                  # 批大小
LEARNING_RATE = 1e-3             # 学习率
CLASS_NAMES = ['circle', 'push', 'static', 'swipe', 'wave']  # 5类手势
SEED = 42                        # 随机种子，保证可复现
MIXUP_ALPHA = 0.2                # MixUp 分布参数（0表示关闭）
LABEL_SMOOTHING = 0.05           # 标签平滑
EARLY_STOP_PATIENCE = 8          # 早停容忍轮数


class ConvGestureClassifier(nn.Module):
    """
    卷积手势分类网络

    架构设计：
    - 特征提取器：3层卷积+ReLU+最大池化，将128×128三通道伪彩图降维至16×16特征图
    - 分类器：全连接层+Dropout，输出5类手势概率

    该结构专为边缘计算优化，在保证精度的同时控制参数量，便于后续部署至嵌入式平台。
    """

    def __init__(self, num_classes: int) -> None:
        """
        初始化网络结构

        Args:
            num_classes: 分类类别数（此处为5）
        """
        super().__init__()

        # 卷积特征提取模块：时空降维与局部特征抽取
        self.feature_extractor = nn.Sequential(
            # 第1层：16通道卷积，保留空间分辨率，后跟2倍下采样
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            # 第2层：32通道，进一步抽象中高层特征
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            # 第3层：64通道，感受野覆盖全局，提取语义信息
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

        # 分类器：将64×16×16特征图展平，映射至类别空间
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 128),   # 全连接降维至128维
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),                # 防止过拟合
            nn.Linear(128, num_classes)     # 输出logits
        )

    def forward(self, x: t.Tensor) -> t.Tensor:
        """
        前向传播

        Args:
            x: 输入张量，形状 (batch, 3, 128, 128)

        Returns:
            形状 (batch, num_classes) 的原始logits
        """
        features = self.feature_extractor(x)
        logits = self.classifier(features)
        return logits


def set_seed(seed: int) -> None:
    """设置随机种子，减少训练结果波动。"""
    random.seed(seed)
    np.random.seed(seed)
    t.manual_seed(seed)
    if t.cuda.is_available():
        t.cuda.manual_seed_all(seed)
    t.backends.cudnn.benchmark = True


def build_dataloaders(device: t.device) -> tuple[DataLoader, DataLoader, int]:
    """构建训练/验证 DataLoader，并返回训练集大小。"""
    # 训练增强：仅做轻量几何扰动，保持毫米波轨迹拓扑结构
    train_transform = tf.Compose([
        tf.RandomAffine(degrees=8, translate=(0.06, 0.06), scale=(0.92, 1.08)),
        tf.ToTensor(),
    ])
    # 验证集不做增强，保持评估一致性
    val_transform = tf.Compose([
        tf.ToTensor(),
    ])

    base_dataset = ds.ImageFolder(DATASET_ROOT)
    n_samples = len(base_dataset)
    if n_samples == 0:
        raise ValueError(f"[ERROR] 数据集为空，请检查目录: {DATASET_ROOT}")

    # 固定随机索引，保证每次划分一致
    g = t.Generator().manual_seed(SEED)
    perm = t.randperm(n_samples, generator=g).tolist()
    val_size = max(1, int(n_samples * 0.2))
    train_size = n_samples - val_size
    train_indices = perm[:train_size]
    val_indices = perm[train_size:]

    # 分别构造训练/验证数据集，使两者拥有不同 transform
    train_full = ds.ImageFolder(DATASET_ROOT, transform=train_transform)
    val_full = ds.ImageFolder(DATASET_ROOT, transform=val_transform)
    train_dataset = Subset(train_full, train_indices)
    val_dataset = Subset(val_full, val_indices)

    num_workers = 0 if os.name == 'nt' else min(4, os.cpu_count() or 2)
    pin_memory = device.type == 'cuda'
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    return train_loader, val_loader, train_size


def build_class_weights(train_loader: DataLoader, num_classes: int, device: t.device) -> t.Tensor:
    """基于训练集统计构建类别权重，缓解类别不平衡。"""
    counts = t.zeros(num_classes, dtype=t.float32)
    for _, labels in train_loader:
        counts += t.bincount(labels, minlength=num_classes).float()
    counts = t.clamp(counts, min=1.0)
    weights = counts.sum() / (num_classes * counts)
    return weights.to(device)


def mixup_batch(images: t.Tensor, labels: t.Tensor, alpha: float) -> tuple[t.Tensor, t.Tensor, t.Tensor, float]:
    """对一个 batch 执行 MixUp，返回混合样本与双标签。"""
    if alpha <= 0:
        return images, labels, labels, 1.0
    lam = np.random.beta(alpha, alpha)
    index = t.randperm(images.size(0), device=images.device)
    mixed = lam * images + (1.0 - lam) * images[index]
    y_a, y_b = labels, labels[index]
    return mixed, y_a, y_b, float(lam)


def mixup_criterion(criterion: nn.Module, logits: t.Tensor, y_a: t.Tensor, y_b: t.Tensor, lam: float) -> t.Tensor:
    """MixUp 对应损失。"""
    return lam * criterion(logits, y_a) + (1.0 - lam) * criterion(logits, y_b)


def train() -> None:
    """
    主训练流程：数据加载、模型初始化、训练与验证、权重保存
    """
    set_seed(SEED)

    # 选择计算设备（GPU优先，实现显存隔离）
    device = t.device('cuda' if t.cuda.is_available() else 'cpu')
    print(f"[INFO] Using device: {device}")

    # 创建 DataLoader（包含增强与固定划分）
    train_loader, val_loader, train_size = build_dataloaders(device)

    # 实例化模型
    model = ConvGestureClassifier(len(CLASS_NAMES)).to(device)

    # 优化器与损失函数（现代训练策略）
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=LEARNING_RATE * 0.05
    )
    class_weights = build_class_weights(train_loader, len(CLASS_NAMES), device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=LABEL_SMOOTHING)
    scaler = t.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))

    best_val_acc = 0.0
    epochs_no_improve = 0
    print("\n==================== Training Started ====================")
    print(f"[INFO] Class weights: {[round(x, 3) for x in class_weights.detach().cpu().tolist()]}")
    for epoch in range(EPOCHS):
        # ---------- 训练阶段 ----------
        model.train()
        train_loss = 0.0
        train_correct = 0.0
        train_total = 0
        for images, labels in train_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            # MixUp 增强（训练阶段）
            mix_images, y_a, y_b, lam = mixup_batch(images, labels, MIXUP_ALPHA)

            optimizer.zero_grad(set_to_none=True)
            with t.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                outputs = model(mix_images)
                loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            preds = outputs.argmax(dim=1)
            # MixUp 下使用软准确率统计，避免用单标签评估混合样本造成偏低
            train_correct += (
                lam * (preds == y_a).sum().item() +
                (1.0 - lam) * (preds == y_b).sum().item()
            )
            train_total += labels.size(0)

        avg_train_loss = train_loss / len(train_loader)
        train_acc = train_correct / max(1, train_total)

        # ---------- 验证阶段 ----------
        model.eval()
        correct = 0
        val_loss = 0.0
        with t.no_grad():                    # 禁用梯度计算，节省显存
            for images, labels in val_loader:
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                with t.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                predictions = outputs.argmax(dim=1)
                correct += (predictions == labels).sum().item()
                val_loss += loss.item()

        scheduler.step()
        avg_val_loss = val_loss / len(val_loader)
        val_acc = correct / len(val_loader.dataset)
        lr_now = optimizer.param_groups[0]['lr']
        print(
            f"Epoch [{epoch+1:2d}/{EPOCHS}] | "
            f"LR: {lr_now:.6f} | "
            f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            t.save({
                'model_state_dict': model.state_dict(),
                'class_names': CLASS_NAMES,
                'best_val_acc': best_val_acc
            }, 'best_radar.pth')
            print(f"  -> Best model saved with accuracy {val_acc:.4f}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= EARLY_STOP_PATIENCE:
                print(f"[INFO] Early stopping triggered at epoch {epoch+1}.")
                break

    print("==================== Training Completed ====================")
    print(f"Best validation accuracy: {best_val_acc:.4f}")


if __name__ == '__main__':
    train()
