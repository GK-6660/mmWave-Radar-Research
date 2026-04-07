import torch
from torch import nn
from torch.utils.data import DataLoader
from src.dataset import RadarGestureDataset
from src.engine import get_device, train, TrainDeps

# 这是一个极其简单的基线模型 (Baseline)，证明流程能跑通
# 以后你可以把它换成复杂的 PointNet 或 ResNet
class SimpleRadarNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # 输入维度是 4 (x, y, z, v)
        self.features = nn.Sequential(
            nn.Conv1d(4, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1) # 把 512 个点压缩成 1 个全局特征
        )
        self.classifier = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.squeeze(-1) # 去掉最后一个维度
        return self.classifier(x)

def main():
    print("🚀 初始化雷达训练管线...")
    
    # 1. 加载数据
    dataset = RadarGestureDataset("data/parsed_npy", max_points=512)
    if len(dataset) == 0:
        print("🛑 没有数据，请先运行 src/data_collector.py 采集数据！")
        return
        
    train_loader = DataLoader(dataset, batch_size=4, shuffle=True)
    num_classes = len(dataset.classes)

    # 2. 准备设备和模型
    device = get_device()
    print(f"⚡ 使用计算设备: {device}")
    
    model = SimpleRadarNet(num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    # 3. 组装引擎依赖
    deps = TrainDeps(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        train_loader=train_loader,
        device=device
    )

    # 4. 开始训练！
    print("🔥 开始训练网络...")
    train(deps=deps, epochs=20)
    print("✅ 训练完成！")

if __name__ == "__main__":
    main()