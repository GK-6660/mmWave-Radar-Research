import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path

class RadarGestureDataset(Dataset):
    def __init__(self, data_dir="data/parsed_npy", max_points=512):
        """
        :param data_dir: 存放 .npy 数据的目录
        :param max_points: 统一对齐的点云数量（深度学习需要维度一致）
        """
        self.data_dir = Path(data_dir)
        self.max_points = max_points
        self.files = list(self.data_dir.glob("*.npy"))
        
        if len(self.files) == 0:
            print(f"⚠️ 警告: 在 {data_dir} 中没有找到任何数据！")
            
        # 从文件名自动提取类别 (例如: "swipe_left_12345.npy" -> "swipe_left")
        self.classes = sorted(list(set(["_".join(f.stem.split("_")[:-1]) for f in self.files])))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        if self.classes:
            print(f"📚 发现类别: {self.class_to_idx}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        label_str = "_".join(file_path.stem.split("_")[:-1])
        label = self.class_to_idx[label_str]

        # 1. 加载矩阵 (每一行: [frame_id, x, y, z, v])
        points = np.load(file_path)
        
        # 2. 丢弃 frame_id，只保留物理特征 [x, y, z, v]
        features = points[:, 1:] 
        
        # 3. 维度对齐 (截断或补零填充)
        if len(features) >= self.max_points:
            # 点太多，随机采样
            choices = np.random.choice(len(features), self.max_points, replace=False)
            features = features[choices]
        else:
            # 点太少，补零
            pad_size = self.max_points - len(features)
            pad_array = np.zeros((pad_size, 4))
            features = np.vstack((features, pad_array))

        # 4. 转为 Tensor: [序列长度, 特征数] -> [特征数, 序列长度] (适配 PyTorch Conv1d)
        features = torch.tensor(features, dtype=torch.float32).transpose(0, 1)
        
        return features, torch.tensor(label, dtype=torch.long)