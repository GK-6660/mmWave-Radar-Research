# 📡 毫米波雷达底层信号处理与手势识别研究
**(mmWave Radar Signal Processing & Gesture Recognition)**

本项目致力于毫米波雷达（mmWave Radar）的底层数字信号处理与特征提取。摒弃高度封装的上层应用，直接从原始二进制数据流出发，完成从**点云数据采集**、**空间降噪**到**深度学习模型训练**的全链路闭环。

---

## 📁 目录结构 (Directory Structure)

```text
MMWAVE-RADAR-RESEARCH/
├── .venv/                   # 虚拟环境 (本地生成，不上云)
├── configs/                 # 雷达底层硬件配置文件 (.cfg)
├── data/                    # 数据集 (受 .gitignore 保护)
│   ├── raw_bin/             # 串口采集的原始二进制流
│   └── parsed_npy/          # 解析并降噪后的 NumPy 点云矩阵
├── notebooks/               # 数据探索与可视化草案
│   └── 01_data_exploration.ipynb
├── src/                     # 核心源码区
│   ├── data_collector.py    # 🎯 入口1：硬件控制与点云数据采集
│   ├── dataset.py           # 工具类：PyTorch 数据加载与维度对齐
│   ├── engine.py            # 工具类：解耦的模型训练与测试引擎
│   └── parser.py            # 工具类：雷达 TLV 协议解析与空间降噪
├── .gitignore               # 防火墙：防止大文件和缓存污染仓库
├── .python-version          # 锁定 Python 版本 (3.10)
├── pyproject.toml           # 项目宪法：定义核心依赖与构建系统
├── train.py                 # 🎯 入口2：神经网络训练主程序
└── uv.lock                  # 环境锁：确保全平台字节级环境一致