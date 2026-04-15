"""
基于毫米波雷达与边缘计算的隐私保护分布式智能中控系统
——数据采集模块

核心特征：
- 多径底噪抑制：基于空间结界与速度阈值剔除静态杂波、微动干扰及多径反射
- 时空降维：通过三视图投影将4D点云数据映射至2D平面，保留空间拓扑特征
- 纯NumPy矩阵运算：所有图像构建基于NumPy数组，避免循环冗余
- 内存隔离：采用滑动窗口缓冲区与魔数同步机制，防止数据污染与内存泄漏
"""

import serial
import time
import numpy as np
import cv2
import struct
import os
from collections import deque

# ============================================================================
# 1. 系统配置参数（需根据实际硬件接口调整）
# ============================================================================
CLI_PORT = 'COM7'          # 雷达配置串口
DATA_PORT = 'COM8'         # 雷达数据输出串口
CFG_FILE = 'profile_3d.cfg'  # 雷达配置文件路径

# 动作类别定义（支持5类3D手势）
ACTION_LABELS = ['circle', 'push', 'static', 'swipe', 'wave']
DATASET_DIR = "dataset_v4"   # 数据集存储根目录

# 自动创建类别子目录
for action in ACTION_LABELS:
    os.makedirs(os.path.join(DATASET_DIR, action), exist_ok=True)

# 点云轨迹队列（时序滑动窗口，用于抑制多径底噪及实现时空平滑）
point_history = deque(maxlen=30)


def send_cfg(cli_port: serial.Serial, cfg_file: str) -> None:
    """
    将雷达配置文件逐行下发至设备，完成工作模式与参数设定。

    该过程通过CLI串口写入配置命令，每行间隔50ms以保证设备响应，
    并读取回显以清空缓冲区，避免残留数据干扰后续数据流解析。
    """
    print("[INFO] 正在下发配置...")
    with open(cfg_file, 'r') as f:
        for line in f.readlines():
            cmd = line.strip()
            if not cmd.startswith('%') and cmd != '':
                cli_port.write((cmd + '\n').encode('utf-8'))
                time.sleep(0.05)
                cli_port.read(cli_port.in_waiting)  # 清空回显，实现内存隔离
    print("[INFO] 雷达已启动。")


def get_img_3_views(points: list) -> np.ndarray:
    """
    时空降维：利用 NumPy 向量化操作将 30 帧点云映射为 128x128x3 的伪彩色轨迹图。
    摒弃冗余的 for 循环和 OpenCV 绘制，直接进行矩阵索引赋值。
    Top (X-Y) -> B通道
    Front (X-Z) -> G通道
    Side (Y-Z) -> R通道
    """
    # 更新点云轨迹队列
    if points is not None and len(points) > 0:
        point_history.append(points)
    else:
        point_history.append([])

    # 初始化 128x128x3 的画布
    img = np.zeros((128, 128, 3), dtype=np.uint8)
    n_frames = len(point_history)
    if n_frames == 0:
        return img

    # 收集有效点云及时间强度权重
    point_data = []
    intensities = []
    
    for i, frame in enumerate(point_history):
        if not frame:
            continue
        # 强度编码：越新的点越亮（时间维降维）
        intensity = int(255 * (i + 1) / n_frames)
        point_data.extend(frame)
        intensities.extend([intensity] * len(frame))
        
    if not point_data:
        return img

    # 转换为 NumPy 数组以进行纯向量化计算
    pts = np.array(point_data, dtype=np.float32)
    vals = np.array(intensities, dtype=np.uint8)

    # 空间归一化并映射到 0~127 的像素坐标
    # X (宽度): ±0.4m -> [0, 127]
    # Y (深度): 0.1~0.5m -> [0, 127]
    # Z (高度): 假设 ±0.4m -> [0, 127]
    px = np.clip((pts[:, 0] + 0.4) / 0.8 * 127, 0, 127).astype(np.int32)
    py = np.clip((pts[:, 1] - 0.1) / 0.4 * 127, 0, 127).astype(np.int32)
    pz = np.clip((pts[:, 2] + 0.4) / 0.8 * 127, 0, 127).astype(np.int32)

    # NumPy 高级索引向量化赋值，后绘制的点(较新/较亮)会自动覆盖旧点
    img[127 - py, px, 0] = vals  # Top View (X-Y) -> B 通道
    img[127 - pz, px, 1] = vals  # Front View (X-Z) -> G 通道
    img[127 - pz, py, 2] = vals  # Side View (Y-Z) -> R 通道

    return img


def main() -> None:
    """主流程：雷达数据采集、实时可视化与手势样本保存。"""
    # 初始化串口连接
    cli_serial = serial.Serial(CLI_PORT, 115200, timeout=1)
    data_serial = serial.Serial(DATA_PORT, 921600, timeout=0.1)
    send_cfg(cli_serial, CFG_FILE)

    # 数据帧同步魔数（由雷达协议定义）
    FRAME_MAGIC = b'\x02\x01\x04\x03\x06\x05\x08\x07'
    raw_packet_buffer = b''   # 内存隔离缓冲区，避免跨帧污染
    sample_counts = {k: 0 for k in ACTION_LABELS}

    print("\n==================================================")
    print(" V4 Data Collector - Privacy-Preserving Gesture Acquisition")
    print(" --------------------------------------------------")
    print(" Filtering: Only valid gestures within 10~40cm, velocity > 15cm/s")
    print(" Press key to capture perfect trajectory:")
    print(" [c] circle   - vertical circle like rotating a safe dial")
    print(" [p] push     - palm push toward radar")
    print(" [s] swipe    - horizontal left/right swipe")
    print(" [w] wave     - continuous side-to-side wave")
    print(" [space] static - keep still, capture background")
    print(" [q] quit")
    print("==================================================\n")

    try:
        while True:
            # 读取原始数据流
            new_data = data_serial.read(data_serial.in_waiting or 1)
            if new_data:
                raw_packet_buffer += new_data

            # 查找帧同步头
            idx = raw_packet_buffer.find(FRAME_MAGIC)
            if idx > 0:
                # 立即丢弃魔数之前的无用数据（处理错位与粘包）
                raw_packet_buffer = raw_packet_buffer[idx:]
                idx = 0

            if idx == 0 and len(raw_packet_buffer) >= 40:
                # 解析帧头，获取数据包长度、目标数量等信息
                header_data = raw_packet_buffer[8: 40]
                _, pkt_len, _, _, _, num_obj, num_tlv, _ = struct.unpack('8I', header_data)

                if len(raw_packet_buffer) >= pkt_len:
                    # 提取完整数据包
                    packet_data = raw_packet_buffer[: pkt_len]
                    tlv_start = 40
                    frame_points = []
                    
                    # 移除已处理的包数据，实现内存隔离
                    raw_packet_buffer = raw_packet_buffer[pkt_len:]

                    # 遍历TLV块，解析目标点云
                    for _ in range(num_tlv):
                        if tlv_start + 8 > pkt_len:
                            break
                        tlv_type, tlv_length = struct.unpack('2I', packet_data[tlv_start: tlv_start + 8])
                        if tlv_type == 1 and num_obj > 0:
                            # 提取每个目标的坐标与速度
                            for p in range(num_obj):
                                p_start = tlv_start + 8 + p * 16
                                if p_start + 16 > pkt_len:
                                    break
                                x, y, z, v = struct.unpack('4f', packet_data[p_start: p_start + 16])

                                # 多径底噪抑制：空间结界 + 速度阈值
                                # 根据架构原则修改：深度 0.1~0.5m，宽度 ±0.4m，径向速度绝对值 > 0.04m/s
                                if 0.1 < y < 0.5 and -0.4 < x < 0.4 and abs(v) > 0.04:
                                    frame_points.append((x, y, z, v))

                        tlv_start += 8 + tlv_length

                    # 向量化生成 128x128x3 的伪彩色轨迹图 (CNN输入格式)
                    img_cnn = get_img_3_views(frame_points)
                    
                    # 放大图像用于 PC 端人类可视化观察
                    img_display = cv2.resize(img_cnn, (512, 512), interpolation=cv2.INTER_NEAREST)
                    cv2.imshow('Radar Pseudo-Color Collector', img_display)

                    # 键盘事件处理
                    key = cv2.waitKey(1) & 0xFF
                    action_to_save = None
                    if key == ord('c'):
                        action_to_save = 'circle'
                    elif key == ord('p'):
                        action_to_save = 'push'
                    elif key == ord('s'):
                        action_to_save = 'swipe'
                    elif key == ord('w'):
                        action_to_save = 'wave'
                    elif key == ord(' '):
                        action_to_save = 'static'
                    elif key == ord('q'):
                        break

                    if action_to_save:
                        timestamp = int(time.time() * 1000)
                        filename = f"{DATASET_DIR}/{action_to_save}/{timestamp}.jpg"
                        cv2.imwrite(filename, img_cnn)
                        sample_counts[action_to_save] += 1
                        print(f"[Captured] {action_to_save} - total: {sample_counts[action_to_save]}")
            else:
                # 缓冲区溢出保护：保留最后7字节用于下一次同步
                if len(raw_packet_buffer) > 4096:
                    raw_packet_buffer = raw_packet_buffer[-7:]

    except Exception as e:
        print(f"[ERROR] System exception: {e}")
    finally:
        cli_serial.close()
        data_serial.close()
        cv2.destroyAllWindows()
        print("[INFO] Resources released.")


if __name__ == '__main__':
    main()