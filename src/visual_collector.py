import serial
import time
import numpy as np
import open3d as o3d
from parser import TIParser

# ================= 配置区 =================
CLI_PORT = 'COM8'   # 控制口 (Standard Port)
DATA_PORT = 'COM7'  # 数据口 (Enhanced Port)
CFG_FILE = 'configs/profile_3d.cfg'
# =========================================

def send_config(port, cfg_path):
    print(f"[INIT] 正在下发配置至 {port}...")
    try:
        with serial.Serial(port, 115200, timeout=1) as ser:
            with open(cfg_path, 'r', encoding='utf-8') as f:
                for line in f:
                    cmd = line.strip()
                    if cmd and not cmd.startswith('%'):
                        ser.write((cmd + '\n').encode())
                        time.sleep(0.05)
            print("✅ 硬件初始化成功。")
    except Exception as e:
        print(f"❌ 串口失败: {e}")

def main():
    # 1. 初始化 3D 画布
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="毫米波雷达实时点云 (关闭窗口结束)", width=800, height=600)
    
    # 创建点云对象
    pcd = o3d.geometry.PointCloud()
    # 初始放一个点，防止坐标系崩塌
    pcd.points = o3d.utility.Vector3dVector(np.array([[0, 0, 0]]))
    vis.add_geometry(pcd)
    
    # 2. 启动硬件
    send_config(CLI_PORT, CFG_FILE)
    
    # 3. 初始化解析器 (把 ROI 开到最大，防止过滤掉你的手)
    parser = TIParser(roi_x=(-2, 2), roi_y=(0.05, 3), roi_z=(-2, 2), v_threshold=0.0)

    print(f"📡 正在监听数据端口 {DATA_PORT}...")
    
    try:
        with serial.Serial(DATA_PORT, 921600, timeout=0.1) as ser:
            ser.flushInput()
            buffer = b''
            
            while True:
                if ser.in_waiting > 0:
                    buffer += ser.read(ser.in_waiting)
                
                # 解析数据
                points, buffer = parser.parse_stream(buffer)
                
                if points:
                    # 提取 x, y, z 坐标 (丢弃 frame_id 和 v)
                    pts_np = np.array(points)[:, 1:4] 
                    
                    # 更新 3D 渲染
                    pcd.points = o3d.utility.Vector3dVector(pts_np)
                    vis.update_geometry(pcd)
                    print(f"✨ 捕获点数: {len(pts_np)}", end='\r')
                
                # 刷新窗口
                vis.poll_events()
                vis.update_renderer()
                
                # 防止 CPU 占用过高
                time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n🛑 停止采集")
    finally:
        vis.destroy_window()

if __name__ == "__main__":
    main()