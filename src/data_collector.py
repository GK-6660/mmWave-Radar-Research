import serial
import time
import numpy as np
from pathlib import Path
from datetime import datetime
from parser import TIParser

# ================= 配置区 =================
CLI_PORT = '/dev/cu.usbmodem1103'   # 修改为你的 CLI 串口
DATA_PORT = '/dev/cu.usbmodem1101'  # 修改为你的 Data 串口
CFG_FILE = 'configs/profile_3d.cfg'
RECORD_SECONDS = 10 
# =========================================

def send_config(cli_port_name, cfg_path):
    print(f"[INIT] 正在下发配置至 {cli_port_name}...")
    try:
        with serial.Serial(cli_port_name, 115200, timeout=1) as ser:
            with open(cfg_path, 'r') as f:
                for line in f:
                    cmd = line.strip()
                    if cmd and not cmd.startswith('%'):
                        ser.write((cmd + '\n').encode())
                        time.sleep(0.05)
            print("✅ 硬件初始化成功。")
    except Exception as e:
        print(f"❌ 串口配置失败: {e}")

def main():
    # 1. 实例化我们的解析大脑
    # 空间结界限定：前 10-50cm，左右 40cm，上下 40cm
    radar_parser = TIParser(
        roi_x=(-0.4, 0.4), 
        roi_y=(0.1, 0.5), 
        roi_z=(-0.4, 0.4), 
        v_threshold=0.08
    )

    # 2. 准备目录
    Path("data/parsed_npy").mkdir(parents=True, exist_ok=True)
    
    # 3. 启动硬件
    send_config(CLI_PORT, CFG_FILE)

    # 4. 进入实时采集循环
    all_samples = []
    print(f"[INFO] 正在开启数据端口 {DATA_PORT}...")
    
    try:
        with serial.Serial(DATA_PORT, 921600, timeout=0.1) as ser:
            ser.flushInput()
            buffer = b''
            start_time = time.time()
            
            print(f"⏺️ 正在采集... (计时 {RECORD_SECONDS}s)")
            
            while time.time() - start_time < RECORD_SECONDS:
                if ser.in_waiting > 0:
                    buffer += ser.read(ser.in_waiting)
                
                # 调用解析类处理当前缓冲区
                new_points, buffer = radar_parser.parse_stream(buffer)
                
                if new_points:
                    all_samples.extend(new_points)

            # 5. 保存结果
            if all_samples:
                result_matrix = np.array(all_samples)
                timestamp = datetime.now().strftime("%H%M%S")
                save_path = f"data/parsed_npy/gesture_{timestamp}.npy"
                np.save(save_path, result_matrix)
                print(f"✅ 采集完成！捕获点数: {len(all_samples)}，已存入: {save_path}")
            else:
                print("⚠️ 未能捕获到任何有效手势点，请检查 ROI 范围。")

    except KeyboardInterrupt:
        print("\n🛑 用户手动停止。")
    except Exception as e:
        print(f"❌ 采集错误: {e}")

if __name__ == "__main__":
    main()