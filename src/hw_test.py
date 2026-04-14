import serial
import time

# --- 请确认控制口 (CLI Port) ---
CLI_PORT = 'COM7'
BAUD_CLI = 115200

def ping_radar():
    print(f"🔍 正在连接雷达控制口 {CLI_PORT}...")
    
    try:
        # 打开串口
        with serial.Serial(CLI_PORT, BAUD_CLI, timeout=2) as ser:
            print("✅ 串口打开成功。")
            
            # 我们发送一个对雷达无害的查询命令 (查询版本号)
            cmd = "version"
            print(f"📡 发送命令: '{cmd}'")
            ser.write((cmd + '\n').encode())
            
            # 等待雷达处理
            time.sleep(0.5)
            
            # 读取所有回显内容
            echo = ser.read_all().decode(errors='ignore').strip()
            
            if echo:
                print("\n🎉 成功！雷达有回应：")
                print("-" * 40)
                print(echo)
                print("-" * 40)
            else:
                print("\n❌ 失败：雷达没有任何回应 (假死状态)。")
                print("👉 建议排查方向：")
                print("   1. 拔掉雷达，重新插到电脑机箱后面的蓝色 USB 3.0 接口。")
                print("   2. 按一下雷达板子上的 RESET 按钮。")
                print("   3. 检查 SOP 拨码开关是否处于正常运行模式 (通常是 011)。")

    except serial.SerialException as e:
        print(f"\n❌ 串口连接失败：{e}")
        print("👉 请检查端口号是否被其他软件 (如 UniFlash, 网页版调试器) 占用。")

if __name__ == "__main__":
    ping_radar()