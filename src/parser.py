import struct
import numpy as np

class TIParser:
    def __init__(self, roi_x=(-0.4, 0.4), roi_y=(0.1, 0.5), roi_z=(-0.4, 0.4), v_threshold=0.08):
        """
        初始化解析器，设置空间结界和速度阈值
        """
        self.MAGIC_WORD = b'\x02\x01\x04\x03\x06\x05\x08\x07'
        self.roi_x = roi_x
        self.roi_y = roi_y
        self.roi_z = roi_z
        self.v_threshold = v_threshold

    def is_in_roi(self, x, y, z, v):
        """物理降噪：检查点是否在 ROI 空间结界内且具有足够动能"""
        return (self.roi_x[0] < x < self.roi_x[1] and 
                self.roi_y[0] < y < self.roi_y[1] and 
                self.roi_z[0] < z < self.roi_z[1] and 
                abs(v) > self.v_threshold)

    def parse_stream(self, buffer):
        # --- 新增调试代码 ---
        if len(buffer) > 0:
            # 看看缓冲区里有没有数据在跑，哪怕对不上 Magic Word
            print(f"DEBUG: 缓冲区大小 = {len(buffer)} 字节", end='\r')
        # ------------------

        idx = buffer.find(self.MAGIC_WORD)
        if idx == -1:
            # 如果缓冲区太大了还没找到同步头，清空一下，防止内存溢出
            if len(buffer) > 8192:
                return [], buffer[-8:] 
            return [], buffer

        # 剩下的逻辑保持不变...
        """
        解析字节流，返回解析出的点云列表和剩余的缓冲区数据
        """
        idx = buffer.find(self.MAGIC_WORD)
        if idx == -1:
            return [], buffer

        # 确保缓冲区至少包含一个完整的帧头 (32字节 + 8字节 Magic)
        if len(buffer) < idx + 40:
            return [], buffer

        # 解析帧头 (Frame Header)
        try:
            header_data = buffer[idx+8 : idx+40]
            header = struct.unpack('<7I2H', header_data)
            packet_len = header[1]
            frame_id = header[3]
            num_tlvs = header[5]
        except struct.error:
            return [], buffer[idx+1:] # 解析失败，跳过一个字节继续找

        # 检查缓冲区是否包含了完整的数据包
        if len(buffer) < idx + packet_len:
            return [], buffer

        packet = buffer[idx : idx + packet_len]
        points = []
        tlv_offset = 40

        for _ in range(num_tlvs):
            if tlv_offset + 8 > packet_len: break
            tlv_type, tlv_len = struct.unpack('<2I', packet[tlv_offset : tlv_offset + 8])
            
            # TLV Type 1: Detected Points
            if tlv_type == 1:
                num_points = (tlv_len - 8) // 16
                for i in range(num_points):
                    p_start = tlv_offset + 8 + i * 16
                    x, y, z, v = struct.unpack('<4f', packet[p_start : p_start + 16])
                    
                    if self.is_in_roi(x, y, z, v):
                        points.append([frame_id, x, y, z, v])
            
            tlv_offset += tlv_len

        # 返回解析出的点，以及剔除当前包后的剩余缓冲区
        remaining_buffer = buffer[idx + packet_len:]
        return points, remaining_buffer