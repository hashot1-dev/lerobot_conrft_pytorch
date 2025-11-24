#!/usr/bin/env python3
import socket
import json
import time
import struct
import threading
from multiprocessing import shared_memory
import numpy as np

class SharedMemoryWriter:
    def __init__(self, shm_name='t265_pose_data', size=1024):
        self.shm_name = shm_name
        self.size = size
        self.shm = None
        self.lock = threading.Lock()
        
    def initialize_shm(self):
        """初始化共享内存"""
        try:
            # 尝试创建新的共享内存，如果已存在则先清理
            try:
                existing_shm = shared_memory.SharedMemory(name=self.shm_name, create=False)
                existing_shm.close()
                existing_shm.unlink()
                print(f"清理已存在的共享内存: {self.shm_name}")
                time.sleep(0.1)
            except FileNotFoundError:
                pass
            
            # 创建新的共享内存
            self.shm = shared_memory.SharedMemory(
                name=self.shm_name, 
                create=True, 
                size=self.size
            )
            
            # 初始化共享内存为0
            self.shm.buf[:] = b'\x00' * self.size
            
            print(f"共享内存 '{self.shm_name}' 初始化成功 (大小: {self.size} 字节)")
            return True
            
        except Exception as e:
            print(f"共享内存初始化失败: {e}")
            return False
    
    def write_pose_data(self, pose_data):
        """写入位姿数据到共享内存"""
        if not self.shm:
            return False
            
        with self.lock:
            try:
                # 修正：使用11个double和1个int，总共92字节
                # 11个double: 位置x3, 旋转x4, 速度x3, 时间戳x1
                # 1个int: 置信度
                data_bytes = struct.pack('11di',
                    pose_data['x'], pose_data['y'], pose_data['z'],      # 位置
                    pose_data['qx'], pose_data['qy'], pose_data['qz'], pose_data['qw'],  # 旋转
                    pose_data['vx'], pose_data['vy'], pose_data['vz'],   # 速度
                    pose_data['timestamp'],                              # 时间戳
                    pose_data['confidence']                              # 置信度
                )
                
                # 写入共享内存
                self.shm.buf[0:len(data_bytes)] = data_bytes
                return True
                
            except Exception as e:
                print(f"写入共享内存失败: {e}")
                return False
    
    def close(self):
        """关闭共享内存"""
        try:
            if self.shm:
                self.shm.close()
                self.shm.unlink()  # 清理共享内存
        except Exception as e:
            print(f"关闭共享内存时出错: {e}")

class T265PoseClientWriter:
    def __init__(self, server_host='localhost', server_port=8888, shm_name='t265_pose_data'):
        self.server_host = server_host
        self.server_port = server_port
        self.shm_writer = SharedMemoryWriter(shm_name)
        self.socket = None
        self.running = False
        self.connection_retries = 3
        self.write_count = 0
        
    def connect_to_server(self):
        """连接到T265服务器"""
        for attempt in range(self.connection_retries):
            try:
                self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.socket.settimeout(5.0)
                self.socket.connect((self.server_host, self.server_port))
                print(f"已连接到T265服务器 {self.server_host}:{self.server_port}")
                return True
            except socket.timeout:
                print(f"连接超时，尝试 {attempt + 1}/{self.connection_retries}")
            except Exception as e:
                print(f"连接尝试 {attempt + 1} 失败: {e}")
            
            if attempt < self.connection_retries - 1:
                time.sleep(2)
        
        return False
    
    def start(self):
        """启动客户端（接收数据并写入共享内存）"""
        # 初始化共享内存
        if not self.shm_writer.initialize_shm():
            print("共享内存初始化失败，退出")
            return
        
        # 连接到服务器
        if not self.connect_to_server():
            print("无法连接到服务器，退出")
            return
            
        self.running = True
        buffer = ""
        last_print_time = time.time()
        
        print("开始接收位姿数据并写入共享内存...")
        print("按 Ctrl+C 停止")
        
        try:
            while self.running:
                try:
                    # 接收数据
                    data = self.socket.recv(4096).decode('utf-8')
                    if not data:
                        print("服务器断开连接")
                        break
                        
                    buffer += data
                    lines = buffer.split('\n')
                    
                    # 处理完整的JSON行
                    for line in lines[:-1]:
                        if line.strip():
                            try:
                                pose_json = json.loads(line)
                                self.process_pose_data(pose_json)
                                self.write_count += 1
                            except json.JSONDecodeError as e:
                                print(f"JSON解析错误: {e}")
                    
                    buffer = lines[-1]
                    
                    # 每秒钟打印一次状态
                    current_time = time.time()
                    if current_time - last_print_time >= 1.0:
                        print(f"已写入 {self.write_count} 帧数据 | 共享内存: {self.shm_writer.shm_name}")
                        last_print_time = current_time
                    
                except socket.timeout:
                    continue  # 超时是正常的，继续循环
                except Exception as e:
                    print(f"接收数据错误: {e}")
                    break
                    
        except KeyboardInterrupt:
            print("\n用户中断")
        except Exception as e:
            print(f"发生错误: {e}")
        finally:
            self.stop()
    
    def process_pose_data(self, pose_json):
        """处理位姿数据并写入共享内存"""
        try:
            # 提取数据
            pos = pose_json['position']
            rot = pose_json['rotation']
            vel = pose_json['velocity']
            
            # 准备共享内存数据
            pose_data = {
                'x': float(pos['x']),
                'y': float(pos['y']),
                'z': float(pos['z']),
                'qx': float(rot['x']),
                'qy': float(rot['y']),
                'qz': float(rot['z']),
                'qw': float(rot['w']),
                'vx': float(vel['x']),
                'vy': float(vel['y']),
                'vz': float(vel['z']),
                'timestamp': time.time(),
                'confidence': int(pose_json['confidence']['tracker'])
            }
            
            # 写入共享内存
            if self.shm_writer.write_pose_data(pose_data):
                # 只在置信度较高时显示详细信息
                if pose_data['confidence'] >= 2 and self.write_count % 10 == 0:
                    print(f"位置: X:{pos['x']:7.3f}, Y:{pos['y']:7.3f}, Z:{pos['z']:7.3f} | "
                          f"置信度: {pose_json['confidence']['tracker']}")
            
        except KeyError as e:
            print(f"数据格式错误，缺少字段: {e}")
        except Exception as e:
            print(f"处理位姿数据错误: {e}")
    
    def stop(self):
        """停止客户端"""
        self.running = False
        if self.socket:
            self.socket.close()
        self.shm_writer.close()
        print(f"客户端写入器已停止，总共写入 {self.write_count} 帧数据")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='T265位姿数据共享内存写入客户端')
    parser.add_argument('--host', default='192.168.1.57', help='T265服务器IP地址')
    parser.add_argument('--port', type=int, default=8888, help='T265服务器端口')
    parser.add_argument('--shm-name', default='t265_pose_data', help='共享内存名称')
    parser.add_argument('--shm-size', type=int, default=1024, help='共享内存大小')
    
    args = parser.parse_args()
    
    client_writer = T265PoseClientWriter(
        server_host=args.host,
        server_port=args.port,
        shm_name=args.shm_name
    )
    
    # 设置共享内存大小
    client_writer.shm_writer.size = args.shm_size
    
    try:
        client_writer.start()
    except Exception as e:
        print(f"客户端运行错误: {e}")
