#!/usr/bin/env python3
import time
import struct
import threading
from multiprocessing import shared_memory
import pyrealsense2 as rs


class SharedMemoryWriter:
    def __init__(self, shm_name='t265_pose_data', size=1024):
        self.shm_name = shm_name
        self.size = size
        self.shm = None
        self.lock = threading.Lock()
        
    def initialize_shm(self):
        """初始化共享内存"""
        try:
            try:
                existing = shared_memory.SharedMemory(name=self.shm_name, create=False)
                existing.close()
                existing.unlink()
                time.sleep(0.05)
            except FileNotFoundError:
                pass
            
            self.shm = shared_memory.SharedMemory(
                name=self.shm_name,
                create=True,
                size=self.size
            )
            self.shm.buf[:] = b'\x00' * self.size
            print(f"共享内存 '{self.shm_name}' 初始化成功 (大小: {self.size} bytes)")
            return True

        except Exception as e:
            print(f"共享内存初始化失败: {e}")
            return False

    def write_pose_data(self, pose):
        """写入共享内存"""
        if not self.shm:
            return False

        with self.lock:
            try:
                data_bytes = struct.pack(
                    '11di',
                    pose['x'], pose['y'], pose['z'],
                    pose['qx'], pose['qy'], pose['qz'], pose['qw'],
                    pose['vx'], pose['vy'], pose['vz'],
                    pose['timestamp'],
                    pose['confidence']
                )

                self.shm.buf[0:len(data_bytes)] = data_bytes
                return True
            
            except Exception as e:
                print(f"写入共享内存失败: {e}")
                return False

    def close(self):
        try:
            if self.shm:
                self.shm.close()
                self.shm.unlink()
        except Exception as e:
            print(f"关闭共享内存出错: {e}")

class T265DirectReader:
    def __init__(self, shm_name='t265_pose_data'):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.pose)
        self.shm_writer = SharedMemoryWriter(shm_name)
        self.running = False
        self.write_count = 0

    def start(self):
        if not self.shm_writer.initialize_shm():
            print("共享内存初始化失败，退出")
            return

        print("启动 RealSense T265 管道...")
        self.pipeline.start(self.config)
        self.running = True

        print("开始从 T265 读取位姿并写入共享内存...")
        print("按 Ctrl+C 停止")

        last_print = time.time()

        try:
            while self.running:
                frames = self.pipeline.wait_for_frames()
                pose_frame = frames.get_pose_frame()
                if pose_frame:
                    pose_data = pose_frame.get_pose_data()

                    pose = {
                        'x': pose_data.translation.x,
                        'y': pose_data.translation.y,
                        'z': pose_data.translation.z,
                        'qx': pose_data.rotation.x,
                        'qy': pose_data.rotation.y,
                        'qz': pose_data.rotation.z,
                        'qw': pose_data.rotation.w,
                        'vx': pose_data.velocity.x,
                        'vy': pose_data.velocity.y,
                        'vz': pose_data.velocity.z,
                        'timestamp': time.time(),
                        'confidence': pose_data.tracker_confidence
                    }

                    self.shm_writer.write_pose_data(pose)
                    self.write_count += 1

                    # 每秒打印一次
                    if time.time() - last_print > 1.0:
                        print(f"[{self.write_count}] X:{pose['x']:.3f} Y:{pose['y']:.3f} Z:{pose['z']:.3f}  conf:{pose['confidence']}")
                        last_print = time.time()

        except KeyboardInterrupt:
            print("\n用户中断")

        finally:
            self.stop()

    def stop(self):
        self.running = False
        try:
            self.pipeline.stop()
        except:
            pass
        self.shm_writer.close()
        print(f"T265 读取器已停止，总写入 {self.write_count} 帧数据")

if __name__ == "__main__":
    reader = T265DirectReader()
    reader.start()
