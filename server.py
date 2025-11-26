#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
import socket
import json
import threading
import time

# ros2  topic echo  /odom --field  pose.pose.position 
# ros2 launch rtabmap_examples  realsense_d435i_infra.launch.py
class OdomServer(Node):
    def __init__(self):
        super().__init__('odom_server')

        # ---- TCP Server ----
        self.server_ip = "0.0.0.0"
        self.server_port = 5000

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((self.server_ip, self.server_port))
        self.sock.listen(1)

        print(f"[SERVER] Listening at {self.server_ip}:{self.server_port}")

        self.conn = None
        self.addr = None

        threading.Thread(target=self.accept_thread, daemon=True).start()

        # ---- ROS Subscriber ----
        self.create_subscription(Odometry, "/odom", self.odom_callback, 10)

    def accept_thread(self):
        while True:
            print("[SERVER] Waiting client...")
            self.conn, self.addr = self.sock.accept()
            print("[SERVER] Client connected:", self.addr)

    def odom_callback(self, msg: Odometry):
        if self.conn is None:
            return  # No client yet

        # ---- Odom pose ----
        pose = {
            "x": msg.pose.pose.position.x,
            "y": msg.pose.pose.position.y,
            "z": msg.pose.pose.position.z,
            "qx": msg.pose.pose.orientation.x,
            "qy": msg.pose.pose.orientation.y,
            "qz": msg.pose.pose.orientation.z,
            "qw": msg.pose.pose.orientation.w,
        }

        # ---- 时间戳 ----
        odom_ts = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        send_ts = time.time()

        packet = {
            "pose": pose,
            "odom_ts": odom_ts,
            "send_ts": send_ts
        }

        try:
            data = json.dumps(packet) + "\n"
            self.conn.sendall(data.encode())
        except Exception as e:
            print("Send error:", e)
            print("[SERVER] Client disconnected")
            self.conn.close()
            self.conn = None


def main():
    rclpy.init()
    node = OdomServer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    if node.conn:
        node.conn.close()

    node.sock.close()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
