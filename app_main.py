#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
import numpy as np
import time
from threading import Lock
from pynput import keyboard
from franka_ee_control import FrankaSimController
# ========== 你的 SO101 控制库 ==========
from lerobot.model.kinematics import RobotKinematics
from lerobot.robots.so101_follower import SO101FollowerConfig, SO101Follower

import pygame

pygame.init()
pygame.joystick.init()

joystick_count = pygame.joystick.get_count()
if joystick_count == 0:
    print("没有检测到手柄")
else:
    joystick = pygame.joystick.Joystick(0)
    joystick.init()
    print(f"手柄: {joystick.get_name()}")

    #'ee.x': 0.39553810410701296, 'ee.y': 0.0025024925733363053, 'ee.z': 0.19295834032125866
# # ====== 初始点 ======
# so101_reset_x = 0.00553810410701296
# so101_reset_y = 0.0025024925733363053
# so101_reset_z = 0.34295834032125866


# ====== 初始点 ======
so101_reset_x = 0.39553810410701296
so101_reset_y = 0.0025024925733363053
so101_reset_z = 0.19295834032125866
so101_t=2

franka_reset_x = 0.21
franka_reset_y = 0.0025024925733363053
franka_reset_z = 0.82  #1.12
franka_t=1

reset_x = franka_reset_x
reset_y = franka_reset_y
reset_z = franka_reset_z
scale_t = franka_t
# ====== 键盘状态 ======
accept_pose = True
paused = False
kb_lock = Lock()
b_real_so101=False
b_sim_franka=True

motor_keys = [
    "shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
    "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"
]

# ==========================================================
#                   SO101 EEPOS 控制器
# ==========================================================

class SO101EEPOSController1:
    def __init__(self, urdf_path, serial_port):
        if b_sim_franka : 
            self.franka = FrankaSimController("/home/mk/Downloads/Genesis/genesis/assets/xml/franka_sim/franka_panda.xml")
            print(">>> Franka MuJoCo 仿真已启动")
        config = SO101FollowerConfig(
            port=serial_port,
            id="follower2"
        )
        self.robot_arm = SO101Follower(config)
        self.robot_arm.connect()

        motor_names = list(self.robot_arm.bus.motors.keys())
        self.kin_solver = RobotKinematics(
            urdf_path, target_frame_name="gripper_frame_link", joint_names=motor_names
        )

        self.current_joint_pos = np.zeros(len(motor_names))

    def eepos_to_matrix(self, x, y, z, roll, pitch, yaw):
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(roll), -np.sin(roll)],
                       [0, np.sin(roll), np.cos(roll)]])
        Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                       [0, 1, 0],
                       [-np.sin(pitch), 0, np.cos(pitch)]])
        Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                       [np.sin(yaw), np.cos(yaw), 0],
                       [0, 0, 1]])

        R = Rz @ Ry @ Rx
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = [x, y, z]
        return T
    def set_frankatarget(self, x, y, z, roll=0, pitch=0, yaw=0):
        if(b_sim_franka):
            self.franka.set_target(x, y, z)
    def move_to_ee(self, x, y, z, roll=0, pitch=0, yaw=0):
        
        if( not b_real_so101):
           return
        
        target_pose = self.eepos_to_matrix(x, y, z, roll, pitch, yaw)
        print("current_joint_pos: ",self.current_joint_pos)
        joint_target_deg = self.kin_solver.inverse_kinematics(
            current_joint_pos=self.current_joint_pos,
            desired_ee_pose=target_pose
        )
        print("joint_target_deg: ",joint_target_deg)
        #print(f"移动: X={x:.3f}, Y={y:.3f}, Z={z:.3f} " ,target_pose," ",joint_target_deg)

        action = {k: v for k, v in zip(motor_keys, joint_target_deg)}
        self.robot_arm.send_action(action)
        present_pos = self.robot_arm.bus.sync_read("Present_Position")
        #print("present_pos: ",present_pos)
        self.current_joint_pos = list([present_pos[name] for name in present_pos])


        #self.current_joint_pos = joint_target_deg

    def go_home(self):
        """恢复到初始点"""
        print(">>> 回到初始位置")
        self.move_to_ee(reset_x, reset_y, reset_z, 0, 0, 0)
        if(b_sim_franka):
            self.franka.set_target(reset_x, reset_y, reset_z)


# ==========================================================
#                  ROS2 Odometry → SO101 控制
# ==========================================================
class OdomToSO101(Node):
    def __init__(self, urdf_path):
        super().__init__('odom_to_so101')
        self.controller = SO101EEPOSController1(urdf_path, "/dev/ttyACM2")

        self.subscription = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10
        )

        self.last_update_time = time.time()
        print(">>> 已启动 /odom 控制机械臂 ...")

    def odom_callback(self, msg):
        global accept_pose, paused

        p = msg.pose.pose.position
        x, y, z = p.x, p.y, p.z

        # 缩放 + 偏移
       
        x_scaled = reset_x - y / scale_t
        y_scaled = reset_y + x / scale_t
        z_scaled = reset_z + z / scale_t

        # 控制频率限制 50Hz
        if time.time() - self.last_update_time < 0.02:
            return

        with kb_lock:
            if paused:
                print(">>> [暂停] 不接收 pose")
                return
            if not accept_pose:
                print(">>> [大写模式] pose 被拒绝")
                return

        # 执行运动
        self.controller.move_to_ee(x_scaled, y_scaled, z_scaled, 0, 0, 0)
        self.controller.set_frankatarget(x_scaled, y_scaled, z_scaled)
        self.last_update_time = time.time()

        print(f"移动: X={x_scaled:.3f}, Y={y_scaled:.3f}, Z={z_scaled:.3f}")


# ==========================================================
#                     键盘监听线程
# ==========================================================
def on_key_press(key):
    global accept_pose, paused
    with kb_lock:
        # 空格暂停/恢复
        if key == keyboard.Key.space:
            paused = not paused
            print(">>> 暂停" if paused else ">>> 恢复")
            return

        # 字母类型
        if hasattr(key, "char") and key.char is not None:
            ch = key.char

            if ch.isalpha() and ch.isupper():
                accept_pose = False
                print(f">>> 检测到大写 [{ch}]：拒绝新的 pose + 回到初始位置")
                # 主线程中访问 controller 更安全，这里仅返回状态
                return

            if ch.isalpha() and ch.islower():
                if paused:
                    print(">>> 小写字母输入，但当前暂停")
                else:
                    accept_pose = True
                    print(f">>> 小写 [{ch}]：接收新的 pose")
                return


def start_keyboard_listener(node: OdomToSO101):
    """
    绑定 node，使得大写时可以调用 go_home()
    """
    def extended_on_key_press(key):
        global accept_pose, paused
        with kb_lock:
            if key == keyboard.Key.space:
                paused = not paused
                print(">>> 暂停" if paused else ">>> 恢复")
                return

            if hasattr(key, "char") and key.char is not None:
                ch = key.char

                if ch.isupper():
                    accept_pose = False
                    print(f">>> 检测到大写 [{ch}]：回到初始 + 拒绝 pose")
                    node.controller.go_home()
                    return

                if ch.islower():
                    if paused:
                        print(">>> 小写字母输入，但当前暂停")
                    else:
                        accept_pose = True
                        print(f">>> 小写 [{ch}]：允许接收 pose")
                    return

    listener = keyboard.Listener(on_press=extended_on_key_press)
    listener.daemon = True
    listener.start()
    print(">>> 键盘监听已启动")


# ==========================================================
#                       主入口
# ==========================================================
def main():
    rclpy.init()
    urdf_path = "./Simulation/SO101/so101_new_calib.urdf"
    node = OdomToSO101(urdf_path)

    start_keyboard_listener(node)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("停止运行")

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
