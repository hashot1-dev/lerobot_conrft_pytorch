#!/usr/bin/env python3
import socket
import json
import threading
import time

from franky import *
import numpy as np
from scipy.spatial.transform import Rotation
from pynput import keyboard  # 需要安装: pip install pynput
from franka_state_reader import FrankaStateReader
import rospy
# 网络通信设置
SERVER_IP = "192.168.2.177"
SERVER_PORT = 5000
scale = 1
ROBOT_IP_L="172.16.0.3"
ROBOT_IP_R="172.16.0.2"

# 角度         实际机器人(度)        网络传来(度)         差值(度)           0.50909722  0.04383123  0.29640619
# --------------------------------------------------------------------------------
# Roll               178.67°          -6.22°         184.89°
# Pitch               -0.95°           1.06°           2.01°
# Yaw                  0.22°         -88.87°          89.09°


# 安全范围限制
SAFE_WORKSPACE_LIMITS = {
    'x_min': 0.37189472,
    'x_max': 0.638869037,
    'y_min': -0.1432448,
    'y_max': 0.400865579,
    'z_min': 0.13485752,
    'z_max': 0.350967397
}

# 全局控制变量
move_enabled = False
pause_event = threading.Event()
current_received_pose = None
pose_lock = threading.Lock()
exit_program = False

# 安全范围检查函数
def check_position_safety(position):
    """
    检查位置是否在安全范围内
    返回: (是否安全, 调整后的位置)
    """
    x, y, z = position
    
    # 检查每个坐标轴是否在安全范围内
    x_safe = max(SAFE_WORKSPACE_LIMITS['x_min'], min(SAFE_WORKSPACE_LIMITS['x_max'], x))
    y_safe = max(SAFE_WORKSPACE_LIMITS['y_min'], min(SAFE_WORKSPACE_LIMITS['y_max'], y))
    z_safe = max(SAFE_WORKSPACE_LIMITS['z_min'], min(SAFE_WORKSPACE_LIMITS['z_max'], z))
    
    # 判断是否需要调整
    needs_adjustment = (x != x_safe) or (y != y_safe) or (z != z_safe)
    
    safe_position = np.array([x_safe, y_safe, z_safe])
    
    if needs_adjustment:
        print(f"⚠️  位置超出安全范围，已调整:")
        print(f"   原始位置: [{x:.4f}, {y:.4f}, {z:.4f}]")
        print(f"   调整后位置: [{x_safe:.4f}, {y_safe:.4f}, {z_safe:.4f}]")
        print(f"   安全范围: X[{SAFE_WORKSPACE_LIMITS['x_min']:.4f}-{SAFE_WORKSPACE_LIMITS['x_max']:.4f}], "
              f"Y[{SAFE_WORKSPACE_LIMITS['y_min']:.4f}-{SAFE_WORKSPACE_LIMITS['y_max']:.4f}], "
              f"Z[{SAFE_WORKSPACE_LIMITS['z_min']:.4f}-{SAFE_WORKSPACE_LIMITS['z_max']:.4f}]")
    
    return not needs_adjustment, safe_position

def get_workspace_center():
    """获取工作空间中心点"""
    center_x = (SAFE_WORKSPACE_LIMITS['x_min'] + SAFE_WORKSPACE_LIMITS['x_max']) / 2
    center_y = (SAFE_WORKSPACE_LIMITS['y_min'] + SAFE_WORKSPACE_LIMITS['y_max']) / 2
    center_z = (SAFE_WORKSPACE_LIMITS['z_min'] + SAFE_WORKSPACE_LIMITS['z_max']) / 2
    return np.array([center_x, center_y, center_z])

# 夹爪控制函数
def open_gripper(gripper):
    """打开夹爪"""
    try:
        speed = 0.1  # [m/s]
        gripper.open(speed)
        print("=== 夹爪已打开 ===")
    except Exception as e:
        print(f"打开夹爪错误: {e}")

def close_gripper(gripper):
    """关闭夹爪"""
    try:
        speed = 0.1  # [m/s]
        force = 20.0  # [N] 夹持力
        success = gripper.grasp(0.0, speed, force, epsilon_outer=0.1, epsilon_inner=0.1)
        print(f"=== 夹爪已关闭 (成功: {success}) ===")
    except Exception as e:
        print(f"关闭夹爪错误: {e}")

def move_gripper(gripper, width):
    """移动夹爪到指定宽度"""
    try:
        speed = 0.1  # [m/s]
        success = gripper.move(width, speed)
        print(f"夹爪移动到宽度 {width:.3f}m (成功: {success})")
    except Exception as e:
        print(f"移动夹爪错误: {e}")

# 机器人控制函数
def get_pose(robot):
    cartesian_state = robot.current_cartesian_state
    robot_pose = cartesian_state.pose
    ee_pose = robot_pose.end_effector_pose
    elbow_pos = robot_pose.elbow_state

    pose = np.concatenate([ee_pose.translation, ee_pose.quaternion])
    pose = pose.flatten()

    return pose

def pose_to_rpy(pose):
    """
    将位姿 [x, y, z, qx, qy, qz, qw] 转换为RPY角度 (弧度)
    返回: (roll, pitch, yaw) 弧度
    """
    quaternion = pose[3:]
    
    # 检查四元数是否有效
    if np.linalg.norm(quaternion) < 1e-6:
        return 0.0, 0.0, 0.0
    
    # 创建旋转对象
    rot = Rotation.from_quat(quaternion)
    
    # 转换为RPY (ZYX顺序对应yaw-pitch-roll)
    yaw, pitch, roll = rot.as_euler('zyx', degrees=False)
    
    return roll, pitch, yaw

def rpy_to_degrees(roll, pitch, yaw):
    """将RPY弧度转换为角度"""
    return np.degrees(roll), np.degrees(pitch), np.degrees(yaw)

def normalize_quaternion(quat):
    """归一化四元数"""
    norm = np.linalg.norm(quat)
    if norm < 1e-6:
        return np.array([0.0, 0.0, 0.0, 1.0])  # 返回单位四元数
    return quat / norm

def goto_pose_smooth(robot, pose, velocity=0.05, acceleration=0.02, jerk=0.1): 
    """平滑移动函数，避免加速度不连续"""
    # 保存当前动态参数
    original_dynamics = robot.relative_dynamics_factor
    
    try:
        # 设置平滑的动态参数
        robot.relative_dynamics_factor = RelativeDynamicsFactor(
            velocity=velocity, 
            acceleration=acceleration, 
            jerk=jerk
        )
        
        pos = pose[:3]
        quat = pose[3:]
        motion = CartesianMotion(Affine(pos, quat))
        move_start_time = time.time()
        robot.move(motion)
        move_execution_time = time.time() - move_start_time   
        return move_execution_time
        
    except Exception as e:
        print(f"移动错误: {e}")
        # 尝试从错误中恢复
        try:
            robot.recover_from_errors()
        except:
            pass
    finally:
        # 恢复原始动态参数
        robot.relative_dynamics_factor = original_dynamics
def update_and_move_pose(robot, received_pose_dict, initial_pose):
    """
    根据接收到的pose字典，设置位置和姿态
    位置：将xyz除以scale后加到初始位置上
    姿态：根据网络传来的RPY进行转换
    """
    # 从字典中提取xyz坐标
    pose_data = received_pose_dict['pose']
    
    # 计算位置
    received_pos = np.array([
        -pose_data['y'],  # 注意：这里仍然有坐标轴转换
        -pose_data['x'], 
        pose_data['z']
    ])
    
    # 将接收到的位置除以scale
    scaled_pos = received_pos / scale
    
    # 在初始位置的基础上加上缩放位置
    initial_position = initial_pose[:3]
    new_position = initial_position + scaled_pos
    
    # 检查位置安全性并调整
    is_safe, safe_position = check_position_safety(new_position)
    
    # 获取网络传来的RPY并转换为目标姿态
    target_roll_deg = target_pitch_deg = target_yaw_deg = 0.0
    
    if 'roll' in pose_data and 'pitch' in pose_data and 'yaw' in pose_data:
        # 直接获取RPY角度
        network_roll_deg = pose_data['roll']
        network_pitch_deg = pose_data['pitch']
        network_yaw_deg = pose_data['yaw']
        
        # 根据要求进行转换：
        # 180 - 网络传过来的roll 设为目标roll
        # 网络传过来的pitch 设为目标pitch  
        # 90 + 网络传过来的yaw 设为目标yaw
        target_roll_deg = 180 - network_roll_deg
        target_pitch_deg = network_pitch_deg
        target_yaw_deg = 90 + network_yaw_deg
        
        print(f"网络RPY: Roll={network_roll_deg:.2f}°, Pitch={network_pitch_deg:.2f}°, Yaw={network_yaw_deg:.2f}°")
        print(f"目标RPY: Roll={target_roll_deg:.2f}°, Pitch={target_pitch_deg:.2f}°, Yaw={target_yaw_deg:.2f}°")
        
    elif 'qx' in pose_data and 'qy' in pose_data and 'qz' in pose_data and 'qw' in pose_data:
        # 从四元数计算RPY
        try:
            network_quat = [pose_data['qx'], pose_data['qy'], pose_data['qz'], pose_data['qw']]
            network_quat = normalize_quaternion(network_quat)
            network_pose = np.concatenate([np.array([0.0, 0.0, 0.0]), network_quat])
            network_roll, network_pitch, network_yaw = pose_to_rpy(network_pose)
            network_roll_deg, network_pitch_deg, network_yaw_deg = rpy_to_degrees(network_roll, network_pitch, network_yaw)
                
            # 根据要求进行转换
            target_roll_deg = 180 - network_roll_deg
            target_pitch_deg = network_pitch_deg
            target_yaw_deg = 90 + network_yaw_deg

            target_roll_deg = - target_roll_deg
            target_pitch_deg = - target_pitch_deg
            target_yaw_deg= - target_yaw_deg
            
            
            print(f"网络RPY(从四元数): Roll={network_roll_deg:.2f}°, Pitch={network_pitch_deg:.2f}°, Yaw={network_yaw_deg:.2f}°")
            print(f"目标RPY: Roll={target_roll_deg:.2f}°, Pitch={target_pitch_deg:.2f}°, Yaw={target_yaw_deg:.2f}°")
            
        except Exception as e:
            print(f"从四元数计算RPY错误: {e}")
            # 如果转换失败，使用初始姿态
            target_quat = initial_pose[3:]
    else:
        # 如果没有RPY信息，使用初始姿态
        print("没有RPY信息，使用初始姿态")
        target_quat = initial_pose[3:]
    
    # 如果有目标RPY，转换为四元数
    if target_roll_deg != 0.0 or target_pitch_deg != 0.0 or target_yaw_deg != 0.0:
        target_roll_rad = np.radians(target_roll_deg)
        target_pitch_rad = np.radians(target_pitch_deg)
        target_yaw_rad = np.radians(target_yaw_deg)
        
        # 创建旋转对象并转换为四元数
        target_rot = Rotation.from_euler('zyx', [target_yaw_rad, target_pitch_rad, target_roll_rad], degrees=False)
        target_quat = target_rot.as_quat()
    
    # 构建新的位姿（位置+姿态）
    new_pose = np.concatenate([safe_position, target_quat])
    
    # 计算时间戳差值
    current_time = time.time()
    odom_ts = received_pose_dict['odom_ts']
    send_ts = received_pose_dict['send_ts']
    
    time_diff_current_odom = current_time - odom_ts
    time_diff_send_odom = send_ts - odom_ts
    
    print(f"接收位置: [{pose_data['x']:.4f}, {pose_data['y']:.4f}, {pose_data['z']:.4f}]")
    
    # 使用平滑移动
    move_start_time = time.time()
    goto_pose_async(robot, new_pose, velocity=0.1, acceleration=0.05, jerk=0.2)
    du = 0.0
    move_execution_time = time.time() - move_start_time
    
    print(f"缩放偏移: [{scaled_pos[0]:.4f}, {scaled_pos[1]:.4f}, {scaled_pos[2]:.4f}] "
          f"数据延迟: {time_diff_current_odom:.4f}秒, "
          f"运动执行: {move_execution_time:.4f}秒 "
          f"1运动执行: {du:.4f}秒")
    
    if not is_safe:
        print("⚠️  注意: 目标位置已调整到安全范围内")
    
    return new_pose

def print_pose_comparison(robot, received_pose_dict):
    """
    打印实际机器人位姿和网络传来的位姿的RPY对比
    """
    # 获取实际机器人位姿和RPY
    actual_pose = get_pose(robot)
    actual_roll, actual_pitch, actual_yaw = pose_to_rpy(actual_pose)
    actual_roll_deg, actual_pitch_deg, actual_yaw_deg = rpy_to_degrees(actual_roll, actual_pitch, actual_yaw)
    
    # 获取网络传来的RPY
    network_roll_deg = network_pitch_deg = network_yaw_deg = 0.0
    has_network_rpy = False
    
    if received_pose_dict and 'pose' in received_pose_dict:
        pose_data = received_pose_dict['pose']
        
        # 检查是否有直接的rpy信息
        if 'roll' in pose_data and 'pitch' in pose_data and 'yaw' in pose_data:
            network_roll_deg = pose_data['roll']
            network_pitch_deg = pose_data['pitch']
            network_yaw_deg = pose_data['yaw']
            has_network_rpy = True
        # 检查是否有四元数信息
        elif 'qx' in pose_data and 'qy' in pose_data and 'qz' in pose_data and 'qw' in pose_data:
            try:
                network_quat = [pose_data['qx'], pose_data['qy'], pose_data['qz'], pose_data['qw']]
                # 归一化四元数
                network_quat = normalize_quaternion(network_quat)
                network_pose = np.concatenate([np.array([0.0, 0.0, 0.0]), network_quat])
                network_roll, network_pitch, network_yaw = pose_to_rpy(network_pose)
                network_roll_deg, network_pitch_deg, network_yaw_deg = rpy_to_degrees(network_roll, network_pitch, network_yaw)
                has_network_rpy = True
            except Exception as e:
                print(f"网络四元数转换错误: {e}")
                has_network_rpy = False
    
    # 打印对比信息
    print("\n" + "="*80)
    print("位姿RPY对比:")
    print("="*80)
    
    if has_network_rpy:
        print(f"{'角度':<10} {'实际机器人(度)':<15} {'网络传来(度)':<15} {'差值(度)':<15}")
        print("-"*80)
        print(f"{'Roll':<10} {actual_roll_deg:>14.2f}° {network_roll_deg:>14.2f}° {abs(actual_roll_deg - network_roll_deg):>14.2f}°")
        print(f"{'Pitch':<10} {actual_pitch_deg:>14.2f}° {network_pitch_deg:>14.2f}° {abs(actual_pitch_deg - network_pitch_deg):>14.2f}°")
        print(f"{'Yaw':<10} {actual_yaw_deg:>14.2f}° {network_yaw_deg:>14.2f}° {abs(actual_yaw_deg - network_yaw_deg):>14.2f}°")
    else:
        print(f"{'角度':<10} {'实际机器人(度)':<15}")
        print("-"*80)
        print(f"{'Roll':<10} {actual_roll_deg:>14.2f}°")
        print(f"{'Pitch':<10} {actual_pitch_deg:>14.2f}°")
        print(f"{'Yaw':<10} {actual_yaw_deg:>14.2f}°")
        print("网络传来的RPY数据不可用")
    
    # 打印位置信息
    print("\n位置信息:")
    print(f"实际机器人位置: [{actual_pose[0]:.4f}, {actual_pose[1]:.4f}, {actual_pose[2]:.4f}]")
    if received_pose_dict and 'pose' in received_pose_dict:
        pose_data = received_pose_dict['pose']
        print(f"网络传来位置: [{pose_data.get('x', 0):.4f}, {pose_data.get('y', 0):.4f}, {pose_data.get('z', 0):.4f}]")
    
    print("="*80)

def on_press(key):
    """键盘按下事件处理"""
    global move_enabled, exit_program
    
    try:
        if key == keyboard.Key.space:
            # 空格键切换移动状态
            move_enabled = not move_enabled
            if move_enabled:
                print("\n=== 开始移动 ===")
                pause_event.set()  # 清除暂停状态
            else:
                print("\n=== 暂停移动 ===")
                pause_event.clear()  # 设置暂停状态
        
        elif key == keyboard.Key.esc:
            # ESC键退出程序
            print("\n=== 退出程序 ===")
            exit_program = True
            return False  # 停止监听
            
        elif hasattr(key, 'char'):
            # 字符键处理
            if key.char == '+':
                # 加号键打开夹爪
                print("\n=== 打开夹爪 ===")
                open_gripper(gripper)
                
            elif key.char == '-':
                # 减号键关闭夹爪
                print("\n=== 关闭夹爪 ===")
                close_gripper(gripper)
                
            elif key.char == 'o':
                # O键打开夹爪（备用）
                print("\n=== 打开夹爪 (O键) ===")
                open_gripper(gripper)
                
            elif key.char == 'c':
                # C键关闭夹爪（备用）
                print("\n=== 关闭夹爪 (C键) ===")
                close_gripper(gripper)
            
    except AttributeError:
        # 忽略特殊键
        pass

def keyboard_listener():
    """键盘监听线程函数"""
    print("键盘监听已启动")
    print("空格键: 开始/暂停移动")
    print("加号键(+): 打开夹爪")
    print("减号键(-): 关闭夹爪")
    print("ESC键: 退出程序")
    print(f"安全范围: X[{SAFE_WORKSPACE_LIMITS['x_min']:.4f}-{SAFE_WORKSPACE_LIMITS['x_max']:.4f}]")
    print(f"          Y[{SAFE_WORKSPACE_LIMITS['y_min']:.4f}-{SAFE_WORKSPACE_LIMITS['y_max']:.4f}]")
    print(f"          Z[{SAFE_WORKSPACE_LIMITS['z_min']:.4f}-{SAFE_WORKSPACE_LIMITS['z_max']:.4f}]")
    
    # 设置键盘监听
    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()

def network_receiver(sock):
    """接收网络数据的线程函数"""
    global current_received_pose
    
    buffer = ""
    
    try:
        while not exit_program:
            try:
                data = sock.recv(1024)
                if not data:
                    print("网络连接断开")
                    break

                buffer += data.decode()

                # 按行解析（每个pose一行）
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    if len(line.strip()) == 0:
                        continue
                    
                    try:
                        received_pose_dict = json.loads(line)
                        #print("received_pose_dict")
                        # 更新当前接收到的pose
                        with pose_lock:
                            current_received_pose = received_pose_dict
                            #print("更新当前接收到的pose")
                        
                    except json.JSONDecodeError as e:
                        print(f"JSON解析错误: {e}")
                    except KeyError as e:
                        print(f"字典键错误，确保包含x,y,z字段: {e}")
                        
            except socket.timeout:
                # 超时是正常的，继续循环
                continue
            except Exception as e:
                if not exit_program:
                    print(f"网络接收错误: {e}")
                break
                    
    except Exception as e:
        if not exit_program:
            print(f"网络接收线程错误: {e}")

def robot_controller(robot, initial_pose):
    """机器人控制线程函数"""
    global move_enabled, current_received_pose, exit_program
    
    print("机器人控制器已启动")
    
    # 添加移动频率控制
    last_move_time = 0
    move_interval = 0.1  # 100ms移动一次，避免过于频繁
    last_print_time = 0
    print_interval = 2.0  # 每2秒打印一次对比信息
    
    while not exit_program:
        current_time = time.time()
        
        # 定期打印位姿对比信息
        if current_time - last_print_time >= print_interval:
            with pose_lock:
                received_pose = current_received_pose
            try:
                print_pose_comparison(robot, received_pose)
            except Exception as e:
                print(f"打印位姿对比时出错: {e}")
            last_print_time = current_time
        
        # 等待移动使能且有新数据
        if move_enabled and pause_event.is_set() and (current_time - last_move_time) >= move_interval:
            
            # 检查是否有新的pose数据
            with pose_lock:
                current_received_pose_copy = current_received_pose  # 复制数据
            
            if current_received_pose_copy is not None:
                try:
                    # 更新并移动机器人
                    current_pose = update_and_move_pose(robot, current_received_pose_copy, initial_pose)
                    last_move_time = current_time
                    
                    # 每次移动后也打印对比信息
                    try:
                        print_pose_comparison(robot, current_received_pose_copy)
                    except Exception as e:
                        print(f"移动后打印位姿对比时出错: {e}")
                    
                except Exception as e:
                    print(f"移动错误: {e}")
                    # 尝试恢复
                    try:
                        robot.recover_from_errors()
                    except:
                        pass
        
        # 短暂休眠避免过度占用CPU
        time.sleep(0.01)

def goto_pose_async(robot, pose, velocity=0.05, acceleration=0.02, jerk=0.1): 
    """异步移动函数，不阻塞主线程"""
    # 保存当前动态参数
    original_dynamics = robot.relative_dynamics_factor
    
    try:
        # 设置平滑的动态参数
        robot.relative_dynamics_factor = RelativeDynamicsFactor(
            velocity=velocity, 
            acceleration=acceleration, 
            jerk=jerk
        )
        
        pos = pose[:3]
        quat = pose[3:]
        motion = CartesianMotion(Affine(pos, quat))
        
        # 使用异步移动
        robot.move(motion, asynchronous=True)
        
    except Exception as e:
        print(f"异步移动错误: {e}")
        try:
            robot.recover_from_errors()
        except:
            pass
    finally:
        # 恢复原始动态参数
        robot.relative_dynamics_factor = original_dynamics

def main():
    global exit_program, gripper
    
    # 初始化机器人
    robot_L = Robot(ROBOT_IP_L)
    gripper = Gripper(ROBOT_IP_L)

    robot_R = Robot(ROBOT_IP_R)
    gripper_R = Gripper(ROBOT_IP_R)

    #robot_L.set_collision_behavior(500)
    #robot_R.set_collision_behavior(500)

    
    # 设置更保守的默认动态参数
    robot_L.relative_dynamics_factor = RelativeDynamicsFactor(
        velocity=0.1, acceleration=0.02, jerk=0.05
    )
    robot_R.relative_dynamics_factor = RelativeDynamicsFactor(
        velocity=0.1, acceleration=0.02, jerk=0.05
    )
    
    robot_L.recover_from_errors()
    robot_R.recover_from_errors()
    
    # 初始化夹爪 - 先打开
    print("初始化夹爪...")
    open_gripper(gripper)
    open_gripper(gripper_R)
    rospy.init_node("MYMY", anonymous=True)
    state_reader_L = FrankaStateReader(robot_L, "franka_L")
    state_reader_R = FrankaStateReader(robot_R, "franka_R")
    #bridge = CvBridge()
    pipelines = []
    
    # 设置初始姿态（确保在安全范围内）
    workspace_center = get_workspace_center()
    initial_pose = np.array([
        workspace_center[0],  # x
        workspace_center[1],  # y  
        workspace_center[2],  # z
       # 9.99902337e-01, -1.91051298e-03, -7.81864347e-03, 1.14252157e-02  # 四元数
       0.99466292 ,-0.08748926 , 0.02287878,-0.04967755
    ])
    
    print(f"工作空间中心: [{workspace_center[0]:.4f}, {workspace_center[1]:.4f}, {workspace_center[2]:.4f}]")
    
    # 先移动到初始位置（使用平滑移动）
    print("移动到初始位置...")
    #goto_pose_smooth(robot, initial_pose, velocity=0.05, acceleration=0.01, jerk=0.02)
    
    # 连接到服务器
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(1.0)  # 设置超时
    print(f"[CLIENT] Connecting to {SERVER_IP}:{SERVER_PORT} ...")
    try:
        sock.connect((SERVER_IP, SERVER_PORT))
        print("[CLIENT] Connected.")
    except Exception as e:
        print(f"连接失败: {e}")
        return
    
    # 设置初始状态为暂停
    pause_event.clear()
    move_enabled = False
        
    try:
        # 创建并启动线程
        keyboard_thread = threading.Thread(target=keyboard_listener, daemon=True)
        network_thread = threading.Thread(target=network_receiver, args=(sock,), daemon=True)
        robot_L_thread = threading.Thread(target=robot_controller, args=(robot_L, initial_pose), daemon=True)
        robot_R_thread = threading.Thread(target=robot_controller, args=(robot_R, initial_pose), daemon=True)
        
        keyboard_thread.start()
        network_thread.start()
        robot_L_thread.start()

        state_reader_L.start_reading()
        state_reader_R.start_reading()
        
        # 你原有的其他启动代码
        

        #robot_R_thread.start()
        
        print("系统已就绪，按空格键开始移动...")
        
        # 主线程等待退出信号
        while not exit_program:
            time.sleep(0.1)
        
    except KeyboardInterrupt:
        print("程序被用户中断")
        exit_program = True
    except Exception as e:
        print(f"发生错误: {e}")
        exit_program = True
    finally:
        # 清理资源
        sock.close()
        # 程序退出前打开夹爪确保安全
        try:
            open_gripper(gripper)
        except:
            pass
        print("连接关闭")
        
        # 等待线程结束
        time.sleep(0.5)
        print("程序退出")

if __name__ == "__main__":
    main()
