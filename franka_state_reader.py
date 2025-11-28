#!/usr/bin/env python

import threading
import time
import numpy as np

# 只有在ROS环境中才导入rospy
try:
    import rospy
    from geometry_msgs.msg import PoseStamped, Pose
    from sensor_msgs.msg import JointState
    from std_msgs.msg import Header
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False
    print("警告: ROS环境未找到，将无法发布ROS话题")

class FrankaStateReader:
    def __init__(self, robot, robot_name="franka", enable_ros_publishing=True):
        """
        初始化状态读取器
        Args:
            robot: Franka机器人实例
            robot_name: 机器人名称，用于话题命名
            enable_ros_publishing: 是否启用ROS发布
        """
        self.robot = robot
        self.robot_name = robot_name
        self.enable_ros_publishing = enable_ros_publishing and ROS_AVAILABLE
        
        # 控制变量
        self.is_running = False
        self.read_thread = None
        
        # 状态缓存
        self.current_pose = None
        self.current_joint_positions = None
        self.current_joint_velocities = None
        self.current_joint_torques = None
        self.last_timestamp = None
        
        # 互斥锁
        self.lock = threading.RLock()
        
        # ROS发布器（如果启用）
        if self.enable_ros_publishing:
            self.pose_pub = rospy.Publisher(f'/{robot_name}/cartesian_pose', PoseStamped, queue_size=10)
            self.joint_pub = rospy.Publisher(f'/{robot_name}/joint_states', JointState, queue_size=10)
            rospy.loginfo(f"Franka状态读取器初始化完成: {robot_name}")
        else:
            self.pose_pub = None
            self.joint_pub = None
            print(f"Franka状态读取器初始化完成: {robot_name} (无ROS发布)")

    def read_robot_state(self):
        """读取机器人状态（线程安全）"""
        with self.lock:
            try:
                # 读取笛卡尔位姿
                cartesian_state = self.robot.current_cartesian_state
                robot_pose = cartesian_state.pose
                ee_pose = robot_pose.end_effector_pose
                
                position = ee_pose.translation.flatten()
                quaternion = ee_pose.quaternion.flatten()
                
                # 读取关节状态 - 使用正确的属性名
                joint_state = self.robot.current_joint_state
                
                # 根据调试输出，属性名是 position 和 velocity
                positions = joint_state.position.flatten()  # 关节位置
                velocities = joint_state.velocity.flatten()  # 关节速度
                
                # 力矩信息可能不可用，设为None
                torques = None
                
                timestamp = time.time()
                
                # 更新缓存
                self.current_pose = {
                    'position': position,
                    'quaternion': quaternion,
                    'timestamp': timestamp
                }
                
                self.current_joint_positions = positions
                self.current_joint_velocities = velocities
                self.current_joint_torques = torques
                self.last_timestamp = timestamp
                
                return True
                
            except Exception as e:
                if self.enable_ros_publishing:
                    rospy.logwarn(f"读取机器人状态失败: {e}")
                else:
                    print(f"读取机器人状态失败: {e}")
                return False

    def publish_robot_state(self):
        """发布机器人状态到ROS"""
        if not self.enable_ros_publishing or self.current_pose is None:
            return
        
        try:
            # 转换为ROS时间
            ros_time = rospy.Time.from_sec(self.current_pose['timestamp'])
            
            # 发布位姿
            pose_msg = PoseStamped()
            pose_msg.header = Header(stamp=ros_time, frame_id=f"{self.robot_name}_link0")
            
            pos = self.current_pose['position']
            quat = self.current_pose['quaternion']
            
            pose_msg.pose.position.x = float(pos[0])
            pose_msg.pose.position.y = float(pos[1])
            pose_msg.pose.position.z = float(pos[2])
            
            pose_msg.pose.orientation.x = float(quat[0])
            pose_msg.pose.orientation.y = float(quat[1])
            pose_msg.pose.orientation.z = float(quat[2])
            pose_msg.pose.orientation.w = float(quat[3])
            
            self.pose_pub.publish(pose_msg)
            
            # 发布关节状态
            if self.current_joint_positions is not None:
                joint_msg = JointState()
                joint_msg.header = Header(stamp=ros_time, frame_id=f"{self.robot_name}_link0")
                
                joint_msg.name = [f'{self.robot_name}_joint{i+1}' for i in range(7)]
                joint_msg.position = [float(p) for p in self.current_joint_positions]
                
                if self.current_joint_velocities is not None:
                    joint_msg.velocity = [float(v) for v in self.current_joint_velocities]
                else:
                    joint_msg.velocity = [0.0] * 7
                    
                # 力矩信息可能不可用，设为0
                joint_msg.effort = [0.0] * 7
                
                self.joint_pub.publish(joint_msg)
            
        except Exception as e:
            if self.enable_ros_publishing:
                rospy.logwarn(f"发布机器人状态失败: {e}")
            else:
                print(f"发布机器人状态失败: {e}")

    def read_and_publish_loop(self):
        """读取和发布循环（30Hz）"""
        seq = 0
        
        while self.is_running and not rospy.is_shutdown():
            try:
                # 读取状态
                if self.read_robot_state():
                    # 发布状态（如果启用ROS）
                    self.publish_robot_state()
                    
                    # 每2秒打印一次状态
                    if seq % 60 == 0:  # 30Hz * 2 = 60
                        if self.current_pose:
                            pos = self.current_pose['position']
                            joint_info = ""
                            if self.current_joint_positions is not None:
                                joint_info = f", 关节: {[f'{p:.3f}' for p in self.current_joint_positions]}"
                            
                            if self.enable_ros_publishing:
                                rospy.loginfo(f"{self.robot_name} - 位置: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]{joint_info}")
                            else:
                                print(f"{self.robot_name} - 位置: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]{joint_info}")
                
                seq += 1
                time.sleep(1.0/30.0)  # 30Hz
                
            except Exception as e:
                if self.enable_ros_publishing:
                    rospy.logwarn(f"状态读取循环出错: {e}")
                else:
                    print(f"状态读取循环出错: {e}")
                time.sleep(1.0/30.0)

    def start_reading(self):
        """开始读取状态"""
        if self.is_running:
            if self.enable_ros_publishing:
                rospy.logwarn(f"{self.robot_name} 读取器已经在运行")
            else:
                print(f"{self.robot_name} 读取器已经在运行")
            return
        
        self.is_running = True
        self.read_thread = threading.Thread(target=self.read_and_publish_loop)
        self.read_thread.daemon = True
        self.read_thread.start()
        
        if self.enable_ros_publishing:
            rospy.loginfo(f"开始读取 {self.robot_name} 状态 (30Hz)")
        else:
            print(f"开始读取 {self.robot_name} 状态 (30Hz)")

    def stop_reading(self):
        """停止读取状态"""
        self.is_running = False
        if self.read_thread and self.read_thread.is_alive():
            self.read_thread.join(timeout=2.0)
        
        if self.enable_ros_publishing:
            rospy.loginfo(f"停止读取 {self.robot_name} 状态")
        else:
            print(f"停止读取 {self.robot_name} 状态")

    def get_current_pose(self):
        """获取当前位姿"""
        return self.current_pose

    def get_current_joint_states(self):
        """获取当前关节状态"""
        return {
            'positions': self.current_joint_positions,
            'velocities': self.current_joint_velocities,
            'torques': self.current_joint_torques,
            'timestamp': self.last_timestamp
        }
