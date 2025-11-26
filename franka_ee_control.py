import mujoco
import mujoco.viewer
import numpy as np
import threading
import time


class FrankaSimController:
    def __init__(self, model_path):
        # 加载模型
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.targetx=0.0
        self.targety=0.0
        self.targetz=0.0
        # 查找名为 hand 的 site
        self.sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "end_effector")
        self.ee_site = self.model.site('end_effector')
        print("site id =", self.sid)

        # 用户目标
        self.target = np.array([0.4, 0.0, 0.4])
        self.running = True

        # 启动模拟线程
        self.sim_thread = threading.Thread(target=self.run_sim)
        self.sim_thread.daemon = True
        self.sim_thread.start()


    import numpy as np

    def move_to_ee_ik_pd(self, target_pose, dt=0.01, kp=50.0, kd=1.0, max_step=0.05):
        """
        1) 使用 IK 得到目标关节角
        2) 用 PD 控制生成 ctrl
        3) 写入 self.data.ctrl 并调用 mujoco.mj_step
        target_pose: 4x4 ndarray or [x,y,z,roll,pitch,yaw]
        dt: 步进时间（秒）
        """
        mkeys = getattr(self, "motor_keys", list(self.robot_arm.bus.motors.keys()))
        has_gripper = ("grip" in mkeys[-1].lower() or "hand" in mkeys[-1].lower())
        n_arm = len(mkeys)-1 if has_gripper else len(mkeys)

        # --- 1) IK 求解 ---
        q_cur = np.asarray(self.current_joint_pos[:n_arm], dtype=float)
        q_des = self.kin_solver.inverse_kinematics(current_joint_pos=self.current_joint_pos,
                                                desired_ee_pose=target_pose)
        q_des = np.asarray(q_des[:n_arm], dtype=float)

        # --- 2) 计算 PD 控制 ---
        # 位置误差
        q_err = q_des - q_cur
        # clip 最大角度步长，避免抖动
        q_err = np.clip(q_err, -max_step, max_step)
        # 速度误差
        q_dot = (q_err) / dt
        # PD 控制
        ctrl = kp * q_err - kd * q_dot

        # --- 3) 写入 self.data.ctrl ---
        for i in range(self.model.nu):
            if i < n_arm:
                self.data.ctrl[i] = float(ctrl[i])
            else:
                # gripper 或多余关节
                self.data.ctrl[i] = 0.0

        # --- 4) MuJoCo 推进一步 ---
        mujoco.mj_step(self.model, self.data)

        # --- 5) 更新 current_joint_pos ---
        self.current_joint_pos[:n_arm] = q_cur + q_err

        # --- 6) 可选：打印信息 ---
        ee_pos = self.data.site_xpos[self.sid]  # 末端位置
        print("target:", target_pose[:3,3] if isinstance(target_pose, np.ndarray) else target_pose[:3],
            "ee_pos:", ee_pos,
            "err:", ee_pos - target_pose[:3,3])

    def _choose_best_ik_solution(self, solutions):
        """
        如果 inverse_kinematics 返回多个解，选择与 current_joint_pos 距离最近的。
        solutions: list or array of shape (k, n)
        """
        cur = np.asarray(self.current_joint_pos, dtype=float)
        sols = np.asarray(solutions, dtype=float)
        if sols.ndim == 1:
            return sols
        # ensure same length: trim/pad cur to sols width
        n = sols.shape[1]
        cur_trim = cur[:n] if cur.size >= n else np.concatenate([cur, np.zeros(n-cur.size)])
        idx = np.argmin(np.linalg.norm(sols - cur_trim, axis=1))
        return sols[idx]

    def move_to_ee_using_ik(self, x, y, z, roll=0.0, pitch=0.0, yaw=0.0, gripper=None):
        """
        1) 调用 inverse_kinematics 得到目标关节角（可以是多个解）
        2) 选择最接近当前关节的解
        3) 用 joint-space PD 平滑跟踪（send_action 发送角度命令）
        外部应以固定频率（20-50Hz）重复调用本函数
        """
        # params (可调)
        JOINT_KP = getattr(self, "JOINT_KP", 6.0)    # P gain (deg or rad unit consistent with send)
        JOINT_KD = getattr(self, "JOINT_KD", 0.5)    # D gain
        MAX_STEP_RAD = getattr(self, "MAX_STEP_RAD", 0.05)  # per-step max angle change (rad)
        POS_THRESH = getattr(self, "POS_THRESH", 0.003)     # m deadzone

        # prepare gripper
        if gripper is not None:
            self.gripper_pos = float(gripper)

        # get current joint vector (arm joints only)
        mkeys = getattr(self, "motor_keys", list(self.robot_arm.bus.motors.keys()))
        has_gripper = ("grip" in mkeys[-1].lower() or "hand" in mkeys[-1].lower())
        n_arm = len(mkeys) - 1 if has_gripper else len(mkeys)
        q_cur = np.asarray(self.current_joint_pos, dtype=float)[:n_arm]

        # quick FK check: if current end-effector already close, optionally just update gripper
        # try FK via kin_solver (many names attempted)
        def try_fk(q):
            names = ["forward_kinematics", "fk", "fkine", "get_end_effector_pose", "get_pose"]
            for name in names:
                if hasattr(self.kin_solver, name):
                    try:
                        res = getattr(self.kin_solver, name)(q)
                        arr = np.array(res)
                        if arr.shape == (4,4):
                            return arr[:3,3]
                        if arr.size == 3:
                            return arr.reshape(3)
                    except Exception:
                        continue
            # not available -> return None
            return None

        pcur = try_fk(q_cur)
        if pcur is not None:
            if np.linalg.norm(np.array([x,y,z]) - pcur) < POS_THRESH:
                # 在死区内，仅发送 gripper变化（若有）
                if has_gripper:
                    action = {}
                    for i,k in enumerate(mkeys[:n_arm]):
                        action[k] = float(q_cur[i])
                    action[mkeys[-1]] = float(self.gripper_pos)
                    self.robot_arm.send_action(action)
                return

        # --- get IK solutions: try to call inverse_kinematics in various ways ---
        ik_args = {"current_joint_pos": self.current_joint_pos, "desired_ee_pose": None}
        # build desired_ee_pose as 4x4 matrix
        T = np.eye(4)
        T[:3,3] = [x,y,z]
        # pack rotation if user expects it (but many inverse_kinematics ignore orientation)
        # For safety pass T or (x,y,z,roll,pitch,yaw)
        sol = None
        try:
            # try direct call forms
            if hasattr(self.kin_solver, "inverse_kinematics"):
                try:
                    sols = self.kin_solver.inverse_kinematics(current_joint_pos=self.current_joint_pos,
                                                            desired_ee_pose=T)
                    sol = sols
                except TypeError:
                    # maybe signature is (T) or (xyz, rpy)
                    try:
                        sols = self.kin_solver.inverse_kinematics(T)
                        sol = sols
                    except Exception:
                        try:
                            sols = self.kin_solver.inverse_kinematics(x, y, z, roll, pitch, yaw)
                            sol = sols
                        except Exception:
                            sol = None
            else:
                # try other names
                for name in ("ik", "solve_ik", "inverse"):
                    if hasattr(self.kin_solver, name):
                        try:
                            sol = getattr(self.kin_solver, name)(T)
                            break
                        except Exception:
                            continue
        except Exception as e:
            print("IK 调用异常:", e)
            sol = None

        if sol is None:
            # 备用: 如果没有 IK 可用，直接返回（或你可以调用数值优化 IK）
            print("没有可用的 inverse_kinematics 接口。跳过 this step.")
            return

        # sol 可以是单解或多解
        sol = np.asarray(sol, dtype=float)
        if sol.ndim == 1:
            q_des = sol[:n_arm]
        else:
            # choose nearest
            # ensure shapes
            if sol.shape[1] < n_arm:
                # pad
                sol = np.hstack([sol, np.zeros((sol.shape[0], n_arm - sol.shape[1]))])
            dists = np.linalg.norm(sol[:, :n_arm] - q_cur.reshape(1,-1), axis=1)
            idx = np.argmin(dists)
            q_des = sol[idx,:n_arm]

        # units: assume kin_solver returns same units as current_joint_pos. We will compute delta in that unit.
        dq = q_des - q_cur

        # clip per-step
        dq = np.clip(dq, -MAX_STEP_RAD, MAX_STEP_RAD)
        q_next = q_cur + dq

        # PD control for smoothness: compute velocity term approximate
        # store last qdot if not exist
        if not hasattr(self, "_last_q"):
            self._last_q = q_cur.copy()
            self._last_time = time.time()
        now = time.time()
        dt = max(1e-3, now - getattr(self, "_last_time", now))
        qdot = (q_next - getattr(self, "_last_q", q_cur)) / dt
        # joint command = q_next (position control) — we send as absolute target (robot expects positions)
        # If robot expects velocity/torque, change interface accordingly.

        # assemble full action
        action = {}
        for i, k in enumerate(mkeys[:n_arm]):
            action[k] = float(q_next[i])
        if has_gripper:
            action[mkeys[-1]] = float(self.gripper_pos)

        # send and update stored
        try:
            self.robot_arm.send_action(action)
        except Exception as e:
            print("send_action error:", e)

        self.current_joint_pos = np.concatenate([q_next, [self.gripper_pos]]) if has_gripper else q_next
        self._last_q = q_next.copy()
        self._last_time = now

    def run_sim(self):
        print(">>> viewer = mujoco.viewer.launch(self.model, self.data).")

        viewer = mujoco.viewer.launch_passive(self.model, self.data)

        print(">>> MuJoCo Franka started.")

        while viewer.is_running() and self.running:

            # pos = self.data.site_xpos[self.sid]
            # err = self.target - pos
            # print("pos:", pos, "target:", self.target, "err:", self.target - pos)
            # ctrl = 200 * err

            # for i in range(self.model.nu):
            #     if i < len(ctrl):
            #         self.data.ctrl[i] = ctrl[i] 
            #     else :
            #         self.data.ctrl[i] = 0

            # mujoco.mj_step(self.model, self.data)
            #self.move_to_ee_using_ik(self.targetx,self.targety,self.targetz)
            self.move_to_ee_ik_pd(self.target)
            viewer.sync()
            #print(">>> viewer.sync()")
        print(">>> viewer.close()")
        viewer.close()

    def set_target(self, x, y, z):
        """实时更新目标末端位置"""
        print(x,y,z)
        self.target = np.array([x, y, z])
        self.targetx = x
        self.targety = y
        self.targetz = z
        #print(self.target)

    def close(self):
        self.running = False