import time

import mujoco.viewer
import mujoco
import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt
import argparse
import numpy.linalg as LA

NUM_MOTOR = 12

def get_gravity_orientation(quaternion):
    qw = quaternion[0]
    qx = quaternion[1]
    qy = quaternion[2]
    qz = quaternion[3]

    gravity_orientation = np.zeros(3)

    gravity_orientation[0] = 2 * (qx * qz - qw * qy)
    gravity_orientation[1] = -2 * (qy * qz + qw * qx)
    gravity_orientation[2] = -1 - 2 * (qx**2 + qy**2)

    return gravity_orientation

def quat_to_rot_matrix(quat: np.ndarray) -> np.ndarray:
    """Convert input quaternion to rotation matrix.

    Args:
        quat (np.ndarray): Input quaternion (w, x, y, z).

    Returns:
        np.ndarray: A 3x3 rotation matrix.
    """
    q = np.array(quat, dtype=np.float64, copy=True)
    nq = np.dot(q, q)
    if nq < 1e-10:
        return np.identity(3)
    q *= np.sqrt(2.0 / nq)
    q = np.outer(q, q)
    return np.array(
        (
            (1.0 - q[2, 2] - q[3, 3], q[1, 2] - q[3, 0], q[1, 3] + q[2, 0]),
            (q[1, 2] + q[3, 0], 1.0 - q[1, 1] - q[3, 3], q[2, 3] - q[1, 0]),
            (q[1, 3] - q[2, 0], q[2, 3] + q[1, 0], 1.0 - q[1, 1] - q[2, 2]),
        ),
        dtype=np.float64,
    )

def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    return (target_q - q) * kp + (target_dq - dq) * kd


from state_estimator.data_types import StateEstimate
from state_estimator.container import StateEstimatorContainer
from state_estimator.estimators.contact import ContactEstimator
from state_estimator.estimators.orientation_cheater import CheaterOrientationEstimator
from state_estimator.estimators.posvel_cheater import CheaterPositionVelocityEstimator

from state_estimator.estimators.orientation_vectornav import VectorNavOrientationEstimator
from state_estimator.estimators.posvel_kf import LinearKFPositionVelocityEstimator

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="config file name in the config folder")
    args = parser.parse_args()
    config_file = args.config_file
    with open(f"{config_file}", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        policy_path = config["policy_path"]
        policy = torch.jit.load(policy_path)
        xml_path = config["xml_path"]
        simulation_duration = config["simulation_duration"]
        simulation_dt = config["simulation_dt"]
        control_decimation = config["control_decimation"]

        kps = np.array(config["kps"], dtype=np.float32)
        kds = np.array(config["kds"], dtype=np.float32)

        default_angles = np.array(config["default_angles"], dtype=np.float32)

        ang_vel_scale = config["ang_vel_scale"]
        dof_pos_scale = config["dof_pos_scale"]
        dof_vel_scale = config["dof_vel_scale"]
        action_scale = config["action_scale"]
        cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)

        num_actions = config["num_actions"]
        num_obs = config["num_obs"]
        one_step_obs_size = config["one_step_obs_size"]
        obs_buffer_size = config["obs_buffer_size"]
        
        cmd = np.array(config["cmd_init"], dtype=np.float32)


    target_dof_pos = default_angles.copy()
    action = np.zeros(num_actions, dtype=np.float32)
    obs = np.zeros(num_obs, dtype=np.float32)

    # Load robot model
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt
    base_body_id = 1

    # Record data
    position_data_list = []
    vBody_data_list = []
    
    cheater_position_list = []
    cheater_ori_list = []
    cheater_rpy_list = []
    cheater_rbody_list = []

    kf_position_list = []
    kf_ori_list = []
    kf_rpy_list = []
    kf_rbody_list = []
    
    lin_vel_data_list = []
    ang_vel_data_list = []
    gravity_b_list = []
    joint_vel_list = []
    action_list = []
    tau_list = []

    all_leg_positions = [[] for _ in range(4)]  # 四條腿的歷史位置

    counter = 0

    state_estimate_cheater = StateEstimate()
    state_estimate_kf = StateEstimate()

    cheater_state = {
        "orientation": np.array([ 0, 0, 0, 0], dtype=np.float32),  # 假設是四元數 [w, x, y, z]
        "omegaBody": np.array([ 0, 0, 0], dtype=np.float32),
        "acceleration": np.array([ 0, 0, 0], dtype=np.float32),  # 沒有實際加速度可以先這樣填
        "position": np.array([ 0, 0, 0], dtype=np.float32),            # 世界座標位置
        "vBody": np.array([ 0, 0, 0], dtype=np.float32)                # 身體速度
    }

    vectornav_data = {
        "quat": np.array([ 0, 0, 0, 0], dtype=np.float32),
        "gyro": np.array([ 0, 0, 0], dtype=np.float32),
        "accelerometer": np.array([ 0, 0, 0], dtype=np.float32)
    }

    leg_data = [{"p": np.zeros(3), "v": np.zeros(3)} for _ in range(4)]

    control_params = {
        "controller_dt": 0.002,
        "imu_process_noise_position": 0.8,
        "imu_process_noise_velocity": 0.5,
        "foot_process_noise_position": 0.0002,
        "foot_sensor_noise_position": 0.0001,
        "foot_sensor_noise_velocity": 0.0001,
        "foot_height_sensor_noise": 0.0001
    }
    
    # create container
    container_cheater = StateEstimatorContainer(
        cheater_state=cheater_state,
        imu_data=vectornav_data,
        leg_data=leg_data,
        state_estimate=state_estimate_cheater,
        control_params=control_params
    )

    container_cheater.add_estimator(ContactEstimator)
    container_cheater.add_estimator(CheaterOrientationEstimator)
    container_cheater.add_estimator(CheaterPositionVelocityEstimator)

    container_kf = StateEstimatorContainer(
        cheater_state=cheater_state,
        imu_data=vectornav_data,
        leg_data=leg_data,
        state_estimate=state_estimate_kf,
        control_params=control_params)
    
    container_kf.add_estimator(ContactEstimator)
    container_kf.add_estimator(VectorNavOrientationEstimator)
    container_kf.add_estimator(LinearKFPositionVelocityEstimator)  # ✅ 這裡傳的是 class

    with mujoco.viewer.launch_passive(m, d) as viewer:
        # Close the viewer automatically after simulation_duration wall-seconds.
        start = time.time()
        while viewer.is_running() and time.time() - start < simulation_duration:
            step_start = time.time()
            
            tau = pd_control(target_dof_pos, d.sensordata[:NUM_MOTOR], kps, np.zeros(12), d.sensordata[NUM_MOTOR:NUM_MOTOR + NUM_MOTOR], kds)

            d.ctrl[:] = tau

            mujoco.mj_step(m, d)
            counter += 1
                        
            if counter == 1:   # ✅ 改成 ==1 （初始化只做一次）
                for est in container_kf._estimators:
                    if isinstance(est, LinearKFPositionVelocityEstimator):
                        est._xhat[0:3,0:1] = d.qpos[0:3].reshape(-1, 1)
                        est._xhat[3:6,0:1] = d.qvel[0:3].reshape(-1, 1)

                        print("[INIT] KF xhat initialized with pos:", est._xhat[0:3])
                continue  # ✅ 初始化完，跳過這一輪

            if counter % control_decimation == 0 and counter > 0:

                # create observation
                position = d.qpos[0:3]            # 世界座標位置
                vBody = d.qvel[0:3]    
                qpos = d.sensordata[:12]
                qvel = d.sensordata[12:24]
                qtorque = d.sensordata[24:36]
                ang_vel_I = d.sensordata[52:55]
                gravity_b = get_gravity_orientation(d.sensordata[36:40])
                cmd_vel = np.array(config["cmd_init"], dtype=np.float32)

                obs_list = []
                obs_list.append(cmd_vel * cmd_scale)
                obs_list.append(ang_vel_I * ang_vel_scale)
                obs_list.append(gravity_b)
                obs_list.append((qpos - default_angles) * dof_pos_scale)
                obs_list.append(qvel * dof_vel_scale)
                obs_list.append(action)
                
                ## Record Data ##
                position_data_list.append(np.asarray(position).flatten())  # 強制攤平成 array
                vBody_data_list.append(np.asarray(vBody).flatten())

                ang_vel_data_list.append(ang_vel_I * ang_vel_scale)
                gravity_b_list.append(gravity_b)
                joint_vel_list.append(qvel * dof_vel_scale)
                action_list.append(action)

                # --- 準備 estimator 輸入資料 ---
                cheater_state = {
                    "orientation": d.sensordata[36:40],   # 假設是四元數 [w, x, y, z]
                    "omegaBody": ang_vel_I,
                    "acceleration": d.sensordata[43:46],  # 沒有實際加速度可以先這樣填
                    "position": position,                 # 世界座標位置
                    "vBody": vBody                        # 身體速度
                }

                vectornav_data = {
                    "quat": d.sensordata[36:40],
                    "gyro": gravity_b,
                    "accelerometer": d.sensordata[43:46]
                }

                sensor_pos_names = ["FR_foot_pos", "FL_foot_pos", "RR_foot_pos", "RL_foot_pos"]
                sensor_vel_names = ["FR_foot_vel", "FL_foot_vel", "RR_foot_vel", "RL_foot_vel"]

                leg_data = []

                for i, (pos_name, vel_name) in enumerate(zip(sensor_pos_names, sensor_vel_names)):
                    # 拿 sensor ID
                    pos_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, pos_name)
                    vel_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, vel_name)

                    # 驗證 ID 是否找到
                    if pos_id == -1 or vel_id == -1:
                        raise RuntimeError(f"[ERROR] Sensor {pos_name} or {vel_name} not found.")

                    # 正確方式：用 m.sensor_adr 查真實資料位址
                    pos_adr = m.sensor_adr[pos_id]
                    vel_adr = m.sensor_adr[vel_id]

                    pos_dim = m.sensor_dim[pos_id]
                    vel_dim = m.sensor_dim[vel_id]

                    # 應為 3 維資料（x, y, z）
                    if pos_dim != 3 or vel_dim != 3:
                        raise ValueError(f"[ERROR] Sensor dimension mismatch: {pos_name}={pos_dim}, {vel_name}={vel_dim}")

                    # 正確讀取資料區段
                    pos = d.sensordata[pos_adr : pos_adr + pos_dim].copy()
                    vel = d.sensordata[vel_adr : vel_adr + vel_dim].copy()

                    leg_data.append({"p": pos, "v": vel})

                    # print(f"[DEBUG] Leg {i}: {pos_name} = {pos}, {vel_name} = {vel}")

                geomid_to_leg = {
                    18: 0,   # FR
                    10: 1,  # FL
                    34: 2,  # RR
                    26: 3   # RL
                }

                contact_phase = [0.0, 0.0, 0.0, 0.0]

                for i in range(d.ncon):
                    contact = d.contact[i]
                    g1, g2 = contact.geom1, contact.geom2
                    for gid in [g1, g2]:
                        if gid in geomid_to_leg:
                            leg_idx = geomid_to_leg[gid]
                            contact_phase[leg_idx] = 1.0

                container_cheater._data.cheaterState = cheater_state
                # container_cheater._data.vectorNavData = vectornav_data
                container_cheater._data.legControllerData = leg_data
                container_cheater._data.contactPhase[:] = contact_phase  # 假設都接觸，或你可以設計 force-based 判斷

                container_kf._data.cheaterState = cheater_state
                container_kf._data.vectorNavData = vectornav_data
                container_kf._data.legControllerData = leg_data
                container_kf._data.contactPhase[:] = contact_phase  # 假設都接觸，或你可以設計 force-based 判斷
                
                container_cheater.run()
                container_kf.run()

                # Compare estimated positions
                cheater_pos = container_cheater._data.result.position.copy()
                kf_pos = container_kf._data.result.position.copy()
                true_pos = d.qpos[0:3].copy()

                # print(f"[DEBUG] cheater_pos = {cheater_pos}")
                # print(f"[DEBUG] kf_pos = {kf_pos}")
                # print(f"[DEBUG] real_pos   = {true_pos}")
                # print("[DEBUG] KF position error:", np.linalg.norm(kf_pos - true_pos))

                cheater_position_list.append(cheater_pos)
                kf_position_list.append(kf_pos)

                ###
                obs_list = [torch.tensor(obs, dtype=torch.float32) if isinstance(obs, np.ndarray) else obs for obs in obs_list]
                obs_tensor_buf = torch.zeros((1, one_step_obs_size * obs_buffer_size))
                obs = torch.cat(obs_list, dim=0).unsqueeze(0)
                obs_tensor = torch.clamp(obs, -100, 100)

                # obs_tensor_buf = torch.cat([
                #     obs_tensor,
                #     obs_tensor_buf[:, :obs_buffer_size * one_step_obs_size - one_step_obs_size]
                # ], dim=1)
                obs_tensor_buf = torch.cat([
                    obs_tensor,
                    obs_tensor_buf[:, :obs_buffer_size * one_step_obs_size - one_step_obs_size]
                ], dim=1)
                # policy inference
                action = policy(obs_tensor_buf).detach().numpy().squeeze()
                # print("action :", action)

                # transform action to target_dof_pos
                if counter < 300:
                    target_dof_pos = default_angles
                else:
                    target_dof_pos = action * action_scale + default_angles
                # target_dof_pos = action * action_scale + default_angles
            # Pick up changes to the physics state, apply perturbations, update options from GUI.
            viewer.sync()

            # Rudimentary time keeping, will drift relative to wall clock.
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

    plt.figure(figsize=(14, 18))

    # 1. 真實位置 real_pos
    plt.subplot(3, 2, 1)
    for i in range(3): 
        plt.plot([step[i] for step in position_data_list], label=f"real_pos[{i}]")
    plt.title("True Position", fontsize=10, pad=10)
    plt.legend()

    # 2. Cheater 預估位置
    plt.subplot(3, 2, 2)
    for i in range(3):
        plt.plot([step[i] for step in cheater_position_list], label=f"cheater_pos[{i}]")
    plt.title("Cheater Estimated Position", fontsize=10, pad=10)
    plt.legend()

    # 3. KF 預估位置
    plt.subplot(3, 2, 3)
    for i in range(3):
        plt.plot([step[i] for step in kf_position_list], label=f"kf_pos[{i}]")
    plt.title("KF Estimated Position", fontsize=10, pad=10)
    plt.legend()

    plt.tight_layout()
    plt.show()


