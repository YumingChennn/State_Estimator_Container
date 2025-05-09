import time
import argparse
import yaml

import mujoco.viewer
import mujoco
import numpy as np
import torch
import matplotlib.pyplot as plt

import pinocchio

from state_estimator.data_types import StateEstimate
from state_estimator.container import StateEstimatorContainer
from state_estimator.estimators.contact import ContactEstimator
from state_estimator.estimators.orientation_cheater import CheaterOrientationEstimator
from state_estimator.estimators.posvel_cheater import CheaterPositionVelocityEstimator
from state_estimator.estimators.orientation_vectornav import VectorNavOrientationEstimator
from state_estimator.estimators.posvel_kf import LinearKFPositionVelocityEstimator

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

def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    return (target_q - q) * kp + (target_dq - dq) * kd

def load_config(config_file_path):
    with open(config_file_path, "r") as f:
        return yaml.load(f, Loader=yaml.FullLoader)
    
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="config file name in the config folder")
    return parser.parse_args()

def load_mujoco_model(xml_path, simulation_dt):
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt
    return m, d

def create_state_estimators(control_params):
    cheater_state = {
        "orientation": np.array([0, 0, 0, 0], dtype=np.float32),
        "omegaBody": np.array([0, 0, 0], dtype=np.float32),
        "acceleration": np.array([0, 0, 0], dtype=np.float32),
        "position": np.array([0, 0, 0], dtype=np.float32),
        "vBody": np.array([0, 0, 0], dtype=np.float32)
    }

    vectornav_data = {
        "quat": np.array([0, 0, 0, 0], dtype=np.float32),
        "gyro": np.array([0, 0, 0], dtype=np.float32),
        "accelerometer": np.array([0, 0, 0], dtype=np.float32)
    }

    leg_data = [{"p": np.zeros(3), "v": np.zeros(3)} for _ in range(4)]

    state_estimate_cheater = StateEstimate()
    state_estimate_kf = StateEstimate()

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
    container_kf.add_estimator(LinearKFPositionVelocityEstimator)
    return container_cheater, container_kf

def collect_leg_sensor_data(m, d):
    sensor_pos_names = ["FR_foot_pos", "FL_foot_pos", "RR_foot_pos", "RL_foot_pos"]
    sensor_vel_names = ["FR_foot_vel", "FL_foot_vel", "RR_foot_vel", "RL_foot_vel"]
    leg_data = []
    for pos_name, vel_name in zip(sensor_pos_names, sensor_vel_names):
        pos_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, pos_name)
        vel_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, vel_name)
        if pos_id == -1 or vel_id == -1:
            raise RuntimeError(f"[Error] Sensor name {pos_name} or {vel_name} not found in mujoco model.")
        pos_adr = m.sensor_adr[pos_id]
        vel_adr = m.sensor_adr[vel_id]
        pos_dim = m.sensor_dim[pos_id]
        vel_dim = m.sensor_dim[vel_id]
        if pos_dim != 3 or vel_dim != 3:
            raise ValueError(f"[ERROR] Sensor dimension mismatch: {pos_name}={pos_dim}, {vel_name}={vel_dim}")
        pos = d.sensordata[pos_adr : pos_adr + pos_dim].copy()
        vel = d.sensordata[vel_adr : vel_adr + vel_dim].copy()
        leg_data.append({"p": pos, "v": vel})
    return leg_data

def update_state_estimators(container_cheater, container_kf, cheater_state, vectornav_data, leg_data, contact_phase):
    container_cheater._data.cheaterState = cheater_state
    container_cheater._data.legControllerData = leg_data
    container_cheater._data.contactPhase[:] = contact_phase

    container_kf._data.cheaterState = cheater_state
    container_kf._data.vectorNavData = vectornav_data
    container_kf._data.legControllerData = leg_data
    container_kf._data.contactPhase[:] = contact_phase

    container_cheater.run()
    container_kf.run()

def plot_simulation_results(position_data_list, cheater_position_list, kf_position_list):
    plt.figure(figsize=(14, 18))

    # 1. Real pos
    plt.subplot(3, 2, 1)
    for i in range(3): 
        plt.plot([step[i] for step in position_data_list], label=f"real_pos[{i}]")
    plt.title("True Position", fontsize=10, pad=10)
    plt.legend()

    # 2. Cheater estimated position
    plt.subplot(3, 2, 2)
    for i in range(3):
        plt.plot([step[i] for step in cheater_position_list], label=f"cheater_pos[{i}]")
    plt.title("Cheater Estimated Position", fontsize=10, pad=10)
    plt.legend()

    # 3. KF estimated position
    plt.subplot(3, 2, 3)
    for i in range(3):
        plt.plot([step[i] for step in kf_position_list], label=f"kf_pos[{i}]")
    plt.title("KF Estimated Position", fontsize=10, pad=10)
    plt.legend()


    plt.tight_layout()
    plt.show()

# === Main Simulation Runner ===
def run_simulation(m, d, container_cheater, container_kf, policy, config):
    # get the parameter from config
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

    counter = 0

    with mujoco.viewer.launch_passive(m, d) as viewer:
        # Close the viewer automatically after simulation_duration wall-seconds.
        start = time.time()
        while viewer.is_running() and time.time() - start < simulation_duration:
            step_start = time.time()

            tau = pd_control(target_dof_pos, d.sensordata[:NUM_MOTOR], kps, np.zeros(12), d.sensordata[NUM_MOTOR:NUM_MOTOR + NUM_MOTOR], kds)

            d.ctrl[:] = tau

            mujoco.mj_step(m, d)

            counter += 1

            if counter == 1:  
                for est in container_kf._estimators:
                    if isinstance(est, LinearKFPositionVelocityEstimator):
                        est._xhat[0:3,0:1] = d.qpos[0:3].reshape(-1, 1)
                        est._xhat[3:6,0:1] = d.qvel[0:3].reshape(-1, 1)

                        print("[INIT] KF xhat initialized with pos:", est._xhat[0:3])
                continue 

            if counter % control_decimation == 0 and counter > 0:

                # create observation
                position = d.qpos[0:3]            # world frame
                vBody = d.qvel[0:3]               # world frame
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

                # --- prepare estimator input date ---
                cheater_state = {
                    "orientation": d.sensordata[36:40],  # quaternions [w, x, y, z]
                    "omegaBody": ang_vel_I,
                    "acceleration": d.sensordata[43:46],  
                    "position": position,                # world frame
                    "vBody": vBody                       # world frame
                }

                vectornav_data = {
                    "quat": d.sensordata[36:40],
                    "gyro": gravity_b,
                    "accelerometer": d.sensordata[43:46]
                }

                leg_data = collect_leg_sensor_data(m,d)

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
                            contact_phase[leg_idx] = 1

                update_state_estimators(container_cheater, container_kf, cheater_state, vectornav_data, leg_data, contact_phase)

                 # Compare estimated positions
                cheater_pos = container_cheater._data.result.position.copy()
                kf_pos = container_kf._data.result.position.copy()
                true_pos = d.qpos[0:3].copy()

                cheater_position_list.append(cheater_pos)
                kf_position_list.append(kf_pos)

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

                # transform action to target_dof_pos
                if counter < 300:
                    target_dof_pos = default_angles
                else:
                    target_dof_pos = action * action_scale + default_angles

            # Pick up changes to the physics state, apply perturbations, update options from GUI.
            viewer.sync()

            # Rudimentary time keeping, will drift relative to wall clock.
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

    plot_simulation_results(position_data_list, cheater_position_list, kf_position_list)

def main():
    args = parse_arguments()
    config = load_config(args.config_file)
    policy = torch.jit.load(config["policy_path"])

    m, d = load_mujoco_model(config["xml_path"], config["simulation_dt"])

    control_params = {
        "controller_dt": 0.002,
        "imu_process_noise_position": 0.8,
        "imu_process_noise_velocity": 0.5,
        "foot_process_noise_position": 0.0002,
        "foot_sensor_noise_position": 0.0001,
        "foot_sensor_noise_velocity": 0.0001,
        "foot_height_sensor_noise": 0.0001
    }

    container_cheater, container_kf = create_state_estimators(control_params)

    run_simulation(m, d, container_cheater, container_kf, policy, config)

if __name__ == "__main__":
    main()


