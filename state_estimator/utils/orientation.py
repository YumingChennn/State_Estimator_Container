from scipy.spatial.transform import Rotation as R
import numpy as np

def quaternion_to_rotation_matrix(q):
    """
    Convert a quaternion [w, x, y, z] to a 3x3 rotation matrix.
    """
    # 轉成 scipy 格式 [x, y, z, w]
    r = R.from_quat([q[1], q[2], q[3], q[0]])
    return r.as_matrix()

def quat_to_rpy(q):
    """
    Convert a quaternion [w, x, y, z] to RPY angles (MuJoCo's ZYX convention).
    """
    r = R.from_quat([q[1], q[2], q[3], q[0]])
    return r.as_euler('zyx', degrees=False)

def rpy_to_quat(rpy):
    """
    Convert RPY angles (ZYX order) back to a quaternion [w, x, y, z].
    """
    q = R.from_euler('zyx', rpy, degrees=False).as_quat()  # [x, y, z, w]
    return np.array([q[3], q[0], q[1], q[2]])  # change into [w, x, y, z]

def quat_product(q1, q2):
    """
    Perform quaternion multiplication q1 * q2. 
    Inputs and output are in [w, x, y, z] format
    """
    r1 = R.from_quat([q1[1], q1[2], q1[3], q1[0]])
    r2 = R.from_quat([q2[1], q2[2], q2[3], q2[0]])
    q_prod = (r1 * r2).as_quat()  # [x, y, z, w]
    return np.array([q_prod[3], q_prod[0], q_prod[1], q_prod[2]])
