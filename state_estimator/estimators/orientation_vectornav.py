from ..base import GenericEstimator
import numpy as np
from ..utils.orientation import quaternion_to_rotation_matrix, quat_to_rpy, quat_product, rpy_to_quat

class VectorNavOrientationEstimator(GenericEstimator):
    def __init__(self):
        super().__init__()
        self._b_first_visit = False
        self._ori_ini_inv = np.array([1.0, 0.0, 0.0, 0.0])

    def setup(self):
        pass

    def run(self):
        result = self._data.result
        imu = self._data.vectorNavData
        result.orientation = np.array(imu["quat"])  # keep [w, x, y, z]

        if self._b_first_visit:
            rpy_ini = quat_to_rpy(result.orientation)
            rpy_ini[0] = 0
            rpy_ini[1] = 0
            self._ori_ini_inv = rpy_to_quat(-rpy_ini)
            self._b_first_visit = False

        result.orientation = quat_product(self._ori_ini_inv, result.orientation)
        result.rpy = quat_to_rpy(result.orientation)
        result.rBody = quaternion_to_rotation_matrix(result.orientation)

        result.omegaBody = np.array(imu["gyro"])
        result.omegaWorld = result.rBody.T @ result.omegaBody
        result.aBody = np.array(imu["accelerometer"])
        result.aWorld = result.rBody.T @ result.aBody

