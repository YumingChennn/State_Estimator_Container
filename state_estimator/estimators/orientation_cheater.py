from ..base import GenericEstimator
import numpy as np
from ..utils.orientation import quaternion_to_rotation_matrix, quat_to_rpy

class CheaterOrientationEstimator(GenericEstimator):
    def setup(self):
        pass

    def run(self):
        result = self._data.result
        cheat = self._data.cheaterState

        result.orientation = np.array(cheat["orientation"])
        result.rBody = quaternion_to_rotation_matrix(result.orientation)
        result.omegaBody = np.array(cheat["omegaBody"])
        result.omegaWorld = result.rBody.T @ result.omegaBody
        result.rpy = quat_to_rpy(result.orientation)
        result.aBody = np.array(cheat["acceleration"])
        result.aWorld = result.rBody.T @ result.aBody