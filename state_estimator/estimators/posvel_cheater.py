from ..base import GenericEstimator
import numpy as np

class CheaterPositionVelocityEstimator(GenericEstimator):
    def setup(self):
        pass

    def run(self):
        result = self._data.result
        cheat = self._data.cheaterState

        result.position = np.array(cheat["position"])
        result.vBody = np.array(cheat["vBody"])
        result.vWorld = result.rBody.T @ result.vBody
