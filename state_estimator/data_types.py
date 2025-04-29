from dataclasses import dataclass
import numpy as np

class StateEstimate:
    def __init__(self):
        self.contact_estimate = np.zeros(4)
        self.position = np.zeros(3)
        self.vBody = np.zeros(3)
        self.orientation = np.array([1.0, 0.0, 0.0, 0.0])  # w, x, y, z
        self.omegaBody = np.zeros(3)
        self.rBody = np.eye(3)
        self.rpy = np.zeros(3)
        self.omegaWorld = np.zeros(3)
        self.vWorld = np.zeros(3)
        self.aBody = np.zeros(3)
        self.aWorld = np.zeros(3)

@dataclass
class StateEstimatorData:
    result: StateEstimate
    vectorNavData: dict
    cheaterState: dict
    legControllerData: list  # list of 4 legs, each with .p and .v
    contactPhase: np.ndarray  # length 4
    parameters: dict
