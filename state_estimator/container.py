class StateEstimatorContainer:
    def __init__(self, cheater_state, imu_data, leg_data, state_estimate, control_params):
        from .data_types import StateEstimatorData
        import numpy as np

        self._data = StateEstimatorData(
            result=state_estimate,
            vectorNavData=imu_data,
            cheaterState=cheater_state,
            legControllerData=leg_data,
            contactPhase=np.zeros(4),
            parameters=control_params
        )
        self._estimators = []

    def add_estimator(self, estimator_class):
        est = estimator_class()
        est.set_data(self._data)
        est.setup()
        self._estimators.append(est)

    def remove_all_estimators(self):
        self._estimators.clear()

    def run(self, visualization=None):
        for estimator in self._estimators:
            estimator.run()
        if visualization:
            visualization.quat = self._data.result.orientation.copy()
            visualization.p = self._data.result.position.copy()