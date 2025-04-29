from ..base import GenericEstimator
import numpy as np

class ContactEstimator(GenericEstimator):
    def setup(self):
        pass

    def run(self):
        # Pass-through contact estimation
        self._data.result.contact_estimate = self._data.contactPhase.copy()
