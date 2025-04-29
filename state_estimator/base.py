class GenericEstimator:
    def __init__(self):
        self._data = None

    def set_data(self, data):
        self._data = data

    def setup(self):
        raise NotImplementedError

    def run(self):
        raise NotImplementedError