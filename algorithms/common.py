import numpy as np


class DemosaicingAlgorithm:
    def apply(self, image: np.ndarray) -> np.ndarray:
        raise NotImplementedError
