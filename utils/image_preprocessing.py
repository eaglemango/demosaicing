import numpy as np
from enum import IntEnum


class Color(IntEnum):
    RED = 0
    GREEN = 1
    BLUE = 2


class BayerFilter:
    def __init__(self, starts_with: Color = Color.GREEN):
        self.filter_size = 2

        if starts_with == Color.GREEN:
            self.filter = np.asarray([[[0, 1, 0], [1, 0, 0]],
                                     [[0, 0, 1], [0, 1, 0]]])
        elif starts_with == Color.RED:
            self.filter = np.asarray([[[1, 0, 0], [0, 1, 0]],
                                      [[0, 1, 0], [0, 0, 1]]])
        elif starts_with == Color.BLUE:
            self.filter = np.asarray([[[0, 0, 1], [0, 1, 0]],
                                      [[0, 1, 0], [1, 0, 0]]])

    def apply(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape

        filter_h = h // self.filter_size + 1
        filter_w = w // self.filter_size + 1

        expanded_filter = np.tile(self.filter, (filter_h, filter_w, 1))[:h, :w, :]

        return np.multiply(image[:, :, np.newaxis], expanded_filter)
