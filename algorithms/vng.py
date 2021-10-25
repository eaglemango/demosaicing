import numpy as np

from enum import IntEnum

from .common import DemosaicingAlgorithm
from utils.image_preprocessing import Color


class Compass(IntEnum):
    N = 0
    E = 1
    S = 2
    W = 3
    NE = 4
    SE = 5
    NW = 6
    SW = 7


class VNG(DemosaicingAlgorithm):
    def __init__(self):
        self.colors = None
        self.intensity = None
        self.nanned = None

    def apply(self, image: np.ndarray) -> np.ndarray:
        # Make a copy in order not to corrupt RBG CFA image
        result = image[2:-2, 2:-2]

        self.colors = np.argmax(image, axis=-1)
        self.intensity = np.max(image, axis=-1)
        self.nanned = np.where(image != 0, image, np.nan)

        # (height, width, channels)
        h, w, c = image.shape

        for i in range(2, h - 2):
            for j in range(2, w - 2):
                # Try to recover current pixel
                result[i - 2][j - 2] = self.recover_pixel(i, j, image).astype("uint64")

        return result

    def recover_pixel(self, i, j, image: np.ndarray) -> np.ndarray:
        # Choose one of four recovery cases
        if self.colors[i][j] == Color.GREEN and self.colors[i - 1][j] == Color.RED:
            return self._recover_green_center_red_above(i, j, image)

        elif self.colors[i][j] == Color.GREEN and self.colors[i - 1][j] == Color.BLUE:
            return self._recover_green_center_blue_above(i, j, image)

        elif self.colors[i][j] == Color.RED:
            return self._recover_red_center(i, j, image)

        elif self.colors[i][j] == Color.BLUE:
            return self._recover_blue_center(i, j, image)

        else:
            raise ValueError("Invalid recovery case")

    def _recover_green_center_red_above(self, i, j, image: np.ndarray) -> np.ndarray:
        grads = self._compute_grads(i, j, image, is_green_center=True)
        threshold = self._compute_threshold(grads)

        chosen_directions = []
        for direction in Compass:
            if grads[direction] < threshold:
                chosen_directions.append(direction)

        if len(chosen_directions) == 0:
            return np.ones(3) * self.intensity[i][j]

        average_color = np.zeros(3)
        for direction in chosen_directions:
            average_color += self._compute_average_color(i, j, image, direction)

        average_color /= len(chosen_directions)

        r_sum, g_sum, b_sum = average_color

        recovered = np.zeros(3)
        recovered[Color.RED] = self.intensity[i][j] + r_sum - g_sum
        recovered[Color.GREEN] = self.intensity[i][j]
        recovered[Color.BLUE] = self.intensity[i][j] + b_sum - g_sum

        return recovered

    def _recover_green_center_blue_above(self, i, j, image: np.ndarray) -> np.ndarray:
        grads = self._compute_grads(i, j, image, is_green_center=True)
        threshold = self._compute_threshold(grads)

        chosen_directions = []
        for direction in Compass:
            if grads[direction] < threshold:
                chosen_directions.append(direction)

        if len(chosen_directions) == 0:
            return np.ones(3) * self.intensity[i][j]

        average_color = np.zeros(3)
        for direction in chosen_directions:
            average_color += self._compute_average_color(i, j, image, direction)

        average_color /= len(chosen_directions)

        r_sum, g_sum, b_sum = average_color

        recovered = np.zeros(3)
        recovered[Color.RED] = self.intensity[i][j] + r_sum - g_sum
        recovered[Color.GREEN] = self.intensity[i][j]
        recovered[Color.BLUE] = self.intensity[i][j] + b_sum - g_sum

        return recovered

    def _recover_red_center(self, i, j, image: np.ndarray) -> np.ndarray:
        grads = self._compute_grads(i, j, image, is_green_center=False)
        threshold = self._compute_threshold(grads)

        chosen_directions = []
        for direction in Compass:
            if grads[direction] < threshold:
                chosen_directions.append(direction)

        if len(chosen_directions) == 0:
            return np.ones(3) * self.intensity[i][j]

        average_color = np.zeros(3)
        for direction in chosen_directions:
            average_color += self._compute_average_color(i, j, image, direction)

        average_color /= len(chosen_directions)

        r_sum, g_sum, b_sum = average_color

        recovered = np.zeros(3)
        recovered[Color.RED] = self.intensity[i][j]
        recovered[Color.GREEN] = self.intensity[i][j] + g_sum - r_sum
        recovered[Color.BLUE] = self.intensity[i][j] + b_sum - r_sum

        return recovered

    def _recover_blue_center(self, i, j, image: np.ndarray) -> np.ndarray:
        grads = self._compute_grads(i, j, image, is_green_center=False)
        threshold = self._compute_threshold(grads)

        chosen_directions = []
        for direction in Compass:
            if grads[direction] < threshold:
                chosen_directions.append(direction)

        if len(chosen_directions) == 0:
            return np.ones(3) * self.intensity[i][j]

        average_color = np.zeros(3)
        for direction in chosen_directions:
            average_color += self._compute_average_color(i, j, image, direction)

        average_color /= len(chosen_directions)

        r_sum, g_sum, b_sum = average_color

        recovered = np.zeros(3)
        recovered[Color.RED] = self.intensity[i][j] + r_sum - b_sum
        recovered[Color.GREEN] = self.intensity[i][j] + g_sum - b_sum
        recovered[Color.BLUE] = self.intensity[i][j]

        return recovered

    def _compute_grads(self, i, j, image: np.ndarray, is_green_center):
        main_grads = self._compute_main_grads(i, j, image)
        additional_grads = self._compute_additional_grads(i, j, image, is_green_center)

        return np.concatenate((main_grads, additional_grads))

    def _compute_main_grads(self, i, j, image: np.ndarray) -> np.ndarray:
        # Gradients for (N, E, S, W) are equal in all cases

        # Gradient N ==========================================================
        n_pixels = self.intensity[i-2:i, j-1:j+2]
        s_pixels = self.intensity[i:i+2, j-1:j+2]
        n_s_diff = np.abs(n_pixels - s_pixels)
        n_s_diff[[0, 1], [1, 1]] *= 2

        grad_n = np.sum(n_s_diff) / 2

        # Gradient E ==========================================================
        e_pixels = self.intensity[i-1:i+2, j+1:j+3]
        w_pixels = self.intensity[i-1:i+2, j-1:j+1]
        e_w_diff = np.abs(e_pixels - w_pixels)
        e_w_diff[[1, 1], [0, 1]] *= 2

        grad_e = np.sum(e_w_diff) / 2

        # Gradient S ==========================================================
        s_pixels = self.intensity[i+1:i+3, j-1:j+2]
        n_pixels = self.intensity[i-1:i+1, j-1:j+2]
        s_n_diff = np.abs(s_pixels - n_pixels)
        s_n_diff[[0, 1], [1, 1]] *= 2

        grad_s = np.sum(s_n_diff) / 2

        # Gradient W ==========================================================
        w_pixels = self.intensity[i-1:i+2, j-2:j]
        e_pixels = self.intensity[i-1:i+2, j:j+2]
        w_e_diff = np.abs(w_pixels - e_pixels)
        w_e_diff[[1, 1], [0, 1]] *= 2

        grad_w = np.sum(w_e_diff) / 2

        return np.array([grad_n, grad_e, grad_s, grad_w])

    def _compute_additional_grads(self, i, j, image: np.ndarray, is_green_center):
        # Gradients for (NE, SE, NW, SW) green and non-green center are different

        if is_green_center:
            # Red center case =================================================

            # Name variables according to paper (zero element is fake and never used)
            r = self.intensity[i + np.array([0, -2, -2, -2, +0, +0, +0, +2, +2, +2]),
                               j + np.array([0, -2, +0, +2, -2, +0, +2, -2, +0, +2])]

            g = self.intensity[i + np.array([0, -2, -2, -1, -1, -1, +0, +0, +1, +1, +1, +2, +2]),
                               j + np.array([0, -1, +1, -2, +0, +2, -1, +1, -2, +0, +2, -1, +1])]

            b = self.intensity[i + np.array([0, -1, -1, +1, +1]),
                               j + np.array([0, -1, +1, -1, +1])]

            # Compute gradients using formulas
            grad_ne = abs(b[2] - b[3]) + abs(r[3] - r[5]) + \
                abs(g[4] - g[6])/2 + abs(g[7] - g[9])/2 + abs(g[2] - g[4])/2 + abs(g[5] - g[7])/2

            grad_se = abs(b[4] - b[1]) + abs(r[9] - r[5]) + \
                abs(g[7] - g[4])/2 + abs(g[9] - g[6])/2 + abs(g[10] - g[7])/2 + abs(g[12] - g[9])/2

            grad_nw = abs(b[1] - b[4]) + abs(r[1] - r[5]) + \
                abs(g[4] - g[7])/2 + abs(g[6] - g[9])/2 + abs(g[1] - g[4])/2 + abs(g[3] - g[6])/2

            grad_sw = abs(b[3] - b[2]) + abs(r[7] - r[5]) + \
                abs(g[6] - g[4])/2 + abs(g[9] - g[7])/2 + abs(g[8] - g[6])/2 + abs(g[11] - g[9])/2

            # Blue center case ================================================

            # Hack: we don't need to consider that case separately, because R and B pixels are swapped,
            # but resulting gradient is the same

        else:
            # Gradient NE =====================================================
            ne_pixels = self.intensity[i-2:i, j+1:j+3]
            sw_pixels = self.intensity[i:i+2, j-1:j+1]
            ne_sw_diff = np.abs(ne_pixels - sw_pixels)

            grad_ne = np.sum(ne_sw_diff)

            # Gradient SE =====================================================
            se_pixels = self.intensity[i+1:i+3, j+1:j+3]
            nw_pixels = self.intensity[i-1:i+1, j-1:j+1]
            se_nw_diff = np.abs(se_pixels - nw_pixels)

            grad_se = np.sum(se_nw_diff)

            # Gradient NW =====================================================
            nw_pixels = self.intensity[i-2:i, j-2:j]
            se_pixels = self.intensity[i:i+2, j:j+2]
            nw_se_diff = np.abs(nw_pixels - se_pixels)

            grad_nw = np.sum(nw_se_diff)

            # Gradient SW =====================================================
            sw_pixels = self.intensity[i+1:i+3, j-2:j]
            ne_pixels = self.intensity[i-1:i+1, j:j+2]
            sw_ne_diff = np.abs(sw_pixels - ne_pixels)

            grad_sw = np.sum(sw_ne_diff)

        return np.array([grad_ne, grad_se, grad_nw, grad_sw])

    # Default values for k1 and k2 are taken from paper
    @staticmethod
    def _compute_threshold(grads: np.ndarray, k_1=1.5, k_2=0.5):
        min_grad = np.min(grads)
        max_grad = np.max(grads)

        # There is a mistake in paper: we need to subtract min_grad from max_grad, not sum them
        threshold = k_1 * min_grad + k_2 * (max_grad - min_grad)

        return threshold

    def _compute_average_color(self, i, j, image: np.ndarray, direction: Compass) -> np.ndarray:
        result = np.zeros(3)
        pixel_color = self.colors[i][j]

        if direction == Compass.N:
            window = self.nanned[i-2:i+1, j-1:j+2, :].reshape((9, 3))
            mean_color = np.nanmean(window, axis=0)

            for i in range(len(Color)):
                if i == pixel_color:
                    result[pixel_color] = (self.intensity[i][j] + self.intensity[i - 2][j]) // 2
                else:
                    result[i] = mean_color[i]

        elif direction == Compass.E:
            window = self.nanned[i-1:i+2, j:j+3, :].reshape((9, 3))
            mean_color = np.nanmean(window, axis=0)

            for i in range(len(Color)):
                if i == pixel_color:
                    result[pixel_color] = (self.intensity[i][j] + self.intensity[i][j + 2]) // 2
                else:
                    result[i] = mean_color[i]

        elif direction == Compass.S:
            window = self.nanned[i:i+3, j-1:j+2, :].reshape((9, 3))
            mean_color = np.nanmean(window, axis=0)

            for i in range(len(Color)):
                if i == pixel_color:
                    result[pixel_color] = (self.intensity[i][j] + self.intensity[i + 2][j]) // 2
                else:
                    result[i] = mean_color[i]

        elif direction == Compass.W:
            window = self.nanned[i-1:i+2, j-2:j+1, :].reshape((9, 3))
            mean_color = np.nanmean(window, axis=0)

            for i in range(len(Color)):
                if i == pixel_color:
                    result[pixel_color] = (self.intensity[i][j] + self.intensity[i][j - 2]) // 2
                else:
                    result[i] = mean_color[i]

        elif direction == Compass.NE:
            window = self.nanned[i-2:i+1, j:j+3, :].reshape((9, 3))
            mean_color = np.nanmean(window, axis=0)

            for i in range(len(Color)):
                if i == pixel_color:
                    if i == self.colors[i - 1][j + 1]:
                        result[pixel_color] = self.intensity[i - 1][j + 1]
                    else:
                        result[pixel_color] = (self.intensity[i][j] + self.intensity[i - 2][j + 2]) // 2
                else:
                    result[i] = mean_color[i]

        elif direction == Compass.SE:
            window = self.nanned[i:i+3, j:j+3, :].reshape((9, 3))
            mean_color = np.nanmean(window, axis=0)

            for i in range(len(Color)):
                if i == pixel_color:
                    if i == self.colors[i + 1][j + 1]:
                        result[pixel_color] = self.intensity[i + 1][j + 1]
                    else:
                        result[pixel_color] = (self.intensity[i][j] + self.intensity[i + 2][j + 2]) // 2
                else:
                    result[i] = mean_color[i]

        elif direction == Compass.NW:
            window = self.nanned[i-2:i+1, j-2:j+1, :].reshape((9, 3))
            mean_color = np.nanmean(window, axis=0)

            for i in range(len(Color)):
                if i == pixel_color:
                    if i == self.colors[i - 1][j - 1]:
                        result[pixel_color] = self.intensity[i - 1][j - 1]
                    else:
                        result[pixel_color] = (self.intensity[i][j] + self.intensity[i - 2][j - 2]) // 2
                else:
                    result[i] = mean_color[i]

        elif direction == Compass.SW:
            window = self.nanned[i:i+3, j-2:j+1, :].reshape((9, 3))
            mean_color = np.nanmean(window, axis=0)

            for i in range(len(Color)):
                if i == pixel_color:
                    if i == self.colors[i + 1][j - 1]:
                        result[pixel_color] = self.intensity[i + 1][j - 1]
                    else:
                        result[pixel_color] = (self.intensity[i][j] + self.intensity[i + 2][j - 2]) // 2
                else:
                    result[i] = mean_color[i]

        else:
            raise ValueError("Unknown direction")

        return result
