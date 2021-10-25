import numpy as np

COLOR_WEIGHTS = np.array([0.299, 0.587, 0.114])
MAX_WEIGHT = 255


# Used for calculating metrics
def get_weighted_image(image: np.ndarray) -> np.ndarray:
    return image @ COLOR_WEIGHTS


def MSE(original_image: np.ndarray, interpolated_image: np.ndarray) -> float:
    weighted_original = get_weighted_image(original_image)
    weighted_interpolated = get_weighted_image(interpolated_image)

    mse = (weighted_original - weighted_interpolated) ** 2
    mse = np.mean(mse)

    return mse


def PSNR(original_image: np.ndarray, interpolated_image: np.ndarray) -> float:
    return 10 * np.log10(MAX_WEIGHT ** 2 / MSE(original_image, interpolated_image))
