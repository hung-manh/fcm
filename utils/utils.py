import numpy as np


def round_float(number: float) -> float:
    return round(number, 3)


def euclidean_cdist(XA: np.ndarray, XB: np.ndarray) -> np.ndarray:
    # from scipy.spatial.distance import cdist
    # return cdist(XA, XB)
    return np.sqrt(((XA[:, np.newaxis, :] - XB) ** 2).sum(axis=2))