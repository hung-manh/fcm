import numpy as np


def round_float(number: float) -> float:
    return round(number, 3)


def euclidean_cdist(XA: np.ndarray, XB: np.ndarray) -> np.ndarray:
    # from scipy.spatial.distance import cdist
    # return cdist(XA, XB)
    return np.sqrt(((XA[:, np.newaxis, :] - XB) ** 2).sum(axis=2))

def extract_labels(U: np.ndarray) -> np.ndarray:
    return np.argmax(U, axis=1)

def extract_clusters(data: np.ndarray, labels: np.ndarray, C: int) -> list:
    return [data[labels == i] for i in range(C)]