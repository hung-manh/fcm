import json
import pandas as pd
from urllib import request, parse
import certifi
import ssl
import time
import numpy as np


# Mã hóa nhãn
class LabelEncoder:
    def __init__(self):
        self.index_to_label = {}
        self.unique_labels = None

    @property
    def classes_(self) -> np.ndarray:
        return self.unique_labels

    def fit_transform(self, labels) -> np.ndarray:
        self.unique_labels = np.unique(labels)
        label_to_index = {label: index for index, label in enumerate(self.unique_labels)}
        self.index_to_label = {index: label for label, index in label_to_index.items()}
        return np.array([label_to_index[label] for label in labels])

    def inverse_transform(self, indices) -> np.ndarray:
        return np.array([self.index_to_label[index] for index in indices])


def random_negative_assignment(labels: np.ndarray, ratio: float = 0.3, val: float = -1) -> np.ndarray:
    _length = len(labels)
    # Tính số lượng phần tử cần gán giá trị -1
    neg_count = int(ratio * _length)
    # Tạo một mảng chỉ số ngẫu nhiên không lặp lại
    random_indices = np.random.choice(_length, neg_count, replace=False)
    # Tạo một bản sao của mảng để không làm thay đổi mảng gốc
    result = np.copy(labels)
    # Gán giá trị -1 vào các vị trí ngẫu nhiên
    result[random_indices] = val
    # print('labeled=========', type(result), result)
    # unique, counts = np.unique(result, return_counts=True)
    # print(dict(zip(unique, counts)))
    # print(f"Tỷ lệ có nhãn = {np.sum(result == -1) / len(result):.2f}")
    return result


def split_data_for_semi_supervised_learning(data: np.ndarray, labels: np.ndarray, n_sites: int = 3, ratio: float = 0.3, val: float = -1) -> list:
    result = []
    datas = np.array_split(data, n_sites)
    labeled = np.array_split(labels, n_sites)
    for i, data in enumerate(datas):
        y_true = labeled[i]
        y_lble = random_negative_assignment(labels=y_true, ratio=ratio, val=val)
        result.append({'X': data, 'Y': y_lble, 'T': y_true})
    return result
