import numpy as np
from utils.utils import euclidean_cdist

class Dfcm:
    def __init__(self, m: float = 2, epsilon: float = 1e-5, maxiter: int = 10000):
        self._m = m  # Mức độ mờ - Degree of fuzziness  
        self._epsilon = epsilon  # Tiêu chuẩn dừng - epsilon 
        self._maxiter = maxiter  # Maximum number of iterations

    # Khởi tạo ma trận thành viên
    @staticmethod
    def __init_membership(N: int, C: int, seed: int = 0) -> np.ndarray:
        if seed > 0:
            np.random.seed(seed)
        U0 = np.random.rand(N, C)
        return U0 / U0.sum(axis=1)[:, None]

    # Cập nhật ma trận tâm cụm
    def update_cluster_centers(self, data: np.ndarray, membership: np.ndarray) -> np.ndarray:
        # Nhân ma trận X với từng độ thuộc cụm của nó
        # Tổng kết quả cho toàn bộ các điểm của mỗi tâm cụm
        
        # # Code cũ 
        # _umT = (membership ** self._m).T
        # V = (_umT[:, :, None] * data).sum(axis=1)
        # return V / (_umT.sum(axis=1)[:, None])
        
        # Code mới
        um = membership ** self._m # u_ik^m
        return (um.T @ data) / um.sum(axis=0)[:, np.newaxis] # (Σ(u_ik^m * x_i)) / (Σ(u_ik^m))
    
    
    # Cập nhật ma trận thành viên, ma trận độ thuộc 
    def __update_membership_matrix(self, distances: np.ndarray) -> np.ndarray:
        epsilon = 1e-10  # small constant to prevent division by zero
        distances = np.maximum(distances, epsilon)  # avoid zero distances
        U = distances[:, :, None] * (1 / distances)[:, None, :]
        U = (U ** (2 / (self._m - 1))).sum(axis=2)
        return 1 / U # d_ik^(-2/(m-1)) / Σ(d_ik^(-2/(m-1)))
        
        # power = 2 / (self._m - 1)
        # return 1 / ((distances[:, :, np.newaxis] / distances[:, np.newaxis, :]) ** power).sum(axis=2)

    # Fuzzy C-means algorithm
    def cmeans(self, data: np.ndarray, C:int = 3, seed: int = 42) -> tuple:
        u = self.__init_membership(len(data), C, seed)
        for step in range(self._maxiter):
            old_u = u.copy()
            v = self.update_cluster_centers(data, old_u)
            sdistances = euclidean_cdist(data, v) # Khoảng các Euclidean giữa các điểm dữ liệu(data) và các tâm cụm(centroids)
            u = self.__update_membership_matrix(sdistances)
            
            # print(str(np.linalg.norm(u - old_u)), '\t ', str((np.abs(u - old_u)).max(axis=(0, 1))))
            # if np.linalg.norm(u - old_u) < self._epsilon:
            if (np.abs(u - old_u)).max(axis=(0, 1)) < self._epsilon:
                break
        return u, v, step + 1