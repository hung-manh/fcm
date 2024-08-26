import numpy as np
from utils.utils import norm_distances
from models.SSFCM import SSfcm
from models.fcm import Dfcm


class SSCfcm:
    def __init__(self, m: float = 2, epsilon: float = 1e-5, maxiter: int = 10000):
        self._m = m  # Mức độ mờ - Degree of fuzziness  
        self._epsilon = epsilon  # Tiêu chuẩn dừng - epsilon 
        self._maxiter = maxiter  # Maximum number of iterations


    def stage1(self, datas: np.ndarray, u_bar: list, C: np.ndarray) -> tuple:
        ssfcm = SSfcm(self._m, self._epsilon, self._maxiter)
        cluster_centers = []
        membership_matrixs = []
        for i in range(len(datas)):
            U, V, step = ssfcm.sscmeans(datas[i], u_bar[i], C[i], seed=42)
            membership_matrixs.append(U)
            cluster_centers.append(V)
        
        return membership_matrixs, cluster_centers

    # def V_tilde(self, cluster_centers_after_fcm: np.ndarray):
    #     return np.mean(cluster_centers_after_fcm, axis=0)

    def V_tilde(self, cluster_centers, C: np.ndarray) -> list:
        dfcm = Dfcm(self._m, self._epsilon, self._maxiter)
        cluster_centers_to_fcm = []
        v_tilde = []

        for V in range(len(cluster_centers)):
            for i in cluster_centers[V]:
                cluster_centers_to_fcm.append(i)
        for i in range(len(cluster_centers)):
            U, V, step = dfcm.cmeans(cluster_centers_to_fcm, C[i], seed=42)
            v_tilde.append(V)
        return v_tilde
    
    def J_fcm(datas: np.ndarray, membership_matrixs: np.ndarray, cluster_centers_after_fcm: np.ndarray):
        pass


    def beta():
        pass

    # Cập nhật ma trận tâm cụm
    def update_cluster_centers(self, datas: list, membership_matrixs: list, u_bar: list, v_tilde: list, beta: float = 0.5) -> np.ndarray:
        '''
            datas: (n_sites x N x D)
            membership_matrixs: (n_sites x N x C)
            u_bar: (n_sites x N x C)
            v_tilde: (C x D)
            output: (n_sites x C x D)
        '''
        result = []
        # result = (membership_matrixs - u_bar) ** 2 #result: (n_sites x N x C)
        for i in range(len(datas)):
            tmp = np.array(membership_matrixs[i]) - np.array(u_bar[i]) ** 2
            result.append(tmp)
        numerator1 = []
        numerator2 = []
        denominator = []

        for i in range(len(datas)):
            tmp = result[i].T @ datas[i] # C x D
            numerator1.append(tmp)
        
        for i in range(len(datas)):
            tmp = beta * (result[i].sum(axis=0)[:, np.newaxis] * v_tilde[i]) # C x D
            numerator2.append(tmp)

        for i in range(len(datas)):
            tmp = (1 + beta) * result[i].sum(axis=0)[:, np.newaxis] # C x 1
            denominator.append(tmp)

        cluster_centers = []
        for i in range(len(datas)):
            tmp = (numerator1[i] + numerator2[i]) / denominator[i]
            cluster_centers.append(tmp)
        return cluster_centers

    
    # Cập nhật ma trận thành viên, ma trận độ thuộc 
    def update_membership_matrix(self, cluster_centers: list, v_tilde: list, distances: list, u_bar: list, beta: float = 0.5) -> list:
        '''     
            distances: (N x C) 
            cluster_centers: (C x D)
            v_tilde: (C x D)
            u_bar: (C x N)
        '''
        numerator = []
        denominator = []
        membership_matrixs = []
        for i in range(len(u_bar)):
            tmp = u_bar[i].sum(axis=1)
            numerator.append((1 - tmp))

        for i in range(len(u_bar)):
            print(cluster_centers[i].shape)
            print(v_tilde[i].shape)
            print(distances[i].shape)
            exit()
            tmp = distances[i] ** 2 + beta * (np.linalg.norm(cluster_centers[i] - v_tilde[i], axis = 1, keepdims=True)).T ** 2
            denominator.append(tmp / tmp.sum(axis=1)[:, np.newaxis])

        numerator = np.array(numerator)
        for i in range(len(u_bar)):
            numerator_tmp = np.repeat(numerator[i][:, np.newaxis], len(v_tilde[i]), axis=1)
            tmp = (numerator_tmp / denominator[i]) + u_bar[i]
            membership_matrixs.append(tmp)
            
        return membership_matrixs
                

    def sscfcmeans(self, datas: np.ndarray, u_bar: list, C: np.ndarray, seed: int = 42) -> tuple:
        membership_matrixs, cluster_centers = self.stage1(datas, u_bar, C=C)
        for step in range(self._maxiter):
            v_tilde = self.V_tilde(cluster_centers, C)
            old_cluster_centers = cluster_centers.copy()
            sdistances = []
            for i in range(len(datas)):
                tmp = norm_distances(datas[i], cluster_centers[i])
                sdistances.append(tmp)

            membership_matrixs = self.update_membership_matrix(cluster_centers, v_tilde, sdistances, u_bar)
            cluster_centers = self.update_cluster_centers(datas, membership_matrixs, u_bar, v_tilde)
            
            if (np.abs(cluster_centers[0] - old_cluster_centers[0])).max(axis=(0, 1)) < self._epsilon and (np.abs(cluster_centers[1] - old_cluster_centers[1])).max(axis=(0, 1)) < self._epsilon and (np.abs(cluster_centers[2] - old_cluster_centers[2])).max(axis=(0, 1)) < self._epsilon:
                break
        return membership_matrixs, cluster_centers, step + 1