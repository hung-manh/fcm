import numpy as np
from utils.utils import norm_distances
from models.fcm import Dfcm


class SSCFCM:
    def __init__(self, datas: list, membership_bar: list, n_clusters: np.ndarray, n_sites: int, m: float = 2, epsilon: float = 1e-5, max_iter: int = 10000):
        self.__n_clusters = n_clusters
        self.__n_sites = n_sites
        self.__m = m
        self.__epsilon = epsilon
        self.__max_iter = max_iter
        self.__datasites = []
        self.__exited_states = [False] * n_sites
        self.__datas = datas
        self.__membership_bar = membership_bar

    # Stage 1: Local clustering at each data site
    def stage1_fcm(self, seed: int = 42) -> list:
        self.__datasites = []
        for i, data in enumerate(self.__datas):
            dsi = Dfcm(m=self.__m, epsilon=self.__epsilon, maxiter=self.__max_iter)
            U, V, step = dsi.cmeans(data=data, C=self.__n_clusters[i], seed=seed)
            self.__datasites.append([U, V])
    
    # Stage 2: Collaboration between data sites
    def stage2_collaborative(self) -> int:
        for step in range(self.__max_iter):
            v_tilde = self.__V_tilde(self.__datasites, self.__n_clusters)

            for i, datasite in enumerate(self.__datasites):
                if not self.__exited_states[i]:  # Check if the data site should still be updated
                    old_centroid = datasite[1].copy()
                    self.__update_datasite(v_tilde=v_tilde, i=i)
                    if self.chk_centroids_break(datasite[1], old_centroid):
                        self.__exited_states[i] = True  # Mark this data site as exited

            if all(self.__exited_states):  # Stop if all data sites have exited
                return step

    def get_datasites(self):
        return self.__datasites

    def fit_transform(self):
        self.stage1_fcm()
        step = self.stage2_collaborative()
        return step
    
    def __V_tilde(self, datasites: list, C: np.ndarray) -> list:
        all_datasite_centroids = [datasite[1] for datasite in datasites]
        all_centroids = np.vstack(all_datasite_centroids)
        v_tilde = []
        fcm = Dfcm(m=self.__m, epsilon=self.__epsilon, maxiter=self.__max_iter)
        for i, dsi in enumerate(datasites):
            U, V, step = fcm.cmeans(data=all_centroids, C=C[i])
            v_tilde.append(V)
        return v_tilde
    
    def chk_centroids_break(self, centroids: np.ndarray, old_centroid: np.ndarray) -> bool:
        return np.linalg.norm(centroids - old_centroid) < self.__epsilon
    
    def __update_datasite(self, v_tilde: np.ndarray, i: int = 0):
        beta = self.__update_beta(i)
        self.update_membership(v_tilde=v_tilde, beta=beta, i=i)
        self.update_centroids(v_tilde=v_tilde, beta=beta, i=i)


    def update_membership(self, v_tilde: list, beta: float, i: int):
        distances = norm_distances(self.__datas[i], self.__datasites[i][1])
        d_tilde = np.linalg.norm(self.__datasites[i][1] - v_tilde[i], axis=1)
        denominator = np.zeros((len(self.__datasites[i][0]), self.__n_clusters[i]))

        for j in range(self.__n_clusters[i]):
            denominator[:, j] = (1 / (distances[:, j] ** 2 + beta * (d_tilde[j] ** 2))) ** (1 / (self.__m - 1))

        sum_denominators = np.sum(denominator, axis=1)

        for r in range(self.__n_clusters[i]):
            numerator = (distances[:, r] ** 2 + beta * (d_tilde[r] ** 2)) ** (-1)
            self.__datasites[i][0][:, r] = numerator / sum_denominators
    

    def update_centroids(self, v_tilde: list, beta: float, i: int):
        um = (self.__datasites[i][0] - self.__membership_bar[i]) ** self.__m
        _V1 = um.T @ self.__datas[i]
        _V2 = beta * np.sum(um, axis=0)[:, None] * v_tilde[i]
        denominator = (1 + beta) * np.sum(um, axis=0)[:, None]
        self.__datasites[i][1] = (_V1 + _V2) / denominator


    def __update_beta(self, i: int) -> float:
        J = np.sum((self.__datasites[i][0] ** self.__m) * (norm_distances(self.__datas[i], self.__datasites[i][1])) ** 2)
        result = 0
        for jj in range(self.__n_sites):
            if jj != i:
                U_tilde = self.U_tilde(data=self.__datas[i], centroids=self.__datasites[jj][1])
                J_tilde = np.sum((U_tilde ** 2) * (norm_distances(self.__datas[i], self.__datasites[jj][1]) ** 2))
                result += min(1, J / J_tilde)
        
        return result / (self.__n_sites - 1)


    def U_tilde(self, data: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        distances = norm_distances(data, centroids)
        return Dfcm.update_membership_matrix_2(distances=distances)
    
