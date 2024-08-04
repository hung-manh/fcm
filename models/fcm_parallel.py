import numpy as np 
from multiprocessing import Pool, cpu_count
from .fcm import Dfcm
from utils.utils import norm_distances

class Dfcm_parallel:
    def __init__(self, num_processes: int = None):
        self._num_processes = num_processes if num_processes is not None else cpu_count() # Số lượng Process tham gia

    def __data_chunks(self, data: np.ndarray):
        return np.array_split(data, self._num_processes)
    
    # Thực hiện phân cụm cộng tác mờ song song trên nhiều process.
    def parallel_cmeans(self, data: np.ndarray, C: int, m: int = 2, epsilon: float = 1e-5, maxiter: int = 1000,  seed: int = 42):
        data_chunks = self.__data_chunks(data)
        fcm = Dfcm(m=m, epsilon=epsilon, maxiter=maxiter)
        
        # Thực hiện phân cụm mờ song song trên nhiều process        
        with Pool(self._num_processes) as pool:
            results = pool.starmap(fcm.cmeans, [(chunk, C, seed) for p, chunk in enumerate(data_chunks)]) 
            
        # Kết hợp kết quả từ các process
        # # Cách 1: cộng U rồi cập nhật V rồi cập nhật U 
        # all_u = np.concatenate([results[0] for results in results], axis=0)
        # all_v = fcm.update_cluster_centers(data, all_u)  
        
        # all_u = fcm.update_membership_matrix(euclidean_cdist(data, all_v))
        
        # cách 2: Chuẩn hoá đc tính toán phân cụm FCM song song, tổng hợp UV bằng cách tìm V mới 
        # bằng cách thực hiện FCM cho V tổng (cộng cơ học theo hàng), sau đó tính lại U mới theo V mới. 
        # KQ các chỉ số đánh giá cũng đẹp, phân cụm ảnh cũng giống nối tiếp với time nhanh hơn.
        all_v = np.concatenate([results[1] for results in results], axis = 0)
        _, V, _ = fcm.cmeans(all_v, C, seed)
        U = fcm.update_membership_matrix(norm_distances(data, V))

        return U, V
        
            
        