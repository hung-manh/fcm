import numpy as np 
from multiprocessing import Pool, cpu_count
from .fcm import Dfcm


class Dfcm_parallel:
    def __init__(self, n_jobs: int = None):
        self._n_jobs = n_jobs if n_jobs is not None else cpu_count()

    # Thực hiện phân cụm cộng tác mờ song song trên nhiều process.
    def parallel_cmeans(self, data: np.ndarray, C: int, m: int = 2, epsilon: float = 1e-5, maxiter: int = 1000):
        n_samples = len(data)   
        chunk_size = n_samples // self._n_jobs
        data_chunks = [data[i:i + chunk_size] for i in range(0, n_samples, chunk_size)]

        fcm = Dfcm(C, m, epsilon, maxiter)
        
        with Pool(self._n_jobs) as pool:
            
            results = pool.starmap(fcm.cmeans, [(chunk, p+1) for p, chunk in enumerate(data_chunks)]) 
            
            u, v, step = zip(*results) 
            u = np.vstack(u) # Gộp các ma trận thành viên của các process
            v = fcm.update_cluster_centers(data, u) # Cập nhật tâm cụm từ ma trận thành viên (bat buoc phai gop u xong moi tinh v)
            
        return u, v, step
        
            
        