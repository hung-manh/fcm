import time
import os 
from models.fcm import Dfcm 
from models.fcm_parallel import Dfcm_parallel
from utils.utils import *
from utils.load_dataset_UCI import fetch_data_from_uci, TEST_CASES
from utils.validity import * 


if __name__ == "__main__":
    # ------------------------------------------
    maxiter = 10000
    m = 2
    epsilon = 1e-5
    seed = 42
    DATA_ID = [53, 109, 602]
    metrics = []
    num_processes = 3
    folder_path = 'data/csv'
    # ------------------------------------------
    _start_time = time.time()
    
    if not os.path.isdir(folder_path):
        print(f'Duong dan thu muc {folder_path} khong ton tai')
        exit()
    
    csv_files = [os.path.join(folder_path ,f) for f in os.listdir(folder_path) if f.endswith('.csv')]
    for i, csvfile in enumerate(csv_files):
        ID = DATA_ID[i]
        _dt = fetch_data_from_uci(name_or_id=ID, file_csv=csvfile)
        C = TEST_CASES[DATA_ID[i]]['n_cluster']
        if not _dt:
            print('Không thể lấy dữ liệu')
            exit()
        print("Thời gian lấy dữ liệu:", round_float(time.time() - _start_time))
        # --------------------------------
        _start_time = time.time()
        # Nối tiếp 
        dfcm = Dfcm(m, epsilon, maxiter)
        U, V, step = dfcm.cmeans(_dt['X'], C, seed)
        labels = extract_labels(U)
        clusters = extract_clusters(_dt['X'], labels, C)
        metric_nt = {
        'FCM': f"{ID} nt ",
        'Size': f"{_dt['X'].shape[0]}x{_dt['X'].shape[1]}",
        'C': C,
        'Time': round_float(time.time() - _start_time),
        'DI': dunn_index(clusters) ,
        'DB': davies_bouldin_index(_dt['X'], labels) ,
        'PC': partition_coefficient(U) ,
        'CE': classification_entropy(labels) ,
        'S': separation_index(_dt['X'], U, V, m) ,
        'CH': calinski_harabasz_index(_dt['X'], labels) ,
        'SI': silhouette_index(_dt['X'], labels) ,
        'FHV': fuzzy_hypervolume(U, m) ,
        'CS': cs_index(_dt['X'], U, V, m) 
        }
        metrics.append(metric_nt)
        
        # Song song
        # --------------------------------
        _start_time = time.time()
        dfcmp = Dfcm_parallel(num_processes)
        U1, V1 = dfcmp.parallel_cmeans(_dt['X'], C, m, epsilon, maxiter, seed)
        labels = extract_labels(U1)
        clusters = extract_clusters(_dt['X'], labels, C)
        
        metric_ss = {
        'FCM': f"{ID} ss",
        'Size': f"{_dt['X'].shape[0]}x{_dt['X'].shape[1]}",
        'C': C,
        'Time': round_float(time.time() - _start_time),
        'DI': dunn_index(clusters),
        'DB': davies_bouldin_index(_dt['X'], labels),
        'PC': partition_coefficient(U1),
        'CE': classification_entropy(labels),
        'S': separation_index(_dt['X'], U1, V1, m),
        'CH': calinski_harabasz_index(_dt['X'], labels),
        'SI': silhouette_index(_dt['X'], labels),
        'FHV': fuzzy_hypervolume(U1, m),
        'CS': cs_index(_dt['X'], U1, V1, m)
        }
        metrics.append(metric_ss)
        
    export_to_latex_data(metrics, 'outputs/logs/fcm_data.txt')
    print("Metrics exported to fcm_data.txt")