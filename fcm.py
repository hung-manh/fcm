import time
from models.fcm import Dfcm 
from utils.utils import *
from utils.load_dataset_UCI import fetch_data_from_uci, TEST_CASES
from utils.validity import (dunn_index,
                            davies_bouldin_index,
                            separation_index,
                           partition_coefficient,
                           classification_entropy,
                           fuzzy_hypervolume,
                           cs_index) 
# --------------------------------Sklearn--------------------------------
from sklearn.metrics import davies_bouldin_score as dbs

if __name__ == "__main__":
    import time
    _start_time = time.time()
    MAX_ITER = 10000000  # 000
    DATA_ID = 602  # 53: Iris, 109: Wine, 602: DryBean
    if DATA_ID in TEST_CASES:
        _dt = fetch_data_from_uci(DATA_ID)
        if not _dt:
            print('Không thể lấy dữ liệu')
            exit()
        print("Thời gian lấy dữ liệu:", round_float(time.time() - _start_time))
        # --------------------------------
        _start_time = time.time()
        dfcm = Dfcm(C=TEST_CASES[DATA_ID]['n_cluster'])
        U, V, step = dfcm.cmeans(data=_dt['X'])
        labels = extract_labels(U)
        clusters = extract_clusters(_dt['X'], labels, TEST_CASES[DATA_ID]['n_cluster'])
        # --------------------------------
        print("Thời gian tính toán", round_float(time.time() - _start_time))
        print("Số bước lặp:", step)
        # print("Ma trận độ thuộc U:", len(U), U[:1], '...')
        # print("Ma tran tâm cụm V:", len(V), V[:1], '...') 
        
        
        # 1.1
        print("Chỉ số Dunn:", dunn_index(clusters))
        
        print("Chỉ số Davies-Bouldin:", davies_bouldin_index(clusters, V))
        # print("Chỉ số Davies-Bouldin Sklearn:", dbs(_dt['X'], labels))
        
        print("Chỉ số SI:", separation_index(clusters, V))
        
        
        # 1.2
        print("Chỉ số PCI:", partition_coefficient(U))
        
        
        # # 1.3
        print("Chỉ số CEI:", classification_entropy(labels))
        # # 1.4 chưa cài được 
        
        # 1.5
        print("Chỉ số FPC:", partition_coefficient(U))
        print("Chỉ số FH:", fuzzy_hypervolume(U))
        print("Chỉ số CS:", cs_index(clusters, V))
        # --------------------------------