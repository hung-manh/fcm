import time
from models.fcm import Dfcm 
from utils.utils import round_float
from utils.load_dataset_UCI import fetch_data_from_uci
from utils.validity import (separation_index,
                           partition_coefficient,
                           classification_entropy,
                           fuzzy_hypervolume,
                           cs_index) 
TEST_CASES = {
    53: {
        'name': 'Iris',
        'n_cluster': 3,
        'test_points': [5.1, 3.5, 1.4, 0.2]
    },
    109: {
        'name': 'Wine',
        'n_cluster': 4,
        'test_points': [14.23, 1.71, 2.43, 15.6,
                        127, 2.80, 3.06, 0.28,
                        2.29, 5.64, 1.04, 3.92,
                        1065]
    },
    602: {
        'name': 'DryBean',
        'n_cluster': 7,
        'test_points': [
            28395, 610.291, 208.178117, 173.888747,
            1.197191, 0.549812, 28715, 190.141097,
            0.763923, 0.988856, 0.958027, 0.913358,
            0.007332, 0.003147, 0.834222, 0.998724]
    }
}

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
        dfcm = Dfcm()
        U, V, step, labels, clusters = dfcm.cmeans(data=_dt['X'])
        # --------------------------------
        print("Thời gian tính toán", round_float(time.time() - _start_time))
        print("Số bước lặp:", step)
        print("Ma trận độ thuộc U:", len(U), U[:1], '...')
        print("Ma tran tâm cụm V:", len(V), V[:1], '...') 
        # 1.1
        # print("Chỉ số Dunn:", dunn_index(clusters))
        # print("Chỉ số Davies-Bouldin:", davies_bouldin_index(clusters, V))
        print("Chỉ số SI:", separation_index(clusters, V))
        # 1.2
        print("Chỉ số PCI:", partition_coefficient(U))
        # 1.3
        print("Chỉ số CEI:", classification_entropy(labels))
        # 1.4 chưa cài được 
        
        # 1.5
        print("Chỉ số FPC:", partition_coefficient(U))
        print("Chỉ số FH:", fuzzy_hypervolume(U))
        print("Chỉ số CS:", cs_index(clusters, V))
        # --------------------------------