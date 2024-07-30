import yaml
import time
from models.fcm_parallel import Dfcm_parallel
from utils.utils import *
from utils.load_dataset_UCI import fetch_data_from_uci, TEST_CASES
from utils.validity import * 


def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


if __name__ == "__main__":
    # ------------------------------------------
    config = load_config('config/fcmp_data.yaml')
    maxiter = config['maxiter']
    m = config['m']
    epsilon = config['epsilon']
    seed = config['seed']
    num_processes = config['num_processes']
    DATA_ID = config['data_id']  
    C = TEST_CASES[DATA_ID]['n_cluster']
    
    # ------------------------------------------
    _start_time = time.time()
    if DATA_ID in TEST_CASES:
        _dt = fetch_data_from_uci(DATA_ID)
        if not _dt:
            print('Không thể lấy dữ liệu')
            exit()
        print("Thời gian lấy dữ liệu:", round_float(time.time() - _start_time))
        # --------------------------------
        _start_time = time.time()
        dfcmp = Dfcm_parallel(num_processes)
        U, V = dfcmp.parallel_cmeans(_dt['X'], C, m, epsilon, maxiter, seed)
        labels = extract_labels(U)
        clusters = extract_clusters(_dt['X'], labels, C)
        
        # ------------------------------------------
        # In thông tin cần thiết
        if config['debug']['print_time']:
            print("Thời gian tính toán", round_float(time.time() - _start_time))
        # if config['debug']['print_steps']:
        #     print("Số bước lặp:", step)
        if config['debug']['print_U']:
            print("Ma trận độ thuộc U:", len(U), U[:1], '...')
        if config['debug']['print_V']:
            print("Ma tran tâm cụm V:", len(V), V[:1], '...')
        
        # ------------------------------------------
        # Tính toán các chỉ số đánh giá dựa trên cấu hình
        if config['validity_indices']['dunn_index']:
            print("Chỉ số Dunn:", dunn_index(clusters))
        if config['validity_indices']['davies_bouldin_index']:
            print("Chỉ số Davies-Bouldin:", davies_bouldin_index(clusters, V))
        if config['validity_indices']['separation_index']:
            print("Chỉ số SI:", separation_index(clusters, V))
        if config['validity_indices']['partition_coefficient']:
            print("Chỉ số PCI:", partition_coefficient(U))
        if config['validity_indices']['classification_entropy']:
            print("Chỉ số CEI:", classification_entropy(labels))
        if config['validity_indices']['fuzzy_hypervolume']:
            print("Chỉ số Fuzzy Hypervolume:", fuzzy_hypervolume(V))
        if config['validity_indices']['cs_index']:
            print("Chỉ số CS:", cs_index(clusters, V))