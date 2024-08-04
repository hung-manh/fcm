import yaml
import time
from models.fcm import Dfcm 
from utils.utils import *
from utils.load_dataset_UCI import fetch_data_from_uci, TEST_CASES
from utils.validity import * 


def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


if __name__ == "__main__":
    # ------------------------------------------
    config = load_config('config/fcm_data.yaml')
    maxiter = config['maxiter']
    m = config['m']
    epsilon = config['epsilon']
    seed = config['seed']
    # seed = 24
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
        dfcm = Dfcm(m, epsilon, maxiter)
        U, V, step = dfcm.cmeans(_dt['X'], C, seed)
        labels = extract_labels(U)
        clusters = extract_clusters(_dt['X'], labels, C)
        
        # ------------------------------------------
        # In thông tin cần thiết
        if config['debug']['print_time']:
            print("Thời gian tính toán", round_float(time.time() - _start_time))
        if config['debug']['print_steps']:
            print("Số bước lặp:", step)
        if config['debug']['print_U']:
            print("Ma trận độ thuộc U:", len(U), U[:1], '...')
        if config['debug']['print_V']:
            print("Ma tran tâm cụm V:", len(V), V[:1], '...')
        
        # ------------------------------------------
        # Tính toán các chỉ số đánh giá dựa trên cấu hình
        # if config['validity_indices']['dunn_index']:
        #     print("Chỉ số Dunn:", dunn_index(clusters))
        # if config['validity_indices']['davies_bouldin_index']:
        #     print("Chỉ số DB:", davies_bouldin_index(_dt['X'], labels))
        # if config['validity_indices']['separation_index']:
        #     print("Chỉ số S:", separation_index(_dt['X'], U, V, m))
        # if config['validity_indices']['calinski_harabasz_index']:
        #     print("Chỉ số CH:", calinski_harabasz_index(_dt['X'], labels))
        # if config['validity_indices']['silhouette_index']:
        #     print("Chỉ số SI:", silhouette_index(_dt['X'], labels))
        if config['validity_indices']['partition_coefficient']:
            print("Chỉ số PC:", partition_coefficient(U))
        # if config['validity_indices']['classification_entropy']:
        #     print("Chỉ số CE:", classification_entropy(U))
        # if config['validity_indices']['fuzzy_hypervolume']:
        #     print("Chỉ số FHV:", fuzzy_hypervolume(U, m))
        # if config['validity_indices']['cs_index']:
        #     print("Chỉ số CS:", cs_index(_dt['X'], U, V, m))
        
        #-------------------------------------------
        y_pred = labels
        y_true = _dt['Y']
        labels2number = {v: k for k, v in enumerate(np.unique(y_true))}
        y_true = np.array([labels2number[i] for i in y_true])
        print("Accuracy:", accuracy_score(y_true, y_pred))
        print("Precision:", precision(y_true, y_pred))
        print("Recall:", recall(y_true, y_pred))
        print("F1 Score:", f1_score(y_true, y_pred))
        print("MSE:", mean_squared_error(y_true, y_pred))
        print("MAE:", mean_absolute_error(y_true, y_pred))
        print("Purity:", purity_score(y_true, y_pred))
        print("NMI:", normalized_mutual_info_score(y_true, y_pred))
        print("SSE:", sum_of_square_error(y_true, y_pred))  