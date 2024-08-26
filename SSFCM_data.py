import yaml
import time
from models.SSFCM import SSfcm 
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
        ssfcm = SSfcm(m, epsilon, maxiter)  

        # split_data_for_semi_supervised_learning()
        data, labeled, u_bar = create_membership_for_semi_supervised_learning(_dt['X'], _dt['Y'], ratio=0.1, C=C)
        metrics = []
        U, V, step = ssfcm.sscmeans(data, u_bar, C, seed)
        labels = extract_labels(U)
        clusters = extract_clusters(data, labels, C)
        
        
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
        if config['validity_indices']['davies_bouldin_index']:
            print("Chỉ số DB:", davies_bouldin_index(data, labels))
        if config['validity_indices']['separation_index']:
            print("Chỉ số S:", separation_index(data, U, V, m))
        if config['validity_indices']['calinski_harabasz_index']:
            print("Chỉ số CH:", calinski_harabasz_index(data, labels))
        if config['validity_indices']['silhouette_index']:
            print("Chỉ số SI:", silhouette_index(data, labels))
        if config['validity_indices']['partition_coefficient']:
            print("Chỉ số PC:", partition_coefficient(U))
        if config['validity_indices']['classification_entropy']:
            print("Chỉ số CE:", classification_entropy(U))
        if config['validity_indices']['fuzzy_hypervolume']:
            print("Chỉ số FHV:", fuzzy_hypervolume(U, m))
        if config['validity_indices']['cs_index']:
            print("Chỉ số CS:", cs_index(data, U, V, m))
        if config['validity_indices']['AC']:
            print("Chỉ số AC:", accuracy_score(labels, labeled))
        if config['validity_indices']['F1']:
            print("Chỉ số F1:", f1_score(labels, labeled))
    
        metric_data = {
            'SSFCM': f"Data",
            'Time': round_float(time.time() - _start_time),
            'DB': davies_bouldin_index(data, labels),
            'PC': partition_coefficient(U),
            'CE': classification_entropy(U) ,
            'S': separation_index(data, U, V, m) ,
            'CH': calinski_harabasz_index(data, labels) ,
            'SI': silhouette_index(data, labels) ,
            'FHV': fuzzy_hypervolume(U, m) ,
            'CS': cs_index(data, U, V, m), 
            'F1': round(accuracy_score(labels, labeled), 2), 
            'AC': round(f1_score(labels, labeled), 2) 
        }
        metrics.append(metric_data)
        export_to_latex_image_v3(metrics, 'outputs/logs/ssfcm_data.txt')
        print("Metrics exported to ssfcm_data.txt")
        