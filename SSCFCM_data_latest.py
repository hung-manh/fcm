import yaml
import time
from models.SSCFCM_latest import SSCFCM
from utils.utils import *
from utils.load_dataset_UCI import fetch_data_from_uci, TEST_CASES
from utils.validity import * 
import os


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

    output_path = "./outputs/images/"
    os.makedirs(output_path, exist_ok=True)

    log_path = "outputs/logs/"
    os.makedirs(log_path, exist_ok=True)
    
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
        C_list = np.array([5, 6, 7])
        datas, labeled, u_bar = create_membership_label(_dt['X'], _dt['Y'], n_sites= 3, ratio=0.3, C=C_list)
        sscfcm = SSCFCM(datas=datas, membership_bar=u_bar, n_clusters=C_list, n_sites=3, m=2, epsilon=epsilon, max_iter=maxiter)
        step = sscfcm.fit_transform()
        datasites = sscfcm.get_datasites()
        U = [x[0] for x in datasites]
        V = [x[1] for x in datasites]
        metrics = []
        for i in range(len(datas)):
            labels = extract_labels(U[i])
            clusters = extract_clusters(datas[i], labels, C_list[i])
            
            # ------------------------------------------
            # In thông tin cần thiết
            if config['debug']['print_time']:
                print("Thời gian tính toán", round_float(time.time() - _start_time))
            if config['debug']['print_steps']:
                print("Số bước lặp:", step)
            if config['debug']['print_U']:
                print("Ma trận độ thuộc U:", len(U[i]), U[i][:1], '...')
            if config['debug']['print_V']:
                print("Ma tran tâm cụm V:", len(V[i]), V[i][:1], '...')
            
            # ------------------------------------------
            # Tính toán các chỉ số đánh giá dựa trên cấu hình
            # if config['alidity_indices']['dunn_index']:
            #     print("Chỉ số Dunn:", dunn_index(clusters))
            if config['validity_indices']['davies_bouldin_index']:
                print("Chỉ số DB:", davies_bouldin_index(datas[i], labels))
            if config['validity_indices']['separation_index']:
                print("Chỉ số S:", separation_index(datas[i], U[i], V[i], m))
            if config['validity_indices']['calinski_harabasz_index']:
                print("Chỉ số CH:", calinski_harabasz_index(datas[i], labels))
            if config['validity_indices']['silhouette_index']:
                print("Chỉ số SI:", silhouette_index(datas[i], labels))
            if config['validity_indices']['partition_coefficient']:
                print("Chỉ số PC:", partition_coefficient(U[i]))
            if config['validity_indices']['classification_entropy']:
                print("Chỉ số CE:", classification_entropy(U[i]))
            if config['validity_indices']['fuzzy_hypervolume']:
                print("Chỉ số FHV:", fuzzy_hypervolume(U[i], m))
            if config['validity_indices']['cs_index']:
                print("Chỉ số CS:", cs_index(datas[i], U[i], V[i], m))
            if config['validity_indices']['AC']:
                print("Chỉ số AC:", accuracy_score(labels, labeled[i]))
            if config['validity_indices']['F1']:
                print("Chỉ số F1:", f1_score(labels, labeled[i]))
            metric_data = {
                'SSCFCM': f"Data with {C_list[i]} center clusters",
                'Time': round_float(time.time() - _start_time),
                'DB': davies_bouldin_index(datas[i], labels),
                'PC': partition_coefficient(U[i]) ,
                'CE': classification_entropy(U[i]) ,
                'S': separation_index(datas[i], U[i], V[i], m) ,
                'CH': calinski_harabasz_index(datas[i], labels) ,
                'SI': silhouette_index(datas[i], labels) ,
                'FHV': fuzzy_hypervolume(U[i], m) ,
                'CS': cs_index(datas[i], U[i], V[i], m), 
                'F1': round(accuracy_score(labels, labeled[i]), 2), 
                'AC': round(f1_score(labels, labeled[i]), 2) 
            }
            metrics.append(metric_data)
        export_to_latex_image_v2(metrics, 'outputs/logs/sscfcm_data.txt')
        print("Metrics exported to fcm_data.txt")
        