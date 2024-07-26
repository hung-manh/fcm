import yaml
import time
from models.fcm_parallel import Dfcm_parallel
from utils.utils import *
from utils.validity import *
from utils.image_data_utils import image2data, data2image


def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


if __name__ == "__main__":
    # ------------------------------------------
    config = load_config('config/fcmp_image.yaml')
    maxiter = config['maxiter']
    n_clusters = config['n_clusters']
    m = config['m']
    epsilon = config['epsilon']
    seed = config['seed']
    num_processes = config['num_processes']
    input_image_path = config['input_image_path']
    
    # ------------------------------------------
    start_time = time.time()
    data = image2data(input_image_path)
    if config['debug']['print_time']:
        print("Thời gian lấy dữ liệu:", round_float(time.time() - start_time))
    if config['debug']['print_image_size']:
        print("Kích thước ảnh:", data.shape)
    
    # ------------------------------------------
    _start_time = time.time()
    dfcmp = Dfcm_parallel(num_processes)
    U, V = dfcmp.parallel_cmeans(data, n_clusters, m, epsilon, maxiter, seed)
    labels = extract_labels(U)
    clusters = extract_clusters(data, labels, n_clusters)
    
    # ------------------------------------------
    if config['debug']['print_time']:
        print("Thời gian tính toán", round_float(time.time() - _start_time))
    # if config['debug']['print_steps']:
    #     print("Số bước lặp:", step)
    if config['debug']['print_U']:
        print("Ma trận độ thuộc U:", len(U), U[:1], '...')
    if config['debug']['print_V']:
        print("Ma tran tâm cụm V:", len(V), V[:1], '...')

    # ------------------------------------------
    data2image(labels, clusters, config['output_image_path'])
    
    # ------------------------------------------
    # Tính toán các chỉ số đánh giá dựa trên cấu hình
    if config['validity_indices']['dunn_index']:
        print("Chỉ số Dunn:", dunn_index(clusters))
    if config['validity_indices']['davies_bouldin_index']:
        print("Chỉ số Davies-Bouldin:", davies_bouldin_index(clusters, V))
    if config['validity_indices']['davies_bouldin_index_sckitlearn']:
        print("Chỉ số Davies-Bouldin scikit-learn:", davies_bouldin_index_sckitlearn(data, labels))
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