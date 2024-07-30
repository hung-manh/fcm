import yaml
import time
from models.fcm import Dfcm
from utils.utils import *
from utils.validity import *
from utils.image_data_utils import image2data, data2image, image_in_folder2data


def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


if __name__ == "__main__":
    # ------------------------------------------
    config = load_config('config/fcm_image.yaml')
    maxiter = config['maxiter']
    n_clusters = config['n_clusters']
    m = config['m']
    epsilon = config['epsilon']
    seed = config['seed']
    input_image_path = config['input_image_path']
    output_image_path = config['output_image_path']
    # ------------------------------------------
    start_time = time.time()
    
    data, image_data_shape = image2data(input_image_path)
    
    if config['debug']['print_time']:
        print("Thời gian lấy dữ liệu:", round_float(time.time() - start_time))
    if config['debug']['print_image_size']:
        print("Kích thước ảnh gốc:", image_data_shape)
        print("Kích thước ảnh cần xử lý:", data.shape)
    
    # ------------------------------------------
    _start_time = time.time()
    dfcm = Dfcm(m, epsilon, maxiter)
    U, V, step = dfcm.cmeans(data, n_clusters, seed)
    labels = extract_labels(U)
    clusters = extract_clusters(data, labels, n_clusters)
    
    # ------------------------------------------
    if config['debug']['print_time']:
        print("Thời gian tính toán", round_float(time.time() - _start_time))
    if config['debug']['print_steps']:
        print("Số bước lặp:", step)
    if config['debug']['print_U']:
        print("Ma trận độ thuộc U:", len(U), U[:1], '...')
    if config['debug']['print_V']:
        print("Ma tran tâm cụm V:", len(V), V[:1], '...')

    # ------------------------------------------
    data2image(labels, clusters, image_data_shape, output_image_path)
    
    # ------------------------------------------
    # Tính toán các chỉ số đánh giá dựa trên cấu hình
    if config['validity_indices']['dunn_index']:
        print("Chỉ số Dunn:", dunn_index(clusters))
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
        print("Chỉ số CE:", classification_entropy(labels))
    if config['validity_indices']['fuzzy_hypervolume']:
        print("Chỉ số FHV:", fuzzy_hypervolume(U, m))
    if config['validity_indices']['cs_index']:
        print("Chỉ số CS:", cs_index(clusters, V))
        