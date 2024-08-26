import time
from models.fcm import Dfcm
from utils.utils import *
from utils.validity import * 
from utils.image_data_utils import image2data, data2image, image_in_folder2data
import os

if __name__ == "__main__":
# ------------------------------------------
    maxiter = 10000
    m = 2
    C = 6
    epsilon = 1e-5
    seed = 42
    metrics = []
    num_processes = 3
    input_image_path_folder = 'data/images/Anh-ve-tinh/Anh-da-pho/HaNoi'
    input_image_path = 'data/images/Labels/subdemo3_1.tif'
    input_image_label_path = 'data/images/Labels/subdemo3_1_water.jpg'
    output_image_path = 'outputs/images/seg.jpg'

    output_path = "./outputs/images/"
    os.makedirs(output_path, exist_ok=True)

    log_path = "outputs/logs/"
    os.makedirs(log_path, exist_ok=True)

    # ------------------------------------------
    start_time = time.time()
    # ------------------------------------------
    # Ảnh màu nối tiếp
    data, image_data_shape1 = image2data(input_image_path)
    # data, labeled, u_bar = create_membership_for_semi_supervised_learning_image(data1, ratio=0.1, C=C)
    _start_time = time.time()
    dfcm = Dfcm(m, epsilon, maxiter)
    U3, V3, step = dfcm.cmeans(data, C, seed)
    labels = extract_labels(U3)
    clusters = extract_clusters(data, labels, C)
    metric_nt = {
    'FCM': f"Ảnh mầu nt",
    'Time': round_float(time.time() - _start_time),
    'DB': davies_bouldin_index(data, labels) ,
    'PC': partition_coefficient(U3) ,
    'CE': classification_entropy(U3) ,
    'S': separation_index(data, U3, V3, m) ,
    'CH': calinski_harabasz_index(data, labels) ,
    'FHV': fuzzy_hypervolume(U3, m) ,
    'CS': cs_index(data, U3, V3, m) 
    }
    metrics.append(metric_nt)
    print("1")
    data2image(labels, clusters, image_data_shape1, output_image_path)
