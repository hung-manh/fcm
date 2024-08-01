import time
from models.fcm import Dfcm 
from models.fcm_parallel import Dfcm_parallel
from utils.utils import *
from utils.validity import * 
from utils.image_data_utils import image2data, data2image, image_in_folder2data


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
    input_image_path = 'data/images/Ha Noi.tif'
    output_image_path = 'outputs/images/seg.jpg'
    # ------------------------------------------
    start_time = time.time()
        
    # ------------------------------------------
    # Ảnh đa phổ nối tiếp
    data, image_data_shape = image_in_folder2data(input_image_path_folder)
    data, image_data_shape1 = image2data(data)
    
    _start_time = time.time()
    dfcm = Dfcm(m, epsilon, maxiter)
    U1, V1, step = dfcm.cmeans(data, C, seed)
    labels = extract_labels(U1)
    clusters = extract_clusters(data, labels, C)
    metric_nt = {
    'FCM': f"Ảnh đa phổ nt",
    'Time': round_float(time.time() - _start_time),
    'DB': davies_bouldin_index(data, labels) ,
    'PC': partition_coefficient(U1) ,
    'CE': classification_entropy(U1) ,
    'S': separation_index(data, U1, V1, m) ,
    'CH': calinski_harabasz_index(data, labels) ,
    'FHV': fuzzy_hypervolume(U1, m) ,
    'CS': cs_index(data, U1, V1, m) 
    }
    metrics.append(metric_nt)
    print("1")
    data2image(labels, clusters, image_data_shape1, output_image_path)
    
    # Ảnh đa phổ song song 
    _start_time = time.time()
    dfcm = Dfcm_parallel(num_processes)
    U2, V2 = dfcm.parallel_cmeans(data, C, m, epsilon, maxiter, seed)
    labels = extract_labels(U2)
    clusters = extract_clusters(data, labels, C)
    metric_ss = {
    'FCM': f"Ảnh đa phổ ss",
    'Time': round_float(time.time() - _start_time),
    'DB': davies_bouldin_index(data, labels) ,
    'PC': partition_coefficient(U2) ,
    'CE': classification_entropy(U2) ,
    'S': separation_index(data, U2, V2, m) ,
    'CH': calinski_harabasz_index(data, labels) ,
    'FHV': fuzzy_hypervolume(U2, m) ,
    'CS': cs_index(data, U2, V2, m) 
    }
    metrics.append(metric_ss)
    data2image(labels, clusters, image_data_shape1, output_image_path)
    print("1")
    
    # ------------------------------------------
    # Ảnh màu nối tiếp
    data, image_data_shape1 = image2data(input_image_path)
    
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
    
    # Ảnh màu song song 
    _start_time = time.time()
    dfcm = Dfcm_parallel(num_processes)
    U4, V4 = dfcm.parallel_cmeans(data, C, m, epsilon, maxiter, seed)
    labels = extract_labels(U4)
    clusters = extract_clusters(data, labels, C)
    metric_ss = {
    'FCM': f"Ảnh mầu ss",
    'Time': round_float(time.time() - _start_time),
    'DB': davies_bouldin_index(data, labels) ,
    'PC': partition_coefficient(U4) ,
    'CE': classification_entropy(U4) ,
    'S': separation_index(data, U4, V4, m) ,
    'CH': calinski_harabasz_index(data, labels) ,
    'FHV': fuzzy_hypervolume(U4, m) ,
    'CS': cs_index(data, U4, V4, m) 
    }
    metrics.append(metric_ss)
    print("1")
    export_to_latex_image(metrics, 'outputs/logs/fcm_data.txt')
    print("Metrics exported to fcm_data.txt")
    data2image(labels, clusters, image_data_shape1, output_image_path)
    