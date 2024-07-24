import time
from models.fcm_parallel import Dfcm_parallel
from utils.utils import *
from utils.validity import (separation_index,
                           partition_coefficient,
                           classification_entropy,
                           fuzzy_hypervolume,
                           cs_index) 
from utils.load_image2data import image2data

if __name__ == "__main__":
    import time
    _start_time = time.time()
    MAX_ITER = 10000000  # 000
    img_path = 'data/images/k1_1024x1024.tif'
    data = image2data(img_path) 
    print("Thời gian lấy dữ liệu:", round_float(time.time() - _start_time))
    print("Kích thước ảnh:", data.shape)
    # --------------------------------
    _start_time = time.time()
    dfcmp = Dfcm_parallel()
    U, V, step = dfcmp.parallel_cmeans(data= data,  C= 3, maxiter=MAX_ITER)
    
    # --------------------------------
    print("Thời gian tính toán", round_float(time.time() - _start_time))
    # print("Số bước lặp:", step)
    print("Ma trận độ thuộc U:", len(U), U[:1], '...')
    print("Ma tran tâm cụm V:", len(V), V[:1], '...') 
    # # 1.1
    # # print("Chỉ số Dunn:", dunn_index(clusters))
    # # print("Chỉ số Davies-Bouldin:", davies_bouldin_index(clusters, V))
    # # print("Chỉ số SI:", separation_index(clusters, V))
    # # 1.2
    print("Chỉ số PCI:", partition_coefficient(U))
    # # 1.3
    # print("Chỉ số CEI:", classification_entropy(labels))
    # # 1.4 chưa cài được 
    
    # # 1.5
    # print("Chỉ số FPC:", partition_coefficient(U))
    # print("Chỉ số FH:", fuzzy_hypervolume(U))
    # print("Chỉ số CS:", cs_index(clusters, V))
    # # --------------------------------