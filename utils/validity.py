import numpy as np 


# 1.1 Chỉ số đo lường mức độ tách biệt giữa các cụm
## 1.1.1 Chỉ số Dunn (DI)
def dunn_index(clusters:np.ndarray)->float:
    """
    Args:
    clusters: Danh sách các cụm, mỗi cụm là một danh sách các điểm dữ liệu.
    Giá trị trả về: Chỉ số Dunn, càng cao càng tốt, càng thể hiện độ tách biệt giữa các cụm.
    
    Đo lường khoảng cách tối thiểu giữa các cụm so với khoảng cách tối đa giữa các điểm trong cùng một cụm.
    - Tính khoảng cách giữa các cụm, tính inter-cluster distance, d_i = min(d(c_i, c_j)), i!=j, c_i, c_j là các cụm.
    - Tính đường kính của mỗi cụm, tính intra-cluster distance, d_j = max(d(x_i, x_j)), x_i, x_j là các điểm trong cụm.
    
    Chỉ số Dunn = min(d_i) / max(d_j)
    Khoảng giá trị: [0, +inf], thuờng nằm trong khoảng [0, 1]
    https://medium.com/@mastmustu/dunn-index-reveals-the-holy-grail-of-optimal-clustering-a48c5bc960e
    """
    # Tính khoảng cách giữa các cụm, tính inter-cluster distance
    ds = []
    for i in range(len(clusters)):
        for j in range(i+1,len(clusters)):
            ds.append(np.min(np.linalg.norm(clusters[i][:,np.newaxis]-clusters[j],axis=2)))
    
    # Tính đường kính của mỗi cụm, tính intra-cluster distance
    diams = []
    for cluster in clusters:
        # np.linalg.norm(cluster[:, np.newaxis] - cluster, axis=2): Tính khoảng cách giữa các điểm trong cụm
        diams.append(np.max(np.linalg.norm(cluster[:,np.newaxis]-cluster,axis=2)))
    return np.min(ds) / np.max(diams)

def davies_bouldin_index(clusters, centroids) -> float:
    """
    Args:
    clusters: Danh sách các cụm, mỗi cụm là một danh sách các điểm dữ liệu.
    centroids: Danh sách các tâm cụm.
    
    Chỉ số DBI đo lường mức độ chồng chéo giữa các cụm, giá trị càng thấp càng tốt. 
    Chỉ số này được tính bằng cách lấy trung bình của tất cả các cặp cụm, trong đó mỗi cụm được đo bằng khoảng cách giữa tâm cụm và trung bình khoảng cách của các điểm trong cụm đó.
    Khoảng giá trị: [0, +inf], càng thấp càng tốt.
    https://www.geeksforgeeks.org/davies-bouldin-index/
    """
    k = len(clusters)
    #  khoảng cách giữa các điểm trong cụm tới tâm cụm
    dispersions = [np.mean([np.linalg.norm(point - centroids[i]) for point in cluster]) for i, cluster in enumerate(clusters)]
        
    # Tính khoảng cách giữa các tâm cụm
    distances = np.linalg.norm(centroids[:, np.newaxis] - centroids, axis=2)
    
    # Tính Davies-Bouldin’s index
    db_index = 0
    for i in range(k):
        max_ratio = 0
        for j in range(k):
            if i != j:
                # Tính tỉ lệ giữa dispersion của 2 cụm và khoảng cách giữa 2 tâm cụm
                ratio = (dispersions[i] + dispersions[j]) / distances[i, j]
                # Tìm tỉ lệ lớn nhất
                if ratio > max_ratio:
                    max_ratio = ratio
        db_index += max_ratio
    
    return db_index / k

## 1.1.3 Chỉ số tách biệt Separation (SI)
def separation_index(clusters:np.ndarray,centroids:np.ndarray)->float:
    """
    Args:
    clusters: Danh sách các cụm, mỗi cụm là một danh sách các điểm dữ liệu.
    centroids: Danh sách các tâm cụm.
    
    Chỉ sô SI đo lường mức độ tách biệt giữa các cụm, càng cao càng tốt.
    """
    # Tính khoảng cách giữa các tâm cụm
    distances = np.linalg.norm(centroids[:, np.newaxis] - centroids, axis=2)
    # Tính đường kính của mỗi cụm
    diams = []
    for cluster in clusters:
        diams.append(np.max(np.linalg.norm(cluster[:, np.newaxis] - cluster, axis=2)))
    return np.min(distances) / np.max(diams)

## 1.1.4 Chỉ số Calinski-Harabasz (CH)

# 1.2 Chỉ số đo lường mức độ rõ ràng của các cụm
## 1.2.1 Hệ số phân vùng Partition Coefficient (PCI), Giống hệt chỉ số FPC
def partition_coefficient(membership:np.ndarray)->float: # Fuzzy Partition Coefficient
    """
    Args:
    membership: Ma trận độ thuộc, mỗi hàng là 1 điểm, mỗi cột là 1 cụm.
    Càng gần 1 càng tốt
    """
    return np.sum(np.square(membership)) / membership.shape[0]
    # return np.trace(np.dot(membership.T, membership)) / membership.shape[0]
    
# 1.3 Entropy phân loại Classification Entropy (CEI)
def classification_entropy(labels: np.ndarray) -> float:
    """
    Args:
    labels: Danh sách nhãn của các điểm dữ liệu.
    """
    # Tính tỉ lệ phần trăm điểm dữ liệu thuộc về mỗi cụm
    _unique_labels, counts = np.unique(labels, return_counts=True)
    proportions = counts / len(labels)
    return - np.sum(proportions * np.log2(proportions))

# 1.4 Chỉ số đo lường mức độ tương tự của các điểm dữ liệu với cụm
# 1.4.1 Chỉ số Silhouette

# 1.5 Các chỉ số khác
# 1.5.1 Fuzzy Partition Coefficient (FPC)
def partition_coefficient(membership:np.ndarray)->float: # Fuzzy Partition Coefficient
    """
    Args:
    membership: Ma trận độ thuộc, mỗi hàng là 1 điểm, mỗi cột là 1 cụm.
    Càng gần 1 càng tốt
    """
    # return np.sum(np.square(membership)) / membership.shape[0]
    return np.trace(np.dot(membership.T, membership)) / membership.shape[0]

# 1.5.2 Thể tích mờ Fuzzy Hypervolume (FH)
def fuzzy_hypervolume(membership:np.ndarray)->float:
    """
    Args:
    membership: Ma trận độ thuộc, mỗi hàng là 1 điểm, mỗi cột là 1 cụm.
    """
    # Tính thể tích của mỗi cụm
    volumes = np.sum(membership,axis=0)
    return np.sum(volumes)

# 1.6 Chỉ số CS
def cs_index(clusters: np.ndarray, centroids: np.ndarray) -> float:
    """
    Args:
    clusters: Danh sách các cụm, mỗi cụm là một danh sách các điểm dữ liệu.
    centroids: Danh sách các tâm cụm.
    """
    # Tính độ lệch chuẩn của mỗi cụm
    stds = [np.std(cluster, axis=0) for cluster in clusters]
    # Tính khoảng cách giữa các tâm cụm
    distances = np.linalg.norm(centroids[:, np.newaxis] - centroids, axis=2)
    result = 0
    for i in range(len(clusters)):
        for j in range(len(clusters)):
            if i != j:
                result += (stds[i] + stds[j]) / distances[i, j]
    return result
# ---------------------------Sklearn---------------------------
def davies_bouldin_index_sckitlearn(X, labels):
    from sklearn.metrics import davies_bouldin_score
    return davies_bouldin_score(X, labels)

def silhouette_score_sckitlearn(X, labels):
    from sklearn.metrics import silhouette_score
    return silhouette_score(X, labels)