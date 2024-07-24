import numpy as np 


# 1.1 Chỉ số đo lường mức độ tách biệt giữa các cụm
## 1.1.1 Chỉ số Dunn (DI)
def dunn_index(clusters:np.ndarray)->float:
    # """
    # Args:
    # clusters: Danh sách các cụm, mỗi cụm là một danh sách các điểm dữ liệu.
    # Giá trị trả về: Chỉ số Dunn, càng cao càng tốt, càng thể hiện độ tách biệt giữa các cụm.
    # """
    # # Tính khoảng cách giữa các cụm
    # ds = []
    # for i in range(len(clusters)):
    #     for j in range(i+1,len(clusters)):
    #         ds.append(np.min(np.linalg.norm(clusters[i][:,np.newaxis]-clusters[j],axis=2)))
    # # Tính đường kính của mỗi cụm
    # diams = []
    # for cluster in clusters:
    #     diams.append(np.max(np.linalg.norm(cluster[:,np.newaxis]-cluster,axis=2)))
    # return np.min(ds) / np.max(diams)
    """
    Calculate the Dunn Index for a set of clusters.
    
    Args:
    clusters: List of clusters, where each cluster is a numpy array of data points.
    
    Returns:
    float: Dunn Index. Higher values indicate better clustering.
    """
    n_clusters = len(clusters)
    
    # Calculate inter-cluster distances
    min_inter_cluster_distances = np.inf
    for i in range(n_clusters):
        for j in range(i+1, n_clusters):
            dist = np.min(np.linalg.norm(clusters[i][:, np.newaxis] - clusters[j], axis=1))
            min_inter_cluster_distances = min(min_inter_cluster_distances, dist)
    
    # Calculate cluster diameters
    max_cluster_diameter = 0
    for cluster in clusters:
        diameter = np.max(np.linalg.norm(cluster[:, np.newaxis] - cluster, axis=1))
        max_cluster_diameter = max(max_cluster_diameter, diameter)
    
    return min_inter_cluster_distances / max_cluster_diameter

## 1.1.2 Chỉ số Davies-Bouldin (DBI)
def davies_bouldin_index(clusters:np.ndarray,centroids:np.ndarray) -> float:
    """
    Args:
    clusters: Danh sách các cụm, mỗi cụm là một danh sách các điểm dữ liệu.
    centroids: Danh sách các tâm cụm.
    """
    # Tính độ lệch chuẩn của mỗi cụm
    stds = [np.std(cluster, axis=0) for cluster in clusters]
    # Tính khoảng cách giữa các tâm cụm
    distances = np.linalg.norm(centroids[:, np.newaxis] - centroids, axis=2)
    # Tính Davies-Bouldin’s index
    davies_bouldin_index = 0
    for i in range(len(clusters)):
        max_ratio = 0
        for j in range(len(clusters)):
            if i != j:
                ratio = (stds[i] + stds[j]) / distances[i, j]
                if ratio > max_ratio:
                    max_ratio = ratio
    davies_bouldin_index += max_ratio
    return davies_bouldin_index / len(clusters)

## 1.1.3 Chỉ số tách biệt Separation (SI)
def separation_index(clusters:np.ndarray,centroids:np.ndarray)->float:
    """
    Args:
    clusters: Danh sách các cụm, mỗi cụm là một danh sách các điểm dữ liệu.
    centroids: Danh sách các tâm cụm.
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

