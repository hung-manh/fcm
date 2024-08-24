#%%
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import udf, col
from pyspark.sql.types import ArrayType, FloatType
from pyspark.sql.window import Window
from pyspark.sql.functions import row_number
from pyspark.sql.functions import col, sum as _sum, row_number
import numpy as np
from utils.validity import * 
import time 

#%% Khởi tạo phiên làm việc Spark
# Initialize Spark session
spark = SparkSession.builder.appName("FCM_PySpark").getOrCreate()
spark
#%% Khởi tạo dữ liệu và dùng phương pháp Naive Sharding để khởi tạo tâm cụm 
# Bước 1: Đọc và chuẩn bị dữ liệu
csv_file_path = "data/csv/602_Dry_Bean.csv"
data = spark.read.csv(csv_file_path, header=True, inferSchema=True)
data = data.drop(data.columns[-1])
for column in data.columns:
    data = data.withColumn(column, col(column).cast(FloatType()))

# ---------Start: Phương pháp Naive Sharding khởi tạo tậm cụm
## Bước 1: Tính tổng các giá trị thuộc tính của một đối tượng và thêm giá trị tổng này thành cột mới vào tập dữ liệu. Thực hiện trên toàn bộ dữ liệu 
data = data.withColumn("sum", sum(col(column) for column in data.columns)) 

## Bước 2: Sắp xếp tập dữ liệu theo cột tổng mới tạo theo thứ tự tăng dần 
data = data.orderBy("sum", ascending=True) 

## Bước 3: Chia tập dữ liệu theo chiều ngang thành k phần bằng nhau, ở đây chia làm 3 dataframe 
window = Window.orderBy("sum")
data_indexed = data.withColumn("row_index", row_number().over(window))

## Define the number of clusters (segments)
k = 7  

## Tính toán số lượng dòng của tập dữ liệu để phân đoạn 
total_rows = data_indexed.count()
segment_size = total_rows // k

## Bước 3: Chia tập dữ liệu theo chiều ngang thành k phần bằng nhau
segments = []
for i in range(k):
    start_idx = i * segment_size + 1
    end_idx = (i + 1) * segment_size
    if i == k - 1:  ## Chắc chắn rằng tất cả các dòng đều được chia hết cho k
        end_idx = total_rows

    segment = data_indexed.filter((col("row_index") >= start_idx) & (col("row_index") <= end_idx)).drop("row_index")
    segments.append(segment)
    
## Bước 4: Đối với mỗi phân đoạn, tính tổng các cột thuộc tính (không bao gồm các cột đã tạo ở bước 1)
## tính giá trị trung bình của nó và đặt vào trong hàng mới. Hàng mới này thực sự là 
## một trong những trọng tâm cụm đã khởi tạo

## Khởi tạo một list rỗng để lưu trữ các trọng tâm cụm   
cluster_centers = []

## Duyệt qua từng phân đoạn
for segment in segments:
    ## Tính tổng các cột thuộc tính 
    segment_sums = segment.agg(*[_sum(col_name).alias(col_name) for col_name in data.columns[:-1]])
    
    ## Tính giá trị trung bình của các cột thuộc tính
    row_count = segment.count()
    cluster_center = [segment_sums.select(col_name).first()[0] / row_count for col_name in data.columns[:-1]]
    
    ## Thêm giá trị trung bình của các cột thuộc tính vào hàng mới
    cluster_centers.append(cluster_center)

print("Initialized Cluster Centers using Naive Sharding:")
for idx, center in enumerate(cluster_centers):
    print(f"Cluster Center {idx + 1}: {center}", len(center))
# ---------End: Phương pháp Naive Sharding khởi tạo tậm cụm


#%%
assembler = VectorAssembler(inputCols=data.columns[:-1], outputCol="features")
data = assembler.transform(data)
data_rdd = data.select("features").rdd.map(lambda x: x[0].toArray())  

my_list = []
for i, j in enumerate(np.array(data_rdd.collect())):    
    my_list.append((i, j.tolist()))

#%% Định nghĩa các hàm tính toán cho FCM
def fuzzy_membership(pixel, centers, m=2):
    pixel = np.array(pixel)
    centers = np.array(centers)
    distances = np.linalg.norm(centers - pixel, axis=1)
    distances = np.maximum(distances, 1e-10)  # Tránh chia cho 0
    u = 1 / (distances ** (2 / (m - 1)))
    u /= np.sum(u)
    return u.tolist()

def compute_centers(features, memberships, k):
    features = np.array(features)
    memberships = np.array(memberships)
    centers = []
    for i in range(k):
        membership = memberships[:, i]
        numerator = np.sum(membership[:, np.newaxis] * features, axis=0)
        denominator = np.sum(membership)
        centers.append(numerator / denominator)
    return centers

# UDF để tính toán độ tin cậy (membership) của pixel cho mỗi cụm
@udf(ArrayType(FloatType()))
def compute_memberships(features):
    centers = initial_centers_broadcast.value
    return fuzzy_membership(features, centers, m=2)

#%%
rdd = spark.sparkContext.parallelize(my_list, numSlices=3) # Mặc định chia full
# Broadcast initial cluster centers V to all worker nodes
cluster_centers = spark.sparkContext.broadcast(cluster_centers)

# Convert RDD to DataFrame to use UDFs
df = rdd.toDF(["id", "features"])

# Step 6: Iteratively update V and U until convergence or max iterations
maxLoop = 100
tolerance = 1e-5
t = 0
#%%
k = 7
m = 2
initial_centers = np.array(cluster_centers.value)

# Broadcast the initial centers
initial_centers_broadcast = spark.sparkContext.broadcast(initial_centers)

# Tính toán độ tin cậy cho mỗi Pixel
df = df.withColumn("memberships", compute_memberships(col("features")))

# Chạy nhiều vòng lặp để cập nhật trọng tâm và độ tin cậy
for _ in range(100):  # Số vòng lặp
    df_memberships = df.select("id", "memberships")
    memberships_rdd = df_memberships.rdd.map(lambda row: (row[0], row[1]))

    # Tính toán tâm cụm mới
    features = df.rdd.map(lambda row: row[1]).collect()
    memberships = np.array([m[1] for m in memberships_rdd.collect()])
    new_centers = compute_centers(features, memberships, k)

    # Cập nhật tâm cụm cho lần lặp tiếp theo
    initial_centers_broadcast.unpersist()
    initial_centers_broadcast = spark.sparkContext.broadcast(new_centers)

    # Tính toán lại ma trận thành viên với tâm cụm mới
    df = df.withColumn("memberships", compute_memberships(col("features")))
    print(_)

# Hiển thị kết quả
df.show()
#%% 
memberships = np.array(df.select("memberships").rdd.collect()).squeeze()
metric_nt = {
        'PC': partition_coefficient(memberships) ,
        'CE': classification_entropy(memberships) ,
        }
metric_nt

#%% 


#%% 
#%% 

# ## -------------------------------------------------------------------------------------------
# # %%
# from pyspark.sql import SparkSession
# from pyspark.sql.functions import udf, col
# from pyspark.sql.types import ArrayType, FloatType
# import numpy as np

# # Tạo phiên làm việc Spark
# spark = SparkSession.builder \
#     .appName("Collaborative Fuzzy Clustering") \
#     .getOrCreate()
# # %%

# # Giả lập dữ liệu ảnh đa phổ
# data = [
#     (0, [0.1, 0.2, 0.3]),
#     (1, [0.4, 0.5, 0.6]),
#     (2, [0.7, 0.8, 0.9]),
#     (3, [0.2, 0.1, 0.4])
# ]

# # Tạo DataFrame từ dữ liệu
# df = spark.createDataFrame(data, ["id", "features"])
# # %%

# # Định nghĩa các hàm tính toán
# def fuzzy_membership(pixel, centers, m=2):
#     pixel = np.array(pixel)
#     centers = np.array(centers)
#     distances = np.linalg.norm(centers - pixel, axis=1)
#     distances = np.maximum(distances, 1e-10)  # Tránh chia cho 0
#     u = 1 / (distances ** (2 / (m - 1)))
#     u /= np.sum(u)
#     return u.tolist()

# def compute_centers(features, memberships, k):
#     features = np.array(features)
#     memberships = np.array(memberships)
#     centers = []
#     for i in range(k):
#         membership = memberships[:, i]
#         numerator = np.sum(membership[:, np.newaxis] * features, axis=0)
#         denominator = np.sum(membership)
#         centers.append(numerator / denominator)
#     return centers

# # UDF để tính toán độ tin cậy (membership) của pixel cho mỗi cụm
# @udf(ArrayType(FloatType()))
# def compute_memberships(features):
#     centers = initial_centers_broadcast.value
#     return fuzzy_membership(features, centers, m=2)

# # Số cụm
# k = 2
# m = 2

# # Giả lập các trọng tâm cụm ban đầu
# initial_centers = np.array([[0.1, 0.2, 0.3], [0.5, 0.6, 0.7]])

# # Broadcast the initial centers
# initial_centers_broadcast = spark.sparkContext.broadcast(initial_centers)

# # Tính toán độ tin cậy cho mỗi Pixel
# df = df.withColumn("memberships", compute_memberships(col("features")))

# # Chạy nhiều vòng lặp để cập nhật trọng tâm và độ tin cậy
# for _ in range(10):  # Số vòng lặp
#     df_memberships = df.select("id", "memberships")
#     memberships_rdd = df_memberships.rdd.map(lambda row: (row[0], row[1]))

#     # Tính toán tâm cụm mới
#     features = df.rdd.map(lambda row: row[1]).collect()
#     memberships = np.array([m[1] for m in memberships_rdd.collect()])
#     new_centers = compute_centers(features, memberships, k)

#     # Cập nhật tâm cụm cho lần lặp tiếp theo
#     initial_centers_broadcast.unpersist()
#     initial_centers_broadcast = spark.sparkContext.broadcast(new_centers)

#     # Tính toán lại ma trận thành viên với tâm cụm mới
#     df = df.withColumn("memberships", compute_memberships(col("features")))
#     print(_)

# # Hiển thị kết quả
# df.show()

# # Dừng phiên làm việc với Spark
# spark.stop()

# # %%
# spark.stop()
# # %%

# %%
