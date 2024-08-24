from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import col
from pyspark.sql.types import FloatType
from pyspark.sql.window import Window
from pyspark.sql.functions import row_number
from pyspark.sql.functions import col, sum as _sum, row_number
import numpy as np
from models.fcm import Dfcm
from utils.validity import * 
import time 


# Initialize Spark session
spark = SparkSession.builder.appName("FCM_PySpark").getOrCreate()
spark

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


