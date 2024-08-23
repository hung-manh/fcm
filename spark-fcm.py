from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import col
from pyspark.sql.types import FloatType
import numpy as np
from models.fcm import Dfcm


# Initialize Spark session
spark = SparkSession.builder.appName("FCM_PySpark").getOrCreate()

# Step 1: Load data from HDFS
csv_file_path ="data/csv/602_Dry_Bean.csv"
data = spark.read.csv(csv_file_path, header=True, inferSchema=True)

## Drop the last column (labels) 
data = data.drop(data.columns[-1])

## Convert all columns to float type    
for column in data.columns:
    data = data.withColumn(column, col(column).cast(FloatType()))

# Step 2: Prepare data for clustering
assembler = VectorAssembler(inputCols=data.columns, outputCol="features")
data_vectorized = assembler.transform(data)

# Step 3: Implement FCM clustering
dfcm = Dfcm()


# Step 4: Distribute FCM computation across nodes
def distribute_fcm(rdd, C:int):
    def fcm_partition(partition):
        local_data = np.array(list(partition))
        print(local_data.shape)
        centers, membership, step = dfcm.cmeans(local_data, C)
        # print(step)
        return centers, membership
    
    results = rdd.mapPartitions(fcm_partition).collect()
    
    # Aggregate results from all partitions
    # global_centers = aggregate_centers(results)
    return results

rdd = data_vectorized.rdd.map(lambda row: row["features"])
final_centers = distribute_fcm(rdd, C=7)
# print(final_centers)    