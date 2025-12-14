from pyspark.sql import SparkSession
from pyspark.sql.functions import col

spark = SparkSession.builder \
                    .appName("Recommender-system") \
                    .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.6") \
                    .getOrCreate()

df = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:29092") \
    .option("subscribe", "example_topic") \
    .option("startingOffsets", "earliest") \
    .load()

df = df.selectExpr("CAST(value AS STRING) AS value")
df = df.withColumn('value', col('value').cast('int') + 10)
df = df.withColumn('value', col('value').cast('string'))

query = df.writeStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:29092") \
    .option("topic", "processed_topic") \
    .outputMode("append") \
    .option("checkpointLocation", "./chk_kafka_out") \
    .start()

query.awaitTermination()