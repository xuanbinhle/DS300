from pyspark.sql import SparkSession
from pyspark.sql.functions import col, regexp_extract, to_json, struct, lit
from model.content_based import get_recommendation_inference_cb
import pandas as pd
import numpy as np
import os
import shutil

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
data_path = os.path.join(project_root, 'data', 'final', 'text_descbook_feat_VisoBert.npy')

print(data_path)
desc_feat = np.load(data_path)
def safe_rm(path: str):
    if os.path.exists(path):
        shutil.rmtree(path)
        print(f"Removed: {path}")

chkpoint_path = "./tmp/chk_recommend_publish"
check_interactions_path = "./tmp/chk_interactions"
store_interactions_path = "./tmp/store/interactions"

for p in [chkpoint_path, check_interactions_path, store_interactions_path]:
    safe_rm(p)

spark = SparkSession.builder \
                    .appName("Recommender-system") \
                    .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.6") \
                    .getOrCreate()

df = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:29092") \
    .option("subscribe", "recsys_topic") \
    .option("startingOffsets", "latest") \
    .load()

df = df.selectExpr("CAST(value AS STRING) AS value")
df_parsed = df \
    .withColumn("product_index", regexp_extract(col("value"), r"book_id:(\d+)", 1).cast("int")) \
    .withColumn("customer_index", regexp_extract(col("value"), r"user_id:(\d+)", 1).cast("int")) \
    .select("product_index", "customer_index")

def recommend_and_publish(batch_df, batch_id: int):
    if batch_df.rdd.isEmpty():
        print(f"[batch {batch_id}] empty batch")
        return
    try:
        hist_df = spark.read.parquet("./tmp/store/interactions")
        all_df = hist_df.unionByName(batch_df)
    except Exception as e:
        print(f"[batch {batch_id}] no history yet or read failed: {e}")
        all_df = batch_df
    pdf = all_df.toPandas()

    out_rows = []
    for user_id in pdf["customer_index"].dropna().unique():
        temp_interaction_df = pdf[pdf["customer_index"] == user_id][["product_index", "customer_index"]]
        temp_interaction_df.drop_duplicates(subset=['product_index', 'customer_index'], inplace=True)
        # print(temp_interaction_df)

        recs = get_recommendation_inference_cb(
            temp_interaction_df,
            desc_feat,
            user_id=int(user_id),
            similarity_name="Cosine"
        )

        rec_list = [int(item[0]) for item in recs[:10]]
        rec_output = ",".join(map(str, rec_list))
        out_rows.append((str(user_id), rec_output, int(batch_id)))

    # Tạo DF output rồi publish Kafka
    out_df = spark.createDataFrame(out_rows, ["user_id", "recommendations", "batch_id"]) \
        .withColumn("event_type", lit("RECOMMENDATION"))

    kafka_df = out_df.select(
        col("user_id").cast("string").alias("key"),
        to_json(struct("event_type", "user_id", "recommendations", "batch_id")).alias("value")
    )

    # kafka_df.show(truncate=False)

    (kafka_df.write
        .format("kafka")
        .option("kafka.bootstrap.servers", "localhost:29092")
        .option("topic", "resys_return_topic")
        .save()
    )

store_interactions_query = (df_parsed.writeStream
    .format("parquet")
    .option("path", "./store/interactions")
    .option("checkpointLocation", "./tmp/chk_interactions")
    .outputMode("append")
    .start()
)

recommend_query = (df_parsed.writeStream
    .foreachBatch(recommend_and_publish)
    .option("checkpointLocation", "./tmp/chk_recommend_publish")
    .outputMode("append")
    .start()
)

spark.streams.awaitAnyTermination()
