# 1. Report and Slides:
Our report and slide are saved in [report](report).

# 2. To build our application demo:

Build the custom Spark image

```bash
docker build -t spark_test .
```
Run docker-compose service
```bash
docker-compose up -d
```

Activate Kafka and Spark
```bash
# Activate kafka-producer
python fastapi_producer/main.py

# Activate kafka-consumer
python fastapi_consumer/main.py

# Activate spark-streaming
python streaming/app.py
```

Activate Webapp
```bash
streamlit run streaming/streamlit_app.py
```
