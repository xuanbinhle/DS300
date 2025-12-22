Build the custom Spark image:

```bash
docker build -t spark_test .
```
Run docker-compose service
```bash
docker-compose up -d
```

Activate Kafka, Spark
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
