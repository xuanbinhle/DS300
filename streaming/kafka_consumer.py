from kafka import KafkaConsumer

topic_name = 'processed_topic'
bootstrap_servers = 'localhost:29092'
consumer = KafkaConsumer(topic_name, bootstrap_servers=bootstrap_servers)

for message in consumer:
    print(f"Processed value: {message.value.decode('utf-8')}")
