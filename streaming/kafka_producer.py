from kafka import KafkaProducer

topic_name = 'example_topic'
bootstrap_servers='localhost:29092'
producer = KafkaProducer(bootstrap_servers=bootstrap_servers)

while True:
    message = input("Enter message to send to Kafka (or 'exit' to quit): ")
    if message.lower() == 'exit':
        break
    producer.send(topic_name, value=message.encode('utf-8'))
    print(f"Sent number: {message}")