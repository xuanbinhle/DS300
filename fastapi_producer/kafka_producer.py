from kafka import KafkaProducer
from fastapi import HTTPException
from producer_schema import ProducerMessage
import json

KAFKA_BROKER = 'localhost:29092'
KAFKA_TOPIC = 'recsys_topic'
PRODUCER_CLIENT_ID = 'fastapi-producer'

def serializer(message):
    return json.dumps(message).encode('utf-8')

producer = KafkaProducer(
    api_version=(0,8,0),
    bootstrap_servers=KAFKA_BROKER,
    client_id=PRODUCER_CLIENT_ID,
    value_serializer=serializer
)

def producer_kafka_message(message: ProducerMessage):
    try:
        producer.send(KAFKA_TOPIC, json.dumps({'message': message.message}))
        producer.flush()
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail="Failed to send message to Kafka")
    return {"status": "Message sent to Kafka successfully"}