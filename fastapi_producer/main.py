from fastapi import FastAPI, BackgroundTasks
from kafka.admin import KafkaAdminClient, NewTopic
from kafka_producer import producer_kafka_message
from producer_schema import ProducerMessage
from contextlib import asynccontextmanager
import uvicorn

KAFKA_BROKER_URL = 'localhost:29092'
KAFKA_TOPIC = 'recsys_topic'
KAFKA_ADMIN_CLIENT_ID = 'fastapi-admin-client'

@asynccontextmanager
async def lifespan(app: FastAPI):
    admin_client = KafkaAdminClient(
        bootstrap_servers=KAFKA_BROKER_URL,
        client_id=KAFKA_ADMIN_CLIENT_ID
    )
    if not KAFKA_TOPIC in admin_client.list_topics():
        topic = NewTopic(
            name=KAFKA_TOPIC,
            num_partitions=1,
            replication_factor=1
        )
        admin_client.create_topics([topic], validate_only=False)
    yield

app = FastAPI(lifespan=lifespan)

@app.post("/produce-message", tags=["Kafka Producer"])
async def producer_message(messageRequest: ProducerMessage, background_tasks: BackgroundTasks):
    background_tasks.add_task(producer_kafka_message, messageRequest)
    return {"status": "Message is being processed in the background"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)