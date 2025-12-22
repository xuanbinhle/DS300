from fastapi import FastAPI
import asyncio
from kafka import KafkaConsumer
import json
import uvicorn
from collections import deque


KAFKA_BROKER_URL = 'localhost:29092'
KAFKA_TOPIC = 'resys_return_topic'
KAFKA_CONSUMER_ID = 'fastapi-consumer'

stop_polling_event = asyncio.Event()
app = FastAPI()

def json_deserializer(message):
    if message is None:
        return None
    
    try:
        return json.loads(message.decode('utf-8'))
    except:
        print("Unable to decode")
        return None
    
def create_kafka_consumer():

    consumer = KafkaConsumer(
        KAFKA_TOPIC,
        bootstrap_servers=[KAFKA_BROKER_URL],
        auto_offset_reset='latest',
        enable_auto_commit=True,
        group_id=KAFKA_CONSUMER_ID,
        value_deserializer=json_deserializer
    )
    return consumer

latest_messages = deque()
queue_lock = asyncio.Lock()
task = None

async def poll_consumer(consumer: KafkaConsumer):
    try:
        while not stop_polling_event.is_set():
            print("Trying to poll messages from Kafka...")
            records = consumer.poll(5000, max_records=10)
            if records:
                for record in records.values():
                    for message in record:
                        m = message.value
                        print(f"[LOGS] Received message: {m} from topic: {message.topic}")

                        async with queue_lock:
                            latest_messages.append(m)
            await asyncio.sleep(5)
    except Exception as e:
        print(f"Error while polling messages: {e}")
    finally:
        print("Closing the consumer...")
        consumer.close()

@app.get("/trigger")
async def trigger_polling(limit: int = 10):
    global task
    if task is None or task.done():
        stop_polling_event.clear()
        task = asyncio.create_task(poll_consumer(consumer=create_kafka_consumer()))
        return {"status": "Kafka polling started"}
    return {"status": "Kafka polling already running"}

@app.get("/drain")
async def drain(limit: int = 10):
    # trả về limit message và xoá khỏi memory
    async with queue_lock:
        n = min(limit, len(latest_messages))
        out = [latest_messages.popleft() for _ in range(n)]
    return {"messages": out}

@app.get("/stop")
async def stop():
    stop_polling_event.set()
    return {"status": "Kafka polling stopping"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)