import requests
from datetime import datetime
import time

def send_to_kafka_via_fastapi(book_id: str, FASTAPI_PRODUCER_URL: str):
    payload = {
        "message": "book_id:" + str(book_id) + " user_id:0"
        # "ts": datetime.utcnow().isoformat() + "Z",
        # nếu ProducerMessage có field khác thì thêm vào đây
    }

    r = requests.post(FASTAPI_PRODUCER_URL, json=payload, timeout=5)
    r.raise_for_status()
    return r.json()

def get_to_kafka_via_fastapi(FASTAPI_CONSUMER_URL: str):
    # deadline = time.time() + 8.0
    r = requests.get(FASTAPI_CONSUMER_URL, timeout=5)
    r.raise_for_status()
    data = r.json()

    while len(data['messages']) == 0:
        time.sleep(0.5)
        r = requests.get(FASTAPI_CONSUMER_URL, timeout=5)
        r.raise_for_status()
        data = r.json()
        
        if len(data['messages']) != 0:
            message = data['messages'][0]
            book_list = [int(book_id) for book_id in message['recommendations'].split(',')]
            print("Received recommendations from FastAPI:", book_list)
            return book_list
    # return []