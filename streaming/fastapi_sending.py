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

def get_to_kafka_via_fastapi(FASTAPI_CONSUMER_URL: str, max_wait_time: int = 120):
    """
    Đợi và lấy recommendations từ FastAPI consumer
    
    Args:
        FASTAPI_CONSUMER_URL: URL của FastAPI consumer
        max_wait_time: Thời gian tối đa để đợi (giây), mặc định 60s
    
    Returns:
        List[int]: Danh sách product_index được recommend
    """
    deadline = time.time() + max_wait_time
    retry_interval = 2  # Đợi 2 giây giữa các lần thử
    
    print(f"Đang đợi recommendations từ FastAPI (tối đa {max_wait_time}s)...")
    
    while time.time() < deadline:
        try:
            r = requests.get(FASTAPI_CONSUMER_URL, timeout=5)
            r.raise_for_status()
            data = r.json()
            
            if len(data['messages']) > 0:
                message = data['messages'][0]
                book_list = [int(book_id) for book_id in message['recommendations'].split(',')]
                print(f"✓ Nhận được {len(book_list)} recommendations từ FastAPI:", book_list)
                return book_list
            
            # Chưa có dữ liệu, đợi và thử lại
            remaining_time = int(deadline - time.time())
            print(f"  Chưa có dữ liệu, đợi thêm... (còn {remaining_time}s)")
            time.sleep(retry_interval)
            
        except Exception as e:
            print(f"  Lỗi khi request: {e}, thử lại sau {retry_interval}s...")
            time.sleep(retry_interval)
    
    print("⚠ Hết thời gian đợi, không nhận được recommendations")
    return []