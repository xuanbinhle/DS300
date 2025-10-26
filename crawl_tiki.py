import requests
import pandas as pd
import time
import re
import os
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import NoSuchElementException, TimeoutException
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import random

URL_PRODUCTS = "https://tiki.vn/api/v2/products"
URL_REVIEWS  = "https://tiki.vn/api/v2/reviews"
FOLDER_DATA = r"./data/raw"
os.makedirs(FOLDER_DATA, exist_ok=True)

FILE_PRODUCTS = os.path.join(FOLDER_DATA, "books.csv")
FILE_REVIEWS = os.path.join(FOLDER_DATA, "reviews.csv")
FILE_CUSTOMERS = os.path.join(FOLDER_DATA, "customers.csv")

# Crawl Description Tiki
text_remove = "Giá sản phẩm trên Tiki đã bao gồm thuế theo luật hiện hành. Bên cạnh đó, tuỳ vào loại sản phẩm, hình thức và địa chỉ giao hàng mà có thể phát sinh thêm chi phí khác như phí vận chuyển, phụ phí hàng cồng kềnh, thuế nhập khẩu (đối với đơn hàng giao từ nước ngoài có giá trị trên 1 triệu đồng)....."
def crawl_decription_typebook(url):
    options = Options()
    
    options.add_argument("--headless")
    options.add_argument("--log-level=3")
    options.add_experimental_option('excludeSwitches', ['enable-logging'])
    driver = webdriver.Edge(options=options)

    driver.get(url)
    time.sleep(1.5)
    # Get type of book
    category = driver.find_element(By.CLASS_NAME, 'breadcrumb').text
    category = category.split("\n")
    type_book = "/".join(category[2:-1])

    driver.execute_script("window.scrollTo(0, document.body.scrollHeight/3.5);")
    time.sleep(0.5)
    last_height = driver.execute_script("return window.pageYOffset;")
    while True:
        try:
            button = driver.find_element(By.CSS_SELECTOR, ".btn-more")
            button.click()
            break
        except NoSuchElementException:
            # Nếu chưa thấy nút => cuộn xuống thêm 500px
            driver.execute_script("window.scrollBy(0, 500);")
            time.sleep(0.4)

        # Kiểm tra nếu đã cuộn đến cuối trang mà vẫn không có nút
        new_height = driver.execute_script("return window.pageYOffset;")
        if new_height == last_height:
            print("Không tìm thấy nút .btn-more — có thể không tồn tại.")
            break
        last_height = new_height

    try:
        description = WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, ".sc-f5219d7f-0.haxTPb"))
        )
        
        cleaned_description = re.sub(re.escape(text_remove), "", description.text, flags=re.IGNORECASE).strip()    
    except NoSuchElementException:
        cleaned_description = None
    except TimeoutException:
        cleaned_description = None

    driver.quit()
    time.sleep(random.choice([1.5, 2, 2.5, 3, 3.5]))
    return cleaned_description, type_book

productsDF = pd.DataFrame(columns=['product_id', 'product_name', 'authors', 'price', 'seller_id', 'seller_type', 'rating_average', 'review_count', 'order_count', 'url', 'image']) if not os.path.exists(FILE_PRODUCTS) else pd.read_csv(FILE_PRODUCTS)
reviewsDF = pd.DataFrame(columns=['customer_id', 'product_id', 'rating', 'content', 'title', 'thank_count']) if not os.path.exists(FILE_REVIEWS) else pd.read_csv(FILE_REVIEWS)
customersDF = pd.DataFrame(columns=['customer_id', 'customer_name', 'customer_full_name', 'region', 'created_time']) if not os.path.exists(FILE_CUSTOMERS) else pd.read_csv(FILE_CUSTOMERS)

keyword = "sách"
pages = 80

sess = requests.Session()
sess.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7",
})

for page in range(1, pages + 1):
    params_search = {"q": keyword, "page": page, "limit": 40}
    headers_search = {"Referer": f"https://tiki.vn/search?q={keyword}"}
    r = sess.get(URL_PRODUCTS, params=params_search, headers=headers_search, timeout=30)
    r.raise_for_status()

    products = r.json().get("data", [])
    if not products:
        print(f"[page {page}] No products found.")
        continue
    
    for product in products:
        product_id = product.get("id")
        if not product_id or product_id in productsDF['product_id'].values:
            continue
        try:
            product_name = product.get("name")
            print(f"📦 Bắt đầu crawl product [{product_id}] – {product_name}")
            url_path = product.get("url_path") or ""
            authors = ""
            for item in product.get('badges_new', []):
                if item['code'] == 'brand_name':
                    authors = item['text']
            referer = f"https://tiki.vn/{url_path}" if url_path else f"https://tiki.vn/product/{product_id}"
            
            description, type_book = crawl_decription_typebook(referer)
            product_info = {
                'product_id': product_id,
                'product_name': product_name,
                'authors': authors,
                'price': product.get("price", 0),
                'seller_id': product.get("seller_id"),
                'seller_type': product.get('visible_impression_info', {}).get('amplitude', {}).get('seller_type'),
                'rating_average': product.get("rating_average"),
                'review_count': product.get("review_count"),
                'order_count': product.get("quantity_sold", {}).get("value"),
                'url': referer,
                'image': product.get('thumbnail_url'),
                'description' : description,
                'type_book': type_book
            }
            
            tempDF_review = pd.DataFrame()
            tempDF_customer = pd.DataFrame()
            page_reviews, get_reviews = 1, 0
            while True:
                params_reviews = {"product_id": product_id, "page": page_reviews, "limit": 5}
                headers_reviews = {"Referer": referer}
                r2 = sess.get(URL_REVIEWS, params=params_reviews, headers=headers_reviews, timeout=30)
                r2.raise_for_status()
                reviews_payload = r2.json()

                total_reviews = reviews_payload['reviews_count']
                get_reviews += len(reviews_payload['data'])

                batch_review = []
                batch_customer = []
                for customer in reviews_payload['data']:
                    batch_review.append({
                        "customer_id": customer.get('customer_id'),
                        "product_id": customer.get('product_id'),
                        "rating": customer.get('rating'),
                        "content": re.sub(r'[\r\n]+', '. ', customer.get('content')).strip(),
                        "title": customer.get('title'),
                        "thank_count": customer.get('thank_count'),
                    })
                    batch_customer.append({
                        'customer_id': customer.get('created_by').get('id'),
                        'customer_name': customer.get('created_by').get('name'),
                        'customer_full_name': customer.get('created_by').get('full_name'),
                        'region': customer.get('created_by').get('region'),
                        'created_time': customer.get('created_by').get('created_time')
                    })
                tempDF_review = pd.concat([tempDF_review, pd.DataFrame(batch_review)], axis=0, ignore_index=True)
                tempDF_customer = pd.concat([tempDF_customer, pd.DataFrame(batch_customer)], axis=0, ignore_index=True)
                print(f"   ├─ Page {page_reviews} scraped: {len(batch_review)} reviews (Total so far: {get_reviews}/{total_reviews})")
                if get_reviews >= total_reviews or len(reviews_payload['data']) == 0:
                    break

                page_reviews += 1
                time.sleep(0.5)
        except:
            continue

        product_info['review_count'] = product_info.get('review_count') if get_reviews == product_info.get('review_count') else get_reviews
        productsDF = pd.concat([productsDF, pd.DataFrame([product_info])], axis=0, ignore_index=True)
        reviewsDF = pd.concat([reviewsDF, tempDF_review], axis=0, ignore_index=True)
        customersDF = pd.concat([customersDF, tempDF_customer], axis=0, ignore_index=True)

        #Remove Duplicate
        customersDF.drop_duplicates(subset=['customer_id'], inplace=True, ignore_index=True)

        productsDF.to_csv(FILE_PRODUCTS, index=False)
        reviewsDF.to_csv(FILE_REVIEWS, index=False)
        customersDF.to_csv(FILE_CUSTOMERS, index=False)
        print(f"✅ Crawl xong product [{product_id}] – {product_name}")
        print("-" * 50)
