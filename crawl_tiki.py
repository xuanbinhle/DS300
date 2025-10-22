import requests
import pandas as pd
from icecream import ic
import time
import re
import os

URL_PRODUCTS = "https://tiki.vn/api/v2/products"
URL_REVIEWS  = "https://tiki.vn/api/v2/reviews"
FILE_PRODUCTS = "books.csv"
FILE_REVIEWS = "reviews.csv"

productsDF = pd.DataFrame(columns=['product_id', 'product_name', 'authors', 'price', 'seller_id', 'seller_type', 'rating_average', 'review_count', 'order_count', 'url', 'image']) if not os.path.exists(FILE_PRODUCTS) else pd.read_csv(FILE_PRODUCTS)
reviewsDF = pd.DataFrame(columns=['customer_id', 'product_id', 'rating', 'content', 'title', 'thank_count']) if not os.path.exists(FILE_REVIEWS) else pd.read_csv(FILE_REVIEWS)
keyword = "sách"
pages = 40

sess = requests.Session()
sess.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7",
})

for page in range(1, pages):
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
        
        product_name = product.get("name")
        print(f"📦 Bắt đầu crawl product [{product_id}] – {product_name}")
        url_path = product.get("url_path") or ""
        authors = ""
        for item in product.get('badges_new', []):
            if item['code'] == 'brand_name':
                authors = item['text']
        referer = f"https://tiki.vn/{url_path}" if url_path else f"https://tiki.vn/product/{product_id}"
        
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
            'image': product.get('thumbnail_url')
        }
        
        tempDF = pd.DataFrame()
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
            for customer in reviews_payload['data']:
                batch_review.append({
                    "customer_id": customer.get('customer_id'),
                    "product_id": customer.get('product_id'),
                    "rating": customer.get('rating'),
                    "content": re.sub(r'[\r\n]+', '. ', customer.get('content')).strip(),
                    "title": customer.get('title'),
                    "thank_count": customer.get('thank_count'),
                })
            
            tempDF = pd.concat([tempDF, pd.DataFrame(batch_review)], axis=0, ignore_index=True)
            print(f"   ├─ Page {page_reviews} scraped: {len(batch_review)} reviews (Total so far: {get_reviews}/{total_reviews})")
            if get_reviews >= total_reviews or len(reviews_payload['data']) == 0:
                break
            
            page_reviews += 1
            time.sleep(0.5)
            
        product_info['review_count'] = product_info.get('review_count') if get_reviews == product_info.get('review_count') else get_reviews

        productsDF = pd.concat([productsDF, pd.DataFrame([product_info])], axis=0, ignore_index=True)
        reviewsDF = pd.concat([reviewsDF, tempDF], axis=0, ignore_index=True)
        productsDF.to_csv(f"books.csv", index=False)
        reviewsDF.to_csv(f"comments.csv", index=False)
        print(f"✅ Crawl xong product [{product_id}] – {product_name}")
        print("-" * 50)
        raise