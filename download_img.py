import requests
import pandas as pd
import os

def download_images(image_url: str, id_book: str, image_dir: str):
    try:
        response = requests.get(image_url, stream=True)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)

        save_path = os.path.join(image_dir, f"{id_book}.png")
        with open(save_path, 'wb') as file:
            file.write(response.content)
            
        print(f"Image downloaded successfully to: {save_path}")
    except requests.exceptions.RequestException as e:
        print(f"Error downloading image: {e}")
        
if __name__ == '__main__':
    image_dir = "./data/book_images"
    books_df = pd.read_csv("./data/preprocessed/new_cleaned_books.csv")
    
    os.makedirs(image_dir, exist_ok=True)
    books_df[['product_id', 'image']].apply(
        lambda row: download_images(
            image_url=row['image'],
            id_book=str(row['product_id']),
            image_dir=image_dir
        ),
        axis=1
    )