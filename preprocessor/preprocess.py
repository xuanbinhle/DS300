import pandas as pd
import os
import glob

def multimodal_filtering(df: pd.DataFrame, downloaded_image_ids: list[int]):
    filtered_df = df[df['product_id'].isin(downloaded_image_ids)].reset_index(drop=True)
    unique_items = sorted(filtered_df["product_id"].unique())
    item2idx = {it: j for j, it in enumerate(unique_items)}
    filtered_df["product_index"] = filtered_df["product_id"].map(item2idx)
    return filtered_df, item2idx


def core_k_filtering(df: pd.DataFrame, k: int, item2idx: dict):
    """ Retain users and items with at least k interactions. """
    filtered_df = df[df['product_id'].isin(list(item2idx.keys()))]
    filtered_df = filtered_df.drop_duplicates(subset=["customer_id", "product_id"], keep="last")
    
    # Loại user có interaction < k & item có interaction < k
    while True:
        user_interactions = filtered_df.groupby('customer_id').size()
        banned_users = user_interactions[user_interactions < k].index
        
        item_interactions = filtered_df.groupby('product_id').size()
        banned_items = item_interactions[item_interactions < k].index
        
        if len(banned_users) == 0 and len(banned_items) == 0:
            break
        filtered_df = filtered_df[~filtered_df['customer_id'].isin(banned_users) & ~filtered_df['product_id'].isin(banned_items)].reset_index(drop=True)
    
    unique_users = sorted(filtered_df["customer_id"].unique())
    user2idx = {u: i for i, u in enumerate(unique_users)}
    return filtered_df, user2idx


def reindexing(df: pd.DataFrame, user2idx, item2idx):
    df["customer_index"] = df["customer_id"].map(user2idx)
    df['product_index'] = df['product_id'].map(item2idx)
    return df
    

if __name__ == '__main__':
    K = 3
    books_df = pd.read_csv(r"data/preprocessed/new_cleaned_books.csv")
    reviews_df = pd.read_csv(r"data/preprocessed/cleaned_reviews.csv")
    print(f"Books Shape: {books_df.shape}, Reviews Shape: {reviews_df.shape}")
    
    downloaded_image_ids = []
    for file in glob.glob("./data/book_images/*.png"):
        downloaded_image_ids.append(int(os.path.splitext(os.path.basename(file))[0])) 
    
    filtered_books_df, item2idx = multimodal_filtering(books_df, downloaded_image_ids)
    # filtered_books_df.to_csv("./data/features/final_cleaned_books.csv", index=False)
    print(f"Multimodal Filtering - Books Shape: {filtered_books_df.shape}")
    
    filtered_reviews_df, user2idx = core_k_filtering(reviews_df, K, item2idx)
    print(f"Core-{K} Filtering - Reviews Shape: {filtered_reviews_df.shape}")
    
    interaction_df = reindexing(filtered_reviews_df, user2idx, item2idx)
    # interaction_df.to_csv("./data/features/final_interactions.csv", index=False)