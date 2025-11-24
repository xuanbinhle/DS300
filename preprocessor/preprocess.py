import pandas as pd
import numpy as np

def core_k_filtering(df: pd.DataFrame, k: int) -> pd.DataFrame:
    """ Retain users and items with at least k interactions. """
    
    # Lọc user có interaction > 5
    user_interactions = df.groupby('customer_id').size()
    valid_users = user_interactions[user_interactions >= k].index
    
    # Lọc item có interaction > 5
    item_interactions = df.groupby('product_id').size()
    valid_items = item_interactions[item_interactions >= k].index
    
    filtered_df = df[df['customer_id'].isin(valid_users) & df['product_id'].isin(valid_items)].reset_index(drop=True)
    return filtered_df


if __name__ == '__main__':
    books_df = pd.read_csv(r"data/preprocessed/cleaned_books.csv")
    reviews_df = pd.read_csv(r"data/raw/reviews.csv")
    print(books_df.shape, reviews_df.shape)
    filtered_reviews_df = core_k_filtering(reviews_df, 5)
    
    merged_df = pd.merge(books_df, filtered_reviews_df, on='product_id', how='left')
    print(merged_df.shape)