import pandas as pd
import numpy as np
import json

def read_file_json(filename):
    data = []
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():  # skip empty lines
                continue
            obj = json.loads(line)
            data.append(obj)
    return data
        

if __name__ == '__main__':
    reviews_df = pd.read_csv("./data/raw/reviews.csv")
    unique_df = pd.read_csv("unique_book.csv")
    duplicate_df = pd.read_csv("duplicate.csv")
    
    id_groups = duplicate_df.groupby('product_name')['product_id'].min()
    duplicate_df["new_id_product"] = duplicate_df["product_name"].map(id_groups)
    duplicate_df["review_count_sum"] = (
        duplicate_df
            .groupby(['new_id_product', 'product_name'])['review_count']
            .transform(sum)
    )
    
    new_duplicate_df = (
        duplicate_df
            .drop_duplicates(['new_id_product', 'product_name'])
            .copy()
            .drop(columns=['product_id', 'review_count'])
            .rename(columns={'new_id_product': 'product_id', 'review_count_sum': 'review_count'})
    )

    books_df = pd.concat([unique_df, new_duplicate_df], axis=0)
    # books_df.to_csv("./data/preprocessed/new_cleaned_books.csv", index=False)
     
    mapping_ids = duplicate_df.set_index("product_id")['new_id_product'].to_dict()
    reviews_df['product_id'] = reviews_df['product_id'].map(mapping_ids).fillna(reviews_df['product_id']).astype(int)
    # reviews_df.to_csv("./data/preprocessed/cleaned_reviews.csv", index=False)
    
    reviews_df = reviews_df.drop(columns=['title', 'thank_count'])
    user_counts = reviews_df.groupby('customer_id')['product_id'].nunique()
    valid_users = user_counts[user_counts < 5].index.to_numpy()
    
    augmented_reviews = read_file_json("./data/augmented_reviews.jsonl")
    augmented_reviews_df = pd.DataFrame(augmented_reviews)
    
    np.random.seed(42)
    filled_augemented_df_list = []
    for pid, group in augmented_reviews_df.groupby('product_id'):
        group = group.copy()
        n = len(group)
        
        # Lọc các uid đã từng rating cho pid
        already_rated = reviews_df.loc[reviews_df['product_id'] == pid, 'customer_id'].unique()
        eligible_users = np.setdiff1d(valid_users, already_rated)
        
        if len(eligible_users) >= n:
            chosen = np.random.choice(eligible_users, size=n, replace=False)
        else:
            chosen = np.random.choice(eligible_users if len(eligible_users) > 0 else valid_users, size=n, replace=True)
        
        group['customer_id'] = chosen
        filled_augemented_df_list.append(group)
    
        # Update reviews_df
        reviews_df = pd.concat([reviews_df, group], ignore_index=True)
    
    reviews_df = reviews_df.drop_duplicates(['customer_id', 'product_id'])
    reviews_df.to_csv("./data/preprocessed/new_cleaned_reviews.csv", index=False)