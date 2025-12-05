import pandas as pd

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
    books_df.to_csv("./data/preprocessed/new_cleaned_books.csv", index=False)
    
    mapping_ids = duplicate_df.set_index("product_id")['new_id_product'].to_dict()
    reviews_df['product_id'] = reviews_df['product_id'].map(mapping_ids).fillna(reviews_df['product_id']).astype(int)
    reviews_df.to_csv("./data/preprocessed/cleaned_reviews.csv", index=False)