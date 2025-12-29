import pandas as pd

products = pd.read_csv("./data/features/final_cleaned_books.csv")
reviews = pd.read_csv("./data/features/final_interactions.csv")
print(products.shape, reviews.shape)



