import os
import json
from tqdm import tqdm
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_core.prompts import PromptTemplate

load_dotenv()

def read_file_json(filename):
    data = []
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():  # skip empty lines
                continue
            obj = json.loads(line)
            data.append(obj)
    return data
        

def write_file_json(filename, data):
    with open(filename, 'w', encoding='utf-8') as f:
        for item in data:
            line = json.dumps(item, ensure_ascii=False)  # giữ Unicode, không \\u
            f.write(line + "\n")


# 1. Define the JSON structure we want
response_schemas = [
    ResponseSchema(
        name="reviews",
        description=(
            "Danh sách các đánh giá sách (reviews). "
            "Mỗi phần tử là một object JSON có các trường: "
            "{'product_id': string, 'rating': int, 'content': string}."
        ),
    )
]

parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = parser.get_format_instructions()

# 2. Create PromptTemplate
prompt = PromptTemplate(
    input_variables=["product_name", "product_id", "description", "quantity", "star_distribution"],
    partial_variables={"format_instructions": format_instructions},
    template="""
# Context:
Hãy tạo danh sách dữ liệu đánh giá cho sách.
Dữ liệu đầu vào:
    - Tên sản phẩm: {product_name}
    - ID sản phẩm: {product_id}
    - Mô tả sản phẩm: {description}
    - Số lượng đánh giá cần tạo: {quantity} (Ví dụ: 5)
    - Yêu cầu phân bổ sao: {star_distribution} (Ví dụ: "1 review 1 sao, 1 review 2 sao, ...")

# Instruction:
- Hãy tạo đúng {quantity} đánh giá sách.
- Mỗi đánh giá tương ứng với một người mua hàng khác nhau (sinh viên, nhân viên văn phòng, người lớn tuổi) để văn phong đa dạng.
- Mỗi đánh giá dài dưới 100 tokens.
- Nội dung review phải phản ánh đúng số sao chấm.

# Requirements cho từng review:
    1. product_id: Giữ nguyên ID đã cung cấp: "{product_id}".
    2. rating: Số nguyên từ 1 đến 5, tuân theo phân bổ {star_distribution}.
    3. content: Nhận xét ngắn gọn (1–3 câu), bằng tiếng Việt, phù hợp với rating.

# Output format (RẤT QUAN TRỌNG):
Trả về DUY NHẤT một JSON hợp lệ, tuân theo hướng dẫn dưới đây, không thêm giải thích, không thêm markdown.

{format_instructions}
"""
)

# 3. Model
llm = ChatOpenAI(
    model="gpt-4o-mini",  # hoặc model khác bạn dùng
    temperature=0.7,
    api_key=os.getenv("OPENAI_API")
)

# 4. Build chain
chain = prompt | llm | parser

if __name__ == '__main__':
    books_df = pd.read_csv("data\\final\\new_cleaned_books.csv")
    reviews_df = pd.read_csv("data\\final\\cleaned_reviews.csv")
    
    item_counts = reviews_df.groupby('product_id')['customer_id'].nunique()
    valid_items = item_counts[item_counts >= 5].index

    filtered_books_df = books_df[~books_df['product_id'].isin(valid_items)].reset_index(drop=True)

    output_path = "./data/augmented_reviews.jsonl"
    appendix_reviews = [] if not os.path.exists(output_path) else json.load(open(output_path, 'r', encoding='utf-8'))
    for idx, row in tqdm(filtered_books_df.iterrows(), total=len(filtered_books_df), desc='Generating Reviews'):
        if idx < 425:
            continue
        product_name = row["product_name"]
        product_id = row["product_id"]
        description = row.get("description", "")

        print(f"Generating reviews for product_id={product_id} ...")
        try:
            result = chain.invoke(
                {
                    "product_name": product_name,
                    "product_id": product_id,
                    "description": description,
                    "quantity": 16,
                    "star_distribution": (
                        "4 review 1 sao, 4 review 2 sao, "
                        "4 review 3 sao, 4 review 4 sao, "
                    )
                }
            )
            if isinstance(result, dict):
                appendix_reviews.extend(result['reviews'])
                write_file_json(output_path, appendix_reviews)
            elif isinstance(result, list):
                appendix_reviews.extend(result)
                write_file_json(output_path, appendix_reviews)
            else:
                print(result)
            
            print(f"✅ Generating Review Successful - ProductID: {product_id}")
        
        except Exception as e:
            print(f"❌ Error for {product_id}: {e}")
            continue
