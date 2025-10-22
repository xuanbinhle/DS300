import pandas as pd
from pathlib import Path
import re

def clean_description(description: str) -> str:
    new_description = []
    if not isinstance(description, str):
        raise ValueError(f"Description is not string: {description}")
    list_sub_desc = re.split(r"[\r\n]+", description)
    for sub_desc in list_sub_desc:
        # Loại bỏ tag #
        pattern_tag = r"\s*#([\wÀ-ỹ-_]+)\b"
        sub_desc = re.sub(pattern_tag, "", sub_desc).strip()
        
        # Loại bỏ thông tin quyền lợi khách hàng
        pattern_benefit = r"(Quyền lợi khách hàng|ưu đãi|1. Đảm bảo|2. Quy cách đóng gói|3. Xử lí đơn|4. Chính sách hỗ trợ|UY TÍN & TRÁCH NHIỆM|Cam kết sách thật|Đóng gói cẩn thận|CAM KẾT CỦA CHÚNG TÔI|Gooda tin rằng cuốn sách sẽ mang lại kiến thức)"
        if re.search(pattern_benefit, sub_desc, re.IGNORECASE):
            continue
        
        # Loại bỏ thông tin kích thước sách
        pattern_size = r"\d+(?:.\d+)?\s*x\s*\d+(?:.\d+)?\s*(?:cm|mm|m|inch|inches)?"
        if re.search(pattern_size, sub_desc, re.IGNORECASE):
            continue
        
        # Loại bỏ thông tin chi tiết về sách (không chứa nội dung)
        pattern_info_book = r"^\s*-?\s*(Về tác giả|GIỚI THIỆU|Tên nhà cung cấp|Nhà cung cấp|Quy cách|Người dịch|Bìa|Hình thức|Xuất xứ|In lần thứ|Khổ sách|Thông tin xuất bản|Thông tin chi tiết|Thông tin sản phẩm|Mô tả sản phẩm|Barcode|Mã EAN|Số trang|Nhà xuất bản|NXB liên kết|Công ty phát hành|Nhà phát hành|Tác giả|Dịch giả|Kích thước|Ngày xuất bản|Ngày XB|Năm xuất bản|Năm XB|Loại bìa|Mã Sách|ISBN|NXB|NPH|Giá bìa|SKU|Trọng lượng)\s*[t:\-]?\s*[\d\w\s,.-]*"
        if re.search(pattern_info_book, sub_desc, re.IGNORECASE):
            continue
        
        # Loại bỏ thông tin ngôn ngữ
        pattern_language = r"Ngôn ngữ( Sách)?\s*[t:\-]?\s*[\w\s,.-]*"
        if re.search(pattern_language, sub_desc, re.IGNORECASE):
            continue
        
        # Loại bỏ thông tin quá ngắn
        if len(sub_desc.split(" ")) < 3:
            continue
        
        new_description.append(sub_desc)

    return "\n".join(new_description)
        

if __name__ == '__main__':
    input_file = Path(__file__).parent.parent / 'data' / "raw" / 'books.csv'
    output_file = Path(__file__).parent.parent / 'data' / "preprocessed" / 'cleaned_books.csv'
    if not input_file.exists():
        print(f"Input file {input_file} does not exist.")
    else:
        df = pd.read_csv(input_file)
        cleaned_df = df[~df["description"].isna()]
        cleaned_df['description'] = cleaned_df['description'].apply(clean_description)
        cleaned_df.to_csv(output_file, index=False)