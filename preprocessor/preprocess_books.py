import pandas as pd
from pathlib import Path
import re

class VietnameseNormalizer:
    """
    Tham khảo: https://github.com/VinAIResearch/BARTpho/blob/main/VietnameseToneNormalization.md
    """
    VINAI_NORMALIZED_TONE = {
        'òa': 'oà', 'Òa': 'Oà', 'ÒA': 'OÀ', 
        'óa': 'oá', 'Óa': 'Oá', 'ÓA': 'OÁ', 
        'ỏa': 'oả', 'Ỏa': 'Oả', 'ỎA': 'OẢ',
        'õa': 'oã', 'Õa': 'Oã', 'ÕA': 'OÃ',
        'ọa': 'oạ', 'Ọa': 'Oạ', 'ỌA': 'OẠ',
        'òe': 'oè', 'Òe': 'Oè', 'ÒE': 'OÈ',
        'óe': 'oé', 'Óe': 'Oé', 'ÓE': 'OÉ',
        'ỏe': 'oẻ', 'Ỏe': 'Oẻ', 'ỎE': 'OẺ',
        'õe': 'oẽ', 'Õe': 'Oẽ', 'ÕE': 'OẼ',
        'ọe': 'oẹ', 'Ọe': 'Oẹ', 'ỌE': 'OẸ',
        'ùy': 'uỳ', 'Ùy': 'Uỳ', 'ÙY': 'UỲ',
        'úy': 'uý', 'Úy': 'Uý', 'ÚY': 'UÝ',
        'ủy': 'uỷ', 'Ủy': 'Uỷ', 'ỦY': 'UỶ',
        'ũy': 'uỹ', 'Ũy': 'Uỹ', 'ŨY': 'UỸ',
        'ụy': 'uỵ', 'Ụy': 'Uỵ', 'ỤY': 'UỴ',
    }
    
    @staticmethod
    def normalize_unicode(text):
        char1252 = r'à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ'
        charutf8 = r'à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ'
        char_map = dict(zip(char1252.split('|'), charutf8.split('|')))
        return re.sub(char1252, lambda x: char_map[x.group()], text.strip())

    @staticmethod
    def normalize_typing(text):
        for wrong_word, correct_word in VietnameseNormalizer.VINAI_NORMALIZED_TONE.items():
            text = text.replace(wrong_word, correct_word)
        return text.strip()

def clean_description(description: str) -> str:
    new_description = []
    if not isinstance(description, str):
        raise ValueError(f"Description is not string: {description}")
    list_sub_desc = re.split(r"[\r\n]+", description)
    for sub_desc in list_sub_desc:
        # Loại bỏ tag #
        pattern_tag = r"\s*#([\wÀ-ỹ-_]+)\b"
        sub_desc = re.sub(pattern_tag, "", sub_desc).strip()
        
        # Chuẩn hóa chuỗi
        sub_desc = VietnameseNormalizer.normalize_unicode(sub_desc)
        sub_desc = VietnameseNormalizer.normalize_typing(sub_desc)
        
        # Loại bỏ thông tin không chứa số + ký tự alphabet
        pattern_no_alpha_allow_newline = re.compile(r"^[^\w]+$", re.MULTILINE | re.DOTALL)
        if re.search(pattern_no_alpha_allow_newline, sub_desc):
            continue
        
        # Loại bỏ thông tin quyền lợi khách hàng & cảm ơn khách hàng
        pattern_benefit = r"(Quyền lợi khách hàng|ưu đãi|1. Đảm bảo|2. Quy cách đóng gói|3. Xử lí đơn|4. Chính sách hỗ trợ|UY TÍN & TRÁCH NHIỆM|Cam kết sách thật|Đóng gói cẩn thận|CAM KẾT CỦA CHÚNG TÔI|Gooda tin rằng cuốn sách sẽ mang lại kiến thức|cảm ơn bạn đã quan tâm|vui lòng liên hệ|xin chân thành cảm ơn|đổi mới cho khách hàng|bán sách thật|Sản phẩm giống ảnh)"
        if re.search(pattern_benefit, sub_desc, re.IGNORECASE):
            continue
        
        # Loại bỏ thông tin kích thước sách
        pattern_size = r"\d+(?:.\d+)?\s*x\s*\d+(?:.\d+)?\s*(?:cm|mm|m|inch|inches|gr)?"
        if re.search(pattern_size, sub_desc, re.IGNORECASE):
            continue
        
        # Loại bỏ thông tin chi tiết về sách (không chứa nội dung)
        pattern_info_book = r"^\s*[-•*\d.]*?\s*(Tóm tắt nội dung|Đơn vị liên kết|Thể loại|Thương hiệu|Định dạng bìa|Thông tin sách|Thông tin mô tả|Thông tin phát hành|Thông tin tác giả|Mã sản phẩm|Khối lượng|Tên tác giả|Về tác giả|GIỚI THIỆU|Tên nhà cung cấp|Nhà cung cấp|Quy cách|Người dịch|Bìa|Hình thức|Xuất xứ|In lần thứ|Khổ sách|Thông tin xuất bản|Thông tin chi tiết|Thông tin sản phẩm|Mô tả sản phẩm|Barcode|Mã EAN|Số trang|Nhà xuất bản|NXB liên kết|Đơn vị phát hành|Công ty phát hành|Nhà phát hành|Tác giả|Dịch giả|Kích thước|Ngày xuất bản|Ngày XB|Năm xuất bản|Năm XB|Loại bìa|Mã Sách|Mã ISBN|Số ISBN|ISBN|NXB|NPH|Giá bìa|SKU|Trọng lượng)\s*[t:\-]?\s*[\d\w\s,.-]*"
        if re.search(pattern_info_book, sub_desc, re.IGNORECASE):
            continue
        
        # Loại bỏ thông tin giá bán (chi tiết)
        pattern_price_vnd = re.compile(
            r"""
            (?<!\w)
            (
                (?:\d{1,3}(?:[.,\s]\d{3})+|\d{5,})  # Số cách nhau bởi [',', '.'] hoặc dính liền
                (?:\s?(?:đ|₫|vnđ|vnd|VNĐ|VND))?     # Đơn vị tiền tệ
                |
                \d{2,}(?:\s?[kK])                   # Dạng rút gọn
            )
            (?!\w)
            """, re.VERBOSE
        )
        if re.search(pattern_price_vnd, sub_desc):
            continue
        
        # Loại bỏ thông tin phát hành (chi tiết)
        pattern_release = re.compile(
            r"""
            (?i)                                     # tương tự re.IGNORECASE
            ^[\s\-–*]*
            (?:ngày|công\s*ty|năm|thời\s*gian)?\s*
            (?:dự\s*kiến)?\s*
            (?:phát\s*hành)
            [\s:,-]*
            (?:dự\s*kiến)?\s*
            (?:bởi\s*[\w\s,.&-]+)?
            (?:[:\-])?\s*                            # dấu cách / gạch / hai chấm
            (?:ngày|năm)?\s*                         # có thể lặp lại chữ 'ngày|năm'
            (?:\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}       # date
            |\d{1,2}[/\-]\d{2,4}
            |(?:Quý\s*\d[/\-]\d{2,4})
            |\d{4}
            |(?:\d{1,2}\s*(?:tháng|\/)\s*\d{1,2}\s*(?:năm)?\s*\d{4}))?
            """, re.VERBOSE
        )
        if re.search(pattern_release, sub_desc):
            continue
        
        # Loại bỏ thông tin ngôn ngữ
        pattern_language = r"Ngôn ngữ( Sách)?\s*[:\-]?\s*[\w\s,.-]*"
        if re.search(pattern_language, sub_desc, re.IGNORECASE):
            continue
        
        # Loại bỏ thông tin quá ngắn
        if len(sub_desc.split(" ")) < 3:
            continue
        
        new_description.append(sub_desc)

    return "\n".join(new_description)
        
def clean_bookname(name_book: str) -> str:
    """
    ' Sách - Thuyết Phục Bất Kỳ Ai' => 'Sách - Thuyết Phục Bất Kỳ Ai'
    'Sách Đời Ngắn Đừng Ngủ Dài ( free bookcare)' => 'Sách Đời Ngắn Đừng Ngủ Dài'
    'Sách - Trở về nhà (Nhã Nam) (tặng kèm bookmark thiết kế)' => 'Sách - Trở về nhà'
    'Sách - Dám Bị Ghét - Free Book Care' => 'Sách - Dám Bị Ghét'
    'Sách Tịch Tịnh - First News' => 'Sách Tịch Tịnh'
    """
    name_book = name_book.lower()
    name_book = name_book.strip()
    name_book = name_book.split("(")[0].strip()

    provider_name = ['first news', '1980books', 'alphabooks - bảnquyền', 'alphabooks', 'nhã nam official', 'firstnews', '1980 books', 'bản quyền', 'free book care', 'thái hà books']
    for name in provider_name:
        if name_book.endswith(name):
            name_book = (name_book.split(name)[-2:])[0]
            name_book = name_book.rstrip().rstrip('-').rstrip()
            break

    # Remove text 'Tặng kèm bookmark thiết kế'
    if name_book.endswith('tặng kèm bookmark thiết kế'):
        name_book = name_book.replace('tặng kèm bookmark thiết kế', '').rstrip().rstrip('-').rstrip()
    
    """
    sách - đọc sách - viết sách - làm sách => sách đọc sách - viết sách - làm sách
    """
    # Chuẩn hóa tên sách
    if not name_book.startswith('sách'):
        name_book = 'sách ' + name_book
    elif name_book.startswith('sách -'):
        name_book = name_book.replace('sách -', 'sách', 1).strip()
    elif name_book.startswith('sách:'):
        name_book = name_book.replace('sách:', 'sách', 1).strip()
    
    return name_book

if __name__ == '__main__':
    input_file = Path(__file__).parent.parent / 'data' / "raw" / 'books.csv'
    output_file = Path(__file__).parent.parent / 'data' / "preprocessed" / 'cleaned_books.csv'
    if not input_file.exists():
        print(f"Input file {input_file} does not exist.")
    else:
        df = pd.read_csv(input_file)
        remove_combo_mask = ~df['product_name'].str.contains('combo', case=False, na=False)
        df_no_combo = df[remove_combo_mask].copy()
        df_no_combo['product_name'] = df_no_combo['product_name'].apply(clean_bookname)

        cleaned_df = df_no_combo[~df_no_combo["description"].isna()].copy()
        cleaned_df['description'] = cleaned_df['description'].apply(clean_description)
        cleaned_df.to_csv(output_file, index=False)