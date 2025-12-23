import re
import unicodedata
import html

class TextPreprocessor:
    def __init__(self):
        pass
    
    def normalize_english(self, text: str) -> str:
        """Chuẩn hóa tiếng Anh."""
        # 0. Giải mã các thực thể HTML (&apos; -> ', &quot; -> ")
        text = html.unescape(text)

        # 1. Làm sạch nhiễu (thẻ HTML, URL)
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'http\S+', '[URL]', text)
        
        # 2. Chuẩn hóa dấu câu (Standardize Punctuation) - Áp dụng các rules an toàn
        text = self.standardize_punctuation(text)
        
        # 3. Xóa khoảng trắng thừa
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    def normalize_vietnamese(self, text: str) -> str:
        """Chuẩn hóa tiếng Việt."""
        # 0. Giải mã các thực thể HTML
        text = html.unescape(text)
        
        # 1. Chuẩn hóa Unicode (NFC - Quan trọng nhất)
        text = unicodedata.normalize('NFC', text)
        
        # 2. Làm sạch nhiễu
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'http\S+', '[URL]', text)
        
        # 3. Chuẩn hóa dấu câu (Standardize Punctuation)
        text = self.standardize_punctuation(text)

        # 4. Xóa khoảng trắng thừa
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    def standardize_punctuation(self, text: str) -> str:
        """
        Gom nhóm dấu câu, số và ký tự đặc biệt về dạng tự nhiên.
        Thứ tự áp dụng cực kỳ quan trọng.
        """
        # 1. Rule Số học (Ưu tiên cao nhất)
        # 12 . 000 -> 12.000, 1 . 5 -> 1.5
        text = re.sub(r'(?<=\d)\s+([.,])\s+(?=\d)', r'\1', text)
        
        # 2. Rule Ký tự đặc biệt
        # % ) ] } -> dính vào từ trước
        text = re.sub(r'\s+([%)\]}])', r'\1', text)
        # ( [ { -> dính vào từ sau
        text = re.sub(r'([(\[{])\s+(?=\S)', r'\1', text)

        # Rule xử lý dấu nháy kép " nội dung " -> "nội dung"
        text = re.sub(r'"\s+(.*?)\s+"', r'"\1"', text)

        # 3. Rule Dấu câu phổ thông (. , ! ? : ;) -> dính vào từ trước
        text = re.sub(r'\s+([.,!?:;])', r'\1', text)
        
        return text

    def process(self, text: str, lang: str) -> str:
        if lang == 'en':
            return self.normalize_english(text)
        elif lang == 'vi':
            return self.normalize_vietnamese(text)
        else:
            return text.strip()
