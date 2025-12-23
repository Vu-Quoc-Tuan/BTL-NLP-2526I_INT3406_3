import os
import json
from typing import List
import logging

# Bắt buộc phải có tokenizers
try:
    import re
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import Whitespace, Punctuation, Sequence
    from tokenizers import decoders
except ImportError:
    raise ImportError("The `tokenizers` library is not installed. Please run `pip install tokenizers`")

logger = logging.getLogger(__name__)

class BilingualTokenizer:
    """
    Bộ tách từ tốc độ cao sử dụng thư viện `tokenizers` (được viết bằng Rust).
    """
    def __init__(self, vocab_size: int = 32768):
        self._vocab_size = vocab_size
        self.tokenizer = Tokenizer(BPE(unk_token="[UNK]", end_of_word_suffix="</w>"))
        self.tokenizer.pre_tokenizer = Sequence([Whitespace(), Punctuation()])
        self.tokenizer.decoder = decoders.BPEDecoder(suffix="</w>")

    def train(self, files: List[str]):
        logger.info("Training Tokenizer (BPE Rust)...")
        trainer = BpeTrainer(
            vocab_size=self._vocab_size, 
            special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"],
            min_frequency=2,
            end_of_word_suffix="</w>" # Giúp decoder ghép các từ lại với nhau
        )
        self.tokenizer.train(files, trainer)
        logger.info(f"Tokenizer trained. Vocab size: {self.tokenizer.get_vocab_size()}")

    def encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text).ids

    def decode(self, ids) -> str:
        return self.tokenizer.decode(ids)

    def detokenize(self, text: str) -> str:
        """
        Hậu xử lý văn bản sau khi decode:
        """
        # 1. Rule Số học (Ưu tiên cao nhất để tránh bị Rule Punctuation "ăn" mất khoảng trắng)
        # Ghép dấu chấm/phẩy nằm giữa 2 số: "12 . 000" -> "12.000", "1 . 5" -> "1.5"
        text = re.sub(r'(?<=\d)\s+([.,])\s+(?=\d)', r'\1', text)
        
        # 2. Rule Ký tự đặc biệt (Symbols)
        # Ghép %, ), ], } vào từ đứng trước: "10 %" -> "10%", "( word )" -> "( word)"
        text = re.sub(r'\s+([%)\]}])', r'\1', text)
        
        # 3. Rule Ngoặc mở (Open Brackets)
        # Ghép (, [, { vào từ đứng sau: "( word" -> "(word"
        text = re.sub(r'([(\[{])\s+(?=\S)', r'\1', text)

        # Rule xử lý dấu nháy kép " nội dung " -> "nội dung"
        # Lưu ý: Rule này nên chạy trước rule dấu câu phổ thông
        text = re.sub(r'"\s+(.*?)\s+"', r'"\1"', text)

        # Rule xử lý dấu gạch ngang kép: "- -" -> "--"
        text = re.sub(r'-\s+-', '--', text)

        # 4. Rule Dấu câu phổ thông (General Punctuation)
        # Tìm khoảng trắng đứng trước dấu câu [.,!?:;] và xóa nó
        # "Hello , world" -> "Hello, world"
        text = re.sub(r'\s+([.,!?:;])', r'\1', text)
        
        # 5. Chuẩn hóa khoảng trắng lần cuối (Kiểm tra an toàn)
        # Đảm bảo không còn double spaces do các replace ở trên hoặc có sẵn
        text = re.sub(r'\s+', ' ', text).strip()

        # 6. Hậu xử lý thẩm mỹ (Viết hoa & Chính tả)
        # Chuyển "i" -> "I" (chỉ tác động tiếng Anh, ít ảnh hưởng tiếng Việt)
        text = re.sub(r"\bi\b", "I", text)
        # Viết hoa chữ cái đầu câu
        if text:
            text = text[0].upper() + text[1:]
        
        return text

    def save(self, path: str):
        self.tokenizer.save(path)

    @classmethod
    def load(cls, path: str, dropout: float = None) -> 'BilingualTokenizer':
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing {path}")
        
        instance = cls()
        
        if dropout is not None:
            # Mẹo: Đọc JSON, chèn giá trị dropout vào cấu hình model, sau đó tải từ chuỗi
            # Điều này giúp ta đổi dropout mà không cần retrain
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if 'model' in data and data['model']['type'] == 'BPE':
                data['model']['dropout'] = dropout
                json_str = json.dumps(data)
                instance.tokenizer = Tokenizer.from_str(json_str)
            else:
                # Dự phòng nếu không phải BPE hoặc cấu trúc lạ
                logger.warning("Tokenizer is not BPE or has unexpected JSON structure. Ignoring dropout setting.")
                instance.tokenizer = Tokenizer.from_file(path)
        else:
            instance.tokenizer = Tokenizer.from_file(path)
            
        return instance

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()
    
    @property
    def pad_token_id(self) -> int: return self.tokenizer.token_to_id("[PAD]")
    @property
    def sos_token_id(self) -> int: return self.tokenizer.token_to_id("[SOS]")
    @property
    def eos_token_id(self) -> int: return self.tokenizer.token_to_id("[EOS]")
