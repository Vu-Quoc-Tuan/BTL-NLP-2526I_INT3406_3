import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Dict, Optional
import os
from src.data.tokenizer import BilingualTokenizer
from src.data.preprocess import TextPreprocessor
from tqdm import tqdm

class BilingualDataset(Dataset):
    """
    Lớp Dataset để tải dữ liệu song ngữ.
    Đọc dữ liệu và Pre-tokenize ngay lúc init để tối ưu tốc độ training.
    Hỗ trợ Disk Caching: Lưu kết quả ra file .pt để lần sau load lại ngay lập tức.
    """
    
    def __init__(self, 
                 src_file: str, 
                 tgt_file: str, 
                 tokenizer: BilingualTokenizer, 
                 max_len: int = 120,
                 cache_path: Optional[str] = None,
                 min_len: int = 3,
                 max_ratio: float = 2.0,
                 filtering: bool = True,
                 dynamic_tokenize: bool = False):
        super().__init__()
        self.src_file = src_file
        self.tgt_file = tgt_file
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.min_len = min_len
        self.max_ratio = max_ratio
        self.filtering = filtering
        self.dynamic_tokenize = dynamic_tokenize
        
        self.data_items = []
        self.preprocessor = TextPreprocessor()
        
        # Kiểm tra cache
        if cache_path and os.path.exists(cache_path):
            print(f"Loading data from cache: {cache_path} ...")
            try:
                self.data_items = torch.load(cache_path)
                print(f"Loaded {len(self.data_items)} items from cache.")
                return # Xong luôn, không cần tokenize lại
            except Exception as e:
                print(f"Failed to load cache: {e}. Re-tokenizing...")
        
        
        # 1. Đọc dữ liệu thô (raw)
        raw_src = self._read_file(src_file)
        raw_tgt = self._read_file(tgt_file)
        
        assert len(raw_src) == len(raw_tgt), \
            f"Mismatched lengths: Src {len(raw_src)} vs Tgt {len(raw_tgt)}"
            
        # 2. Tiền xử lý (Pre-tokenize)
        dropped_empty = 0
        dropped_outlier = 0
        dropped_ratio_mismatched = 0
        dropped_too_short = 0
        
        if self.dynamic_tokenize:
             print(f"Dynamic Tokenization enabled. Skipping pre-tokenization for {len(raw_src)} sentences.")
             # Chỉ clean data trước
             for src_text, tgt_text in tqdm(zip(raw_src, raw_tgt), total=len(raw_src), desc="Pre-cleaning"):
                # Làm sạch dữ liệu
                src_clean = self.preprocessor.process(src_text, 'en')
                tgt_clean = self.preprocessor.process(tgt_text, 'vi')
                
                if not src_clean or not tgt_clean:
                    dropped_empty += 1
                    continue
                    
                # Chỉ lưu văn bản. Việc Tokenization sẽ diễn ra trong __getitem__
                self.data_items.append({
                    "src_text": src_clean,
                    "tgt_text": tgt_clean,
                    "src_ids": None, # Sẽ được tính sau
                    "tgt_ids": None
                })
        else:
            print(f"Pre-tokenizing {len(raw_src)} sentences...")
            for src_text, tgt_text in tqdm(zip(raw_src, raw_tgt), total=len(raw_src), desc="Tokenizing"):
                # Làm sạch dữ liệu
                src_clean = self.preprocessor.process(src_text, 'en')
                tgt_clean = self.preprocessor.process(tgt_text, 'vi')
                
                # Loại bỏ cặp câu rỗng
                if not src_clean or not tgt_clean:
                    dropped_empty += 1
                    continue
                
                # Mã hóa (Encode)
                src_ids = self.tokenizer.encode(src_clean)
                tgt_ids = self.tokenizer.encode(tgt_clean)
                
                src_len = len(src_ids)
                tgt_len = len(tgt_ids)
                
                if src_len == 0 or tgt_len == 0:
                    dropped_empty += 1
                    continue
    
                # Lọc theo thống kê (Chỉ áp dụng cho dữ liệu Train)
                if self.filtering:
                    # Lọc câu quá ngắn
                    if src_len < self.min_len or tgt_len < self.min_len:
                        dropped_too_short += 1
                        continue
                    
                    # Lọc theo tỷ lệ độ dài
                    if src_len / tgt_len > self.max_ratio or tgt_len / src_len > self.max_ratio:
                        dropped_ratio_mismatched += 1
                        continue
                    
                    # Loại bỏ Outliers (Nếu dài hơn max_len thì bỏ luôn, không truncate)
                    # +2 cho SOS và EOS
                    if len(src_ids) + 2 > self.max_len or len(tgt_ids) + 2 > self.max_len:
                        dropped_outlier += 1
                        continue
                
                # Lưu lại kết quả đã encode
                self.data_items.append({
                    "src_ids": src_ids,
                    "tgt_ids": tgt_ids,
                    "src_text": src_clean,
                    "tgt_text": tgt_clean
                })
            
        print(f"Finished processing. Dropped {dropped_empty} empty items, {dropped_too_short} too short, {dropped_ratio_mismatched} ratio mismatched, and {dropped_outlier} outliers.")
        print(f"Retained {len(self.data_items)} valid items.")
            
        # 3. Lưu Cache nếu được yêu cầu
        if cache_path:
            print(f"Saving data to cache: {cache_path} ...")
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            torch.save(self.data_items, cache_path)
            print("Cache saved.")

    def _read_file(self, path: str) -> List[str]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        with open(path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f.readlines()]
        return lines

    def __len__(self):
        return len(self.data_items)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        item = self.data_items[idx]
        src_text = item["src_text"]
        tgt_text = item["tgt_text"]
        
        if self.dynamic_tokenize:
            # Tokenize trực tiếp (on-the-fly) (áp dụng BPE Dropout nếu có)
            src_ids = self.tokenizer.encode(src_text)
            tgt_ids = self.tokenizer.encode(tgt_text)
        else:
            # Dùng cache
            src_ids = item["src_ids"]
            tgt_ids = item["tgt_ids"]
        
        # Các bước xử lý Tensor (Padding, Masking)
        sos = [self.tokenizer.sos_token_id]
        eos = [self.tokenizer.eos_token_id]
        pad = self.tokenizer.pad_token_id
        
        enc_src = sos + src_ids + eos
        enc_tgt = sos + tgt_ids + eos
        
        if len(enc_src) > self.max_len:
            enc_src = enc_src[:self.max_len]
        
        if len(enc_tgt) > self.max_len:
            enc_tgt = enc_tgt[:self.max_len]
            
        src_padding = [pad] * (self.max_len - len(enc_src))
        tgt_padding = [pad] * (self.max_len - len(enc_tgt))
        
        enc_src_padded = enc_src + src_padding
        enc_tgt_padded = enc_tgt + tgt_padding
        
        encoder_input = torch.tensor(enc_src_padded, dtype=torch.long)
        decoder_input = torch.tensor(enc_tgt_padded, dtype=torch.long)
        
        label_ids = tgt_ids + eos
        if len(label_ids) > self.max_len:
             label_ids = label_ids[:self.max_len]
        label_padding = [pad] * (self.max_len - len(label_ids))
        label_padded = label_ids + label_padding
        label = torch.tensor(label_padded, dtype=torch.long)
        
        encoder_mask = (encoder_input != pad).unsqueeze(0).unsqueeze(0).int() 
        decoder_padding_mask = (decoder_input != pad).unsqueeze(0).unsqueeze(0).int()
        causal_mask = torch.tril(torch.ones((1, self.max_len, self.max_len))).int()
        decoder_mask = decoder_padding_mask & causal_mask
        
        return {
            "encoder_input": encoder_input,
            "decoder_input": decoder_input,
            "encoder_mask": encoder_mask,
            "decoder_mask": decoder_mask,
            "label": label,
            "src_text": src_text,
            "tgt_text": tgt_text
        }
