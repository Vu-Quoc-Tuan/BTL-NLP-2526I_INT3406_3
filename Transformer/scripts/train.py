import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

# Thêm thư mục gốc dự án vào path để import module

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.core.config import AppConfig
from src.data.tokenizer import BilingualTokenizer
from src.data.dataset import BilingualDataset
from src.models.transformer import build_transformer
from src.training.trainer import Trainer
import logging
import math
from torch.optim.lr_scheduler import LambdaLR

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5, last_epoch=-1):
    """
    Tạo lịch trình thay đổi learning rate giảm theo hàm cosine (từ 0 đến pi * cycles)
    sau giai đoạn khởi động (warmup) tăng tuyến tính.
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)

# Cấu hình logging (ghi nhật ký)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    # 1. Load Cấu hình
    config_path = "configs/default.yaml"
    app_config = AppConfig.load(config_path)
    model_cfg = app_config.model
    train_cfg = app_config.training
    data_cfg = app_config.data
    logger.info(f"Loaded config from {config_path}")
    
    # Set Seed
    torch.manual_seed(train_cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(train_cfg.seed)
    logger.info(f"Random seed set to: {train_cfg.seed}")

    # 2. Cấu hình Thiết bị (MPS cho Mac M1/M2, CUDA cho Nvidia, còn lại là CPU)
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    logger.info(f"Using device: {device}")

    # 3. Load Tokenizer (Bộ tách từ)
    tokenizer_path = "tokenizer.json"
    if not os.path.exists(tokenizer_path):
        logger.error(f"Tokenizer file not found at {tokenizer_path}. Run train_tokenizer.py first.")
        return
    
    # Tải 2 tokenizer riêng biệt cho Training (có Dropout) và Eval (Cố định/Deterministic)
    # BPE Dropout giúp mô hình học biểu diễn từ phụ (subword) tốt hơn, tăng khả năng robust
    tokenizer_train = BilingualTokenizer.load(tokenizer_path, dropout=data_cfg.bpe_dropout)
    tokenizer_eval = BilingualTokenizer.load(tokenizer_path, dropout=None)
    
    logger.info(f"Loaded tokenizer. Vocab size: {tokenizer_train.vocab_size}")

    # 4. Khởi tạo Datasets & Dataloaders
    # --- TRAIN SET ---
    if os.path.exists("data/clean/train.en") and os.path.exists("data/clean/train.vi"):
        train_file_en = "data/clean/train.en"
        train_file_vi = "data/clean/train.vi"
        logger.info("Detected FULL Train dataset (train.en/vi)")
    else:
        # Fallback raw
        train_file_en = "data/raw/train.en"
        train_file_vi = "data/raw/train.vi"
        logger.warning(f"Clean data not found. Falling back to {train_file_en}")

    train_cache = os.path.join("data/cache", os.path.basename(train_file_en).replace(".en", ".pt"))
    
    # Sử dụng Dynamic Tokenization cho Training để enable BPE Dropout randomness
    train_ds = BilingualDataset(
        train_file_en, train_file_vi, 
        tokenizer=tokenizer_train, # Use Dropout Tokenizer
        max_len=model_cfg.max_len, cache_path=train_cache,
        min_len=data_cfg.min_len, max_ratio=data_cfg.max_ratio,
        dynamic_tokenize=True
    )

    # --- DEV (VALIDATION) SET ---
    # User yêu cầu dùng dev riêng
    if os.path.exists("data/clean/dev.en") and os.path.exists("data/clean/dev.vi"):
        val_file_en = "data/clean/dev.en"
        val_file_vi = "data/clean/dev.vi"
        logger.info("Detected DEV dataset (dev.en/vi) - Using for Validation")
        
        val_cache = os.path.join("data/cache", os.path.basename(val_file_en).replace(".en", ".pt"))
        # Validation set: KHÔNG NÊN lọc quá chặt, để phản ánh hiệu năng thực tế.
        # Ta đặt min_len=1 và max_ratio=100 để vô hiệu hóa các bộ lọc này một cách hiệu quả.
        val_ds = BilingualDataset(
            val_file_en, val_file_vi, 
            tokenizer=tokenizer_eval, # Use Eval Tokenizer (No Dropout)
            max_len=model_cfg.max_len, cache_path=val_cache,
            min_len=1, max_ratio=100.0, filtering=False,
            dynamic_tokenize=False
        )
    else:
        logger.warning("Dev dataset (dev.en/vi) not found!")
        logger.warning("Using Random Split 95/5 from Train set as fallback.")
        # Lưu ý: Khi random split từ train_ds, val_ds cũng sẽ kế thừa tokenizer_train và dynamic_tokenize=True
        # Điều này hơi không tối ưu cho validation (kết quả sẽ giao động nhẹ), nhưng chấp nhận được nếu thiếu dev set.
        train_size = int(0.95 * len(train_ds))
        val_size = len(train_ds) - train_size
        generator = torch.Generator().manual_seed(train_cfg.seed)
        train_ds, val_ds = random_split(train_ds, [train_size, val_size], generator=generator)

    
    train_loader = DataLoader(
        train_ds, 
        batch_size=train_cfg.batch_size, 
        shuffle=True,
        num_workers=train_cfg.num_workers,
        pin_memory=True if device == "cuda" else False
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size=train_cfg.batch_size, 
        shuffle=False,
        num_workers=train_cfg.num_workers,
        pin_memory=True if device == "cuda" else False
    )
    
    logger.info(f"Final sizes -> Train: {len(train_ds)}, Val: {len(val_ds)}")

    # 5. Xây dựng Model
    model = build_transformer(
        src_vocab_size=tokenizer_train.vocab_size,
        tgt_vocab_size=tokenizer_train.vocab_size,
        src_seq_len=model_cfg.max_len,
        tgt_seq_len=model_cfg.max_len,
        d_model=model_cfg.d_model,
        N=model_cfg.n_layers,
        h=model_cfg.heads,
        dropout=model_cfg.dropout,
        d_ff=model_cfg.d_model * 4 # Standard Transformer: d_ff = 4 * d_model
    )
    
    # 6. Cấu hình Optimizer & Loss
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=train_cfg.lr, 
        betas=(0.9, 0.98), 
        eps=1e-9, 
        weight_decay=0.0001
    )
    
    # Loss: ignore_index là padding token id
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer_train.pad_token_id, label_smoothing=0.1)

    # 7. Cấu hình Scheduler (Warmup + Cosine Decay)
    total_steps = len(train_loader) * train_cfg.epochs
    num_warmup_steps = int(total_steps * train_cfg.warmup_ratio)
    
    logger.info(f"Total training steps: {total_steps}")
    logger.info(f"Warmup steps: {num_warmup_steps} ({train_cfg.warmup_ratio * 100}%)")
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=num_warmup_steps, 
        num_training_steps=total_steps
    )

    # 8. Bắt đầu Training
    train_machine = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_loader=train_loader,
        val_loader=val_loader,
        config=app_config,
        scheduler=scheduler,
        device=device,
        tokenizer=tokenizer_eval, # Dùng Eval Tokenizer cho Decoding
        max_len=model_cfg.max_len 
    )
    
    train_machine.fit()

    # 8. Sau khi train xong: Test ngẫu nhiên 5 câu trên tập Validation và Test (nếu có)
    logger.info("=== SAMPLE TRANSLATIONS ON VALIDATION SET ===")
    train_machine.sample_translation(loader=val_loader, num_samples=5)

    # Kiểm tra xem có tập Test riêng không
    test_file_en = "data/clean/test.en"
    test_file_vi = "data/clean/test.vi"
    
    if os.path.exists(test_file_en) and os.path.exists(test_file_vi):
        logger.info(f"Detected TEST dataset ({test_file_en})")
        logger.info("=== SAMPLE TRANSLATIONS ON TEST SET ===")
        
        # Test set: Không lọc (No filtering) ngoại trừ làm sạch cơ bản
        test_ds = BilingualDataset(
            test_file_en, 
            test_file_vi, 
            tokenizer=tokenizer_eval, 
            max_len=model_cfg.max_len,
            cache_path=None, # Không cần cache test set vì nó nhỏ
            min_len=1, max_ratio=100.0, filtering=False
        )
        test_loader = DataLoader(test_ds, batch_size=1, shuffle=True)
        train_machine.sample_translation(loader=test_loader, num_samples=5)
    else:
        logger.info("No explicit test set found (test.en/vi). Skipping test set verification.")

if __name__ == "__main__":
    main()
