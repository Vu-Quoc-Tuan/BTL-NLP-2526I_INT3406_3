import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import argparse

# Thêm thư mục gốc vào path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.core.config import AppConfig, TrainingConfig
from src.data.tokenizer import BilingualTokenizer
from src.data.dataset import BilingualDataset
from src.models.transformer import build_transformer
from src.training.trainer import Trainer
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Test model on Validation and Test sets")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint file (.pt)")
    parser.add_argument("--data_dir", type=str, default="data/clean", help="Directory containing clean data")
    args = parser.parse_args()

    # 1. Load Cấu hình & Device
    config_path = "configs/default.yaml"
    if not os.path.exists(config_path):
        logger.error(f"Config file not found: {config_path}")
        return
        
    app_config = AppConfig.load(config_path)
    model_cfg = app_config.model
    # Cập nhật configs từ checkpoint nếu cần thiết, nhưng ở đây ta dùng config gốc để build model
    
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Set Seed
    train_cfg = app_config.training
    torch.manual_seed(train_cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(train_cfg.seed)

    # 2. Load Tokenizer
    tokenizer_path = "tokenizer.json"
    if not os.path.exists(tokenizer_path):
        logger.error("Tokenizer not found!")
        return
    tokenizer = BilingualTokenizer.load(tokenizer_path)

    # 3. Load Checkpoint & Determine Model Config
    if not os.path.exists(args.checkpoint):
        logger.error(f"Checkpoint not found: {args.checkpoint}")
        return
        
    logger.info(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    
    # Ưu tiên lấy config từ checkpoint để build model đúng kiến trúc
    if 'config' in checkpoint and hasattr(checkpoint['config'], 'model'):
        logger.info("Using model architecture config from CHECKPOINT.")
        model_cfg = checkpoint['config'].model
    else:
        logger.warning("Config not found in checkpoint! Falling back to default.yaml")
        # model_cfg đã được load từ yaml ở trên

    # 4. Build Model
    model = build_transformer(
        src_vocab_size=tokenizer.vocab_size,
        tgt_vocab_size=tokenizer.vocab_size,
        src_seq_len=model_cfg.max_len,
        tgt_seq_len=model_cfg.max_len,
        d_model=model_cfg.d_model,
        N=model_cfg.n_layers,
        h=model_cfg.heads,
        dropout=model_cfg.dropout,
        d_ff=model_cfg.d_model * 4
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    # 5. Khởi tạo Dummy Trainer (chỉ để dùng hàm sample_translation)
    # Không cần optimizer hay loss
    trainer = Trainer(
        model=model,
        optimizer=torch.optim.Adam(model.parameters()), # Giả lập, không dùng
        criterion=nn.CrossEntropyLoss(), # Giả lập, không dùng
        train_loader=None,
        val_loader=None, # Sẽ truyền trực tiếp vào hàm sample
        config=TrainingConfig(), # Giả lập
        device=device,
        tokenizer=tokenizer,
        max_len=model_cfg.max_len
    )

    # 6. Load Data & Test
    # --- VALIDATION SET ---
    data_file_en = os.path.join(args.data_dir, "train.en")
    data_file_vi = os.path.join(args.data_dir, "train.vi")
    
    # Fallback to mini if full not found
    if not os.path.exists(data_file_en):
        data_file_en = os.path.join(args.data_dir, "mini.en")
        data_file_vi = os.path.join(args.data_dir, "mini.vi")

    if os.path.exists(data_file_en):
        logger.info(f"Loading Validation samples from {data_file_en}...")
        # Lấy file cache tương ứng
        cache_name = os.path.basename(data_file_en).replace(".en", ".pt")
        cache_path = os.path.join("data/cache", cache_name)
        
        full_ds = BilingualDataset(
            data_file_en, data_file_vi, tokenizer, 
            max_len=model_cfg.max_len, 
            cache_path=cache_path,
            filtering=False
        )
        
        # Tách Validation set y hệt như lúc train (10% cuối)
        train_size = int(0.9 * len(full_ds))
        val_size = len(full_ds) - train_size
        # Generator seed
        generator = torch.Generator().manual_seed(train_cfg.seed)
        _, val_ds = random_split(full_ds, [train_size, val_size], generator=generator)
        
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=True) # Shuffle để lấy ngẫu nhiên
        
        logger.info("=== KẾT QUẢ DỊCH THỬ TRÊN TẬP VALIDATION (5 samples) ===")
        trainer.sample_translation(loader=val_loader, num_samples=5)
    else:
        logger.warning(f"Validation data not found at {data_file_en}")

    # --- TEST SET ---
    test_file_en = os.path.join(args.data_dir, "test.en")
    test_file_vi = os.path.join(args.data_dir, "test.vi")
    
    if os.path.exists(test_file_en):
        logger.info(f"Loading Test samples from {test_file_en}...")
        test_ds = BilingualDataset(
            test_file_en, test_file_vi, tokenizer, 
            max_len=model_cfg.max_len, 
            cache_path=None, # Test thường nhỏ, ko cần cache
            filtering=False
        )
        test_loader = DataLoader(test_ds, batch_size=1, shuffle=True)
        
        logger.info("=== KẾT QUẢ DỊCH THỬ TRÊN TẬP TEST (5 samples) ===")
        trainer.sample_translation(loader=test_loader, num_samples=5)
    else:
        logger.info("No test set found (test.en). Skipping.")

if __name__ == "__main__":
    main()
