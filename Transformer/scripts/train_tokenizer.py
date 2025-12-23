import sys
import os
import argparse
from pathlib import Path

# Thêm path để import module src
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.core.logging_utils import setup_logging
from src.core.logging_utils import setup_logging
from src.core.logging_utils import setup_logging
from src.data.tokenizer import BilingualTokenizer
from src.core.config import AppConfig

def main():
    # Load config defaults first
    config_path = "configs/default.yaml"
    default_vocab_size = 32768
    
    if os.path.exists(config_path):
        try:
            app_config = AppConfig.load(config_path)
            default_vocab_size = app_config.model.vocab_size
        except Exception as e:
            print(f"Warning: Could not load config from {config_path}: {e}")

    parser = argparse.ArgumentParser(description="Clean and Train BPE Tokenizer")
    parser.add_argument("--src_lang", type=str, default="en", help="Source language extension (e.g., en)")
    parser.add_argument("--tgt_lang", type=str, default="vi", help="Target language extension (e.g., vi)")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing train files")
    parser.add_argument("--vocab_size", type=int, default=default_vocab_size, help=f"Vocabulary size (default: {default_vocab_size} from config)")
    parser.add_argument("--save_path", type=str, default="tokenizer.json", help="Path to save trained tokenizer")
    
    args = parser.parse_args()
    logger = setup_logging()
    
    logger.info(f"Starting Tokenizer training with vocab_size={args.vocab_size}")
    
    # Tìm các file dữ liệu (Chỉ dùng tập train để tránh data leakage)
    data_path = Path(args.data_dir)
    files = [str(f) for f in data_path.glob("train.*") if f.suffix in [f".{args.src_lang}", f".{args.tgt_lang}"]]
    
    if not files:
        logger.error(f"No files found in {args.data_dir} with extensions .{args.src_lang} or .{args.tgt_lang}")
        sys.exit(1)
        
    logger.info(f"Found {len(files)} files to train on: {files}")
    
    logger.info(f"Found {len(files)} files to train on: {files}")
    
    # Khởi tạo và train
    print("Initializing Tokenizer...")
    tokenizer = BilingualTokenizer(vocab_size=args.vocab_size)

    try:
        tokenizer.train(files)
        logger.info("Training completed successfully!")
        
        tokenizer.save(args.save_path)
        logger.info(f"Tokenizer saved to {args.save_path}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
