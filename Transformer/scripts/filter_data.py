import sys
import os
import argparse
import shutil
import logging

# Thêm thư mục gốc vào path để import các module trong src
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.core.config import AppConfig
from src.data.tokenizer import BilingualTokenizer
from src.data.dataset import BilingualDataset

# Cấu hình logging ra màn hình console
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Script to run Data Filtering independently and create Cache.")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config file")
    parser.add_argument("--force", action="store_true", help="If set, will delete old cache to force re-filtering")
    parser.add_argument("--export-text", action="store_true", help="If set, will save filtered data to text files (filtered.en/vi) for inspection")
    
    args = parser.parse_args()
    
    # 1. Load Configuration
    if not os.path.exists(args.config):
        logger.error(f"Config file not found at {args.config}")
        return

    app_config = AppConfig.load(args.config)
    model_cfg = app_config.model
    data_cfg = app_config.data
    
    logger.info(f"Loaded config from {args.config}")
    logger.info(f"Filter config: min_len={data_cfg.min_len}, max_ratio={data_cfg.max_ratio}")
    logger.info(f"Model config: max_len={model_cfg.max_len}")

    # 2. Load Tokenizer
    tokenizer_path = "tokenizer.json"
    if not os.path.exists(tokenizer_path):
        logger.error(f"Tokenizer file not found at {tokenizer_path}. Run train_tokenizer.py first.")
        return
        
    tokenizer = BilingualTokenizer.load(tokenizer_path)
    logger.info(f"Loaded Tokenizer (Vocab: {tokenizer.vocab_size})")

    # 3. Determine data file paths
    train_file_en = "data/clean/train.en"
    train_file_vi = "data/clean/train.vi"
    
    if not os.path.exists(train_file_en) or not os.path.exists(train_file_vi):
        # Fallback raw if clean not found (but clean is recommended)
        train_file_en = "data/raw/train.en"
        train_file_vi = "data/raw/train.vi"
        logger.warning(f"Clean data not found. Using raw: {train_file_en}")
        
    if not os.path.exists(train_file_en):
        logger.error("Original train data not found (raw or clean).")
        return

    # Cache file path
    cache_path = os.path.join("data/cache", os.path.basename(train_file_en).replace(".en", ".pt"))

    # 4. Handle Force logic (Delete old cache)
    if args.force:
        if os.path.exists(cache_path):
            logger.info(f"Deleting old cache at {cache_path} as requested by --force...")
            os.remove(cache_path)
            logger.info("Old cache deleted.")
        else:
            logger.info("No old cache found to delete.")
    
    # 5. Initialize Dataset (Filtering process happens here)
    logger.info("Starting filtering and caching process...")
    
    # Note: We pass min_len and max_ratio from config here
    dataset = BilingualDataset(
        src_file=train_file_en,
        tgt_file=train_file_vi,
        tokenizer=tokenizer,
        max_len=model_cfg.max_len,
        cache_path=cache_path,
        min_len=data_cfg.min_len,
        max_ratio=data_cfg.max_ratio
    )
    
    logger.info(f"Completed! Number of data samples after filtering: {len(dataset)}")
    logger.info(f"Cache file saved at: {cache_path}")

    # 6. Export text if requested
    if args.export_text:
        export_en = "data/clean/filtered.en"
        export_vi = "data/clean/filtered.vi"
        logger.info(f"Exporting filtered data to text: {export_en} and {export_vi} ...")
        
        with open(export_en, 'w', encoding='utf-8') as f_en, \
             open(export_vi, 'w', encoding='utf-8') as f_vi:
            for item in dataset.data_items:
                f_en.write(item["src_text"] + "\n")
                f_vi.write(item["tgt_text"] + "\n")
                
        logger.info("Text export finished.")

if __name__ == "__main__":
    main()
