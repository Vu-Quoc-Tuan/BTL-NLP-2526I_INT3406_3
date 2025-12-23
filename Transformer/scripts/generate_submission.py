import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Thêm path để import module từ src
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.core.config import AppConfig
from src.data.tokenizer import BilingualTokenizer
from src.data.dataset import BilingualDataset
from src.models.transformer import build_transformer
from src.training.trainer import Trainer
import logging

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

import argparse

def main():
    parser = argparse.ArgumentParser(description="Generate translation submission file")
    parser.add_argument("--test_src", type=str, default="data/clean/test.en", help="Path to source test file (English)")
    parser.add_argument("--output", type=str, default="predictions.txt", help="Path to output prediction file")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to specific model checkpoint. If None, picks the latest in save_dir.")
    args = parser.parse_args()

    # 1. Load Cấu hình
    config_path = args.config
    if not os.path.exists(config_path):
        logger.error(f"Config file not found at {config_path}")
        return

    app_config = AppConfig.load(config_path)
    model_cfg = app_config.model
    # inference_cfg = app_config.inference # Nếu có section inference
    
    # 2. Thiết bị
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # 3. Load Tokenizer
    tokenizer_path = "tokenizer.json"
    if not os.path.exists(tokenizer_path):
        logger.error(f"Tokenizer not found at {tokenizer_path}")
        return
    
    # Dùng tokenizer không dropout cho inference
    tokenizer = BilingualTokenizer.load(tokenizer_path, dropout=None)
    logger.info(f"Loaded Tokenizer. Vocab size: {tokenizer.vocab_size}")

    # 4. Load Model Checkpoint
    if args.checkpoint_path:
        ckpt_path = args.checkpoint_path
        if not os.path.exists(ckpt_path):
            logger.error(f"Checkpoint not found at {ckpt_path}")
            return
        logger.info(f"Loading specific checkpoint: {ckpt_path}")
    else:
        # Tìm checkpoint mới nhất trong thư mục save_dir
        save_dir = app_config.training.save_dir
        if not os.path.exists(save_dir):
            logger.error(f"Checkpoints directory not found: {save_dir}")
            return
            
        checkpoints = sorted([f for f in os.listdir(save_dir) if f.endswith(".pt")])
        if not checkpoints:
            logger.error("No checkpoints found!")
            return
        
        latest_ckpt = checkpoints[-1]
        ckpt_path = os.path.join(save_dir, latest_ckpt)
        logger.info(f"Loading latest checkpoint: {ckpt_path}")
    
    # PyTorch 2.6+ defaults weights_only=True which blocks custom objects like AppConfig
    # We set weights_only=False because we trust our own checkpoints
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    # Build model structure
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
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # 5. Load Test Data (Source Only for Contest)
    test_src = args.test_src
    
    if not os.path.exists(test_src):
        logger.error(f"Test source file not found: {test_src}")
        return
        
    # Tạo dummy target file vì dataset cần cặp (không cần nội dung đúng)
    # Lưu vào thư mục tmp hoặc cùng chỗ với test_src
    test_tgt = test_src + ".dummy" 
    
    # Tạo file dummy nếu chưa tồn tại
    if not os.path.exists(test_tgt):
        logger.info(f"Creating dummy target file at {test_tgt}")
        # Count lines in source file
        with open(test_src, 'r', encoding='utf-8') as f:
            src_lines = f.readlines()
            
        with open(test_tgt, 'w', encoding='utf-8') as f:
            for _ in src_lines:
                f.write("a \n") # Write "a" to ensure non-empty token list after cleaning
                
    # Dataset & Loader
    # Validation/Test: KHÔNG lọc theo ratio/min_len để giữ đúng số lượng câu của đề bài
    test_ds = BilingualDataset(
        test_src, test_tgt, 
        tokenizer=tokenizer, 
        max_len=model_cfg.max_len,
        cache_path=None, # Không cache
        min_len=0, max_ratio=100.0, filtering=False, # Quan trọng: Không lọc
        dynamic_tokenize=False
    )
    
    # Quan trọng: Không shuffle để giữ thứ tự output khớp với input (cho chấm điểm)
    # Tăng batch_size để tối ưu tốc độ (model nhẹ nên 32 là an toàn)
    test_batch_size = 32
    # Nếu config có khai báo eval_batch_size thì dùng
    if hasattr(app_config, 'inference') and app_config.inference.eval_batch_size:
        test_batch_size = app_config.inference.eval_batch_size
        
    test_loader = DataLoader(test_ds, batch_size=test_batch_size, shuffle=False) 
    
    logger.info(f"Test Set Size: {len(test_ds)}")
    logger.info(f"Using Batch Size: {test_batch_size}")

    # 6. Inference Loop
    output_file = args.output
    logger.info(f"Starting inference... Output will be saved to {output_file}")
    
    # Khởi tạo Trainer (chỉ để dùng hàm decode tiện ích)
    trainer = Trainer(
        model=model, 
        optimizer=None, criterion=None, 
        train_loader=None, val_loader=None, 
        config=app_config, tokenizer=tokenizer, 
        device=device, max_len=model_cfg.max_len
    )
    
    predictions = []
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader, desc="Translating")):
            encoder_input = batch['encoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)
            
            # Beam Search (hoặc Greedy)
            # Lấy beam_size từ config nếu có
            beam_size = 3 
            if hasattr(app_config, 'inference') and app_config.inference.beam_size:
                beam_size = app_config.inference.beam_size

            if beam_size > 1:
                model_out_ids = trainer.batched_beam_search_decode(
                    encoder_input, encoder_mask, 
                    beam_size=beam_size, 
                    max_len=model_cfg.max_len + 5,
                    start_symbol=tokenizer.sos_token_id
                )
            else:
                model_out_ids = trainer.greedy_decode(
                    encoder_input, encoder_mask, 
                    max_len=model_cfg.max_len + 5, 
                    start_symbol=tokenizer.sos_token_id
                )
            
            # Decode & Detokenize
            # Model out ids is (Batch, Seq)
            # Duyệt từng câu trong batch
            batch_out_list = model_out_ids.tolist()
            
            for pred_ids in batch_out_list:
                # Cắt tại EOS
                if tokenizer.eos_token_id in pred_ids:
                    pred_ids = pred_ids[:pred_ids.index(tokenizer.eos_token_id)]
                    
                pred_text = tokenizer.decode(pred_ids)
                
                if hasattr(tokenizer, 'detokenize'):
                    pred_text = tokenizer.detokenize(pred_text)
                
                predictions.append(pred_text)
            
    # 7. Write to file
    with open(output_file, "w", encoding="utf-8") as f:
        for line in predictions:
            f.write(line + "\n")
            
    logger.info("Done! Predictions saved.")

    # Dọn dẹp file dummy
    if os.path.exists(test_tgt):
        os.remove(test_tgt)

if __name__ == "__main__":
    main()
