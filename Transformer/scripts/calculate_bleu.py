import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import argparse
import sacrebleu
from tqdm import tqdm

# Thêm thư mục gốc vào path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.core.config import AppConfig, TrainingConfig
from src.data.tokenizer import BilingualTokenizer
from src.data.dataset import BilingualDataset
from src.models.transformer import build_transformer
from src.training.trainer import Trainer
import logging

import re

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

def post_process(text: str) -> str:
    """
    Sửa lỗi dấu câu bị tách rời (Ví dụ: 'hello .' -> 'hello.')
    Để tránh warning của sacrebleu và đưa về dạng text tự nhiên.
    """
    # Xóa khoảng trắng trước dấu câu: . , ! ? : ;
    # Regex tìm: khoảng trắng + (dấu câu) + (khoảng trắng hoặc hết dòng)
    text = re.sub(r'\s+([.,!?:;])', r'\1', text)
    return text.strip()

def main():
    parser = argparse.ArgumentParser(description="Calculate BLEU Score for NMT Model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint file (.pt)")
    parser.add_argument("--data_dir", type=str, default="data/clean", help="Directory containing clean data")
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"], help="Split to evaluate on (val/test)")
    parser.add_argument("--beam_size", type=int, help="Override beam size in config")
    parser.add_argument("--decoding_method", type=str, choices=["greedy", "beam"], help="Override decoding method")
    args = parser.parse_args()

    # 1. Load Cấu hình & Device
    config_path = "configs/default.yaml"
    if not os.path.exists(config_path):
        logger.error(f"Config file not found: {config_path}")
        return
        
    app_config = AppConfig.load(config_path)
    model_cfg = app_config.model
    
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Set Seed
    train_cfg = app_config.training
    torch.manual_seed(train_cfg.seed)
    torch.manual_seed(train_cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(train_cfg.seed)

    inference_cfg = app_config.inference
    # Override từ CLI nếu được cung cấp
    if args.beam_size: inference_cfg.beam_size = args.beam_size
    if args.decoding_method: inference_cfg.decoding_method = args.decoding_method
    
    logger.info(f"Decoding Method: {inference_cfg.decoding_method}, Beam Size: {inference_cfg.beam_size}")

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
        # Lưu ý: Các tham số inference (beam_size, batch_size) vẫn dùng từ file yaml (inference_cfg)
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
    model.eval()

    # 5. Load Data
    if args.split == 'test':
        data_file_en = os.path.join(args.data_dir, "test.en")
        data_file_vi = os.path.join(args.data_dir, "test.vi")
    elif args.split == 'val':
        data_file_en = os.path.join(args.data_dir, "dev.en")
        data_file_vi = os.path.join(args.data_dir, "dev.vi")
    else:
        # Fallback hoặc custom
        data_file_en = os.path.join(args.data_dir, f"{args.split}.en")
        data_file_vi = os.path.join(args.data_dir, f"{args.split}.vi")


    if not os.path.exists(data_file_en):
        logger.error(f"Data file not found: {data_file_en}")
        return

    logger.info(f"Loading data from {data_file_en}...")
    cache_path = None # Không cache cho test để tránh phức tạp
    if args.split == 'val':
         cache_name = os.path.basename(data_file_en).replace(".en", ".pt")
         cache_path = os.path.join("data/cache", cache_name)

    ds = BilingualDataset(
        data_file_en, data_file_vi, tokenizer, 
        max_len=model_cfg.max_len, 
        cache_path=cache_path,
        filtering=False
    )

    # Batch size từ config
    eval_batch_size = inference_cfg.eval_batch_size
    logger.info(f"Using Eval Batch Size: {eval_batch_size}")
    
    loader = DataLoader(ds, batch_size=eval_batch_size, shuffle=False)

    
    # 6. Run Inference & Calculate BLEU
    logger.info(f"Evaluating BLEU on {len(ds)} sentences...")
    
    # Dùng Trainer để reuse hàm greedy_decode
    trainer = Trainer(
        model=model,
        optimizer=None, criterion=None, 
        train_loader=None, val_loader=None,
        config=TrainingConfig(), device=device,
        tokenizer=tokenizer, max_len=model_cfg.max_len
    )

    hypotheses = []
    references = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Translating"):
            encoder_input = batch['encoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)
            tgt_text = batch['tgt_text'] # List of strings

            # Decode
            if inference_cfg.decoding_method == "beam" and inference_cfg.beam_size > 1:
                model_out_ids = trainer.batched_beam_search_decode(
                    encoder_input, 
                    encoder_mask, 
                    beam_size=inference_cfg.beam_size, 
                    max_len=model_cfg.max_len + 5,
                    length_penalty_alpha=inference_cfg.length_penalty
                )
            else:
                # Greedy Decode đã hỗ trợ batch
                model_out_ids = trainer.greedy_decode(
                    encoder_input, 
                    encoder_mask, 
                    max_len=model_cfg.max_len + 5, 
                    start_symbol=tokenizer.sos_token_id
                )
            
            # Detokenize & Post-process cho cả batch
            model_out_ids_list = model_out_ids.tolist()
            
            for i, out_ids in enumerate(model_out_ids_list):
                 # Cutoff tại EOS để tránh token nhiễu
                 if tokenizer.eos_token_id in out_ids:
                     eos_idx = out_ids.index(tokenizer.eos_token_id)
                     out_ids = out_ids[:eos_idx]
                     
                 out_text = tokenizer.decode(out_ids)
                 hyp = post_process(out_text)
                 ref = post_process(tgt_text[i])
                 
                 hypotheses.append(hyp)
                 references.append(ref)


    # 7. Compute BLEU
    bleu = sacrebleu.corpus_bleu(hypotheses, [references])
    
    print("-" * 50)
    print(f"BLEU Score: {bleu.score:.2f}")
    print(f"Details: {bleu}")
    print("-" * 50)

if __name__ == "__main__":
    main()
