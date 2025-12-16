# VLSP Medical Translation Pipeline

## Cài đặt

```bash
pip install -r requirements.txt
```

## Full Pipeline

> 🔴 = **Required** | 🟡 = **Optional**

```bash
# 🔴 1. Preprocessing & Splitting
python scripts/preprocess_vlsp.py \
    --src_in data/raw/train.en --tgt_in data/raw/train.vi \
    --out_dir data/clean \
    --min_len 3 --max_len 256 --max_ratio 3.0 \
    --dev_size 1000 --test_size 1000

# 🟡 2. Deduplication (nếu data lớn, có nhiều duplicate)
python scripts/dedup_minhash.py \
    --src data/clean/train.en --tgt data/clean/train.vi \
    --out_dir data/dedup \
    --threshold 0.8 --dedup_by both --rep_strategy longest

# 🟡 3. Analyze Vocabulary (kiểm tra OOV)
python scripts/analyze_vocab.py \
    --train data/clean/train.en --test data/clean/test.en --top_oov 50

# 🔴 4. SFT Training
python scripts/train_qwen_lora.py \
    --model_name Qwen/Qwen2.5-3B-Instruct \
    --direction en2vi \
    --src data/clean/train.en --tgt data/clean/train.vi \
    --val_src data/clean/dev.en --val_tgt data/clean/dev.vi \
    --run_id qwen_en2vi_v1 \
    --lora_r 32 --lora_alpha 64 \
    --lr 2e-4 --batch_size 4 --grad_accum 4 --epochs 3 \
    --neftune_alpha 5.0 --label_smoothing 0.1 --early_stopping_patience 3

# 🟡 5. RL Training (GRPO) - cải thiện thêm sau SFT
python scripts/rl_train_grpo.py \
    --model_name Qwen/Qwen2.5-3B-Instruct \
    --sft_adapter runs/qwen_en2vi_v1/lora_en2vi_sft \
    --init_adapter runs/qwen_en2vi_v1/lora_en2vi_sft \
    --rl_src data/clean/train.en --rl_tgt data/clean/train.vi \
    --run_id qwen_en2vi_v1_rl --direction en2vi \
    --epochs 1 --batch_size 4 --grad_accum_steps 4 --lr 3e-6 --kl_coef 0.01

# 🔴 6. Generate Translations
python scripts/generate.py \
    --model_name Qwen/Qwen2.5-3B-Instruct \
    --adapter_path runs/qwen_en2vi_v1/lora_en2vi_sft \
    --direction en2vi \
    --input data/clean/test.en --output outputs/test.hyp.vi \
    --batch_size 8 --num_beams 4

# 🔴 7. Evaluation
python scripts/eval_bleu.py \
    --hyp outputs/test.hyp.vi --ref data/clean/test.vi

# 🟡 7b. Evaluation với Gemini Score
python scripts/eval_bleu.py \
    --hyp outputs/test.hyp.vi --ref data/clean/test.vi --src data/clean/test.en \
    --gemini --gemini_api_key $GEMINI_API_KEY --gemini_samples 100 --direction en2vi

# 🟡 Quick eval pipeline (generate + eval)
python scripts/run_eval_all.py \
    --run_id qwen_en2vi_v1 --direction en2vi --model_name Qwen/Qwen2.5-3B-Instruct
```

## VI→EN (thay đổi direction và swap src/tgt)

```bash
python scripts/train_qwen_lora.py \
    --model_name Qwen/Qwen2.5-3B-Instruct \
    --direction vi2en \
    --src data/clean/train.vi --tgt data/clean/train.en \
    --val_src data/clean/dev.vi --val_tgt data/clean/dev.en \
    --run_id qwen_vi2en_v1 \
    --lora_r 32 --lr 2e-4 --batch_size 4 --grad_accum 4 --epochs 3
```

## Project Structure

```
vlsp-mt/
├── data/
│   ├── raw/          # Dữ liệu gốc
│   ├── dedup/        # Sau dedup (optional)
│   └── clean/        # Train/dev/test splits
├── runs/{run_id}/
│   ├── lora_{direction}_sft/
│   └── meta.json
├── outputs/          # Generated translations
└── scripts/
```

## Hyperparameters

| Data Size | lora_r | batch | grad_accum | epochs | lr |
|-----------|--------|-------|------------|--------|-----|
| <10k | 16 | 4 | 2 | 5-10 | 2e-4 |
| 10k-100k | 32 | 4-8 | 4 | 3-5 | 2e-4 |
| >100k | 64 | 8-16 | 4-8 | 2-3 | 1e-4 |

## Troubleshooting

- **OOM**: Giảm `batch_size`, tăng `grad_accum`, dùng `--max_len 128`
- **Loss không giảm**: Giảm lr, tăng `lora_r`
- **BLEU thấp**: Train thêm epochs, dùng RL fine-tuning
