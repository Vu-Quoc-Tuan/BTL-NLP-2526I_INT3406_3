# FULL PIPELINE - VLSP Medical Translation
# Tu A den Z: Preprocess -> Dedup -> Train -> RL -> Generate -> Eval

## TONG QUAN PIPELINE

```
[Raw Data] 
    |
    v
[1. Preprocess] --> Clean, filter, split train/dev/test
    |
    v
[2. Dedup] --> Loai bo duplicate (optional)
    |
    v
[3. SFT Training] --> Train LoRA adapter (EN->VI va VI->EN)
    |
    v
[4. RL Training] --> Fine-tune them bang GRPO (optional)
    |
    v
[5. Back-Translation] --> Tang data (optional)
    |
    v
[6. Generate] --> Dich test set
    |
    v
[7. Evaluate] --> BLEU, chrF, Gemini Score
```

---

## BUOC 0: SETUP

```bash
cd vlsp-mt

# Install dependencies
pip install -r requirements.txt
pip install evaluate

# Kiem tra GPU
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"
```

---

## BUOC 1: PREPROCESSING (Da chay roi - co the skip)

```bash
python scripts/preprocess_vlsp.py \
    --src_in data/raw/train.en.txt \
    --tgt_in data/raw/train.vi.txt \
    --out_dir data/clean \
    --min_len 3 \
    --max_len 256 \
    --max_ratio 3.0 \
    --dedup both \
    --dev_size 1000 \
    --test_size 1000 \
    --seed 42
```

**Output:**
- `data/clean/train.en`, `data/clean/train.vi` (337,489 cau)
- `data/clean/dev.en`, `data/clean/dev.vi` (1,000 cau)
- `data/clean/test.en`, `data/clean/test.vi` (1,000 cau)

---

## BUOC 2: DEDUPLICATION (Optional - neu data co nhieu duplicate)

```bash
python scripts/dedup_minhash.py \
    --src data/clean/train.en \
    --tgt data/clean/train.vi \
    --out_dir data/dedup \
    --threshold 0.8 \
    --dedup_by both \
    --rep_strategy longest \
    --smart_threshold
```

**Giai thich:**
- `--smart_threshold`: 0.95 cho cau co so lieu y khoa (mg, ml...), 0.85 cho cau thuong
- `--rep_strategy longest`: Giu cau dai nhat trong cluster duplicate

---

## BUOC 3: SFT TRAINING (QUAN TRONG NHAT)

### 3A. Train EN->VI

```bash
python scripts/train_qwen_lora.py \
    --model_name Qwen/Qwen2.5-3B-Instruct \
    --direction en2vi \
    --src data/clean/train.en \
    --tgt data/clean/train.vi \
    --val_src data/clean/dev.en \
    --val_tgt data/clean/dev.vi \
    --run_id en2vi_v1 \
    --epochs 2 \
    --batch_size 8 \
    --grad_accum 4 \
    --lr 1.5e-4 \
    --lora_r 32 \
    --lora_alpha 64 \
    --neftune_alpha 3.0 \
    --eval_bleu \
    --bleu_sample_size 200 \
    --eval_steps 1000 \
    --early_stopping_patience 3 \
    --max_len 256
```

### 3B. Train VI->EN

```bash
python scripts/train_qwen_lora.py \
    --model_name Qwen/Qwen2.5-3B-Instruct \
    --direction vi2en \
    --src data/clean/train.vi \
    --tgt data/clean/train.en \
    --val_src data/clean/dev.vi \
    --val_tgt data/clean/dev.en \
    --run_id vi2en_v1 \
    --epochs 2 \
    --batch_size 8 \
    --grad_accum 4 \
    --lr 1.5e-4 \
    --lora_r 32 \
    --lora_alpha 64 \
    --neftune_alpha 3.0 \
    --eval_bleu \
    --bleu_sample_size 200 \
    --eval_steps 1000 \
    --early_stopping_patience 3 \
    --max_len 256
```

**Thoi gian uoc tinh:** 4-6 gio/direction tren A100 40GB

---

## BUOC 4: RL TRAINING - GRPO (Optional - Tang chat luong)

Sau khi co SFT model, dung RL de fine-tune them:

```bash
python scripts/rl_train_grpo.py \
    --model_name Qwen/Qwen2.5-3B-Instruct \
    --sft_adapter runs/en2vi_v1/lora_en2vi_sft \
    --init_adapter runs/en2vi_v1/lora_en2vi_sft \
    --rl_src data/clean/train.en \
    --rl_tgt data/clean/train.vi \
    --run_id en2vi_v1_rl \
    --direction en2vi \
    --epochs 1 \
    --batch_size 4 \
    --grad_accum_steps 8 \
    --lr 3e-6 \
    --kl_coef 0.02 \
    --temperature 0.8 \
    --max_new_tokens 64
```

**Giai thich:**
- `--sft_adapter`: Model SFT lam reference (frozen)
- `--init_adapter`: Model khoi tao cho policy (trainable)
- `--kl_coef`: He so KL penalty (0.01-0.05)
- Reward = 0.5*BLEU + 0.3*chrF + 0.2*chrF++

---

## BUOC 5: BACK-TRANSLATION (Optional - Tang data)

Dung model VI->EN de tao them data cho EN->VI:

```bash
# 5.1 Chuan bi monolingual Vietnamese data
# (Lay tu nguon khac hoac dung chinh train.vi)

# 5.2 Back-translate VI -> EN
python scripts/back_translate.py \
    --model_name Qwen/Qwen2.5-3B-Instruct \
    --adapter_path runs/vi2en_v1/lora_vi2en_sft \
    --input data/monolingual/vi.txt \
    --output data/augment/bt.en \
    --direction vi2en \
    --batch_size 16 \
    --temperature 0.7

# 5.3 Gop data
# Windows:
type data\clean\train.en data\augment\bt.en > data\augment\train_aug.en
type data\clean\train.vi data\monolingual\vi.txt > data\augment\train_aug.vi

# Linux:
# cat data/clean/train.en data/augment/bt.en > data/augment/train_aug.en
# cat data/clean/train.vi data/monolingual/vi.txt > data/augment/train_aug.vi

# 5.4 Train lai voi augmented data
python scripts/train_qwen_lora.py \
    --direction en2vi \
    --src data/augment/train_aug.en \
    --tgt data/augment/train_aug.vi \
    --val_src data/clean/dev.en \
    --val_tgt data/clean/dev.vi \
    --run_id en2vi_v2_aug \
    ...
```

---

## BUOC 6: GENERATE TRANSLATIONS

### 6A. Generate EN->VI

```bash
python scripts/generate.py \
    --model_name Qwen/Qwen2.5-3B-Instruct \
    --adapter_path runs/en2vi_v1/lora_en2vi_sft \
    --direction en2vi \
    --input data/clean/test.en \
    --output outputs/test_en2vi.hyp.vi \
    --batch_size 16 \
    --num_beams 4 \
    --repetition_penalty 1.1
```

### 6B. Generate VI->EN

```bash
python scripts/generate.py \
    --model_name Qwen/Qwen2.5-3B-Instruct \
    --adapter_path runs/vi2en_v1/lora_vi2en_sft \
    --direction vi2en \
    --input data/clean/test.vi \
    --output outputs/test_vi2en.hyp.en \
    --batch_size 16 \
    --num_beams 4 \
    --repetition_penalty 1.1
```

**Tham so generate:**
- `--num_beams 4`: Beam search (chat luong cao hon greedy)
- `--repetition_penalty 1.1`: Tranh lap tu
- `--do_sample`: Dung sampling thay vi beam search (da dang hon)

---

## BUOC 7: EVALUATION

### 7A. Basic Evaluation (BLEU, chrF, TER, METEOR)

```bash
# EN->VI
python scripts/eval_bleu.py \
    --hyp outputs/test_en2vi.hyp.vi \
    --ref data/clean/test.vi \
    --src data/clean/test.en \
    --show_worst 10

# VI->EN
python scripts/eval_bleu.py \
    --hyp outputs/test_vi2en.hyp.en \
    --ref data/clean/test.en \
    --src data/clean/test.vi \
    --show_worst 10
```

### 7B. Gemini Evaluation (LLM-as-Judge)

```bash
# Set API key
set GEMINI_API_KEY=your-api-key-here

# Evaluate voi Gemini
python scripts/eval_bleu.py \
    --hyp outputs/test_en2vi.hyp.vi \
    --ref data/clean/test.vi \
    --src data/clean/test.en \
    --gemini \
    --gemini_samples 100 \
    --gemini_batch_size 10 \
    --direction en2vi \
    --gemini_verbose
```

### 7C. Save Results to JSON

```bash
python scripts/eval_bleu.py \
    --hyp outputs/test_en2vi.hyp.vi \
    --ref data/clean/test.vi \
    --src data/clean/test.en \
    --output outputs/eval_results_en2vi.json \
    --gemini --gemini_samples 100 --direction en2vi
```

---

## BUOC 8: PUBLIC TEST (NOP BAI THI)

Public test la bo test chinh thuc de nop bai. File nam o `data/raw/public_test.*.txt`

### 8A. Generate Public Test EN->VI

```bash
python scripts/generate.py \
    --model_name Qwen/Qwen2.5-3B-Instruct \
    --adapter_path runs/en2vi_v1/lora_en2vi_sft \
    --direction en2vi \
    --input data/raw/public_test.en.txt \
    --output outputs/public_test.hyp.vi \
    --batch_size 16 \
    --num_beams 4 \
    --repetition_penalty 1.1
```

### 8B. Generate Public Test VI->EN

```bash
python scripts/generate.py \
    --model_name Qwen/Qwen2.5-3B-Instruct \
    --adapter_path runs/vi2en_v1/lora_vi2en_sft \
    --direction vi2en \
    --input data/raw/public_test.vi.txt \
    --output outputs/public_test.hyp.en \
    --batch_size 16 \
    --num_beams 4 \
    --repetition_penalty 1.1
```

### 8C. Evaluate Public Test (neu co reference)

```bash
# EN->VI
python scripts/eval_bleu.py \
    --hyp outputs/public_test.hyp.vi \
    --ref data/raw/public_test.vi.txt \
    --src data/raw/public_test.en.txt \
    --output outputs/public_test_en2vi_results.json

# VI->EN
python scripts/eval_bleu.py \
    --hyp outputs/public_test.hyp.en \
    --ref data/raw/public_test.en.txt \
    --src data/raw/public_test.vi.txt \
    --output outputs/public_test_vi2en_results.json
```

### 8D. Neu dung RL model

```bash
# EN->VI voi RL adapter
python scripts/generate.py \
    --model_name Qwen/Qwen2.5-3B-Instruct \
    --adapter_path runs/en2vi_v1_rl/final_model \
    --direction en2vi \
    --input data/raw/public_test.en.txt \
    --output outputs/public_test_rl.hyp.vi \
    --batch_size 16 \
    --num_beams 4
```

---

## QUICK REFERENCE - FULL PIPELINE (Copy-paste)

```bash
# ========== TRAIN EN->VI ==========
python scripts/train_qwen_lora.py \
    --direction en2vi \
    --src data/clean/train.en --tgt data/clean/train.vi \
    --val_src data/clean/dev.en --val_tgt data/clean/dev.vi \
    --run_id en2vi_v1 \
    --epochs 2 --batch_size 8 --grad_accum 4 \
    --eval_bleu --early_stopping_patience 3

# ========== TRAIN VI->EN ==========
python scripts/train_qwen_lora.py \
    --direction vi2en \
    --src data/clean/train.vi --tgt data/clean/train.en \
    --val_src data/clean/dev.vi --val_tgt data/clean/dev.en \
    --run_id vi2en_v1 \
    --epochs 2 --batch_size 8 --grad_accum 4 \
    --eval_bleu --early_stopping_patience 3

# ========== GENERATE PUBLIC TEST ==========
python scripts/generate.py \
    --model_name Qwen/Qwen2.5-3B-Instruct \
    --adapter_path runs/en2vi_v1/lora_en2vi_sft \
    --direction en2vi \
    --input data/raw/public_test.en.txt \
    --output outputs/public_test.hyp.vi \
    --num_beams 4

python scripts/generate.py \
    --model_name Qwen/Qwen2.5-3B-Instruct \
    --adapter_path runs/vi2en_v1/lora_vi2en_sft \
    --direction vi2en \
    --input data/raw/public_test.vi.txt \
    --output outputs/public_test.hyp.en \
    --num_beams 4

# ========== EVALUATE ==========
python scripts/eval_bleu.py \
    --hyp outputs/public_test.hyp.vi \
    --ref data/raw/public_test.vi.txt \
    --src data/raw/public_test.en.txt
```

---

## TROUBLESHOOTING

| Van de | Giai phap |
|--------|-----------|
| OOM | `--batch_size 4 --grad_accum 8` hoac `--max_len 192` |
| Train qua lau | `--eval_steps 2000` hoac `--epochs 1` |
| BLEU thap | Tang `--epochs 3`, dung RL, hoac back-translation |
| Loss khong giam | Giam `--lr 1e-4`, tang `--lora_r 64` |
| Generate lap tu | Tang `--repetition_penalty 1.2` |

---

## OUTPUT STRUCTURE

```
vlsp-mt/
├── data/
│   ├── raw/              # Data goc
│   ├── clean/            # Sau preprocessing (train/dev/test)
│   ├── dedup/            # Sau dedup (optional)
│   └── augment/          # Back-translation data (optional)
├── runs/
│   ├── en2vi_v1/
│   │   ├── lora_en2vi_sft/   # LoRA adapter
│   │   ├── meta.json         # Training config
│   │   └── training_history.json
│   └── vi2en_v1/
│       └── ...
└── outputs/
    ├── test_en2vi.hyp.vi     # Generated translations
    └── eval_results.json     # Evaluation results
```
