# VLSP Medical Translation Pipeline

Complete guide to train and evaluate medical translation models (EN↔VI).

---

## 📦 1. Installation

```bash
pip install -r requirements.txt
```

**Required packages:**
- transformers
- peft
- torch
- datasets
- sacrebleu
- datasketch
- tqdm

---

## 🗂️ 2. Data Preparation

### 2.1 Deduplication (MinHash LSH)

Remove near-duplicate sentence pairs:

```bash
python scripts/dedup_minhash.py \
    --src data/raw/train.en \
    --tgt data/raw/train.vi \
    --out_dir data/dedup \
    --threshold 0.8 \
    --dedup_by both \
    --rep_strategy longest
```

**Output example:**
```
============================================================
DEDUPLICATION RESULTS
============================================================
  Original pairs:     150,000
  Kept pairs:         127,845
  Removed pairs:      22,155
  Keep ratio:         85.23%
  Number of clusters: 127,845
  Avg cluster size:   1.17
  Max cluster size:   15
  Singleton clusters: 118,234
============================================================
```

### 2.2 Preprocessing & Splitting

Clean data and split into train/dev/test:

```bash
python scripts/preprocess_vlsp.py \
    --src_in data/dedup/train.en \
    --tgt_in data/dedup/train.vi \
    --out_dir data/clean \
    --min_len 3 \
    --max_len 256 \
    --max_ratio 3.0 \
    --dev_size 1000 \
    --test_size 1000
```

**Output example:**
```
============================================================
VLSP Medical Translation Preprocessing
============================================================

Loading data from:
  Source: data/dedup/train.en
  Target: data/dedup/train.vi
Loaded 127,845 pairs

Cleaning...
After cleaning: 127,102 pairs

Filtering...
After filtering: 125,456 pairs

Deduplicating by 'both'...
After dedup: 124,890 pairs

Splitting (dev=1000, test=1000)...

Train:
  Pairs: 122,890
  Src length: min=3, max=256, avg=24.5
  Tgt length: min=3, max=248, avg=28.2

Dev:
  Pairs: 1,000
  Src length: min=4, max=198, avg=23.8
  Tgt length: min=4, max=215, avg=27.5

Test:
  Pairs: 1,000
  Src length: min=3, max=201, avg=24.1
  Tgt length: min=4, max=220, avg=27.9

============================================================
Preprocessing complete!
Output saved to: data/clean
============================================================
```

### 2.3 Create RL Subset (Optional)

```bash
# Windows
powershell -Command "Get-Content data/clean/train.en -Head 50000 | Set-Content data/rl_subset/en.txt"
powershell -Command "Get-Content data/clean/train.vi -Head 50000 | Set-Content data/rl_subset/vi.txt"

# Linux/Mac
head -n 50000 data/clean/train.en > data/rl_subset/en.txt
head -n 50000 data/clean/train.vi > data/rl_subset/vi.txt
```

---


## 🎯 3. SFT Training (LoRA)

### 3.1 Train EN→VI

```bash
python scripts/train_qwen_lora.py \
    --model_name Qwen/Qwen2.5-3B-Instruct \
    --direction en2vi \
    --src data/clean/train.en \
    --tgt data/clean/train.vi \
    --val_src data/clean/dev.en \
    --val_tgt data/clean/dev.vi \
    --run_id qwen_en2vi_v1 \
    --lora_r 32 \
    --lora_alpha 64 \
    --lr 2e-4 \
    --batch_size 4 \
    --grad_accum 4 \
    --epochs 3 \
    --neftune_alpha 5.0 \
    --label_smoothing 0.1 \
    --early_stopping_patience 3
```

**Output example:**
```
Loading tokenizer: Qwen/Qwen2.5-3B-Instruct
Loading model: Qwen/Qwen2.5-3B-Instruct
Using dtype: torch.bfloat16
Trainable params: 41,943,040 (1.35%)

Loading training data...
Training samples: 122,890
Loading validation data...
Validation samples: 1,000

Effective batch size: 16
Steps per epoch: 7,680
Total steps: 23,040

============================================================
Starting training...
============================================================

{'loss': 1.8234, 'learning_rate': 2e-05, 'epoch': 0.01}
{'loss': 1.5123, 'learning_rate': 8e-05, 'epoch': 0.05}
{'loss': 1.2456, 'learning_rate': 1.5e-04, 'epoch': 0.10}
...
{'eval_loss': 0.9823, 'epoch': 1.0}
{'eval_loss': 0.8912, 'epoch': 2.0}
{'eval_loss': 0.8756, 'epoch': 3.0}

============================================================
Training complete!
Adapter saved to: runs/qwen_en2vi_v1/lora_en2vi_sft
============================================================
```

### 3.2 Train VI→EN

```bash
python scripts/train_qwen_lora.py \
    --model_name Qwen/Qwen2.5-3B-Instruct \
    --direction vi2en \
    --src data/clean/train.vi \
    --tgt data/clean/train.en \
    --val_src data/clean/dev.vi \
    --val_tgt data/clean/dev.en \
    --run_id qwen_vi2en_v1 \
    --lora_r 32 \
    --lr 2e-4 \
    --batch_size 4 \
    --grad_accum 4 \
    --epochs 3
```

---

## 🔄 4. RL Training (GRPO)

Fine-tune with reinforcement learning using BLEU/chrF rewards:

```bash
python scripts/rl_train_grpo.py \
    --model_name Qwen/Qwen2.5-3B-Instruct \
    --sft_adapter runs/qwen_en2vi_v1/lora_en2vi_sft \
    --init_adapter runs/qwen_en2vi_v1/lora_en2vi_sft \
    --rl_src data/rl_subset/en.txt \
    --rl_tgt data/rl_subset/vi.txt \
    --run_id qwen_en2vi_v1_rl \
    --direction en2vi \
    --epochs 1 \
    --batch_size 4 \
    --grad_accum_steps 4 \
    --lr 3e-6 \
    --kl_coef 0.01 \
    --num_samples 2 \
    --temperature 0.8
```

**Output example:**
```
Using device: cuda
Loading SFT reference model...
SFT model loaded and frozen.
Loading trainable model...
Total params: 3,090,000,000
Trainable params: 41,943,040
Loaded RL data: 50,000 pairs

Epoch 1/1: 100%|██████████| 3125/3125 [2:15:30<00:00]
  loss: 0.0234, reward: 0.4523, baseline: 0.4456, lr: 3.00e-06

Epoch 1 completed. Avg reward: 0.4523
New best model saved to runs/qwen_en2vi_v1_rl/best_model

============================================================
RL training completed. Final model saved to runs/qwen_en2vi_v1_rl/final_model
============================================================
```

---


## 📝 5. Generate Translations

### 5.1 Generate with SFT Model

```bash
python scripts/generate.py \
    --model_name Qwen/Qwen2.5-3B-Instruct \
    --adapter_path runs/qwen_en2vi_v1/lora_en2vi_sft \
    --direction en2vi \
    --input data/clean/dev.en \
    --output outputs/dev.hyp.sft.vi \
    --batch_size 8 \
    --num_beams 4
```

**Output example:**
```
Loading model: Qwen/Qwen2.5-3B-Instruct
Loading adapter: runs/qwen_en2vi_v1/lora_en2vi_sft
Model loaded on cuda:0, dtype=torch.bfloat16
Loaded 1000 sentences from data/clean/dev.en

Generating: 100%|██████████| 125/125 [02:34<00:00]

Saved 1000 translations to outputs/dev.hyp.sft.vi
```

### 5.2 Generate with RL Model

```bash
python scripts/generate.py \
    --model_name Qwen/Qwen2.5-3B-Instruct \
    --adapter_path runs/qwen_en2vi_v1_rl/best_model \
    --direction en2vi \
    --input data/clean/dev.en \
    --output outputs/dev.hyp.rl.vi \
    --batch_size 8
```

### 5.3 Generate with Sampling (for diversity)

```bash
python scripts/generate.py \
    --model_name Qwen/Qwen2.5-3B-Instruct \
    --adapter_path runs/qwen_en2vi_v1/lora_en2vi_sft \
    --direction en2vi \
    --input data/clean/dev.en \
    --output outputs/dev.hyp.sample.vi \
    --do_sample \
    --temperature 0.7 \
    --top_p 0.9
```

---

## 📊 6. Evaluation

### 6.1 Basic Evaluation

```bash
python scripts/eval_bleu.py \
    --hyp outputs/dev.hyp.sft.vi \
    --ref data/clean/dev.vi
```

**Output example:**
```
Loading hypothesis: outputs/dev.hyp.sft.vi
Loading reference: data/clean/dev.vi

Evaluating 1000 sentence pairs...

==================================================
EVALUATION RESULTS
==================================================
  BLEU:    32.45
  chrF++:  58.72
  chrF:    55.31
  TER:     48.23 (lower is better)
==================================================
```

### 6.2 Detailed Evaluation with Error Analysis

```bash
python scripts/eval_bleu.py \
    --hyp outputs/dev.hyp.sft.vi \
    --ref data/clean/dev.vi \
    --src data/clean/dev.en \
    --show_worst 10 \
    --output results/eval_sft.json
```

**Output example:**
```
==================================================
EVALUATION RESULTS
==================================================
  BLEU:    32.45
  chrF++:  58.72
  chrF:    55.31
  TER:     48.23 (lower is better)
==================================================

10 WORST TRANSLATIONS (by BLEU):
--------------------------------------------------

[1] Line 234 | BLEU: 2.3 | chrF: 18.5
  SRC: The patient presented with acute myocardial infarction.
  HYP: Bệnh nhân có triệu chứng nhồi máu cơ tim.
  REF: Bệnh nhân nhập viện với chẩn đoán nhồi máu cơ tim cấp.

[2] Line 567 | BLEU: 3.1 | chrF: 21.2
  SRC: Administer 500mg of amoxicillin every 8 hours.
  HYP: Cho uống amoxicillin 500mg.
  REF: Cho bệnh nhân uống 500mg amoxicillin mỗi 8 giờ.
...

Results saved to: results/eval_sft.json
```

### 6.3 Quick Evaluation Pipeline

```bash
python scripts/run_eval_all.py \
    --run_id qwen_en2vi_v1 \
    --direction en2vi \
    --model_name Qwen/Qwen2.5-3B-Instruct
```

---


## 🚀 7. Full Pipeline (Copy-Paste Ready)

### Quick Start (Small Test Run)

```bash
# 1. Preprocess (skip dedup for speed)
python scripts/preprocess_vlsp.py \
    --src_in data/raw/train.en \
    --tgt_in data/raw/train.vi \
    --out_dir data/clean \
    --dev_size 500 \
    --test_size 500

# 2. Train SFT (small subset)
python scripts/train_qwen_lora.py \
    --model_name Qwen/Qwen2.5-3B-Instruct \
    --direction en2vi \
    --src data/clean/train.en \
    --tgt data/clean/train.vi \
    --run_id test_run \
    --subset 5000 \
    --epochs 1 \
    --batch_size 4

# 3. Generate & Evaluate
python scripts/run_eval_all.py --run_id test_run --direction en2vi
```

### Full Training Pipeline

```bash
# Step 1: Deduplication
python scripts/dedup_minhash.py \
    --src data/raw/train.en \
    --tgt data/raw/train.vi \
    --out_dir data/dedup \
    --threshold 0.8

# Step 2: Preprocessing
python scripts/preprocess_vlsp.py \
    --src_in data/dedup/train.en \
    --tgt_in data/dedup/train.vi \
    --out_dir data/clean \
    --dev_size 1000 \
    --test_size 1000

# Step 3: SFT Training
python scripts/train_qwen_lora.py \
    --model_name Qwen/Qwen2.5-3B-Instruct \
    --direction en2vi \
    --src data/clean/train.en \
    --tgt data/clean/train.vi \
    --val_src data/clean/dev.en \
    --val_tgt data/clean/dev.vi \
    --run_id qwen_en2vi_full \
    --lora_r 32 \
    --epochs 3 \
    --neftune_alpha 5.0

# Step 4: Evaluate SFT
python scripts/run_eval_all.py \
    --run_id qwen_en2vi_full \
    --direction en2vi \
    --adapter_name lora_en2vi_sft

# Step 5: RL Training (Optional)
python scripts/rl_train_grpo.py \
    --model_name Qwen/Qwen2.5-3B-Instruct \
    --sft_adapter runs/qwen_en2vi_full/lora_en2vi_sft \
    --init_adapter runs/qwen_en2vi_full/lora_en2vi_sft \
    --rl_src data/clean/train.en \
    --rl_tgt data/clean/train.vi \
    --run_id qwen_en2vi_full_rl \
    --direction en2vi \
    --epochs 1

# Step 6: Evaluate RL
python scripts/run_eval_all.py \
    --run_id qwen_en2vi_full_rl \
    --direction en2vi \
    --adapter_name best_model
```

---

## 📁 Project Structure

```
vlsp-mt/
├── data/
│   ├── raw/                    # Original data
│   │   ├── train.en
│   │   └── train.vi
│   ├── dedup/                  # After deduplication
│   │   ├── train.en
│   │   └── train.vi
│   └── clean/                  # Final preprocessed
│       ├── train.en / train.vi
│       ├── dev.en / dev.vi
│       └── test.en / test.vi
├── runs/                       # Training outputs
│   └── {run_id}/
│       ├── lora_en2vi_sft/     # SFT adapter
│       ├── best_model/         # Best RL model
│       ├── meta.json
│       └── training_history.json
├── outputs/                    # Generated translations
│   └── dev.hyp.{run_id}.vi
├── results/                    # Evaluation results
│   └── eval_{run_id}.json
└── scripts/
    ├── preprocess_vlsp.py
    ├── dedup_minhash.py
    ├── train_qwen_lora.py
    ├── rl_train_grpo.py
    ├── generate.py
    ├── eval_bleu.py
    └── run_eval_all.py
```

---

## ⚙️ Hyperparameter Recommendations

| Parameter | Small Data (<10k) | Medium (10k-100k) | Large (>100k) |
|-----------|-------------------|-------------------|---------------|
| lora_r | 16 | 32 | 64 |
| lora_alpha | 32 | 64 | 128 |
| batch_size | 4 | 4-8 | 8-16 |
| grad_accum | 2 | 4 | 4-8 |
| epochs | 5-10 | 3-5 | 2-3 |
| lr | 2e-4 | 2e-4 | 1e-4 |
| neftune_alpha | 5.0 | 5.0 | 5.0 |

---

## 🔧 Troubleshooting

### Out of Memory (OOM)
- Reduce `batch_size`
- Increase `grad_accum` to maintain effective batch size
- Use `--max_len 128` instead of 256

### Training Loss Not Decreasing
- Check data quality (run preprocessing again)
- Try lower learning rate
- Increase `lora_r` for more capacity

### Poor BLEU Score
- Train longer (more epochs)
- Use RL fine-tuning
- Check for data leakage between train/dev/test
