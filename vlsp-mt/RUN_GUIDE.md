# VLSP Medical Translation - Hướng dẫn
Model Qwen 2.5 3B - Được train trên gg colab A100 VRAM 80, Ram 167

## Cấu trúc thư mục

```
vlsp-mt/
├── data/
│   ├── raw/                      # Data gốc từ VLSP
│   │   ├── train.en.txt          # Training data (EN)
│   │   ├── train.vi.txt          # Training data (VI)
│   │   ├── public_test.en.txt    # Public test (EN)
│   │   ├── public_test.vi.txt    # Public test (VI)
│   │   ├── test_unseen_v3.en.txt # Unseen test (EN)
│   │   └── test_unseen_v3.vi.txt # Unseen test (VI)
│   ├── clean/                    # Sau preprocessing
│   │   ├── train.{en,vi}         # Train set đã clean
│   │   ├── dev.{en,vi}           # Validation set
│   │   └── test.{en,vi}          # Test set
│   ├── dedup/                    # Sau deduplication
│   │   └── train.{en,vi}
│   ├── rl_subset/                # Subset cho RL training
│   │   └── {en,vi}.txt
│   └── medical_vocab.txt         # Từ vựng y khoa
│
├── scripts/
│   ├── preprocess_vlsp.py        # Tiền xử lý data
│   ├── dedup_minhash.py          # Loại bỏ duplicate
│   ├── train_qwen_lora.py        # SFT training với LoRA
│   ├── rl_train_grpo.py          # RL training (GRPO)
│   ├── generate.py               # Generate translations
│   ├── eval_bleu.py              # Đánh giá BLEU, chrF, Gemini
│   ├── back_translate.py         # Back-translation augmentation
│   └── file train/               # Jupyter notebooks mẫu
│       ├── en2vi__52.ipynb       # Train EN→VI (52 BLEU)
│       └── vi2en.ipynb           # Train VI→EN
│
├── runs/                         # Model checkpoints lưu ở huggingface
│  
│
├── outputs/                      # Generated translations
├── File chạy code/               # Notebooks chạy trên Colab
│   ├── train.ipynb               # Notebook train
│   └── redict.ipynb              # Notebook generate & eval
│
├── requirements.txt
├── KETQUA.md                     # Kết quả thí nghiệm
└── RUN_GUIDE.md                  # File này
```

---

## Chạy nhanh

Xem chi tiết trong 2 notebook:
- **`File chạy code/train.ipynb`** - Train model
- **`File chạy code/redict.ipynb`** - Generate & Evaluate

---

## Các lệnh chính

### 1. Train SFT

```bash
python scripts/train_qwen_lora.py \
    --direction en2vi \
    --src data/clean/train.en --tgt data/clean/train.vi \
    --val_src data/clean/dev.en --val_tgt data/clean/dev.vi \
    --run_id en2vi_v1 \
    --epochs 2 --batch_size 4 --grad_accum 4 \
    --eval_bleu --early_stopping_patience 3
```

### 2. Generate

```bash
python scripts/generate.py \
    --model_name Qwen/Qwen2.5-3B-Instruct \
    --adapter_path runs/en2vi/52/lora_en2vi_sft \
    --direction en2vi \
    --input data/raw/public_test.en.txt \
    --output outputs/public_test.hyp.vi \
    --num_beams 4
```

### 3. Evaluate

```bash
python scripts/eval_bleu.py \
    --hyp outputs/public_test.hyp.vi \
    --ref data/raw/public_test.vi.txt \
    --src data/raw/public_test.en.txt
```

---

## Troubleshooting

| Vấn đề | Giải pháp |
|--------|-----------|
| OOM | `--batch_size 2 --grad_accum 8` |
| BLEU thấp | Tăng epochs, dùng RL, BT |
| Generate lặp từ | `--repetition_penalty 1.2` |
