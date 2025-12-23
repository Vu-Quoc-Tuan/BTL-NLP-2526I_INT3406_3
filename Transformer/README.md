# Professional Transformer (Xây dựng Model Dịch máy từ A-Z)

Dự án triển khai mô hình Transformer từ con số 0 phục vụ bài toán Dịch máy.

## 1. Cấu trúc Dự án
```
ProfessionalTransformer/
├── configs/            # Chứa file cấu hình (YAML) - Dễ dàng chỉnh sửa mà không cần sửa code
├── data/               # Nơi chứa dữ liệu (Raw, Clean, Cache)
├── src/                # Mã nguồn chính (Core, Data, Models, Training)
├── scripts/            # Các script chạy lệnh (Train, Clean, Test)
├── notebooks/          # Notebook chạy trên Colab/Kaggle
└── checkpoints/        # Nơi lưu model đã train
```

## 2. Hướng dẫn chạy

### 2.1. Chạy trên Colab / Kaggle
Copy nội dung 2 file notebook sau và upload lên Colab hoặc tạo Kernel mới trên Kaggle:
- **Colab:** `colab_train.ipynb`
- **Kaggle:** `kaggle_train.ipynb`

Cả 2 notebook đều đã được thiết kế sẵn quy trình chuẩn: Cài đặt -> Clean -> Build Tokenizer -> Train -> Test.

### 2.2. Chạy Local (Máy cá nhân)
Nếu chạy trên máy local, bạn có thể sử dụng trực tiếp file `colab_train.ipynb`.
> **Lưu ý:** Hãy bỏ qua bước mount Google Drive (truy cập Drive) trong notebook.

## 3. Cấu hình tham số
Bạn có thể tùy chỉnh các tham số model và training theo 2 cách:
- **Trực tiếp trong Notebook:** Sửa các biến tại cell cấu hình (nếu train bằng notebook).
- **Thông qua file Config:** Sửa file `configs/default.yaml`.

## 4. Hướng dẫn chạy dự đoán (Inference)
Để chạy dự đoán với input tùy chỉnh và xuất ra kết quả `prediction.txt`:

1. **Chuẩn bị file:**
   - Upload file input (ví dụ `test.en`) vào thư mục `/data`.
   - Upload file model (`.pt`) vào thư mục `/checkpoints`.
   - **Quan trọng:** Upload file `tokenizer.json` vào thư mục gốc.

2. **Thực hiện:**
   - Mở notebook (`colab_train.ipynb`) và đi đến phần tiêu đề cuối cùng (Generate Submission / Inference).
   - Sửa đường dẫn của `input_file`, `output_file` và `checkpoint_path` cho phù hợp.
   - Chạy cell đó để thực hiện dịch và xuất kết quả.

## 5. Kết quả thực nghiệm (Benchmark)
### Dataset: IWSLT15
**Cấu hình chung:** `max_len = 160`, `vocab_size = 32768`.

#### 5.1. Mô hình có Weight Tying
**Cấu hình:** `train_dropout = 0.2`

| Epoch | Greedy BLEU | Beam5 BLEU | Beam5 – Greedy |
| :---: | :---: | :---: | :---: |
| 15 | 25.99 | 26.71 | +0.72 |
| 16 | 25.86 | 26.92 | +1.06 |
| 17 | 26.26 | 27.14 | +0.88 |
| 18 | 26.32 | 27.05 | +0.73 |
| 19 | 26.33 | 27.17 | +0.84 |
| 20 | 26.29 | 27.15 | +0.86 |

#### 5.2. Cải tiến: GELU + BPE Dropout
**Cấu hình:** `train_dropout = 0.1`, `bpe_dropout = 0.1`, Activation: `GELU`

| Epoch | Greedy BLEU | Beam5 BLEU | Beam5 – Greedy |
| :---: | :---: | :---: | :---: |
| 15 | 16.95 | 17.02 | +0.07 |
| 20 | **27.09** | **27.95** | **+0.86** |

### Dataset: VLSP (Cải tiến - GELU + BPE Dropout)
**Cấu hình:** `max_len = 192`, `vocab_size = 50000`, `bpe_dropout = 0.1`, `train_dropout = 0.1`

#### Public Test
| Epoch | Greedy BLEU | Beam5 BLEU | Beam5 – Greedy |
| :---: | :---: | :---: | :---: |
| 1 | 20.70 | 21.94 | +1.24 |
| 5 | 40.66 | 41.84 | +1.18 |
| 10 | 43.83 | 44.72 | +0.89 |
| 17 | **46.44** | **47.28** | **+0.84** |

#### Unseen Test (Public nhưng loại bỏ câu trùng trong train)
| Epoch | Greedy BLEU | Beam5 BLEU | Beam5 – Greedy |
| :---: | :---: | :---: | :---: |
| 1 | 21.64 | 23.11 | +1.47 |
| 5 | 39.38 | 40.69 | +1.31 |
| 10 | 41.61 | 42.57 | +0.96 |
| 15 | 42.84 | 43.75 | +0.91 |
| 17 | **43.27** | **44.14** | **+0.87** |
