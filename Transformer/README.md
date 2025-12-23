# Professional Transformer (Xây dựng Model Dịch máy từ A-Z)

Dự án triển khai mô hình Transformer chuẩn chuyên nghiệp ("Production-Ready") từ con số 0 (From Scratch) phục vụ bài toán Dịch máy (Machine Translation).

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

## 2. Các Script Quan Trọng

### 2.1. Làm sạch Dữ liệu (`clean_data.py`)
Chuyển đổi dữ liệu thô (raw) sang dạng chuẩn (đã chuẩn hóa Unicode, chữ thường).
```bash
python scripts/clean_data.py
```

### 2.2. Huấn luyện Tokenizer (`train_tokenizer.py`)
Học bộ từ vựng từ dữ liệu sạch và tạo file `tokenizer.json`.
```bash
python scripts/train_tokenizer.py --data_dir data/clean
```

### 2.3. Huấn luyện Model (`train.py`)
Bắt đầu quá trình training model.
```bash
python scripts/train.py
```

### 2.4. Kiểm tra Model (`test_model.py`)
Load một checkpoint bất kỳ và dịch thử ngẫu nhiên trên tập Validation và Test.
```bash
python scripts/test_model.py --checkpoint checkpoints/checkpoint_epoch_20.pt
```

## 3. Cách chạy trên Colab / Kaggle
Copy nội dung 2 file notebook sau và upload lên Colab hoặc tạo Kernel mới trên Kaggle:
- **Colab:** `colab_train.ipynb`
- **Kaggle:** `kaggle_train.ipynb`

Cả 2 notebook đều đã được thiết kế sẵn quy trình chuẩn: Cài đặt -> Clean -> Build Tokenizer -> Train -> Test.
