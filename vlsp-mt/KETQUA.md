# Kết quả thực nghiệm

Phần này trình bày kết quả thực nghiệm của mô hình dịch máy y sinh sử dụng **Qwen 2.5 + LoRA + Supervised Fine-Tuning (SFT) + Reinforcement Learning (GRPO)**.  
Mô hình được đánh giá trên hai hướng dịch **Anh → Việt** và **Việt → Anh**, với ba tập dữ liệu: **validation**, **public test** và **unseen test**.
---
## Anh → Việt (EN → VI)

### Tập validation

| Model | BLEU | chrF++ | TER ↓ |
|------|------|--------|-------|
| SFT + RL (GRPO) | 52.80 | 66.13 | 40.87 |

| Model | METEOR | Gemini |
|------|--------|--------|
| SFT + RL (GRPO) | 74.84 | 3.7 |

**Nhận xét**

- Mô hình đạt điểm BLEU và METEOR cao, cho thấy khả năng dịch chính xác và bảo toàn ngữ nghĩa tốt.
- Giá trị TER thấp cho thấy số lượng chỉnh sửa cần thiết để đạt bản dịch tham chiếu là nhỏ.
- Kết quả cho thấy mô hình đã hội tụ tốt trong quá trình huấn luyện.

---

### Tập public test

| Model | BLEU | chrF++ | TER ↓ |
|------|------|--------|-------|
| SFT + RL (GRPO) | 54.19 | 67.29 | 39.22 |

| Model | METEOR | Gemini |
|------|--------|--------|
| SFT + RL (GRPO) | 75.85 | 4.0 |

**Nhận xét**

- Kết quả trên tập public test tiếp tục được cải thiện so với tập validation.
- Chỉ số chrF++ và METEOR tăng cho thấy mô hình xử lý tốt hơn các biến thể hình thái và mức độ trôi chảy của câu dịch.
- Điểm Gemini cao hơn phản ánh chất lượng dịch được đánh giá tốt hơn theo tiêu chí gần với con người.

---

### Tập unseen test

Tập unseen bao gồm các câu chưa từng xuất hiện trong cả tập validation và public test.

| Model | BLEU | chrF++ | TER ↓ |
|------|------|--------|-------|
| SFT + RL (GRPO) | 51.08 | 64.93 | 41.72 |

| Model | METEOR | Gemini |
|------|--------|--------|
| SFT + RL (GRPO) | 74.72 | 3.4 |

**Nhận xét**

- Mặc dù điểm số giảm nhẹ so với các tập đã biết, hiệu năng vẫn duy trì ở mức cao.
- Điều này cho thấy mô hình có **khả năng tổng quát hóa tốt**, hạn chế hiện tượng overfitting.

---

## Việt → Anh (VI → EN)

### Tập validation

| Model | BLEU | chrF++ | TER ↓ |
|------|------|--------|-------|
| SFT + RL (GRPO) | 43.91 | 64.16 | 48.84 |

| Model | METEOR | Gemini |
|------|--------|--------|
| SFT + RL (GRPO) | 69.94 | 3.2 |

**Nhận xét**

- Hiệu năng thấp hơn so với hướng Anh → Việt do sự khác biệt về cú pháp và cách biểu đạt.
- Tuy vậy, mô hình vẫn giữ được mức độ tương đồng ngữ nghĩa tương đối ổn định.

---

### Tập public test

| Model | BLEU | chrF++ | TER ↓ |
|------|------|--------|-------|
| SFT + RL (GRPO) | 42.88 | 63.30 | 51.68 |

| Model | METEOR | Gemini |
|------|--------|--------|
| SFT + RL (GRPO) | 69.35 | 3.3 |

**Nhận xét**

- Đây là hướng dịch khó hơn, đặc biệt trong miền ngôn ngữ y sinh chuyên ngành.

---

## Tổng kết

- Hướng dịch **Anh → Việt** đạt kết quả tốt và ổn định trên cả ba tập dữ liệu.
- Hướng **Việt → Anh** khó hơn và có sự giảm nhẹ ở các thước đo.
- Việc kết hợp **SFT và Reinforcement Learning (GRPO)** giúp cải thiện đáng kể chất lượng dịch, đồng thời tăng khả năng tổng quát hóa trên dữ liệu chưa từng thấy.
