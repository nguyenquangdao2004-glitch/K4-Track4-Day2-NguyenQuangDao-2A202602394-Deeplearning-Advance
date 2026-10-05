# Báo Cáo Thực Nghiệm Lab Day 2 — Phân Loại Cỏ Dại DeepWeeds

- **Họ và tên**: Nguyễn Quang Đạo
- **MSSV**: 2A202602394
- **Lớp / Track**: AI20K - Track 4 (Deep Learning Advance) - Day 2
- **Môi trường thực nghiệm**: Kaggle Notebooks (GPU NVIDIA Tesla T4 x2 / P100)

---

## 1. Liên kết Notebook Thực Nghiệm (Kaggle)

- **Kaggle Notebook Link**: [https://www.kaggle.com/code/daokevin/track4-lab2](https://www.kaggle.com/code/daokevin/track4-lab2)
- **Chế độ thực thi**: GPU T4 x2, Internet: ON, Persistence: Variables & Files.

---

## 2. Cấu Trúc Thư Mục Nộp Bài

```
submissions/2A202602394_NguyenQuangDao/
├── README.md                  # Hướng dẫn chạy lại, môi trường, thông tin sinh viên
├── results.xlsx               # Bảng tổng hợp chi tiết toàn bộ thí nghiệm (7 sheet)
├── report.md                  # Báo cáo kết quả và phân tích chi tiết (4-8 trang)
├── curves/                    # Đồ thị huấn luyện (loss & F1 theo epoch) cho từng exp_id
│   ├── B01_resnet50.png
│   ├── B02_convnext_tiny.png
│   ├── ...
├── predictions/               # File dự đoán test và val cho chung kết & mốc
│   ├── F01_seed0_test.csv
│   ├── F01_seed1_test.csv
│   ├── F01_seed2_test.csv
│   ├── T00_seed0_test.csv
│   └── ...
└── code/                      # Mã nguồn hoàn chỉnh
    ├── dataset.py
    ├── model.py
    ├── losses.py
    ├── train.py
    ├── inference.py
    ├── benchmark.py
    └── lab_day2.ipynb         # Notebook đầy đủ các bước thực nghiệm
```

---

## 3. Môi Trường & Phiên Bản Thư Viện

- **Python**: `>= 3.10`
- **PyTorch**: `>= 2.0.0` (hỗ trợ CUDA, AMP)
- **torchvision**: `>= 0.15.0`
- **timm**: `>= 0.9.0`
- **pandas**: `>= 2.0.0`
- **numpy**: `>= 1.24.0`
- **scikit-learn**: `>= 1.2.0`
- **openpyxl**: `>= 3.1.0`

Cài đặt nhanh trên môi trường mới:
```bash
pip install -q timm openpyxl scikit-learn pandas numpy matplotlib
```

---

## 4. Thứ Tự & Lệnh Chạy Thí Nghiệm

### Bước 0: Tải dữ liệu và chuẩn bị
1. Tải ảnh `images.zip` từ Zenodo và kiểm tra MD5 (`b7b30f96d466fba86016aa5a26606e0f`).
2. Tải nhãn Fold 0 từ GitHub DeepWeeds tác giả vào `data/labels/`.
3. Kiểm tra tính toàn vẹn và sanity check pipeline bằng `check_split`.

### Bước 1: Khảo sát Backbone (≥ 5 mô hình)
Chạy công thức nền `T00` trên các backbone:
- `B01`: `resnet50` (ResNet - Baseline mốc)
- `B02`: `convnext_tiny` (ConvNeXt - Kiến trúc hiện đại)
- `B03`: `resnext50_32x4d` (ResNeXt - Cardinality)
- `B04`: `swin_tiny_patch4_window7_224` (Vision Transformer - Shifted Windows)
- `B05`: `mobilenetv3_large_100` (Mạng nhẹ - Mobile/Edge)

```bash
python code/train.py --set exp_id=B01 backbone=resnet50 seed=0
python code/train.py --set exp_id=B02 backbone=convnext_tiny seed=0
python code/train.py --set exp_id=B03 backbone=resnext50_32x4d seed=0
python code/train.py --set exp_id=B04 backbone=swin_tiny_patch4_window7_224 seed=0
python code/train.py --set exp_id=B05 backbone=mobilenetv3_large_100 seed=0
```

### Bước 2: Tối ưu Công thức Huấn luyện (Ablation)
Trên backbone tốt nhất (`convnext_tiny` hoặc `resnet50`), khảo sát các trục:
- `T01`: Khởi tạo (scratch vs finetune)
- `T02`: Augmentation (basic vs RandAugment vs CutMix)
- `T03`: Loss (Cross-Entropy vs Focal Loss vs Label Smoothing)
- `T04`: Class Sampler (None vs WeightedRandomSampler)
- `T05`: EMA weights & LR Scaling

### Bước 3: Đánh giá Phương pháp Suy luận & Benchmark Độ Trễ
Đo trên tập `val` không train lại:
- `I00`: 1-view chuẩn (mốc)
- `I01`: TTA lật ngang (Horizontal Flip, K=2)
- `I02`: TTA Multi-scale / Multi-crop
- `I03`: Dò độ phân giải (FixRes: 256x256)
- `I04`: Temperature Scaling (tối ưu ECE trên val)
- `I05`: Gộp BatchNorm vào Conv (Inference Speedup)
- Đo benchmark độ trễ (warmup 10 lần, synchronize GPU, 50-100 iterations, p50/p95/p99).

### Bước 4: Vòng Chung Kết & Đánh Giá Test (≥ 3 Seed)
Huấn luyện lại cấu hình tốt nhất (`F01`) và cấu hình mốc (`T00`) với 3 seed (0, 1, 2), bật cờ `save_test_predictions=True`.
Chạy đánh giá chính thức qua `eval.py`:
```bash
# Chấm điểm chỉ số
python eval.py score --pred "predictions/F01_seed*_test.csv" \
    --test-csv data/labels/test_subset0.csv --labels data/labels/labels.csv --tag F01 --out eval_out

# Tự chấm điểm Rubric Phần I
python eval.py grade --final "predictions/F01_seed*_test.csv" \
    --baseline "predictions/T00_seed*_test.csv" \
    --uncal "predictions/F01_uncal_seed*_test.csv" \
    --final-val "predictions/F01_seed*_val.csv" \
    --latency-p95-ms 42.0 --latency-method proper \
    --test-csv data/labels/test_subset0.csv --labels data/labels/labels.csv
```
