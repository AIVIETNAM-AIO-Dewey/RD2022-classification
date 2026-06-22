# Hướng Dẫn Đóng Góp & Hướng Dẫn Sử Dụng (Contributing & Usage Guide)

Chào mừng bạn đến với dự án **Image Preprocessing & Classification**. Dưới đây là mô tả chi tiết cấu trúc thư mục, hướng dẫn tải dữ liệu từ Google Drive, xử lý cắt ảnh (crop patches), huấn luyện và chạy baseline.

---

## 1. Chi Tiết Vai Trò Của Từng Thư Mục

Dưới đây là sơ đồ tổ chức thư mục của dự án:

```text
├── configs/            # Chứa các tệp cấu hình thực nghiệm (YAML)
│   └── baseline.yaml            # Cấu hình baseline (Resize + Normalize)
├── data/               # Chứa dữ liệu đầu vào (được đưa vào .gitignore)
│   ├── raw/            # Chứa dữ liệu gốc (tải và giải nén từ Drive)
│   └── processed/      # Chứa các patch ảnh cắt từ bounding box chia theo train/val/test
├── notebooks/          # Jupyter Notebooks phục vụ EDA và thử nghiệm nhanh
├── outputs/            # Kết quả đầu ra sau khi chạy huấn luyện
│   └── baseline/       # Checkpoint và confusion matrix của baseline
├── src/                # Mã nguồn chính của dự án
│   ├── data/
│   │   ├── download_data.py    # Tải và giải nén dữ liệu từ Google Drive
│   │   └── process_patches.py  # Cắt ảnh hư hại từ file XML & phân chia dữ liệu
│   ├── preprocessing/
│   │   └── transforms.py       # Pipeline tiền xử lý ảnh (Đã cấu hình baseline, chứa TODOs thực nghiệm)
│   ├── models/
│   │   └── baseline_cnn.py     # Định nghĩa kiến trúc ResNet18
│   ├── training/
│   │   ├── dataset.py          # PyTorch Dataset load dữ liệu từ data/processed/
│   │   └── train.py            # Vòng lặp huấn luyện mô hình
│   └── evaluation/
│       └── evaluate.py         # Đánh giá mô hình trên tập test & vẽ confusion matrix
├── .gitignore          # Cấu hình bỏ qua data/, outputs/, venv/
├── README.md           # Giới thiệu tổng quan về dự án
└── CONTRIBUTING.md     # Hướng dẫn chi tiết dành cho nhà phát triển (tệp này)
```

---

## 2. Hướng Dẫn Sử Dụng & Luồng Phát Triển (Usage Workflow)

### Bước 1: Thiết lập môi trường ảo và cài đặt thư viện
Khởi tạo môi trường ảo Python và cài đặt các dependencies cần thiết từ `requirements.txt`:
```bash
# Tạo môi trường ảo
python -m venv venv

# Kích hoạt môi trường ảo (Windows PowerShell)
.\venv\Scripts\Activate.ps1
# Hoặc Windows CMD:
.\venv\Scripts\activate.bat
# Hoặc Linux/macOS:
source venv/bin/activate

# Cài đặt thư viện
pip install -r requirements.txt
```

### Bước 2: Tải và Giải nén dữ liệu tự động
Chạy script để tải bộ dữ liệu RDD2022 India từ Google Drive của nhóm về thư mục `data/raw/`:
```bash
python src/data/download_data.py
```
*Lưu ý: Bộ dữ liệu sẽ tự động được tải và giải nén vào `data/raw/India/`.*

### Bước 3: Cắt ảnh (Crop Patches) theo Bounding Box
Bộ dữ liệu gốc của RDD2022 ở định dạng Object Detection (ảnh kèm file XML chứa bounding box). Chúng ta sẽ tiến hành chạy script để:
1. Parse toàn bộ file XML.
2. Cắt các vùng hư hại thành các patch ảnh kích thước nhỏ cho 3 lớp: `D00` (Nứt dọc), `D20` (Nứt lưới), `D40` (Ổ gà).
3. Trích xuất ngẫu nhiên các vùng đường không có hư hại làm lớp `Normal`.
4. Chia tập dữ liệu thành `train`, `val`, `test` ở cấp độ ảnh gốc để tránh rò rỉ dữ liệu (data leakage).

Chạy lệnh sau:
```bash
python src/data/process_patches.py
```
*Sau khi chạy, dữ liệu đã xử lý sẽ nằm trong `data/processed/` với các thư mục con tương ứng.*

### Bước 4: Chạy huấn luyện Baseline
Chạy huấn luyện baseline (chỉ dùng Resize + Normalize):
```bash
python src/training/train.py --config configs/baseline.yaml
```
*Mô hình tốt nhất và lịch sử huấn luyện sẽ được lưu tại thư mục `outputs/baseline/`.*

### Bước 5: Đánh giá mô hình Baseline
Sau khi kết thúc huấn luyện, chạy script đánh giá trên tập kiểm thử (test set):
```bash
python src/evaluation/evaluate.py --config configs/baseline.yaml --model_path outputs/baseline/best_model.pth
```

---

## 3. Thêm các kỹ thuật Tiền xử lý (Thử nghiệm)

Để so sánh ảnh hưởng của **CLAHE**, **Resize Letterbox** hoặc **Grayscale + Bilateral Filter** với Baseline:

1. **Implement hàm preprocessing**: Mở tệp `src/preprocessing/transforms.py` và hoàn thiện các hàm OpenCV (`apply_clahe`, `apply_letterbox_resize`, `apply_grayscale_bilateral`) bằng cách uncomment code gợi ý.
2. **Chạy từng pipeline**: Mỗi pipeline đã có sẵn config riêng:
   - Pipeline 1 (Letterbox): `python src/training/train.py --config configs/pipeline1_letterbox.yaml`
   - Pipeline 2 (CLAHE): `python src/training/train.py --config configs/pipeline2_clahe.yaml`
   - Pipeline 3 (Grayscale+Bilateral): `python src/training/train.py --config configs/pipeline3_grayscale_bilateral.yaml`
   - Pipeline 4 (Kết hợp cả 3): `python src/training/train.py --config configs/pipeline4_combined.yaml`
3. **Đánh giá**: Chạy evaluate tương tự, ví dụ: `python src/evaluation/evaluate.py --config configs/pipeline2_clahe.yaml --model_path outputs/pipeline2_clahe/best_model.pth`

