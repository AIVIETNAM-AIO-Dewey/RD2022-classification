# Hướng Dẫn Đóng Góp & Hướng Dẫn Sử Dụng (Contributing & Usage Guide)

Chào mừng bạn đến với dự án **Image Preprocessing & Classification**. Dưới đây là đánh giá cấu trúc thư mục hiện tại, hướng dẫn chi tiết về chức năng của từng thư mục, và quy trình sử dụng/đóng góp vào dự án Python Machine Learning này.

---

## 1. Đánh Giá Cấu Trúc Dự Án

Cấu trúc dự án hiện tại **rất tốt và chuẩn hóa** đối với một dự án Python Machine Learning & Computer Vision. Việc chia tách rõ ràng giữa cấu hình (`configs`), dữ liệu (`data`), mã nguồn (`src`), thực nghiệm nhanh (`notebooks`), và kết quả (`outputs`) giúp dự án:
- **Dễ dàng quản lý và mở rộng** khi thêm các kiến trúc mô hình hoặc phương pháp tiền xử lý mới.
- **Độc lập và tái sử dụng mã nguồn** nhờ cấu trúc module hóa trong `src`.
- **Dễ dàng tái lập kết quả thực nghiệm** (Reproducibility) thông qua việc lưu cấu hình độc lập.

---

## 2. Chi Tiết Vai Trò Của Từng Thư Mục

Dưới đây là chi tiết những gì mỗi thư mục cần chứa và quản lý:

```text
├── configs/            # Chứa các tệp cấu hình (YAML, JSON)
├── data/               # Chứa dữ liệu đầu vào (không đưa lên Git)
│   ├── raw/            # Dữ liệu gốc chưa qua xử lý (ví dụ: ảnh RDD2022 gốc)
│   ├── processed/      # Dữ liệu sau khi đã chạy qua các bước tiền xử lý
│   └── external/       # Dữ liệu bổ sung hoặc trọng số mô hình tải về từ bên ngoài
├── notebooks/          # Jupyter Notebooks phục vụ cho EDA, thử nghiệm nhanh
├── outputs/            # Kết quả đầu ra của quá trình huấn luyện và đánh giá
│   ├── models/         # Các checkpoint, tệp trọng số mô hình đã huấn luyện (.pth, .onnx)
│   ├── plots/          # Biểu đồ loss, accuracy, confusion matrix, kết quả trực quan hóa
│   └── logs/           # Nhật ký chạy (logs), TensorBoard logs
├── src/                # Mã nguồn chính của dự án (dưới dạng thư viện nội bộ)
│   ├── preprocessing/  # Các kỹ thuật tiền xử lý ảnh (Resize, Contrast, Noise Reduction, Augmentation)
│   ├── models/         # Định nghĩa các kiến trúc mạng Neural (Custom CNN, ResNet, MobileNet, v.v.)
│   ├── training/       # Vòng lặp huấn luyện (train loop), hàm tối ưu (optimizer), loss function, dataloader
│   └── evaluation/     # Các tập lệnh đánh giá mô hình, tính toán các chỉ số (Precision, Recall, F1, mAP)
├── experiments/        # Các kịch bản chạy thử nghiệm, so sánh các phiên bản tiền xử lý khác nhau
├── .gitignore          # Cấu hình bỏ qua các thư mục nặng như data/, outputs/, venv/
├── README.md           # Giới thiệu tổng quan về dự án và kết quả đạt được
└── CONTRIBUTION.md     # Hướng dẫn chi tiết dành cho nhà phát triển (tệp này)
```

---

## 3. Hướng Dẫn Sử Dụng & Luồng Phát Triển (Usage Workflow)

Để phát triển hoặc chạy dự án, hãy tuân theo quy trình tiêu chuẩn dưới đây:

### Bước 1: Thiết lập môi trường ảo
Khởi tạo và kích hoạt môi trường ảo Python để cài đặt các thư viện cần thiết:
```bash
# Tạo môi trường ảo
python -m venv venv

# Kích hoạt trên Windows (PowerShell)
.\venv\Scripts\Activate.ps1
# Hoặc trên Command Prompt
.\venv\Scripts\activate.bat

# Cài đặt các thư viện phụ thuộc (nếu có requirements.txt)
pip install -r requirements.txt
```

### Bước 2: Chuẩn bị dữ liệu
1. Tải bộ dữ liệu (ví dụ: RDD2022) và giải nén vào thư mục `data/raw/`.
2. Giữ nguyên cấu trúc thư mục gốc của bộ dữ liệu để các script đọc đúng đường dẫn.

### Bước 3: Thử nghiệm và Tiền xử lý dữ liệu
1. Sử dụng thư mục `notebooks/` để tạo các file `.ipynb` phân tích dữ liệu (Exploratory Data Analysis - EDA) và thử nghiệm các thuật toán tiền xử lý ảnh.
2. Khi các thuật toán tiền xử lý đã chạy ổn định, chuyển chúng thành các module Python sạch sẽ lưu trong `src/preprocessing/` (ví dụ: `src/preprocessing/enhancement.py`).

### Bước 4: Định nghĩa cấu hình (Configuration)
Trước khi chạy huấn luyện, tạo một tệp cấu hình trong thư mục `configs/` (ví dụ: `configs/baseline_config.yaml`) chứa các thông số:
- Đường dẫn dữ liệu đầu vào/đầu ra.
- Kích thước ảnh, phương pháp tiền xử lý áp dụng.
- Hyperparameters: `batch_size`, `learning_rate`, `epochs`, `optimizer`, v.v.

### Bước 5: Định nghĩa và Huấn luyện mô hình
1. Định nghĩa cấu trúc mạng Neural trong `src/models/` (ví dụ: `src/models/cnn_classifier.py`).
2. Viết mã nguồn huấn luyện tại `src/training/` và thực hiện huấn luyện bằng cách trỏ tới tệp cấu hình:
   ```bash
   python src/training/train.py --config configs/baseline_config.yaml
   ```
3. Trọng số mô hình sau khi huấn luyện sẽ tự động được lưu vào `outputs/models/`.

### Bước 6: Đánh giá mô hình
1. Chạy các lệnh kiểm thử và đánh giá mô hình trên tập kiểm thử (test set) bằng mã nguồn trong `src/evaluation/`:
   ```bash
   python src/evaluation/evaluate.py --model_path outputs/models/best_model.pth --config configs/baseline_config.yaml
   ```
2. Lưu các biểu đồ kết quả trực quan (confusion matrix, PR curve) vào `outputs/plots/`.

---

## 4. Quy Định Đóng Góp Mã Nguồn (Contributing Guidelines)

Nếu bạn muốn đóng góp mã nguồn hoặc phát triển tính năng mới:
1. **Tuân thủ chuẩn Code Style**: Viết mã nguồn tuân thủ tiêu chuẩn [PEP 8](https://peps.python.org/pep-0008/). Sử dụng các công cụ format như `black` hoặc `flake8` trước khi commit.
2. **Không commit dữ liệu và outputs**: Đảm bảo thư mục `data/` và `outputs/` luôn nằm trong `.gitignore` để tránh việc đẩy các file dung lượng lớn lên GitHub.
3. **Tạo nhánh (Branch) mới**: Luôn tạo nhánh mới từ nhánh `main` khi phát triển tính năng: `git checkout -b feature/ten-tinh-nang`.
4. **Viết Docstring và Chú thích**: Mỗi hàm, class mới cần có docstring mô tả chi tiết chức năng, tham số đầu vào và kiểu dữ liệu trả về.
