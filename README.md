# Phục chế ảnh cũ tự động bằng học sâu

Đồ án môn Xử lý ảnh số, Trường Đại học Công Thương TP. Hồ Chí Minh. GVHD: ThS. Đỗ Trung Dũng.

Ứng dụng web nhận ảnh cũ, tiền xử lý bằng OpenCV, làm rõ khuôn mặt bằng **GFPGAN**, phóng to ×2/×4 và làm nét nền bằng **Real-ESRGAN**, rồi đánh giá chất lượng bằng **PSNR / SSIM**. Không xóa vết xước, không tô màu.

## Phân công

| Thành viên | Vai trò | Mã nguồn |
|---|---|---|
| Nguyễn Thu Thảo (nhóm trưởng) | Dữ liệu & tiền xử lý | `src/build_test_set.py`, `src/preprocessing.py`, `test_images/` |
| Nguyễn Phan Nhật Huy | Lõi thuật toán & mô hình AI | `src/model_core.py` |
| Nguyễn Thị Kiều Trang | Giao diện & ứng dụng | `src/app.py` |
| Đỗ Thị Hồng Quyên | Kiểm thử, đánh giá & báo cáo | `src/evaluate.py`, `docs/` |

## Cấu trúc thư mục

```
├── src/
│   ├── build_test_set.py    (TV1) tạo test_images/ từ data/raw_archive/
│   ├── preprocessing.py     (TV1) chuẩn hóa màu, cân bằng sáng (gamma/CLAHE), resize + padding
│   ├── model_core.py        (TV2) GFPGAN + Real-ESRGAN, quản lý bộ nhớ, xử lý lỗi
│   ├── app.py               (TV3) giao diện Streamlit
│   ├── evaluate.py          (TV4) PSNR / SSIM trên tập test
│   └── restore_photo.py     giữ lại cho lệnh cũ, gọi sang model_core.py
├── data/
│   ├── raw_archive/         archive.zip đã giải nén: 146 ảnh sạch, 54 ảnh hỏng, file phân loại CSV
│   ├── input/, output/      3 ảnh Wikimedia Commons và kết quả của giai đoạn đầu
├── test_images/             bộ dữ liệu chuẩn dùng chung
│   ├── real_damaged/        54 ảnh hỏng thật
│   ├── gt/                  40 ảnh gốc (cạnh dài 512 px)
│   ├── degraded/            40 ảnh gốc bị làm hỏng nhân tạo (mờ + thu nhỏ ×2 + nhiễu + JPEG)
│   └── manifest.csv
├── results/                 kết quả evaluate.py: metrics.csv, summary.csv, real_times.csv, ảnh
├── docs/                    báo cáo Word, hình vẽ (docs/figures/), đề xuất, báo cáo thử nghiệm cũ
└── .streamlit/config.toml   theme của giao diện
```

## Cài đặt

Cần Python 3.10 hoặc 3.12 (3.13 lỗi khi cài `basicsr`). Chạy từ thư mục gốc của repo:

```bash
python -m venv .venv
```

Windows: `.venv\Scripts\activate` · macOS/Linux: `source .venv/bin/activate`

```bash
pip install -r requirements.txt
```

`requirements.txt` cài PyTorch bản CPU và cài `basicsr` từ wheel có sẵn trong `wheels/` (bản trên PyPI phải build và tải thêm cả PyTorch, rất lâu). Máy có GPU NVIDIA thì cài trước torch bản CUDA theo hướng dẫn tại pytorch.org.

## Triển khai lên Streamlit Community Cloud

- Main file: `src/app.py`, nhánh `main`.
- Advanced settings → Python 3.10 hoặc 3.11.
- `packages.txt` cài thêm thư viện hệ thống Linux mà OpenCV cần.
- Lần chạy đầu trên cloud sẽ tải khoảng 600 MB model. Máy chủ miễn phí chỉ có CPU và RAM hạn chế, nên mỗi ảnh có thể mất vài phút; nếu app bị khởi động lại khi phục chế, hãy chọn ảnh nhỏ hoặc giảm kích thước ô (tile).

Không cần tải model thủ công: lần chạy đầu tự tải khoảng 600 MB (cần Internet). Nếu tải bị ngắt giữa chừng (lỗi `unexpected EOF`), xóa file `.pth` dở dang rồi chạy lại.

## Cách chạy

Giao diện web (mở http://localhost:8501):

```bash
streamlit run src/app.py
```

Phục chế một ảnh bằng dòng lệnh (`--scale 4` để phóng ×4, `--face none` để bỏ bước làm rõ mặt, `--balance clahe|gamma|none` để chọn cách cân bằng sáng):

```bash
python src/model_core.py data/input/bride.jpg results/bride_restored.png
```

Tạo lại bộ dữ liệu `test_images/` từ `data/raw_archive/`:

```bash
python src/build_test_set.py
```

Tính PSNR / SSIM (thêm `--limit 5` để chạy thử nhanh). Trên CPU toàn bộ mất khoảng 30 đến 60 phút:

```bash
python src/evaluate.py
```

## Nguồn dữ liệu

- `data/raw_archive/`: bộ ảnh độ phân giải thấp từ Kaggle (archive.zip), 200 ảnh kèm `image_classification_results.csv`.
- `data/input/`: 3 ảnh public domain từ Wikimedia Commons, đã thu nhỏ còn khoảng 1280 px.
