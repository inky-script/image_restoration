# Báo cáo thử nghiệm

**Ngày:** 11/10/2026 · **Chương trình:** `src/restore_photo.py` (GFPGAN v1.4 + Real-ESRGAN x2plus)

## Môi trường
- Mac, chạy CPU (không GPU, không MPS), Python 3.13, torch 2.14.1, torchvision 0.29.1.
- `basicsr` 1.4.2 gặp 2 lỗi trên môi trường mới:
  1. Build lỗi `KeyError: '__version__'` trên **Python 3.13**. Dùng **Python 3.12** thì cài bình thường.
  2. Import `torchvision.transforms.functional_tensor` (module đã bị gỡ). Đã xử lý bằng 3 dòng trong `src/restore_photo.py`, không cần sửa thư viện.
- **Chạy thử từ đầu:** bản sao sạch của dự án (không có model), venv Python 3.12 mới, `pip install -r requirements.txt`, rồi chạy `restore_photo.py`. Chương trình tự tải 4 file model và ra ảnh `bride` 2560×3442 trong 3 phút 53 giây (gồm thời gian tải).

## Ảnh thử (public domain, Wikimedia Commons, thu nhỏ còn ~1280px)

| Ảnh trong `data/input/` | Kích thước vào → ra | Thời gian |
|---|---|---|
| `travers.jpg`: Travers as Titania | 1280×1739 → 2560×3478 | 82 giây (chạy riêng) |
| `bride.jpg`: cô dâu, Tati Photo Studios (c.1964) | 1280×1721 → 2560×3442 | 117 giây cho cả hai (chạy song song) |
| `poe.jpg`: Edgar Allan Poe (c.1849, **bản đã phục chế sẵn**) | 1280×1733 → 2560×3466 | |

Kết quả: `data/output/*_restored.jpg`; so sánh cận mặt (trái = gốc phóng ×2, phải = kết quả): `data/output/*_compare.jpg`.

## Nhận xét (đánh giá bằng mắt, mới xem ảnh cô dâu và Travers)
- **Tốt:** nhiễu hạt giảm rõ; khuôn mặt, mắt, răng, tóc và vòng lá/hoa sắc nét hơn; ren váy vẫn giữ chi tiết.
- **Chú ý:** da mặt và phông nền mịn hơn bản gốc; chưa so sánh danh tính khuôn mặt bằng số đo.
- **Chưa đánh giá:** ảnh Poe chưa xem kỹ vì đã là bản phục chế sẵn.
- **Chưa kiểm tra được:** ảnh bị xước nặng hoặc rách, vì cả ba ảnh thử đều khá sạch; chương trình cũng không xử lý xước.

## Thử nghiệm bị loại: dò xước bằng OpenCV
Bản đầu tiên dò xước bằng top-hat + inpaint. Trên ảnh cô dâu và Travers, nó đánh dấu nhầm 1,6 đến 4,1% diện tích (ren, tóc, môi, hoa), làm mờ ren và chữ ký studio, nên đã gỡ khỏi chương trình.
