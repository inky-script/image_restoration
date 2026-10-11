# Đồ án: Phục chế ảnh cũ tự động bằng học sâu

## 1. Vấn đề

Ảnh cũ thường bị nhiễu hạt, mờ, mất chi tiết khuôn mặt và độ phân giải thấp. Phục chế thủ công bằng Photoshop mất nhiều thời gian và cần kỹ năng. Đồ án xây dựng một chương trình Python nhận một ảnh cũ và tự động trả ra ảnh rõ hơn, không cần thao tác thủ công.

## 2. Giải pháp

Dùng hai model học sâu có sẵn (không tự huấn luyện):

| Model | Việc làm | Vì sao chọn |
|---|---|---|
| **GFPGAN v1.4** | Dò khuôn mặt, khôi phục chi tiết mặt | Chuyên cho khuôn mặt, dùng đơn giản |
| **Real-ESRGAN x2plus** | Phóng to ×2 và làm rõ phần nền | Phổ biến, ổn định, GFPGAN tích hợp sẵn |

```
Ảnh cũ → [GFPGAN dò mặt + làm rõ mặt] ┐
      → [Real-ESRGAN phóng to nền ×2] ┴→ ghép lại → Ảnh phục chế
```

Toàn bộ nằm trong một file `src/restore_photo.py` (khoảng 55 dòng), chạy trên máy cá nhân, không gửi ảnh ra ngoài.

## 3. Cách thực hiện

1. Cài thư viện (`pip install -r requirements.txt`, Python 3.12). Model (khoảng 600 MB) tự tải ở lần chạy đầu.
2. Chạy từ thư mục gốc: `python src/restore_photo.py anh_cu.jpg anh_moi.jpg`.
3. Thử trên 3 ảnh public domain (Wikimedia Commons), so sánh trước/sau bằng mắt.

## 4. Kết quả thử nghiệm

Chạy thật trên CPU (không GPU), ảnh 1280×1740 phóng thành 2560×3480:

| Ảnh | Thời gian |
|---|---|
| Travers (chạy riêng) | 82 giây |
| Cô dâu và Poe (chạy song song) | 117 giây cho cả hai |

Quan sát bằng mắt: nhiễu hạt giảm rõ, khuôn mặt, tóc và vòng hoa sắc nét hơn, chi tiết ren vẫn được giữ. Chi tiết xem ở `docs/test_report.md` và thư mục `data/output/`.

## 5. Hạn chế

- **Không xóa vết xước.** Mình đã thử dò xước bằng OpenCV nhưng nó bắt nhầm cả ren, tóc và chữ ký trên ảnh thật, nên đã bỏ. Xóa xước là hướng mở rộng (ví dụ cho người dùng vẽ mask rồi inpaint).
- **Không tô màu.**
- **Có thể làm mịn da quá mức** và khuôn mặt sau phục chế có thể khác nhẹ so với thật, vì model "vẽ lại" chi tiết. Chưa có phép đo định lượng, mới chỉ đánh giá bằng mắt trên 3 ảnh.
- **Ảnh thử đã khá sạch** (không có xước nặng), nên chưa kiểm tra được ảnh hư hại nặng.
- **Chậm trên CPU** (vài chục giây đến hai phút mỗi ảnh). Có GPU sẽ nhanh hơn nhiều (chưa đo).
- **Giấy phép:** GFPGAN và Real-ESRGAN dùng giấy phép mở, nhưng nên kiểm tra lại nếu dùng thương mại.

## 6. Hướng phát triển

Xóa xước bằng mask do người dùng vẽ; tô màu ảnh đen trắng (ví dụ DDColor); giao diện web đơn giản để tải ảnh lên; đo định lượng chất lượng trên ảnh có ground truth.

## 7. Tài liệu tham khảo

- GFPGAN: <https://github.com/TencentARC/GFPGAN>
- Real-ESRGAN: <https://github.com/xinntao/Real-ESRGAN>
- Bringing Old Photos Back to Life (tham khảo hướng xử lý xước): <https://github.com/microsoft/Bringing-Old-Photos-Back-to-Life>
