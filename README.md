# Phục chế ảnh cũ tự động

Chương trình Python nhận một ảnh cũ và trả ra ảnh rõ hơn: làm rõ khuôn mặt bằng **GFPGAN** và phóng to nền ×2 bằng **Real-ESRGAN**. Không xóa vết xước, không tô màu (xem `docs/proposal.md`, mục Hạn chế).

## Cấu trúc thư mục

```
photo_restoration/
├── README.md              Hướng dẫn này
├── requirements.txt       Danh sách thư viện Python cần cài
├── .gitignore             Loại gfpgan/ khỏi git
├── src/
│   └── restore_photo.py   Chương trình chính (đọc ảnh, phục chế, ghi ảnh)
├── docs/
│   ├── proposal.md        Đề xuất đồ án (bản nguồn)
│   ├── proposal.pdf       Đề xuất đồ án (bản PDF, sinh từ proposal.md)
│   └── test_report.md     Báo cáo kết quả chạy thử
├── data/
│   ├── input/             Ảnh cũ dùng để thử
│   │   ├── bride.jpg      Ảnh cô dâu, Tati Photo Studios (khoảng 1964)
│   │   ├── travers.jpg    Ảnh P.L. Travers as Titania
│   │   └── poe.jpg        Ảnh Edgar Allan Poe (khoảng 1849), bản đã phục chế sẵn
│   └── output/            Kết quả chương trình
│       ├── *_restored.jpg Ảnh sau phục chế (kích thước gấp đôi ảnh gốc)
│       └── *_compare.jpg  Ảnh so sánh cận mặt: trái = gốc phóng ×2, phải = kết quả
└── gfpgan/                (tự sinh khi chạy, không nộp) model dò mặt tải về, khoảng 200 MB
```

Ba ảnh trong `data/input/` là ảnh public domain lấy từ Wikimedia Commons, đã thu nhỏ còn khoảng 1280px.

## Cài đặt

Cần Python 3.9 trở lên. Nên dùng môi trường ảo.

```bash
pip install -r requirements.txt
```

Không cần tải model thủ công: lần chạy đầu chương trình tự tải khoảng 600 MB (cần Internet). Model lưu trong thư mục của thư viện (`site-packages`) và trong `gfpgan/weights/`, các lần sau dùng lại. Nếu quá trình tải bị ngắt giữa chừng, xóa file dở dang rồi chạy lại (lỗi thường là `unexpected EOF`).

**Lưu ý phiên bản Python:** dùng Python 3.12 (đã chạy thử). Trên Python 3.13, `pip install basicsr` lỗi `KeyError: '__version__'` khi build.

## Cách chạy

Chạy từ thư mục `photo_restoration/` (không chạy từ trong `src/`), vì model dò mặt được lưu vào `gfpgan/` theo thư mục hiện tại.

```bash
python src/restore_photo.py data/input/bride.jpg data/output/bride_restored.jpg
```

Lần chạy đầu sẽ tải model nên lâu hơn. Trên CPU mỗi ảnh mất khoảng 1 đến 2 phút.

## Kết quả thử

Xem `docs/test_report.md` (thời gian chạy, nhận xét) và các ảnh `*_compare.jpg` trong `data/output/`.
