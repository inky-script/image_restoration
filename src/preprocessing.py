"""Tiền xử lý ảnh đầu vào bằng OpenCV (phần việc Thành viên 1).

Các bước, theo đúng thứ tự trong preprocess():
    1. Chuẩn hóa không gian màu: ảnh xám / ảnh có kênh alpha -> BGR 3 kênh, uint8.
    2. Cân bằng sáng tự động: sửa gamma nếu ảnh quá tối/quá sáng, CLAHE nếu tương phản thấp.
    3. Chuẩn hóa kích thước: thu nhỏ nếu cạnh dài vượt ngưỡng, rồi padding cho
       chiều cao/rộng chia hết cho một số (mô hình chạy ổn định hơn với kích thước chẵn).

Sau khi phục chế, gọi unpad() để cắt bỏ phần padding (đã nhân theo tỷ lệ phóng to).

Ví dụ:
    img, info = preprocess(cv2.imread("anh.jpg", cv2.IMREAD_UNCHANGED))
    out = model(img)                     # ảnh đã phóng to x scale
    out = unpad(out, info["pad"], scale)
"""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class PreprocessConfig:
    balance: str = "auto"  # "auto" | "clahe" | "gamma" | "none"
    max_side: int = 2000  # 0 = không giới hạn
    pad_multiple: int = 8  # 0/1 = không padding
    clahe_clip: float = 2.0


# ---------------------------------------------------------------------------
# 1. Không gian màu
# ---------------------------------------------------------------------------
def to_bgr3(img):
    """Đưa mọi kiểu ảnh OpenCV đọc được về BGR 3 kênh, uint8."""
    if img is None or img.size == 0:
        raise ValueError("Ảnh rỗng hoặc không đọc được.")
    if img.dtype == np.uint16:  # một số ảnh scan PNG 16 bit
        img = (img / 257).astype(np.uint8)
    elif img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        # ghép lên nền trắng thay vì bỏ alpha (vùng trong suốt không thành đen)
        alpha = img[..., 3:4].astype(np.float32) / 255
        rgb = img[..., :3].astype(np.float32)
        return (rgb * alpha + 255 * (1 - alpha)).astype(np.uint8)
    return img


def is_grayscale(img, tol=3.0):
    """Ảnh 3 kênh nhưng thực chất là ảnh đen trắng (3 kênh gần bằng nhau)."""
    b, g, r = (img[..., i].astype(np.int16) for i in range(3))
    return float(np.abs(b - g).mean()) < tol and float(np.abs(g - r).mean()) < tol


def bgr_to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def rgb_to_bgr(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


# ---------------------------------------------------------------------------
# 2. Cân bằng sáng
# ---------------------------------------------------------------------------
def brightness_stats(img):
    """Độ sáng trung bình và độ tương phản (độ lệch chuẩn) của kênh L, thang 0-1."""
    L = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32) / 255
    return float(L.mean()), float(L.std())


def gamma_correct(img, target_mean=0.5):
    """Chọn gamma sao cho độ sáng trung bình kênh L về gần target_mean."""
    mean, _ = brightness_stats(img)
    mean = min(max(mean, 0.02), 0.98)
    gamma = float(np.clip(np.log(target_mean) / np.log(mean), 0.5, 2.0))
    lut = ((np.arange(256) / 255.0) ** gamma * 255).astype(np.uint8)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab[..., 0] = cv2.LUT(lab[..., 0], lut)  # chỉ đổi độ sáng, giữ nguyên màu
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR), gamma


def clahe(img, clip=2.0, grid=8):
    """CLAHE trên kênh L của không gian LAB: tăng tương phản cục bộ, không đổi màu."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab[..., 0] = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid)).apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def auto_balance(img, mode="auto", clip=2.0):
    """Trả về (ảnh, danh sách thao tác đã làm)."""
    steps = []
    if mode == "none":
        return img, steps
    mean, std = brightness_stats(img)
    if mode == "gamma" or (mode == "auto" and not 0.30 <= mean <= 0.70):
        img, g = gamma_correct(img)
        steps.append(f"gamma={g:.2f} (độ sáng TB {mean:.2f})")
    if mode == "clahe" or (mode == "auto" and std < 0.18):
        img = clahe(img, clip)
        steps.append(f"CLAHE clip={clip} (tương phản {std:.2f})")
    return img, steps


# ---------------------------------------------------------------------------
# 3. Kích thước
# ---------------------------------------------------------------------------
def limit_size(img, max_side):
    h, w = img.shape[:2]
    if max_side <= 0 or max(h, w) <= max_side:
        return img
    k = max_side / max(h, w)
    return cv2.resize(img, (round(w * k), round(h * k)), interpolation=cv2.INTER_AREA)


def pad_to_multiple(img, m):
    """Padding phản chiếu ở cạnh dưới/phải. Trả về (ảnh, (pad_bottom, pad_right))."""
    if m <= 1:
        return img, (0, 0)
    h, w = img.shape[:2]
    pb, pr = (-h) % m, (-w) % m
    if pb or pr:
        img = cv2.copyMakeBorder(img, 0, pb, 0, pr, cv2.BORDER_REFLECT_101)
    return img, (pb, pr)


def unpad(img, pad, scale=1):
    """Cắt bỏ padding sau khi ảnh đã được phóng to scale lần."""
    pb, pr = pad
    h, w = img.shape[:2]
    return img[: h - pb * scale, : w - pr * scale]


# ---------------------------------------------------------------------------
# Toàn bộ quy trình
# ---------------------------------------------------------------------------
def preprocess(img, cfg=None):
    """Chạy đủ 3 bước. Trả về (ảnh BGR đã xử lý, info) để hiển thị và để unpad về sau."""
    cfg = cfg or PreprocessConfig()
    info = {"orig_shape": img.shape, "steps": []}

    img = to_bgr3(img)
    info["grayscale"] = is_grayscale(img)
    info["steps"].append("Chuẩn hóa màu: BGR 3 kênh" + (" (ảnh đen trắng)" if info["grayscale"] else ""))

    img, steps = auto_balance(img, cfg.balance, cfg.clahe_clip)
    info["steps"] += steps or ["Cân bằng sáng: không cần chỉnh"]

    h, w = img.shape[:2]
    img = limit_size(img, cfg.max_side)
    if img.shape[:2] != (h, w):
        info["steps"].append(f"Thu nhỏ {w}×{h} → {img.shape[1]}×{img.shape[0]}")

    img, pad = pad_to_multiple(img, cfg.pad_multiple)
    info["pad"] = pad
    if any(pad):
        info["steps"].append(f"Padding dưới {pad[0]}px, phải {pad[1]}px (bội số {cfg.pad_multiple})")
    return img, info
