"""Đóng gói bộ dữ liệu kiểm thử chuẩn test_images/ (phần việc Thành viên 1).

Nguồn: data/raw_archive/ (giải nén từ archive.zip)
    01_Clean_Candidates_GT/   146 ảnh tương đối sạch
    02_Damaged_Testing_Set/   54 ảnh hỏng thật (mờ, vỡ pixel, xước, ố, nhòe chuyển động)

Kết quả: test_images/
    real_damaged/  ảnh hỏng thật, chỉ để đánh giá bằng mắt (không có ảnh gốc)
    gt/            ảnh sạch đã chuẩn hóa (cạnh dài 512 px) -> ảnh gốc để tính PSNR/SSIM
    degraded/      ảnh gt bị làm hỏng nhân tạo (mờ + thu nhỏ x2 + nhiễu + nén JPEG)
    manifest.csv   danh sách ảnh và tham số làm hỏng của từng ảnh

Chạy từ thư mục gốc dự án:
    python src/build_test_set.py            # mặc định 40 cặp gt/degraded
    python src/build_test_set.py --pairs 50
"""

import argparse
import csv
import re
import shutil
from pathlib import Path

import cv2
import numpy as np

from preprocessing import limit_size, to_bgr3

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw_archive"
OUT = ROOT / "test_images"

GT_SIDE = 512  # cạnh dài ảnh gốc; ảnh hỏng nhỏ hơn 2 lần (256 px)
SCALE = 2
SEED = 2026  # cố định để mọi thành viên tạo ra cùng một bộ dữ liệu


def img_number(path):
    m = re.search(r"img(\d+)", path.name)
    return int(m.group(1)) if m else 0


def normalize_gt(img):
    """Ảnh sạch -> cạnh dài GT_SIDE, kích thước chia hết cho SCALE*4."""
    img = to_bgr3(img)
    h, w = img.shape[:2]
    k = GT_SIDE / max(h, w)
    img = cv2.resize(img, (round(w * k), round(h * k)), interpolation=cv2.INTER_AREA)
    m = SCALE * 4
    h, w = img.shape[:2]
    return img[: h - h % m, : w - w % m]


def degrade(gt, rng):
    """Làm hỏng ảnh theo mô hình x = [(y ⊛ k)↓s + n]_JPEG. Trả về (ảnh hỏng, tham số)."""
    p = {
        "blur_sigma": round(float(rng.uniform(0.6, 1.6)), 2),
        "noise_sigma": round(float(rng.uniform(2, 10)), 1),
        "jpeg_quality": int(rng.integers(35, 76)),
    }
    x = cv2.GaussianBlur(gt, (0, 0), p["blur_sigma"])
    h, w = gt.shape[:2]
    x = cv2.resize(x, (w // SCALE, h // SCALE), interpolation=cv2.INTER_AREA)
    x = x.astype(np.float32) + rng.normal(0, p["noise_sigma"], x.shape).astype(np.float32)
    x = np.clip(x, 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", x, [cv2.IMWRITE_JPEG_QUALITY, p["jpeg_quality"]])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR), p


def build(pairs=40, real_max_side=1024):
    if not RAW.exists():
        raise SystemExit(f"Không thấy {RAW}. Hãy giải nén archive.zip vào data/raw_archive/.")
    if OUT.exists():
        shutil.rmtree(OUT)
    for sub in ("real_damaged", "gt", "degraded"):
        (OUT / sub).mkdir(parents=True)
    rows = []

    # --- ảnh hỏng thật ---
    for i, f in enumerate(sorted((RAW / "02_Damaged_Testing_Set").glob("*.jpg"), key=img_number), 1):
        img = limit_size(to_bgr3(cv2.imread(str(f), cv2.IMREAD_UNCHANGED)), real_max_side)
        name = f"real_{i:03d}.png"
        cv2.imwrite(str(OUT / "real_damaged" / name), img)
        rows.append({"id": name[:-4], "set": "real_damaged", "file": f"real_damaged/{name}",
                     "source": f.name, "width": img.shape[1], "height": img.shape[0]})

    # --- cặp gt / degraded: chọn ảnh sạch đủ lớn để thu nhỏ x2 vẫn còn chi tiết ---
    rng = np.random.default_rng(SEED)
    candidates = []
    for f in sorted((RAW / "01_Clean_Candidates_GT").glob("*.jpg"), key=img_number):
        img = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
        if img is not None and min(img.shape[:2]) >= 350 and max(img.shape[:2]) >= GT_SIDE:
            candidates.append((f, img))
    if len(candidates) < pairs:
        print(f"Chỉ có {len(candidates)} ảnh sạch đủ lớn, dùng hết.")
    for i, (f, img) in enumerate(candidates[:pairs], 1):
        gt = normalize_gt(img)
        bad, p = degrade(gt, rng)
        name = f"pair_{i:03d}.png"
        cv2.imwrite(str(OUT / "gt" / name), gt)
        cv2.imwrite(str(OUT / "degraded" / name), bad)
        rows.append({"id": name[:-4], "set": "pair", "file": f"degraded/{name}", "gt_file": f"gt/{name}",
                     "source": f.name, "width": bad.shape[1], "height": bad.shape[0], **p})

    fields = ["id", "set", "file", "gt_file", "source", "width", "height",
              "blur_sigma", "noise_sigma", "jpeg_quality"]
    with open(OUT / "manifest.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    n_real = sum(r["set"] == "real_damaged" for r in rows)
    print(f"Xong: {n_real} ảnh hỏng thật, {len(rows) - n_real} cặp gt/degraded -> {OUT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Tạo bộ dữ liệu kiểm thử test_images/")
    ap.add_argument("--pairs", type=int, default=40, help="số cặp gt/degraded (30-50)")
    build(ap.parse_args().pairs)
