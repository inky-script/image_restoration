"""Đánh giá định lượng PSNR / SSIM trên tập test (phần việc Thành viên 4).

Với mỗi cặp trong test_images/ (gt = ảnh gốc, degraded = ảnh bị làm hỏng nhỏ hơn 2 lần),
phóng ảnh hỏng về lại kích thước gốc bằng 4 phương pháp rồi so với ảnh gốc:
    bicubic      nội suy bicubic (mốc so sánh thấp nhất)
    opencv       khử nhiễu Non-Local Means + bicubic + unsharp mask (xử lý ảnh truyền thống)
    realesrgan   chỉ Real-ESRGAN x2
    gfpgan       GFPGAN v1.4 + Real-ESRGAN x2 (hệ thống của đồ án)
Ảnh hỏng thật (real_damaged/) không có ảnh gốc nên chỉ được phục chế để xem bằng mắt.

Chạy từ thư mục gốc dự án:
    python src/evaluate.py                 # toàn bộ
    python src/evaluate.py --limit 5       # thử nhanh 5 cặp
    python src/evaluate.py --skip-real     # bỏ phần ảnh hỏng thật
Kết quả: results/metrics.csv, results/summary.csv, results/real_times.csv, ảnh trong results/
"""

import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np
from skimage.metrics import structural_similarity

ROOT = Path(__file__).resolve().parent.parent
TEST = ROOT / "test_images"
RESULTS = ROOT / "results"
SCALE = 2
METHODS = ["bicubic", "opencv", "realesrgan", "gfpgan"]
METHOD_LABELS = {
    "bicubic": "Bicubic",
    "opencv": "OpenCV truyền thống",
    "realesrgan": "Real-ESRGAN",
    "gfpgan": "GFPGAN + Real-ESRGAN",
}


# ---------------------------------------------------------------------------
# Độ đo
# ---------------------------------------------------------------------------
def to_y(img):
    """Kênh độ sáng Y (chuẩn ITU-R BT.601), thang 16-235, như các bài báo siêu phân giải."""
    img = img.astype(np.float64) / 255.0
    b, g, r = img[..., 0], img[..., 1], img[..., 2]
    return 16.0 + 65.481 * r + 128.553 * g + 24.966 * b


def psnr(ref, out, border=4):
    """PSNR = 10·log10(255² / MSE), tính trên kênh Y, bỏ viền `border` px."""
    a, b = to_y(ref), to_y(out)
    if border:
        a, b = a[border:-border, border:-border], b[border:-border, border:-border]
    mse = np.mean((a - b) ** 2)
    return float("inf") if mse == 0 else float(10 * np.log10(255.0**2 / mse))


def ssim(ref, out, border=4):
    """SSIM trên kênh Y (cửa sổ Gauss 11x11, σ=1.5 như bài báo gốc của Wang et al. 2004)."""
    a, b = to_y(ref), to_y(out)
    if border:
        a, b = a[border:-border, border:-border], b[border:-border, border:-border]
    return float(structural_similarity(a, b, data_range=255.0, gaussian_weights=True,
                                       sigma=1.5, use_sample_covariance=False))


_lpips_net = None


def lpips_dist(ref, out):
    """LPIPS (Zhang et al. 2018): khoảng cách cảm nhận dựa trên đặc trưng AlexNet.
    Càng THẤP càng giống ảnh gốc theo cảm nhận của người xem. Cần: pip install lpips"""
    global _lpips_net
    import lpips
    import torch

    if _lpips_net is None:
        _lpips_net = lpips.LPIPS(net="alex", verbose=False)

    def to_t(img):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 127.5 - 1
        return torch.from_numpy(rgb).permute(2, 0, 1)[None]

    with torch.inference_mode():
        return float(_lpips_net(to_t(ref), to_t(out)).item())


def laplacian_var(img):
    """Độ nét không tham chiếu: phương sai Laplacian của ảnh xám."""
    return float(cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())


# ---------------------------------------------------------------------------
# Các phương pháp không dùng AI
# ---------------------------------------------------------------------------
def upscale_bicubic(img, scale=SCALE):
    h, w = img.shape[:2]
    return cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)


def upscale_opencv(img, scale=SCALE):
    den = cv2.fastNlMeansDenoisingColored(img, None, 7, 7, 7, 21)
    up = upscale_bicubic(den, scale)
    blur = cv2.GaussianBlur(up, (0, 0), 1.5)
    return cv2.addWeighted(up, 1.6, blur, -0.6, 0)  # unsharp mask


# ---------------------------------------------------------------------------
# Chạy đánh giá
# ---------------------------------------------------------------------------
def load_manifest():
    with open(TEST / "manifest.csv", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def match_size(out, ref):
    """Kết quả mô hình có thể lệch 1-2 px do làm tròn; cắt/phóng cho khớp ảnh gốc."""
    if out.shape[:2] == ref.shape[:2]:
        return out
    return cv2.resize(out, (ref.shape[1], ref.shape[0]), interpolation=cv2.INTER_CUBIC)


def run_pairs(rows, methods, log=print):
    """Trả về danh sách dict {id, method, psnr, ssim, seconds}. Ghi ảnh ra results/pairs/."""
    from preprocessing import PreprocessConfig

    # Không cân bằng sáng khi đo: đổi độ sáng sẽ làm lệch so với ảnh gốc một cách không công bằng
    cfg = PreprocessConfig(balance="none", max_side=0)
    out_rows = []
    for method in methods:
        (RESULTS / "pairs" / method).mkdir(parents=True, exist_ok=True)
        restorer = None
        if method in ("realesrgan", "gfpgan"):
            from model_core import build_restorer, restore_image  # chỉ nạp torch khi cần

            restorer = build_restorer(face_model="v1.4" if method == "gfpgan" else None, scale=SCALE)
        for i, r in enumerate(rows, 1):
            gt = cv2.imread(str(TEST / r["gt_file"]))
            bad = cv2.imread(str(TEST / r["file"]))
            t0 = time.perf_counter()
            faces = ""
            if method == "bicubic":
                out = upscale_bicubic(bad)
            elif method == "opencv":
                out = upscale_opencv(bad)
            else:
                out, info = restore_image(restorer, bad, cfg)
                faces = "" if info["faces"] is None else info["faces"]
            sec = time.perf_counter() - t0
            out = match_size(out, gt)
            cv2.imwrite(str(RESULTS / "pairs" / method / f"{r['id']}.png"), out)
            out_rows.append({"id": r["id"], "method": method, "psnr": round(psnr(gt, out), 3),
                             "ssim": round(ssim(gt, out), 4), "lpips": round(lpips_dist(gt, out), 4),
                             "seconds": round(sec, 2), "faces": faces})
            log(f"[{method}] {i}/{len(rows)} {r['id']}: PSNR {out_rows[-1]['psnr']:.2f} dB, "
                f"SSIM {out_rows[-1]['ssim']:.3f}, {sec:.1f}s")
        del restorer
    return out_rows


def summarize(metric_rows):
    out = []
    for m in METHODS:
        rs = [r for r in metric_rows if r["method"] == m]
        if not rs:
            continue
        p = np.array([float(r["psnr"]) for r in rs])
        s = np.array([float(r["ssim"]) for r in rs])
        t = np.array([float(r["seconds"]) for r in rs])
        row = {"method": m, "label": METHOD_LABELS[m], "n": len(rs),
               "psnr_mean": round(float(p.mean()), 2), "psnr_std": round(float(p.std()), 2),
               "ssim_mean": round(float(s.mean()), 4), "ssim_std": round(float(s.std()), 4),
               "seconds_mean": round(float(t.mean()), 2)}
        if all(r.get("lpips") not in (None, "") for r in rs):
            lp = np.array([float(r["lpips"]) for r in rs])
            row.update(lpips_mean=round(float(lp.mean()), 4), lpips_std=round(float(lp.std()), 4))
        out.append(row)
    return out


def add_lpips():
    """Tính LPIPS cho các ảnh đã có trong results/pairs/ rồi cập nhật metrics.csv, summary.csv."""
    with open(RESULTS / "metrics.csv", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for i, r in enumerate(rows, 1):
        gt = cv2.imread(str(TEST / "gt" / f"{r['id']}.png"))
        out = cv2.imread(str(RESULTS / "pairs" / r["method"] / f"{r['id']}.png"))
        r["lpips"] = round(lpips_dist(gt, out), 4)
        if i % 20 == 0:
            print(f"LPIPS {i}/{len(rows)}")
    write_csv(RESULTS / "metrics.csv", rows)
    write_csv(RESULTS / "summary.csv", summarize(rows))
    for s in summarize(rows):
        print(f"  {s['label']:<22} LPIPS {s['lpips_mean']:.4f} ± {s['lpips_std']:.4f}")


def run_real(rows, log=print):
    """Phục chế ảnh hỏng thật bằng hệ thống đầy đủ (có cân bằng sáng tự động)."""
    from model_core import build_restorer, restore_image
    from preprocessing import PreprocessConfig

    (RESULTS / "real").mkdir(parents=True, exist_ok=True)
    restorer = build_restorer(face_model="v1.4", scale=SCALE)
    cfg = PreprocessConfig(balance="auto", max_side=768)
    out_rows = []
    for i, r in enumerate(rows, 1):
        img = cv2.imread(str(TEST / r["file"]))
        t0 = time.perf_counter()
        out, info = restore_image(restorer, img, cfg)
        sec = time.perf_counter() - t0
        cv2.imwrite(str(RESULTS / "real" / f"{r['id']}.png"), out)
        base = upscale_bicubic(info["preprocessed"])
        out_rows.append({"id": r["id"], "in_w": info["preprocessed"].shape[1],
                         "in_h": info["preprocessed"].shape[0], "faces": info["faces"],
                         "seconds": round(sec, 2), "lap_bicubic": round(laplacian_var(base), 1),
                         "lap_restored": round(laplacian_var(out), 1),
                         "preprocess": "; ".join(info["steps"])})
        log(f"[real] {i}/{len(rows)} {r['id']}: {info['faces']} mặt, {sec:.1f}s")
    return out_rows


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description="Tính PSNR/SSIM trên test_images/")
    ap.add_argument("--limit", type=int, default=0, help="chỉ chạy N cặp đầu (0 = tất cả)")
    ap.add_argument("--methods", nargs="+", default=METHODS, choices=METHODS)
    ap.add_argument("--skip-real", action="store_true", help="bỏ qua ảnh hỏng thật")
    ap.add_argument("--add-lpips", action="store_true",
                    help="chỉ tính thêm LPIPS cho kết quả đã có trong results/ (không chạy lại mô hình)")
    args = ap.parse_args()
    if args.add_lpips:
        add_lpips()
        return

    rows = load_manifest()
    pairs = [r for r in rows if r["set"] == "pair"]
    real = [r for r in rows if r["set"] == "real_damaged"]
    if args.limit:
        pairs, real = pairs[: args.limit], real[: args.limit]

    metrics = run_pairs(pairs, args.methods)
    write_csv(RESULTS / "metrics.csv", metrics)
    summary = summarize(metrics)
    write_csv(RESULTS / "summary.csv", summary)
    print("\nTrung bình trên", len(pairs), "cặp:")
    for s in summary:
        print(f"  {s['label']:<22} PSNR {s['psnr_mean']:6.2f} ± {s['psnr_std']:.2f} dB   "
              f"SSIM {s['ssim_mean']:.4f} ± {s['ssim_std']:.4f}   {s['seconds_mean']:.1f} s/ảnh")

    if not args.skip_real:
        write_csv(RESULTS / "real_times.csv", run_real(real))


if __name__ == "__main__":
    main()
