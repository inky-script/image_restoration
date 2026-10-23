"""Vẽ hình kết quả cho báo cáo từ thư mục results/ (chạy sau src/evaluate.py).

    python docs/figures/make_result_figures.py
"""

from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RES, TEST, OUT = ROOT / "results", ROOT / "test_images", Path(__file__).parent
plt.rcParams.update({"font.family": "Times New Roman", "font.size": 12})
ORDER = ["bicubic", "opencv", "realesrgan", "gfpgan"]
COLORS = ["#9aa5b1", "#b9a27a", "#5b7fa6", "#1f3a5f"]


def rgb(p):
    return cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)


summary = pd.read_csv(RES / "summary.csv").set_index("method").loc[ORDER]
metrics = pd.read_csv(RES / "metrics.csv")
real = pd.read_csv(RES / "real_times.csv")

# ---- Biểu đồ PSNR / SSIM ----
fig, axes = plt.subplots(1, 3, figsize=(15, 3.9))
for ax, key, title, fmt in [(axes[0], "psnr", "PSNR (dB), cao hơn là tốt hơn", "{:.2f}"),
                            (axes[1], "ssim", "SSIM, cao hơn là tốt hơn", "{:.3f}"),
                            (axes[2], "lpips", "LPIPS, thấp hơn là tốt hơn", "{:.3f}")]:
    vals, errs = summary[f"{key}_mean"], summary[f"{key}_std"]
    bars = ax.bar(summary["label"], vals, yerr=errs, color=COLORS, capsize=4, error_kw={"lw": 1})
    if key == "lpips":
        ax.set_ylim(0, (vals + errs).max() * 1.1)
    else:
        lo = (vals - errs).min()
        ax.set_ylim(lo - (vals.max() - lo) * 0.4, (vals + errs).max() * 1.01)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, fmt.format(v), ha="center", va="bottom", fontsize=10,
                bbox=dict(fc="white", ec="none", pad=1))
    ax.set_title(title, fontsize=12)
    ax.tick_params(axis="x", labelsize=10)
    ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "psnr_ssim.png", dpi=200)
plt.close(fig)

# ---- Ví dụ các cặp ảnh ----
piv = metrics.pivot(index="id", columns="method", values="psnr")
lp = metrics.pivot(index="id", columns="method", values="lpips")
gain = lp["bicubic"] - lp["gfpgan"]  # LPIPS giảm càng nhiều càng tốt
faces = metrics[metrics["method"] == "gfpgan"].set_index("id")["faces"].fillna(0)
with_face = gain[faces.reindex(gain.index).fillna(0) > 0].sort_values()
pick = []
if len(with_face):
    pick.append(with_face.index[-1])  # cặp có mặt, GFPGAN tăng nhiều nhất
srt = gain.sort_values()
for idx in (srt.index[len(srt) // 2], srt.index[-1], srt.index[0]):  # trung vị, tốt nhất, kém nhất
    if idx not in pick:
        pick.append(idx)
pick = pick[:3]
cols = ["gt", "bicubic", "opencv", "realesrgan", "gfpgan"]
names = {"gt": "Ảnh gốc", "bicubic": "Bicubic", "opencv": "OpenCV", "realesrgan": "Real-ESRGAN",
         "gfpgan": "GFPGAN + Real-ESRGAN"}
fig, axes = plt.subplots(len(pick), len(cols), figsize=(13, 3.1 * len(pick)))
for r, pid in enumerate(pick):
    for c, m in enumerate(cols):
        ax = axes[r, c]
        img = rgb(TEST / "gt" / f"{pid}.png") if m == "gt" else rgb(RES / "pairs" / m / f"{pid}.png")
        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])
        label = names[m] if m == "gt" else f"{names[m]}\n{piv.loc[pid, m]:.2f} dB · LPIPS {lp.loc[pid, m]:.3f}"
        ax.set_title(label if r == 0 or m != "gt" else "", fontsize=10.5)
        if c == 0:
            ax.set_ylabel(pid, fontsize=10)
fig.tight_layout()
fig.savefig(OUT / "pairs_examples.png", dpi=170)
plt.close(fig)

# ---- Ví dụ ảnh hỏng thật ----
cand = real[real["faces"] > 0]
if len(cand) < 6:
    cand = real
step = max(1, len(cand) // 6)
ids = list(cand["id"].iloc[::step][:6])
fig, axes = plt.subplots(2, len(ids), figsize=(2.3 * len(ids), 5.2))
for c, rid in enumerate(ids):
    after = rgb(RES / "real" / f"{rid}.png")
    before = rgb(TEST / "real_damaged" / f"{rid}.png")
    before = cv2.resize(before, (after.shape[1], after.shape[0]), interpolation=cv2.INTER_CUBIC)
    for r, img in enumerate((before, after)):
        axes[r, c].imshow(img)
        axes[r, c].set_xticks([])
        axes[r, c].set_yticks([])
    axes[0, c].set_title(rid, fontsize=10)
axes[0, 0].set_ylabel("Trước", fontsize=11)
axes[1, 0].set_ylabel("Sau", fontsize=11)
fig.tight_layout()
fig.savefig(OUT / "real_examples.png", dpi=170)
plt.close(fig)
print("ok:", pick, ids)
