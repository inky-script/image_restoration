"""Vẽ sơ đồ khối cho báo cáo: luồng xử lý và kiến trúc phần mềm.

    python docs/figures/make_diagrams.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = Path(__file__).parent
plt.rcParams.update({"font.family": "Times New Roman", "font.size": 12})
BLUE, BROWN = ("#e8eef5", "#1f3a5f"), ("#f5ede3", "#7a5230")
GREEN = ("#eaf3ea", "#2e6b34")


def box(ax, x, y, w, h, text, c=BLUE, fs=11):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=c[0], ec=c[1], lw=1.4))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)


def arrow(ax, x0, y0, x1, y1):
    ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="-|>", color="#1f3a5f", lw=1.4))


def canvas(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    return fig, ax


# ---- Luồng xử lý một ảnh ----
fig, ax = canvas(12, 4.6)
box(ax, 0.1, 1.8, 1.4, 1.0, "Ảnh cũ", BROWN)
box(ax, 1.9, 1.5, 2.0, 1.6, "Tiền xử lý (TV1)\nchuẩn hóa màu\ncân bằng sáng\nresize + padding", GREEN, 10.5)
box(ax, 4.4, 3.2, 2.4, 1.0, "Real-ESRGAN\nphóng to nền ×2/×4")
box(ax, 4.4, 0.4, 2.4, 1.0, "RetinaFace\ndò mặt + 5 điểm mốc")
box(ax, 7.2, 0.4, 1.9, 1.0, "Căn chỉnh affine\n→ 512×512")
box(ax, 9.4, 0.4, 1.4, 1.0, "GFPGAN\nkhôi phục mặt")
box(ax, 9.0, 3.2, 2.2, 1.0, "Affine ngược\n+ ghép mặt nạ mềm")
box(ax, 9.4, 1.85, 1.4, 0.9, "Bỏ padding", GREEN, 10.5)
box(ax, 11.0, 1.85, 0.95, 0.9, "Ảnh\nra", BROWN)
arrow(ax, 1.5, 2.3, 1.9, 2.3)
arrow(ax, 3.9, 2.7, 4.4, 3.6)
arrow(ax, 3.9, 1.9, 4.4, 0.9)
arrow(ax, 6.8, 0.9, 7.2, 0.9)
arrow(ax, 9.1, 0.9, 9.4, 0.9)
arrow(ax, 10.1, 1.4, 10.1, 1.85)
arrow(ax, 6.8, 3.7, 9.0, 3.7)
arrow(ax, 10.1, 3.2, 10.1, 2.75)
arrow(ax, 10.8, 2.3, 11.0, 2.3)
ax.text(7.8, 2.1, "model_core.py (TV2)", ha="center", fontsize=11, color="#1f3a5f", style="italic")
fig.tight_layout()
fig.savefig(OUT / "pipeline.png", dpi=200)
plt.close(fig)

# ---- Kiến trúc phần mềm ----
fig, ax = canvas(11, 5.6)
box(ax, 0.2, 4.4, 2.6, 0.9, "Người dùng\n(trình duyệt)", BROWN)
box(ax, 3.6, 4.4, 3.8, 0.9, "app.py, giao diện Streamlit (TV3)\n5 tab: Phục chế · Tiền xử lý · Đánh giá · Dữ liệu · Nhóm", fs=10)
box(ax, 8.1, 4.4, 2.7, 0.9, "Cache mô hình\nst.cache_resource")
box(ax, 0.2, 2.6, 2.6, 1.1, "build_test_set.py (TV1)\narchive.zip → test_images/", GREEN, 10)
box(ax, 3.6, 2.6, 2.3, 1.1, "preprocessing.py\n(TV1)", GREEN)
box(ax, 6.3, 2.6, 2.1, 1.1, "model_core.py\n(TV2)")
box(ax, 8.8, 2.6, 2.0, 1.1, "evaluate.py (TV4)\nPSNR / SSIM", ("#f3eaf3", "#6b2e66"), 10.5)
box(ax, 1.4, 0.4, 2.6, 0.9, "OpenCV, NumPy")
box(ax, 4.4, 0.4, 3.0, 0.9, "realesrgan, basicsr,\ngfpgan, facexlib")
box(ax, 7.8, 0.4, 3.0, 0.9, "PyTorch (CPU / CUDA)\nscikit-image")
arrow(ax, 2.8, 4.85, 3.6, 4.85)
arrow(ax, 7.4, 4.85, 8.1, 4.85)
arrow(ax, 4.75, 4.4, 4.75, 3.7)
arrow(ax, 6.0, 4.4, 7.35, 3.7)
arrow(ax, 7.0, 4.4, 9.8, 3.7)
arrow(ax, 2.8, 3.15, 3.6, 3.15)
arrow(ax, 5.9, 3.15, 6.3, 3.15)
arrow(ax, 8.8, 3.15, 8.4, 3.15)
arrow(ax, 4.2, 2.6, 2.7, 1.3)
arrow(ax, 7.35, 2.6, 6.0, 1.3)
arrow(ax, 7.6, 2.6, 8.8, 1.3)
fig.tight_layout()
fig.savefig(OUT / "architecture.png", dpi=200)
plt.close(fig)
print("ok")
