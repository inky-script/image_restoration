"""Phục chế ảnh cũ: làm rõ khuôn mặt (GFPGAN) + phóng to x2 (Real-ESRGAN).

Dùng (chạy từ thư mục photo_restoration/, không chạy từ src/):
    python src/restore_photo.py data/input/bride.jpg data/output/bride_restored.jpg
Weights (model) tự tải về ở lần chạy đầu (khoảng 400 MB), không cần tải tay.
"""

import sys

import cv2
import torchvision.transforms.functional as F

# basicsr còn import module cũ torchvision.transforms.functional_tensor (torchvision
# mới đã bỏ). Gán tên cũ trỏ sang module mới để khỏi phải sửa thư viện.
sys.modules.setdefault("torchvision.transforms.functional_tensor", F)

from basicsr.archs.rrdbnet_arch import RRDBNet
from gfpgan import GFPGANer
from realesrgan import RealESRGANer

# Truyền URL thì thư viện tự tải model về lần đầu rồi dùng lại.
GFPGAN_URL = (
    "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth"
)
ESRGAN_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"


def restore(in_path, out_path):
    img = cv2.imread(in_path)
    if img is None:
        sys.exit(f"Không đọc được ảnh: {in_path}")

    # Real-ESRGAN phóng to phần nền; tile=400 chia ảnh nhỏ để đỡ tốn RAM.
    net = RRDBNet(
        num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2
    )  # cấu hình cố định của model x2plus
    background = RealESRGANer(scale=2, model_path=ESRGAN_URL, model=net, tile=400)

    # GFPGAN dò mặt, làm rõ rồi ghép lại vào ảnh nền đã phóng to.
    restorer = GFPGANer(
        model_path=GFPGAN_URL,
        upscale=2,
        arch="clean",
        channel_multiplier=2,
        bg_upsampler=background,
    )
    _, _, result = restorer.enhance(img, paste_back=True)

    cv2.imwrite(out_path, result)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("Dùng: python src/restore_photo.py <ảnh_vào> <ảnh_ra>")
    restore(sys.argv[1], sys.argv[2])
