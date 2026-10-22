"""Phục chế ảnh cũ: làm rõ khuôn mặt (GFPGAN) + phóng to x2/x4 (Real-ESRGAN).

Dùng (chạy từ thư mục photo_restoration/, không chạy từ src/):
    python src/restore_photo.py data/input/bride.jpg data/output/bride_restored.jpg
    python src/restore_photo.py data/input/bride.jpg data/output/bride_x4.jpg --scale 4
Weights (model) tự tải về ở lần chạy đầu (khoảng 400 MB), không cần tải tay.

Dùng như thư viện (ví dụ trong src/app.py):
    restorer = build_restorer(face_model="v1.4", scale=2)   # nạp model 1 lần
    result = restore_array(restorer, img_bgr)                # gọi nhiều lần
"""

import argparse
import gc
import sys

import cv2
import torch
import torchvision.transforms.functional as F

# basicsr còn import module cũ torchvision.transforms.functional_tensor (torchvision
# mới đã bỏ). Gán tên cũ trỏ sang module mới để khỏi phải sửa thư viện.
sys.modules.setdefault("torchvision.transforms.functional_tensor", F)

from basicsr.archs.rrdbnet_arch import RRDBNet
from gfpgan import GFPGANer
from realesrgan import RealESRGANer

# Truyền URL thì thư viện tự tải model về lần đầu rồi dùng lại.
GFPGAN_URLS = {
    "v1.4": "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth",
    "v1.3": "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth",
}
ESRGAN_URLS = {
    2: "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
    4: "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
}


class RestoreError(Exception):
    """Lỗi có thông báo dễ hiểu cho người dùng (CLI in ra, GUI hiện lên màn hình)."""


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def free_memory():
    """Thu dọn RAM và VRAM sau khi xử lý ảnh lớn."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def build_restorer(face_model="v1.4", scale=2, tile=400):
    """Nạp model một lần. Trả về đối tượng có .enhance() để dùng lại cho nhiều ảnh.

    face_model: "v1.4", "v1.3" hoặc None (bỏ qua bước làm rõ mặt, chỉ dùng Real-ESRGAN).
    scale: 2 hoặc 4.
    tile: chia ảnh thành ô tile x tile để đỡ tốn bộ nhớ (0 = không chia).
    """
    if scale not in ESRGAN_URLS:
        raise RestoreError(f"Chỉ hỗ trợ phóng to x2 hoặc x4, nhận được x{scale}.")
    if face_model is not None and face_model not in GFPGAN_URLS:
        raise RestoreError(f"Không có model GFPGAN '{face_model}'.")

    device = get_device()
    half = device.type == "cuda"  # fp16 chỉ chạy được trên GPU, giảm ~một nửa VRAM

    # Real-ESRGAN phóng to phần nền; tile chia ảnh nhỏ để đỡ tốn RAM.
    net = RRDBNet(
        num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=scale
    )  # cấu hình cố định của model x2plus / x4plus
    try:
        background = RealESRGANer(
            scale=scale,
            model_path=ESRGAN_URLS[scale],
            model=net,
            tile=tile,
            tile_pad=10,
            half=half,
            device=device,
        )
        if face_model is None:
            return background

        # GFPGAN dò mặt, làm rõ rồi ghép lại vào ảnh nền đã phóng to.
        return GFPGANer(
            model_path=GFPGAN_URLS[face_model],
            upscale=scale,
            arch="clean",
            channel_multiplier=2,
            bg_upsampler=background,
            device=device,
        )
    except (OSError, RuntimeError) as e:
        # Hay gặp nhất: mất mạng khi tải lần đầu hoặc file .pth tải dở bị hỏng.
        raise RestoreError(
            "Không nạp được model. Kiểm tra kết nối Internet; nếu file model tải dở, "
            f"hãy xóa nó rồi chạy lại.\nChi tiết: {e}"
        ) from e


def restore_array(restorer, img):
    """Phục chế một ảnh BGR (numpy, như cv2.imread trả về). Trả về ảnh BGR đã phục chế."""
    if img is None or img.size == 0:
        raise RestoreError("Ảnh rỗng hoặc không đọc được.")
    if img.ndim == 2:  # ảnh xám 1 kênh -> 3 kênh
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:  # bỏ kênh alpha của PNG
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    try:
        with torch.inference_mode():  # không lưu gradient -> tiết kiệm bộ nhớ
            if isinstance(restorer, GFPGANer):
                _, _, result = restorer.enhance(img, paste_back=True)
            else:  # chỉ có Real-ESRGAN
                result, _ = restorer.enhance(img, outscale=restorer.scale)
    except torch.cuda.OutOfMemoryError as e:
        raise RestoreError(
            "GPU hết bộ nhớ. Hãy giảm kích thước ô (tile) hoặc chọn phóng to x2."
        ) from e
    except MemoryError as e:
        raise RestoreError("Hết RAM. Hãy thu nhỏ ảnh đầu vào hoặc chọn phóng to x2.") from e
    finally:
        free_memory()

    if result is None:
        raise RestoreError("Model không trả về kết quả.")
    return result


def restore(in_path, out_path, face_model="v1.4", scale=2):
    """Giữ nguyên cách dùng cũ: đọc file -> phục chế -> ghi file."""
    img = cv2.imread(in_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RestoreError(f"Không đọc được ảnh: {in_path}")

    restorer = build_restorer(face_model=face_model, scale=scale)
    try:
        result = restore_array(restorer, img)
    finally:
        del restorer  # trả model khỏi RAM/VRAM ngay
        free_memory()

    if not cv2.imwrite(out_path, result):
        raise RestoreError(f"Không ghi được ảnh ra: {out_path} (thư mục có tồn tại không?)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phục chế ảnh cũ bằng GFPGAN + Real-ESRGAN")
    parser.add_argument("input", help="ảnh vào")
    parser.add_argument("output", help="ảnh ra")
    parser.add_argument("--scale", type=int, choices=[2, 4], default=2)
    parser.add_argument("--face", choices=["v1.4", "v1.3", "none"], default="v1.4")
    args = parser.parse_args()

    try:
        restore(
            args.input,
            args.output,
            face_model=None if args.face == "none" else args.face,
            scale=args.scale,
        )
    except RestoreError as e:
        sys.exit(str(e))
