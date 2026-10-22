"""Giao diện web cho đồ án Phục chế ảnh cũ (Streamlit).

Chạy (từ thư mục photo_restoration/, giống restore_photo.py):
    streamlit run src/app.py
Trình duyệt tự mở http://localhost:8501.

Phần lõi (nạp model, phục chế) nằm hết trong restore_photo.py; file này chỉ lo giao diện:
    build_restorer(...)  -> nạp model, được cache để chỉ nạp 1 lần
    restore_array(...)   -> phục chế 1 ảnh numpy BGR
"""

import base64
import math
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import torch

from restore_photo import RestoreError, build_restorer, free_memory, get_device, restore_array

# ---------------------------------------------------------------------------
# Thông tin đồ án: sửa ở đây
# ---------------------------------------------------------------------------
PROJECT_TITLE = "Phục chế ảnh cũ tự động bằng học sâu"
STUDENT_NAME = "Nguyễn Văn A"  # TODO: tên sinh viên
STUDENT_ID = "MSSV: 0000000"  # TODO: mã số sinh viên
SUPERVISOR = "GVHD: ThS. ..."  # TODO: giảng viên hướng dẫn

# Nhãn hiển thị -> tham số face_model của build_restorer()
FACE_MODELS = {
    "GFPGAN v1.4 + Real-ESRGAN (khuyên dùng)": "v1.4",
    "GFPGAN v1.3 + Real-ESRGAN": "v1.3",
    "Chỉ Real-ESRGAN (không xử lý mặt)": None,
}
PREVIEW_MAX_SIDE = 1400  # ảnh xem trước thu nhỏ cho nhẹ trang; ảnh tải về vẫn full

st.set_page_config(page_title="Phục chế ảnh cũ", page_icon="🖼️", layout="wide")

st.markdown(
    """
    <style>
      .block-container { padding-top: 2rem; max-width: 1300px; }
      .hero {
        padding: 1.4rem 1.6rem; border-radius: 16px; margin-bottom: 1.2rem;
        background: linear-gradient(135deg, #3b2a1a 0%, #7a5230 55%, #c08a4a 100%);
        color: #fff8ee;
      }
      .hero h1 { margin: 0; font-size: 1.9rem; color: #fff8ee; }
      .hero p  { margin: .35rem 0 0; opacity: .85; }
      .info-card {
        padding: .9rem 1rem; border-radius: 12px; font-size: .92rem; line-height: 1.55;
        background: rgba(192, 138, 74, .12); border: 1px solid rgba(192, 138, 74, .35);
      }
      div[data-testid="stMetricValue"] { font-size: 1.3rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Các hàm hỗ trợ
# ---------------------------------------------------------------------------
@st.cache_resource(max_entries=1, show_spinner=False)
def load_restorer(face_model, scale, tile):
    """Nạp model 1 lần cho mỗi bộ cài đặt.

    max_entries=1: đổi cài đặt thì model cũ bị bỏ khỏi cache, tránh giữ 2 bộ model
    cùng lúc trong RAM/VRAM.
    """
    free_memory()
    return build_restorer(face_model=face_model, scale=scale, tile=tile)


def decode_upload(file_bytes):
    """bytes -> ảnh BGR numpy (giống cv2.imread)."""
    arr = np.frombuffer(file_bytes, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def limit_size(img, max_side):
    """Thu nhỏ nếu cạnh dài vượt max_side (0 = giữ nguyên)."""
    h, w = img.shape[:2]
    if max_side <= 0 or max(h, w) <= max_side:
        return img
    k = max_side / max(h, w)
    return cv2.resize(img, (round(w * k), round(h * k)), interpolation=cv2.INTER_AREA)


def to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def to_data_url(img, quality=90):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()


def estimate_seconds(img, scale, on_gpu):
    """Ước lượng thời gian để chạy thanh tiến trình (theo báo cáo: ~80 s/ảnh 2.2 MP x2 trên CPU)."""
    megapixels = img.shape[0] * img.shape[1] / 1e6
    per_mp = 4 if on_gpu else 37
    return max(3.0, megapixels * per_mp * (scale / 2) ** 2)


def compare_slider(before, after, height=620):
    """Thanh trượt so sánh trước/sau, viết bằng HTML thuần (không cần cài thêm thư viện).

    before được phóng cho bằng kích thước after để hai ảnh khớp nhau.
    """
    h, w = after.shape[:2]
    k = min(1.0, PREVIEW_MAX_SIDE / max(h, w))
    size = (round(w * k), round(h * k))
    after_small = cv2.resize(after, size, interpolation=cv2.INTER_AREA)
    before_big = cv2.resize(before, size, interpolation=cv2.INTER_CUBIC)

    html = f"""
    <style>
      body {{ margin: 0; display: flex; justify-content: center; font-family: sans-serif; }}
      .cmp {{ position: relative; display: inline-block; line-height: 0; user-select: none;
              border-radius: 12px; overflow: hidden; box-shadow: 0 6px 24px rgba(0,0,0,.25); }}
      .cmp img {{ max-height: {height - 20}px; max-width: 100%; display: block; }}
      .cmp .before {{ position: absolute; inset: 0; width: 100%; height: 100%;
                      clip-path: inset(0 50% 0 0); }}
      .line {{ position: absolute; top: 0; bottom: 0; left: 50%; width: 3px;
               background: #fff; transform: translateX(-50%); pointer-events: none; }}
      .knob {{ position: absolute; top: 50%; left: 50%; width: 38px; height: 38px;
               border-radius: 50%; background: #fff; transform: translate(-50%, -50%);
               box-shadow: 0 2px 8px rgba(0,0,0,.4); pointer-events: none;
               display: flex; align-items: center; justify-content: center;
               font: 700 16px sans-serif; color: #7a5230; line-height: 1; }}
      .tag {{ position: absolute; top: 10px; padding: 4px 10px; border-radius: 6px;
              background: rgba(0,0,0,.55); color: #fff; font: 600 13px sans-serif;
              line-height: 1.2; pointer-events: none; }}
      input[type=range] {{ position: absolute; inset: 0; width: 100%; height: 100%;
                           margin: 0; opacity: 0; cursor: ew-resize; }}
    </style>
    <div class="cmp">
      <img src="{to_data_url(after_small)}" alt="Sau phục chế">
      <img class="before" id="before" src="{to_data_url(before_big)}" alt="Ảnh gốc">
      <div class="line" id="line"></div><div class="knob" id="knob">&#x2194;</div>
      <span class="tag" style="left:10px">Ảnh gốc</span>
      <span class="tag" style="right:10px">Sau phục chế</span>
      <input type="range" min="0" max="100" value="50" step="0.1" id="s">
    </div>
    <script>
      const s = document.getElementById('s');
      s.addEventListener('input', () => {{
        const v = s.value;
        document.getElementById('before').style.clipPath = `inset(0 ${{100 - v}}% 0 0)`;
        document.getElementById('line').style.left = v + '%';
        document.getElementById('knob').style.left = v + '%';
      }});
    </script>
    """
    components.html(html, height=height)


# ---------------------------------------------------------------------------
# Sidebar: thông tin đồ án + cài đặt
# ---------------------------------------------------------------------------
device = get_device()
on_gpu = device.type == "cuda"

with st.sidebar:
    st.markdown("## 🖼️ Thông tin đồ án")
    st.markdown(
        f"""<div class="info-card">
        <b>{PROJECT_TITLE}</b><br>
        👤 {STUDENT_NAME}<br>🆔 {STUDENT_ID}<br>🎓 {SUPERVISOR}
        </div>""",
        unsafe_allow_html=True,
    )

    st.markdown("## ⚙️ Cài đặt")
    model_label = st.selectbox("Mô hình", list(FACE_MODELS))
    scale = st.radio("Tỷ lệ phóng to", [2, 4], format_func=lambda s: f"x{s}", horizontal=True)

    with st.expander("Nâng cao"):
        tile = st.select_slider(
            "Kích thước ô (tile)",
            options=[0, 200, 300, 400, 600, 800],
            value=400,
            help="Chia ảnh thành ô nhỏ để xử lý, đỡ tốn bộ nhớ. Giảm nếu bị lỗi hết bộ nhớ. "
            "0 = không chia (nhanh hơn nhưng tốn bộ nhớ nhất).",
        )
        max_side = st.number_input(
            "Giới hạn cạnh dài ảnh vào (px, 0 = không giới hạn)",
            min_value=0,
            max_value=8000,
            value=2000,
            step=100,
            help="Ảnh lớn hơn sẽ được thu nhỏ trước khi phục chế, tránh hết RAM khi phóng x4.",
        )

    st.divider()
    if on_gpu:
        st.success(f"Thiết bị: GPU ({torch.cuda.get_device_name(0)})")
    else:
        st.info("Thiết bị: CPU (khoảng 1 đến 2 phút/ảnh)")
    st.caption("Lần chạy đầu tự tải model (khoảng 600 MB), cần Internet.")


# ---------------------------------------------------------------------------
# Khu vực chính
# ---------------------------------------------------------------------------
st.markdown(
    f"""<div class="hero"><h1>{PROJECT_TITLE}</h1>
    <p>GFPGAN dò và khôi phục khuôn mặt · Real-ESRGAN phóng to và làm nét nền</p></div>""",
    unsafe_allow_html=True,
)

uploaded = st.file_uploader(
    "Kéo thả ảnh cũ vào đây (JPG, PNG, BMP, WEBP)",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
)

if uploaded is None:
    st.session_state.pop("result", None)
    st.info("⬆️ Tải một ảnh cũ lên để bắt đầu.")
    st.stop()

img = decode_upload(uploaded.getvalue())
if img is None:
    st.error("Không đọc được file ảnh này. Hãy thử ảnh JPG hoặc PNG khác.")
    st.stop()

src = limit_size(img, max_side)
face_model = FACE_MODELS[model_label]
# Kết quả cũ chỉ còn đúng khi cùng ảnh và cùng cài đặt
job_key = (uploaded.file_id, face_model, scale, tile, max_side)
if st.session_state.get("result", {}).get("key") != job_key:
    st.session_state.pop("result", None)

col_img, col_info = st.columns([2, 3], gap="large")
with col_img:
    st.image(to_rgb(src), caption="Ảnh gốc", width="stretch")
with col_info:
    h, w = img.shape[:2]
    st.markdown(f"**{uploaded.name}**")
    st.markdown(f"Kích thước gốc: `{w} × {h}`")
    if src is not img:
        st.warning(f"Ảnh được thu nhỏ còn `{src.shape[1]} × {src.shape[0]}` trước khi xử lý.")
    st.markdown(
        f"Ảnh kết quả: `{src.shape[1] * scale} × {src.shape[0] * scale}` · "
        f"{model_label}"
    )
    start = st.button("✨ Bắt đầu phục chế", type="primary", width="stretch")

# ---------------------------------------------------------------------------
# Xử lý
# ---------------------------------------------------------------------------
if start:
    progress = st.progress(0, text="Đang nạp mô hình...")
    t0 = time.perf_counter()
    try:
        restorer = load_restorer(face_model, scale, tile)
        progress.progress(15, text="Đã nạp mô hình. Đang phục chế...")

        # Chạy model ở luồng phụ để luồng chính cập nhật được thanh tiến trình.
        # Model không báo tiến độ thật, nên thanh chạy theo thời gian ước lượng
        # và dừng ở 95% cho tới khi xong.
        expected = estimate_seconds(src, scale, on_gpu)
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(restore_array, restorer, src)
            started = time.perf_counter()
            while not future.done():
                elapsed = time.perf_counter() - started
                pct = 15 + int(80 * (1 - math.exp(-elapsed / expected)))
                progress.progress(pct, text=f"Đang phục chế... {elapsed:.0f} giây")
                time.sleep(0.3)
            result = future.result()  # ném lại lỗi nếu luồng phụ bị lỗi

        progress.progress(97, text="Đang mã hóa ảnh kết quả...")
        ok, png = cv2.imencode(".png", result)
        if not ok:
            raise RestoreError("Không mã hóa được ảnh kết quả.")

        faces = getattr(getattr(restorer, "face_helper", None), "restored_faces", None)
        st.session_state["result"] = {
            "key": job_key,
            "image": result,
            "png": png.tobytes(),
            "seconds": time.perf_counter() - t0,
            "faces": None if faces is None else len(faces),
        }
        progress.progress(100, text="Hoàn tất!")
        st.toast("Phục chế xong!", icon="✅")
    except RestoreError as e:
        progress.empty()
        st.error(str(e))
    except Exception as e:  # lỗi bất ngờ: hiện chi tiết để dễ báo lỗi
        progress.empty()
        st.error("Có lỗi không mong muốn khi phục chế.")
        st.exception(e)
    finally:
        free_memory()

# ---------------------------------------------------------------------------
# Kết quả
# ---------------------------------------------------------------------------
res = st.session_state.get("result")
if res:
    out = res["image"]
    st.divider()
    st.subheader("Kết quả")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Kích thước gốc", f"{src.shape[1]}×{src.shape[0]}")
    m2.metric("Kích thước mới", f"{out.shape[1]}×{out.shape[0]}")
    m3.metric("Thời gian", f"{res['seconds']:.1f} s")
    m4.metric("Khuôn mặt phục chế", "—" if res["faces"] is None else res["faces"])

    tab_slider, tab_side = st.tabs(["🔀 Thanh trượt so sánh", "🖼️ Đặt cạnh nhau"])
    with tab_slider:
        compare_slider(src, out)
        st.caption("Kéo sang trái/phải để so sánh. Ảnh gốc được phóng bằng bicubic cho cùng kích thước.")
    with tab_side:
        c1, c2 = st.columns(2)
        c1.image(to_rgb(src), caption="Ảnh gốc", width="stretch")
        c2.image(to_rgb(out), caption="Sau phục chế", width="stretch")

    stem = Path(uploaded.name).stem
    d1, d2 = st.columns(2)
    d1.download_button(
        "⬇️ Tải ảnh về (PNG, không nén mất dữ liệu)",
        data=res["png"],
        file_name=f"{stem}_restored_x{scale}.png",
        mime="image/png",
        type="primary",
        width="stretch",
    )
    ok, jpg = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 95])
    d2.download_button(
        "⬇️ Tải ảnh về (JPG 95%, nhẹ hơn)",
        data=jpg.tobytes(),
        file_name=f"{stem}_restored_x{scale}.jpg",
        mime="image/jpeg",
        width="stretch",
    )
