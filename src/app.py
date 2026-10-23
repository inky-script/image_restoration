"""Ứng dụng web Phục chế ảnh cũ (Streamlit), phần việc Thành viên 3.

Chạy từ thư mục gốc dự án:
    streamlit run src/app.py
Trình duyệt tự mở http://localhost:8501.

App chỉ lo giao diện, mọi xử lý gọi sang các mô-đun của nhóm:
    preprocessing.py  (TV1)  tiền xử lý OpenCV
    model_core.py     (TV2)  GFPGAN + Real-ESRGAN
    evaluate.py       (TV4)  PSNR / SSIM
"""

import base64
import io
import math
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

import evaluate as ev
from model_core import RestoreError, build_restorer, free_memory, get_device, restore_image
from preprocessing import PreprocessConfig, brightness_stats, preprocess, to_bgr3

ROOT = Path(__file__).resolve().parent.parent
TEST = ROOT / "test_images"
RESULTS = ROOT / "results"

# ---------------------------------------------------------------------------
# Thông tin đồ án
# ---------------------------------------------------------------------------
SCHOOL = "Trường Đại học Công Thương TP. Hồ Chí Minh"
COURSE = "Đồ án môn Xử lý ảnh số"
PROJECT_TITLE = "Phục chế ảnh cũ tự động bằng học sâu"
SUPERVISOR = "ThS. Đỗ Trung Dũng"
TEAM = [
    ("Nguyễn Thu Thảo (nhóm trưởng)", "2045240281", "Dữ liệu & tiền xử lý",
     "Thu thập, chuẩn hóa test set; tiền xử lý OpenCV; đóng gói test_images/"),
    ("Nguyễn Phan Nhật Huy", "2045240107", "Lõi thuật toán & mô hình AI",
     "Cấu hình GFPGAN + Real-ESRGAN; model_core.py; xử lý lỗi VRAM/RAM"),
    ("Nguyễn Thị Kiều Trang", "2045240312", "Giao diện & ứng dụng",
     "App Streamlit: upload, xử lý, so sánh trước/sau, tải ảnh; nhúng lõi AI"),
    ("Đỗ Thị Hồng Quyên", "2045240255", "Kiểm thử, đánh giá & báo cáo",
     "Tính PSNR/SSIM; báo cáo, sơ đồ khối; slide, GitHub, video demo"),
]

FACE_MODELS = {
    "GFPGAN v1.4 + Real-ESRGAN (khuyên dùng)": "v1.4",
    "GFPGAN v1.3 + Real-ESRGAN": "v1.3",
    "Chỉ Real-ESRGAN (không xử lý mặt)": None,
}
BALANCE = {"Tự động": "auto", "CLAHE": "clahe", "Gamma": "gamma", "Tắt": "none"}
PREVIEW_MAX_SIDE = 1400

st.set_page_config(page_title="Phục chế ảnh cũ", page_icon="🖼️", layout="wide")
st.markdown(
    """
    <style>
      .block-container { padding-top: 1.6rem; max-width: 1350px; }
      .hero { padding: 1.3rem 1.6rem; border-radius: 16px; margin-bottom: 1rem;
              background: linear-gradient(135deg, #3b2a1a 0%, #7a5230 55%, #c08a4a 100%);
              color: #fff8ee; }
      .hero h1 { margin: 0; font-size: 1.85rem; color: #fff8ee; }
      .hero p { margin: .35rem 0 0; opacity: .88; }
      .info-card { padding: .85rem 1rem; border-radius: 12px; font-size: .9rem; line-height: 1.55;
                   background: rgba(192,138,74,.12); border: 1px solid rgba(192,138,74,.35); }
      .step { padding: .45rem .7rem; margin: .25rem 0; border-left: 3px solid #c08a4a;
              background: rgba(192,138,74,.08); border-radius: 0 8px 8px 0; font-size: .92rem; }
      div[data-testid="stMetricValue"] { font-size: 1.3rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Hàm hỗ trợ
# ---------------------------------------------------------------------------
@st.cache_resource(max_entries=1, show_spinner=False)
def load_restorer(face_model, scale, tile):
    """Nạp model 1 lần cho mỗi bộ cài đặt; đổi cài đặt thì model cũ bị bỏ khỏi bộ nhớ."""
    free_memory()
    return build_restorer(face_model=face_model, scale=scale, tile=tile)


def decode(file_bytes):
    return cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_UNCHANGED)


@st.cache_data(show_spinner=False)
def read_image(path):
    return cv2.imread(str(path), cv2.IMREAD_UNCHANGED)


@st.cache_data(show_spinner=False)
def load_manifest():
    p = TEST / "manifest.csv"
    return pd.read_csv(p) if p.exists() else None


def load_csv(name):
    p = RESULTS / name
    return pd.read_csv(p) if p.exists() else None


def to_data_url(img, quality=90):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()


def png_bytes(img):
    return cv2.imencode(".png", img)[1].tobytes()


def estimate_seconds(img, scale, on_gpu):
    """Ước lượng thời gian để chạy thanh tiến trình (CPU ~37 s / triệu điểm ảnh khi x2)."""
    mp = img.shape[0] * img.shape[1] / 1e6
    return max(3.0, mp * (4 if on_gpu else 37) * (scale / 2) ** 2)


def compare_slider(before, after, height=600):
    """Thanh trượt Trước/Sau bằng HTML thuần; ảnh trước được phóng cho bằng ảnh sau."""
    h, w = after.shape[:2]
    k = min(1.0, PREVIEW_MAX_SIDE / max(h, w))
    size = (round(w * k), round(h * k))
    a = cv2.resize(after, size, interpolation=cv2.INTER_AREA)
    b = cv2.resize(before, size, interpolation=cv2.INTER_CUBIC)
    html = f"""
    <style>
      body {{ margin:0; display:flex; justify-content:center; }}
      .cmp {{ position:relative; display:inline-block; line-height:0; user-select:none;
              border-radius:12px; overflow:hidden; box-shadow:0 6px 24px rgba(0,0,0,.25); }}
      .cmp img {{ max-height:{height - 20}px; max-width:100%; display:block; }}
      .cmp .before {{ position:absolute; inset:0; width:100%; height:100%; clip-path:inset(0 50% 0 0); }}
      .line {{ position:absolute; top:0; bottom:0; left:50%; width:3px; background:#fff;
               transform:translateX(-50%); pointer-events:none; }}
      .knob {{ position:absolute; top:50%; left:50%; width:38px; height:38px; border-radius:50%;
               background:#fff; transform:translate(-50%,-50%); box-shadow:0 2px 8px rgba(0,0,0,.4);
               pointer-events:none; display:flex; align-items:center; justify-content:center;
               font:700 16px sans-serif; color:#7a5230; line-height:1; }}
      .tag {{ position:absolute; top:10px; padding:4px 10px; border-radius:6px; background:rgba(0,0,0,.55);
              color:#fff; font:600 13px sans-serif; line-height:1.2; pointer-events:none; }}
      input[type=range] {{ position:absolute; inset:0; width:100%; height:100%; margin:0; opacity:0;
                           cursor:ew-resize; }}
    </style>
    <div class="cmp">
      <img src="{to_data_url(a)}" alt="Sau">
      <img class="before" id="b" src="{to_data_url(b)}" alt="Trước">
      <div class="line" id="l"></div><div class="knob" id="k">&#x2194;</div>
      <span class="tag" style="left:10px">Trước</span><span class="tag" style="right:10px">Sau</span>
      <input type="range" min="0" max="100" value="50" step="0.1" id="s">
    </div>
    <script>
      const s = document.getElementById('s');
      s.addEventListener('input', () => {{
        const v = s.value;
        document.getElementById('b').style.clipPath = `inset(0 ${{100 - v}}% 0 0)`;
        document.getElementById('l').style.left = v + '%';
        document.getElementById('k').style.left = v + '%';
      }});
    </script>"""
    components.html(html, height=height)


def run_with_progress(fn, expected, label, start_pct=15):
    """Chạy fn ở luồng phụ, luồng chính cập nhật thanh tiến trình theo thời gian ước lượng.

    Model không báo tiến độ thật nên thanh tăng nhanh rồi chậm dần, dừng ở 95% tới khi xong.
    """
    progress = st.progress(start_pct, text=label)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(fn)
        t0 = time.perf_counter()
        while not future.done():
            el = time.perf_counter() - t0
            pct = start_pct + int((95 - start_pct) * (1 - math.exp(-el / expected)))
            progress.progress(pct, text=f"{label} {el:.0f} giây")
            time.sleep(0.3)
        try:
            return future.result()  # ném lại lỗi của luồng phụ (nếu có)
        finally:
            progress.empty()


def test_choices():
    """Danh sách ảnh trong test_images/ để chọn nhanh: {nhãn: (đường dẫn, id cặp hoặc None)}."""
    m = load_manifest()
    if m is None:
        return {}
    out = {}
    for _, r in m.iterrows():
        if r["set"] == "real_damaged":
            out[f"Ảnh hỏng thật · {r['id']}"] = (TEST / r["file"], None)
        else:
            out[f"Ảnh làm hỏng (có ảnh gốc) · {r['id']}"] = (TEST / r["file"], r["id"])
    return out


def pick_image(key):
    """Widget chọn ảnh dùng chung: tải lên hoặc lấy từ bộ test. Trả về (ảnh, tên, id cặp)."""
    src = st.radio("Nguồn ảnh", ["Tải ảnh lên", "Chọn từ bộ test"], horizontal=True, key=f"{key}_src")
    if src == "Tải ảnh lên":
        f = st.file_uploader("Kéo thả ảnh cũ vào đây (JPG, PNG, BMP, WEBP)",
                             type=["jpg", "jpeg", "png", "bmp", "webp"], key=f"{key}_up")
        if f is None:
            return None, None, None
        img = decode(f.getvalue())
        if img is None:
            st.error("Không đọc được file ảnh này. Hãy thử ảnh JPG hoặc PNG khác.")
            return None, None, None
        return img, f.name, None
    choices = test_choices()
    if not choices:
        st.warning("Chưa có test_images/. Chạy `python src/build_test_set.py` trước.")
        return None, None, None
    label = st.selectbox("Ảnh trong bộ test", list(choices), key=f"{key}_sel")
    path, pair_id = choices[label]
    return read_image(path), path.name, pair_id


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
device = get_device()
on_gpu = device.type == "cuda"

with st.sidebar:
    st.markdown("## 🖼️ Thông tin đồ án")
    st.markdown(
        f"""<div class="info-card"><b>{PROJECT_TITLE}</b><br>{COURSE}<br>{SCHOOL}<br>
        🎓 GVHD: {SUPERVISOR}<br>👥 Nhóm 4 sinh viên (xem tab Nhóm)</div>""",
        unsafe_allow_html=True,
    )
    st.markdown("## ⚙️ Cài đặt phục chế")
    model_label = st.selectbox("Mô hình", list(FACE_MODELS))
    scale = st.radio("Tỷ lệ phóng to", [2, 4], format_func=lambda s: f"x{s}", horizontal=True)
    st.markdown("**Tiền xử lý**")
    balance_label = st.selectbox("Cân bằng sáng", list(BALANCE),
                                 help="Tự động: sửa gamma khi ảnh quá tối/sáng, CLAHE khi tương phản thấp.")
    max_side = st.number_input("Giới hạn cạnh dài (px, 0 = không giới hạn)", 0, 8000, 1600, 100,
                               help="Ảnh lớn hơn sẽ được thu nhỏ, tránh hết RAM khi phóng x4.")
    with st.expander("Nâng cao"):
        pad_multiple = st.select_slider("Padding về bội số", [1, 4, 8, 16, 32], value=8)
        tile = st.select_slider("Kích thước ô (tile)", [0, 200, 300, 400, 600, 800], value=400,
                                help="Chia ảnh thành ô nhỏ để đỡ tốn bộ nhớ. Giảm nếu bị lỗi hết bộ nhớ.")
    st.divider()
    if on_gpu:
        import torch

        st.success(f"Thiết bị: GPU ({torch.cuda.get_device_name(0)})")
    else:
        st.info("Thiết bị: CPU (khoảng 1 đến 2 phút cho ảnh 1 MP)")
    st.caption("Lần chạy đầu tự tải model (khoảng 600 MB), cần Internet.")

face_model = FACE_MODELS[model_label]
pre_cfg = PreprocessConfig(balance=BALANCE[balance_label], max_side=int(max_side), pad_multiple=pad_multiple)

st.markdown(
    f"""<div class="hero"><h1>{PROJECT_TITLE}</h1>
    <p>Tiền xử lý OpenCV → GFPGAN khôi phục khuôn mặt → Real-ESRGAN phóng to, làm nét nền
    → đánh giá PSNR / SSIM / LPIPS</p></div>""",
    unsafe_allow_html=True,
)
tab_restore, tab_pre, tab_eval, tab_data, tab_team = st.tabs(
    ["🖼️ Phục chế", "🧪 Tiền xử lý", "📊 Đánh giá PSNR/SSIM", "🗂️ Bộ dữ liệu", "👥 Nhóm & phân công"]
)

# ---------------------------------------------------------------------------
# Tab 1: Phục chế
# ---------------------------------------------------------------------------
with tab_restore:
    mode = st.radio("Chế độ", ["Một ảnh", "Hàng loạt"], horizontal=True, label_visibility="collapsed")

    if mode == "Một ảnh":
        img, name, pair_id = pick_image("restore")
        if img is not None:
            job = (name, img.shape, face_model, scale, tile, str(pre_cfg))
            if st.session_state.get("result", {}).get("job") != job:
                st.session_state.pop("result", None)

            c1, c2 = st.columns([2, 3], gap="large")
            with c1:
                st.image(to_bgr3(img), channels="BGR", caption="Ảnh đầu vào", width="stretch")
            with c2:
                h, w = img.shape[:2]
                st.markdown(f"**{name}** · `{w} × {h}`")
                st.markdown(f"Mô hình: {model_label} · phóng **x{scale}** · cân bằng sáng: {balance_label}")
                start = st.button("✨ Bắt đầu phục chế", type="primary", width="stretch")

            if start:
                t0 = time.perf_counter()
                try:
                    with st.spinner("Đang nạp mô hình..."):
                        restorer = load_restorer(face_model, scale, tile)
                    pre_preview, _ = preprocess(img, pre_cfg)
                    out, info = run_with_progress(
                        lambda: restore_image(restorer, img, pre_cfg),
                        estimate_seconds(pre_preview, scale, on_gpu), "Đang phục chế...")
                    res = {"job": job, "image": out, "info": info, "seconds": time.perf_counter() - t0}
                    if pair_id is not None and scale == 2:  # ảnh có ảnh gốc -> tính luôn PSNR/SSIM
                        gt = read_image(TEST / "gt" / f"{pair_id}.png")
                        o = ev.match_size(out, gt)
                        b = ev.match_size(ev.upscale_bicubic(info["preprocessed"]), gt)
                        res["metrics"] = {"psnr": ev.psnr(gt, o), "ssim": ev.ssim(gt, o),
                                          "psnr_b": ev.psnr(gt, b), "ssim_b": ev.ssim(gt, b), "gt": gt}
                    st.session_state["result"] = res
                    st.toast("Phục chế xong!", icon="✅")
                except RestoreError as e:
                    st.error(str(e))
                except Exception as e:  # lỗi bất ngờ: hiện chi tiết để dễ báo lỗi
                    st.error("Có lỗi không mong muốn khi phục chế.")
                    st.exception(e)
                finally:
                    free_memory()

            res = st.session_state.get("result")
            if res:
                out, info = res["image"], res["info"]
                pre = info["preprocessed"]
                st.divider()
                st.subheader("Kết quả")
                with st.expander("Các bước tiền xử lý đã áp dụng", expanded=False):
                    for s in info["steps"]:
                        st.markdown(f'<div class="step">{s}</div>', unsafe_allow_html=True)
                m = st.columns(4)
                m[0].metric("Kích thước vào", f"{pre.shape[1]}×{pre.shape[0]}")
                m[1].metric("Kích thước ra", f"{out.shape[1]}×{out.shape[0]}")
                m[2].metric("Thời gian", f"{res['seconds']:.1f} s")
                m[3].metric("Khuôn mặt phục chế", "—" if info["faces"] is None else info["faces"])
                if "metrics" in res:
                    mm = res["metrics"]
                    k = st.columns(2)
                    k[0].metric("PSNR so với ảnh gốc", f"{mm['psnr']:.2f} dB",
                                f"{mm['psnr'] - mm['psnr_b']:+.2f} dB so với bicubic")
                    k[1].metric("SSIM so với ảnh gốc", f"{mm['ssim']:.4f}",
                                f"{mm['ssim'] - mm['ssim_b']:+.4f} so với bicubic")

                views = ["🔀 Thanh trượt", "🖼️ Đặt cạnh nhau"] + (["🎯 So với ảnh gốc"] if "metrics" in res else [])
                tabs = st.tabs(views)
                with tabs[0]:
                    compare_slider(pre, out)
                    st.caption("Kéo sang trái/phải để so sánh. Ảnh trước được phóng bicubic cho cùng kích thước.")
                with tabs[1]:
                    a, b = st.columns(2)
                    a.image(pre, channels="BGR", caption="Trước (sau tiền xử lý)", width="stretch")
                    b.image(out, channels="BGR", caption="Sau phục chế", width="stretch")
                if "metrics" in res:
                    with tabs[2]:
                        a, b = st.columns(2)
                        a.image(res["metrics"]["gt"], channels="BGR", caption="Ảnh gốc (ground truth)", width="stretch")
                        b.image(out, channels="BGR", caption="Kết quả phục chế", width="stretch")

                stem = Path(name).stem
                d1, d2 = st.columns(2)
                d1.download_button("⬇️ Tải ảnh về (PNG, không mất dữ liệu)", png_bytes(out),
                                   f"{stem}_restored_x{scale}.png", "image/png", type="primary", width="stretch")
                d2.download_button("⬇️ Tải ảnh về (JPG 95%, nhẹ hơn)",
                                   cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes(),
                                   f"{stem}_restored_x{scale}.jpg", "image/jpeg", width="stretch")

    else:  # Hàng loạt
        files = st.file_uploader("Kéo thả nhiều ảnh cùng lúc", type=["jpg", "jpeg", "png", "bmp", "webp"],
                                 accept_multiple_files=True, key="batch_up")
        st.caption(f"Mỗi ảnh mất khoảng 1 đến 2 phút trên CPU. Đã chọn {len(files or [])} ảnh.")
        if files and st.button("✨ Phục chế tất cả", type="primary"):
            restorer = load_restorer(face_model, scale, tile)
            buf, log = io.BytesIO(), []
            bar = st.progress(0, text="Bắt đầu...")
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                for i, f in enumerate(files):
                    bar.progress(i / len(files), text=f"Đang xử lý {i + 1}/{len(files)}: {f.name}")
                    img = decode(f.getvalue())
                    if img is None:
                        log.append({"Ảnh": f.name, "Trạng thái": "Không đọc được"})
                        continue
                    t0 = time.perf_counter()
                    try:
                        out, info = restore_image(restorer, img, pre_cfg)
                        zf.writestr(f"{Path(f.name).stem}_restored_x{scale}.png", png_bytes(out))
                        log.append({"Ảnh": f.name, "Trạng thái": "Xong", "Khuôn mặt": info["faces"],
                                    "Thời gian (s)": round(time.perf_counter() - t0, 1)})
                    except RestoreError as e:
                        log.append({"Ảnh": f.name, "Trạng thái": f"Lỗi: {e}"})
            bar.progress(1.0, text="Hoàn tất!")
            st.dataframe(pd.DataFrame(log), width="stretch", hide_index=True)
            st.download_button("⬇️ Tải tất cả (ZIP)", buf.getvalue(), "restored_images.zip",
                               "application/zip", type="primary")

# ---------------------------------------------------------------------------
# Tab 2: Tiền xử lý
# ---------------------------------------------------------------------------
with tab_pre:
    st.markdown("Xem từng bước tiền xử lý của **Thành viên 1** trước khi ảnh được đưa vào mô hình AI.")
    img, name, _ = pick_image("pre")
    if img is not None:
        pre, info = preprocess(img, pre_cfg)
        for s in info["steps"]:
            st.markdown(f'<div class="step">{s}</div>', unsafe_allow_html=True)
        before = to_bgr3(img)
        a, b = st.columns(2)
        a.image(before, channels="BGR", caption=f"Gốc · {before.shape[1]}×{before.shape[0]}", width="stretch")
        b.image(pre, channels="BGR", caption=f"Sau tiền xử lý (có padding) · {pre.shape[1]}×{pre.shape[0]}",
                width="stretch")
        m0, s0 = brightness_stats(before)
        m1, s1 = brightness_stats(pre)
        k = st.columns(2)
        k[0].metric("Độ sáng trung bình (kênh L)", f"{m1:.2f}", f"{m1 - m0:+.2f}")
        k[1].metric("Độ tương phản (độ lệch chuẩn L)", f"{s1:.2f}", f"{s1 - s0:+.2f}")

        def hist_L(x):
            L = cv2.cvtColor(x, cv2.COLOR_BGR2LAB)[..., 0]
            return np.bincount(L.ravel(), minlength=256) / L.size

        st.markdown("**Histogram kênh độ sáng L** (không gian màu LAB)")
        st.line_chart(pd.DataFrame({"Gốc": hist_L(before), "Sau tiền xử lý": hist_L(pre)}),
                      height=230, color=["#9aa5b1", "#c08a4a"])

# ---------------------------------------------------------------------------
# Tab 3: Đánh giá
# ---------------------------------------------------------------------------
with tab_eval:
    st.markdown(
        "Ảnh gốc sạch được **làm hỏng nhân tạo** (mờ Gauss + thu nhỏ x2 + nhiễu + nén JPEG), rồi phục chế "
        "về lại kích thước gốc bằng 4 phương pháp. **PSNR**, **SSIM** (kênh Y, càng cao càng tốt) và **LPIPS** "
        "(khoảng cách cảm nhận, càng thấp càng tốt) đo độ giống với ảnh gốc."
    )
    summary, metrics = load_csv("summary.csv"), load_csv("metrics.csv")
    if summary is None:
        st.info("Chưa có kết quả đánh giá. Bấm nút bên dưới hoặc chạy `python src/evaluate.py`.")
    else:
        best_p = summary.loc[summary["psnr_mean"].idxmax()]
        msg = (f"Trên {int(summary['n'].iloc[0])} cặp ảnh: PSNR cao nhất là **{best_p['label']}** "
               f"({best_p['psnr_mean']:.2f} dB)")
        if "lpips_mean" in summary:
            best_l = summary.loc[summary["lpips_mean"].idxmin()]
            msg += f"; LPIPS thấp nhất (giống ảnh gốc nhất theo cảm nhận) là **{best_l['label']}** ({best_l['lpips_mean']:.3f})"
        st.success(msg + ".")
        st.caption("PSNR/SSIM đo độ trùng khớp từng điểm ảnh nên thường ưu ái ảnh mờ như Bicubic; LPIPS đo độ giống "
                   "theo cảm nhận của người xem. Cần đọc cả ba chỉ số cùng với ảnh minh họa.")
        show = summary.rename(columns={"label": "Phương pháp", "n": "Số ảnh", "psnr_mean": "PSNR TB (dB)",
                                       "psnr_std": "± PSNR", "ssim_mean": "SSIM TB", "ssim_std": "± SSIM",
                                       "lpips_mean": "LPIPS TB ↓", "lpips_std": "± LPIPS",
                                       "seconds_mean": "Giây/ảnh"}).drop(columns="method")
        st.dataframe(show, width="stretch", hide_index=True)
        charts = [("psnr_mean", "PSNR trung bình (dB) ↑", "#c08a4a"), ("ssim_mean", "SSIM trung bình ↑", "#7a5230")]
        if "lpips_mean" in summary:
            charts.append(("lpips_mean", "LPIPS trung bình ↓", "#5b7fa6"))
        for c, (key, title, color) in zip(st.columns(len(charts)), charts):
            c.markdown(f"**{title}**")
            c.bar_chart(summary.set_index("label")[key], height=220, color=color, horizontal=True, sort=False)

        if metrics is not None:
            st.markdown("#### Chi tiết từng ảnh")
            piv = metrics.pivot(index="id", columns="method", values="psnr")
            piv = piv[[m for m in ev.METHODS if m in piv.columns]].rename(columns=ev.METHOD_LABELS)
            st.dataframe(piv.round(2), width="stretch", height=260)
            pid = st.selectbox("Xem ảnh", list(piv.index))
            row = metrics[metrics["id"] == pid].set_index("method")
            cols = st.columns(2 + len(row))
            cols[0].image(read_image(TEST / "gt" / f"{pid}.png"), channels="BGR", caption="Ảnh gốc", width="stretch")
            cols[1].image(read_image(TEST / "degraded" / f"{pid}.png"), channels="BGR",
                          caption="Ảnh làm hỏng (nhỏ hơn 2 lần)", width="stretch")
            for c, m in zip(cols[2:], [m for m in ev.METHODS if m in row.index]):
                p = RESULTS / "pairs" / m / f"{pid}.png"
                if p.exists():
                    extra = f" · LPIPS {row.loc[m, 'lpips']:.3f}" if "lpips" in row.columns else ""
                    c.image(read_image(p), channels="BGR", width="stretch",
                            caption=f"{ev.METHOD_LABELS[m]}\n{row.loc[m, 'psnr']:.2f} dB · SSIM {row.loc[m, 'ssim']:.3f}{extra}")

    with st.expander("▶️ Chạy lại đánh giá"):
        man = load_manifest()
        n_pairs = 0 if man is None else int((man["set"] == "pair").sum())
        if n_pairs == 0:
            st.warning("Chưa có cặp ảnh. Chạy `python src/build_test_set.py` trước.")
        else:
            limit = st.slider("Số cặp ảnh", 1, n_pairs, min(5, n_pairs))
            methods = st.multiselect("Phương pháp", ev.METHODS, ev.METHODS, format_func=ev.METHOD_LABELS.get)
            st.caption("Trên CPU, mỗi cặp mất khoảng 10 đến 30 giây cho 2 phương pháp AI.")
            if st.button("Chạy đánh giá", type="primary") and methods:
                rows = man[man["set"] == "pair"].head(limit).to_dict("records")
                logs = []
                box = st.empty()
                with ThreadPoolExecutor(max_workers=1) as pool:
                    fut = pool.submit(ev.run_pairs, rows, methods, logs.append)
                    while not fut.done():
                        box.code("\n".join(logs[-8:]) or "Đang nạp mô hình...")
                        time.sleep(0.5)
                    try:
                        res = fut.result()
                        ev.write_csv(RESULTS / "metrics.csv", res)
                        ev.write_csv(RESULTS / "summary.csv", ev.summarize(res))
                        st.success("Xong! Đang tải lại kết quả...")
                        st.rerun()
                    except Exception as e:
                        st.exception(e)

    real = load_csv("real_times.csv")
    if real is not None:
        st.markdown("#### Ảnh hỏng thật (không có ảnh gốc, đánh giá bằng mắt)")
        k = st.columns(3)
        k[0].metric("Số ảnh", len(real))
        k[1].metric("Thời gian TB", f"{real['seconds'].mean():.1f} s/ảnh")
        k[2].metric("Ảnh dò được mặt", f"{(real['faces'] > 0).sum()}/{len(real)}")
        rid = st.selectbox("Chọn ảnh hỏng thật", list(real["id"]))
        p = RESULTS / "real" / f"{rid}.png"
        if p.exists():
            r = real.set_index("id").loc[rid]
            before, _ = preprocess(read_image(TEST / "real_damaged" / f"{rid}.png"),
                                   PreprocessConfig(balance="auto", max_side=768, pad_multiple=1))
            compare_slider(before, read_image(p), height=520)
            st.caption(f"Khuôn mặt: {r['faces']} · {r['seconds']} s · độ nét (phương sai Laplacian) "
                       f"{r['lap_bicubic']} → {r['lap_restored']} · tiền xử lý: {r['preprocess']}")

# ---------------------------------------------------------------------------
# Tab 4: Bộ dữ liệu
# ---------------------------------------------------------------------------
with tab_data:
    man = load_manifest()
    if man is None:
        st.warning("Chưa có test_images/. Chạy `python src/build_test_set.py` trước.")
    else:
        k = st.columns(3)
        k[0].metric("Ảnh hỏng thật", int((man["set"] == "real_damaged").sum()))
        k[1].metric("Cặp gốc / làm hỏng", int((man["set"] == "pair").sum()))
        k[2].metric("Tổng số ảnh", len(man) + int((man["set"] == "pair").sum()))
        st.caption("Nguồn: bộ ảnh Kaggle *low-resolution photographs* (archive.zip), đã được Thành viên 1 "
                   "phân loại và chuẩn hóa vào thư mục test_images/.")
        which = st.radio("Xem", ["Ảnh hỏng thật", "Ảnh gốc", "Ảnh làm hỏng"], horizontal=True)
        folder = {"Ảnh hỏng thật": "real_damaged", "Ảnh gốc": "gt", "Ảnh làm hỏng": "degraded"}[which]
        files = sorted((TEST / folder).glob("*.png"))
        cols = st.columns(6)
        for i, f in enumerate(files):
            cols[i % 6].image(read_image(f), channels="BGR", caption=f.stem, width="stretch")
        with st.expander("manifest.csv"):
            st.dataframe(man, width="stretch", hide_index=True)

# ---------------------------------------------------------------------------
# Tab 5: Nhóm
# ---------------------------------------------------------------------------
with tab_team:
    st.markdown(f"### {COURSE}\n**{SCHOOL}** · GVHD: **{SUPERVISOR}**")
    st.dataframe(pd.DataFrame(TEAM, columns=["Họ và tên", "MSSV", "Vai trò", "Nhiệm vụ"]),
                 width="stretch", hide_index=True)
    st.markdown("#### Kiến trúc hệ thống")
    st.markdown(
        """
| Bước | Mô-đun | Thành viên |
|---|---|---|
| Dữ liệu kiểm thử `test_images/` | `build_test_set.py` | TV1 |
| Chuẩn hóa màu, cân bằng sáng, resize + padding | `preprocessing.py` | TV1 |
| GFPGAN dò & khôi phục mặt, Real-ESRGAN phóng to nền | `model_core.py` | TV2 |
| Giao diện, so sánh Trước/Sau, tải ảnh | `app.py` | TV3 |
| PSNR / SSIM, báo cáo | `evaluate.py` | TV4 |
"""
    )
