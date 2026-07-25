"""
=============================================================================
 app.py — AUTOMATED BOTTLE QUALITY INSPECTION (v3)
 Smart Retail Quality Inspection — Deteksi Kerusakan Botol Minuman
 Versi: Streamlit Community Cloud (deploy dari GitHub, model dari HF Hub)
=============================================================================
PENTING - KONSISTENSI PIPELINE (TIDAK BOLEH BERBEDA DARI TAHAP 5 & TAHAP 6):
  Semua hal berikut PERSIS SAMA dengan Tahap 5 (Modeling) & Tahap 6 (Evaluasi):
    - CLASS_NAMES_LIST & urutannya, CLASS_COLOR (lihat utils.py)
    - Threshold operasional default = 0.40 (sama dengan VIZ_CONF_THRESHOLD
      di Tahap 6, dipakai utk confusion matrix/P/R/F1 di laporan evaluasi)
    - Cara memanggil model: RFDETRNano(pretrain_weights=...) lalu
      model.predict(image, threshold=...) — lihat utils.run_inference()
  File ini HANYA mengurus: UI Streamlit, loading model (HF Hub), dan
  pipeline prediksi yang memanggil utils.py. Tidak ada logic deteksi baru.

v3 — PERUBAHAN DARI v2 (murni tampilan & UX, TIDAK menyentuh logic inferensi):
  - Desain ulang total: gaya "industrial AI inspection dashboard", bukan
    tampilan default Streamlit. Tanpa emoji, tanpa ikon robot AI.
  - Sidebar DIHAPUS — semua kontrol (threshold, legenda kelas) dipindah ke
    panel konfigurasi horizontal di halaman utama.
  - Bounding box lebih tebal + label lebih besar & mudah dibaca (lihat
    utils.draw_detections / utils.get_font).
  - Confidence score ditampilkan besar & jelas (angka utama + bar per kelas
    via utils.render_detection_table_html), bukan angka kecil di tabel.
  - Status akhir eksplisit: PASS (tidak ada cacat di atas threshold) /
    FAIL (ada cacat terdeteksi) — ditampilkan sbg kartu status besar.
=============================================================================
"""

import os
import io
import time
import json
import shutil
import zipfile

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from utils import (
    CLASS_NAMES_LIST, CLASS_COLOR, VALID_IMAGE_EXTS,
    run_inference, draw_detections, pil_to_bytes,
    render_detection_table_html, render_defect_chips_html,
    extract_images_from_zip,
)

# =============================================================================
# 0. KONFIGURASI — TIDAK DIUBAH DARI VERSI SEBELUMNYA
# =============================================================================

HF_MODEL_REPO_ID = "dody019/rfdetr-nano-bottle-defect"
HF_CHECKPOINT_FILENAME = "best.pth"
HF_METADATA_FILENAME = "metadata.json"

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")
MODEL_PATH = os.path.join(MODEL_DIR, "best.pth")
METADATA_PATH = os.path.join(MODEL_DIR, "metadata.json")
EXAMPLES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "examples")

DEFAULT_CONF_THRESHOLD = 0.40
IOU_MATCH_THRESHOLD = 0.50
PRIMARY_FIX_TARGET_CLASS = "Dent"
PRIMARY_FIX_TARGET_MIN_AP_WARNING = 0.10
MAX_ZIP_IMAGES = 200


# =============================================================================
# 1. MODEL LOADING — TIDAK DIUBAH DARI VERSI SEBELUMNYA
# =============================================================================

@st.cache_resource(show_spinner=False)
def ensure_checkpoint():
    os.makedirs(MODEL_DIR, exist_ok=True)
    if not os.path.isfile(MODEL_PATH):
        from huggingface_hub import hf_hub_download
        downloaded = hf_hub_download(repo_id=HF_MODEL_REPO_ID, filename=HF_CHECKPOINT_FILENAME)
        shutil.copy(downloaded, MODEL_PATH)
    if not os.path.isfile(METADATA_PATH):
        try:
            from huggingface_hub import hf_hub_download
            downloaded_meta = hf_hub_download(repo_id=HF_MODEL_REPO_ID, filename=HF_METADATA_FILENAME)
            shutil.copy(downloaded_meta, METADATA_PATH)
        except Exception:
            pass
    return True


@st.cache_resource(show_spinner=False)
def load_model():
    from rfdetr import RFDETRNano
    return RFDETRNano(pretrain_weights=MODEL_PATH)


@st.cache_data(show_spinner=False)
def load_metadata():
    if os.path.isfile(METADATA_PATH):
        with open(METADATA_PATH) as f:
            return json.load(f)
    return {}


# =============================================================================
# 2. PAGE CONFIG + DESIGN SYSTEM (CSS)
# =============================================================================

st.set_page_config(
    page_title="Automated Bottle Quality Inspection",
    page_icon="▪",
    layout="wide",
    initial_sidebar_state="collapsed",
)

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

:root{
    --bg:#F3F5F8;
    --surface:#FFFFFF;
    --border:#E2E6EC;
    --navy:#0F1F3D;
    --navy-soft:#16294D;
    --accent:#2F6FED;
    --accent-soft:#EAF1FF;
    --text:#10131A;
    --text-muted:#5B6472;
    --pass:#14804A;
    --pass-bg:#E7F6EE;
    --fail:#C4293B;
    --fail-bg:#FDEAEC;
    --warn:#B4790A;
    --warn-bg:#FDF3E1;
    --radius:10px;
    --shadow:0 1px 2px rgba(16,19,26,0.04), 0 1px 12px rgba(16,19,26,0.05);
    --shadow-hover:0 4px 20px rgba(16,19,26,0.10);
}

html, body, [class*="css"]{
    font-family:'IBM Plex Sans', -apple-system, sans-serif;
    color:var(--text);
}
.stApp{ background:var(--bg); overflow-x:hidden; }
#MainMenu, footer{ visibility:hidden; height:0; }
header[data-testid="stHeader"]{ display:none !important; }
.block-container{ padding-top:0 !important; max-width:1180px; }

h1,h2,h3,h4{ font-family:'IBM Plex Sans', sans-serif; font-weight:700; letter-spacing:-0.01em; color:var(--text); }
code, .mono{ font-family:'IBM Plex Mono', monospace; }

/* ---------- Top status bar ---------- */
.topbar{
    background:var(--navy); color:#C9D4E8; padding:9px 4px; margin:0 -100vw;
    padding-left:calc(100vw - 100% + 4px); padding-right:calc(100vw - 100% + 4px);
    font-family:'IBM Plex Mono', monospace; font-size:12px; letter-spacing:0.04em;
    display:flex; justify-content:space-between; align-items:center;
}
.topbar .brand{ color:#FFFFFF; font-weight:600; letter-spacing:0.08em; }
.status-dot{ display:inline-block; width:7px; height:7px; border-radius:50%; margin-right:6px; background:#3ECF8E; box-shadow:0 0 0 3px rgba(62,207,142,0.18); }

/* ---------- Hero ---------- */
.hero-wrap{ background:var(--navy); margin:0 -100vw 28px; padding:44px calc(100vw - 100% + 4px) 52px; }
.hero-inner{ max-width:1180px; margin:0 auto; display:flex; gap:40px; align-items:center; flex-wrap:wrap; }
.hero-left{ flex:1 1 460px; }
.hero-eyebrow{ font-family:'IBM Plex Mono',monospace; font-size:12px; letter-spacing:0.12em; color:#7C9BE0; text-transform:uppercase; margin-bottom:14px; }
.hero-title{ font-size:38px; line-height:1.15; color:#FFFFFF; font-weight:700; margin:0 0 14px; letter-spacing:-0.02em; }
.hero-sub{ font-size:15.5px; color:#B7C3DC; line-height:1.6; max-width:480px; margin-bottom:22px; }
.tag-row{ display:flex; gap:8px; flex-wrap:wrap; }
.tag-chip{ font-size:12px; font-family:'IBM Plex Mono',monospace; color:#DCE5F5; background:rgba(255,255,255,0.07); border:1px solid rgba(255,255,255,0.14); padding:6px 12px; border-radius:100px; }

.hero-right{ flex:1 1 340px; display:flex; justify-content:center; }
.scan-card{ position:relative; width:280px; height:280px; background:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.12); border-radius:16px; overflow:hidden; display:flex; align-items:center; justify-content:center; }
.scan-line{ position:absolute; left:0; right:0; height:2px; background:linear-gradient(90deg, transparent, #63A6FF, transparent); animation:scan 3.2s linear infinite; }
@keyframes scan{ 0%{ top:8%; } 50%{ top:88%; } 100%{ top:8%; } }
.scan-corner{ position:absolute; width:22px; height:22px; border-color:#63A6FF; }
.scan-corner.tl{ top:14px; left:14px; border-top:2px solid; border-left:2px solid; }
.scan-corner.tr{ top:14px; right:14px; border-top:2px solid; border-right:2px solid; }
.scan-corner.bl{ bottom:14px; left:14px; border-bottom:2px solid; border-left:2px solid; }
.scan-corner.br{ bottom:14px; right:14px; border-bottom:2px solid; border-right:2px solid; }
@media (prefers-reduced-motion: reduce){ .scan-line{ animation:none; top:50%; } }

/* ---------- Cards / containers ---------- */
div[data-testid="stVerticalBlockBorderWrapper"]{
    background:var(--surface); border:1px solid var(--border) !important;
    border-radius:var(--radius) !important; box-shadow:var(--shadow);
    transition:box-shadow .18s ease, transform .18s ease;
}
div[data-testid="stVerticalBlockBorderWrapper"]:hover{ box-shadow:var(--shadow-hover); }

.section-title{ font-size:13px; font-family:'IBM Plex Mono',monospace; letter-spacing:0.1em; text-transform:uppercase; color:var(--text-muted); margin:34px 5px 12px; padding-bottom:8px; border-bottom:1px solid var(--border); }

/* ---------- Config panel ---------- */
.config-label{ font-size:12px; font-family:'IBM Plex Mono',monospace; letter-spacing:0.06em; text-transform:uppercase; color:var(--text-muted); margin-bottom:6px; }
.legend-row{ display:flex; gap:8px; flex-wrap:wrap; margin-top:4px; }
.legend-chip{ display:flex; align-items:center; gap:7px; font-size:12.5px; color:var(--text); background:var(--bg); border:1px solid var(--border); padding:5px 10px; border-radius:100px; }
.legend-dot{ width:9px; height:9px; border-radius:50%; }

/* ---------- Status card ---------- */
.status-card{ display:flex; align-items:center; gap:18px; padding:20px 22px; border-radius:var(--radius); border:1px solid var(--border); margin-bottom:16px; }
.status-card.pass{ background:var(--pass-bg); border-color:#BEE6CE; }
.status-card.fail{ background:var(--fail-bg); border-color:#F3C3CB; }
.status-bar{ width:5px; align-self:stretch; border-radius:4px; }
.status-bar.pass{ background:var(--pass); }
.status-bar.fail{ background:var(--fail); }
.status-label{ font-family:'IBM Plex Mono',monospace; font-size:12px; letter-spacing:0.1em; color:var(--text-muted); text-transform:uppercase; }
.status-value{ font-family:'IBM Plex Mono',monospace; font-size:30px; font-weight:600; letter-spacing:0.02em; line-height:1.2; }
.status-value.pass{ color:var(--pass); }
.status-value.fail{ color:var(--fail); }
.status-meta{ margin-left:auto; text-align:right; }
.status-meta .conf-num{ font-family:'IBM Plex Mono',monospace; font-size:26px; font-weight:600; color:var(--text); }
.status-meta .conf-label{ font-size:11.5px; color:var(--text-muted); font-family:'IBM Plex Mono',monospace; letter-spacing:0.06em; text-transform:uppercase; }

.defect-chip{ display:inline-block; font-size:12px; font-family:'IBM Plex Mono',monospace; padding:5px 11px; border-radius:100px; border:1px solid; margin:2px 5px 2px 0; }
.defect-chip-none{ color:var(--text-muted); background:var(--bg); border-color:var(--border); }

/* ---------- Detection table (custom) ---------- */
.det-table{ display:flex; flex-direction:column; gap:9px; margin-top:6px; }
.det-row{ display:grid; grid-template-columns:150px 1fr 60px; align-items:center; gap:12px; }
.det-class{ display:flex; align-items:center; gap:8px; font-size:13.5px; font-weight:500; }
.det-dot{ width:9px; height:9px; border-radius:50%; flex-shrink:0; }
.det-bar-track{ background:var(--bg); border:1px solid var(--border); border-radius:6px; height:10px; overflow:hidden; }
.det-bar-fill{ height:100%; border-radius:6px 0 0 6px; }
.det-pct{ font-family:'IBM Plex Mono',monospace; font-size:13px; text-align:right; color:var(--text); }

/* ---------- Metric tiles (batch summary) ---------- */
.metric-tile{ background:var(--surface); border:1px solid var(--border); border-radius:var(--radius); padding:16px 18px; box-shadow:var(--shadow); }
.metric-tile .metric-value{ font-family:'IBM Plex Mono',monospace; font-size:26px; font-weight:600; color:var(--text); }
.metric-tile .metric-label{ font-size:12px; color:var(--text-muted); margin-top:2px; }
.metric-tile.pass .metric-value{ color:var(--pass); }
.metric-tile.fail .metric-value{ color:var(--fail); }

/* ---------- Model info table ---------- */
.info-table{ width:100%; border-collapse:collapse; }
.info-table tr{ border-bottom:1px solid var(--border); }
.info-table tr:last-child{ border-bottom:none; }
.info-table td{ padding:11px 4px; font-size:13.5px; vertical-align:top; }
.info-table td.k{ width:190px; font-family:'IBM Plex Mono',monospace; font-size:12px; color:var(--text-muted); letter-spacing:0.04em; text-transform:uppercase; }
.info-table td.v{ color:var(--text); }

/* ---------- Streamlit widget overrides ---------- */
.stTabs [data-baseweb="tab-list"]{ gap:4px; border-bottom:1px solid var(--border); }
.stTabs [data-baseweb="tab"]{ height:40px; font-family:'IBM Plex Sans'; font-weight:500; font-size:13.5px; color:var(--text-muted); }
.stTabs [aria-selected="true"]{ color:var(--accent) !important; }
.stTabs [data-baseweb="tab-highlight"]{ background-color:var(--accent) !important; }

.stButton>button{ border-radius:8px; border:1px solid var(--border); font-weight:500; font-size:13.5px; transition:transform .12s ease, box-shadow .12s ease; }
.stButton>button:hover{ transform:translateY(-1px); box-shadow:0 3px 10px rgba(16,19,26,0.08); }
.stButton>button[kind="primary"]{ background:var(--navy); border-color:var(--navy); }

[data-testid="stFileUploaderDropzone"]{ background:var(--bg); border:1.5px dashed #C6CFDC !important; border-radius:var(--radius); }
[data-testid="stFileUploaderDropzone"]:hover{ border-color:var(--accent) !important; }

[data-testid="stDataFrame"]{ border-radius:8px; overflow:hidden; border:1px solid var(--border); }

.footer-note{ text-align:center; font-size:12px; color:var(--text-muted); font-family:'IBM Plex Mono',monospace; padding:30px 0 10px; letter-spacing:0.03em; }
/* --- FIX: hilangkan garis dekorasi default & padding atas Streamlit --- */
div[data-testid="stDecoration"]{ display:none !important; }
div[data-testid="stMainBlockContainer"],
div[data-testid="stAppViewBlockContainer"],
.main .block-container{
    padding-top:0 !important;
}

/* --- FIX: hilangkan scroll horizontal di semua level container --- */
html, body,
section[data-testid="stMain"],
div[data-testid="stAppViewContainer"]{
    overflow-x:hidden !important;
}

div[data-testid="stAppViewContainer"]{ padding-top:0 !important; }
div[data-testid="stAppViewBlockContainer"],
div[data-testid="stMainBlockContainer"]{ padding-top:0 !important; margin-top:0 !important; }
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)

# =============================================================================
# 3. LOAD MODEL (dgn status bar di atas)
# =============================================================================

if "<username>" in HF_MODEL_REPO_ID:
    st.error(
        "HF_MODEL_REPO_ID di app.py masih placeholder. Ganti dengan repo model "
        "Hugging Face yang berisi best.pth, lalu deploy ulang."
    )
    st.stop()

try:
    with st.spinner("Preparing inspection model..."):
        ensure_checkpoint()
        model = load_model()
        metadata = load_metadata()
except Exception as e:
    st.markdown(
        '<div class="topbar"><span><span class="status-dot" '
        'style="background:#E4574A;box-shadow:0 0 0 3px rgba(228,87,74,0.18);"></span>'
        'MODEL STATUS: OFFLINE</span><span class="brand">QUALITY INSPECTION SYSTEM</span></div>',
        unsafe_allow_html=True,
    )
    st.error(
        f"Failed to load model checkpoint from Hugging Face Hub: {e}\n\n"
        "Check that HF_MODEL_REPO_ID is correct, the model repo is public "
        "(or a valid token is set in Streamlit Secrets), and best.pth exists in that repo."
    )
    st.stop()

st.markdown(
    '<div class="topbar"><span><span class="status-dot"></span>MODEL STATUS: ONLINE — RF-DETR-NANO</span>'
    '<span class="brand">QUALITY INSPECTION SYSTEM</span></div>',
    unsafe_allow_html=True,
)

# =============================================================================
# 4. HERO SECTION
# =============================================================================

st.markdown(
    """
    <div class="hero-wrap">
      <div class="hero-inner">
        <div class="hero-left">
          <div class="hero-eyebrow">Retail Packaging &middot; Visual Inspection</div>
          <div class="hero-title">Automated Bottle<br>Quality Inspection</div>
          <div class="hero-sub">AI-powered visual inspection system for detecting packaging
          defects in retail products &mdash; dent, label damage, missing cap, and seal damage &mdash;
          from a single product photo or a full inspection batch.</div>
          <div class="tag-row">
            <span class="tag-chip">Computer Vision</span>
            <span class="tag-chip">Object Detection</span>
            <span class="tag-chip">Quality Control Automation</span>
          </div>
        </div>
        <div class="hero-right">
          <div class="scan-card">
            <div class="scan-corner tl"></div><div class="scan-corner tr"></div>
            <div class="scan-corner bl"></div><div class="scan-corner br"></div>
            <div class="scan-line"></div>
            <svg width="120" height="180" viewBox="0 0 120 180" fill="none" xmlns="http://www.w3.org/2000/svg">
              <rect x="44" y="8" width="32" height="18" rx="3" stroke="#7C9BE0" stroke-width="2.5"/>
              <path d="M46 26 L46 44 L36 62 L36 168 Q36 174 42 174 L78 174 Q84 174 84 168 L84 62 L74 44 L74 26"
                    stroke="#B7C3DC" stroke-width="2.5" stroke-linejoin="round"/>
              <line x1="36" y1="94" x2="84" y2="94" stroke="#3A5586" stroke-width="1.5" stroke-dasharray="3 3"/>
            </svg>
          </div>
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# =============================================================================
# 5. CONFIGURATION PANEL (menggantikan sidebar)
# =============================================================================

st.markdown('<div class="section-title">Inspection Configuration</div>', unsafe_allow_html=True)
with st.container(border=True):
    cfg_col1, cfg_col2, cfg_col3 = st.columns([1.1, 1.6, 0.9])
    with cfg_col1:
        st.markdown('<div class="config-label">Confidence Threshold</div>', unsafe_allow_html=True)
        conf_threshold = st.slider(
            "Confidence threshold", min_value=0.05, max_value=0.95,
            value=DEFAULT_CONF_THRESHOLD, step=0.05, label_visibility="collapsed",
        )
        st.caption(f"Matches the operational threshold used in final test-set evaluation ({DEFAULT_CONF_THRESHOLD:.2f}).")
    with cfg_col2:
        st.markdown('<div class="config-label">Defect Classes</div>', unsafe_allow_html=True)
        legend_html = "".join(
            f'<div class="legend-chip"><span class="legend-dot" style="background:{CLASS_COLOR[n]};"></span>{n}</div>'
            for n in CLASS_NAMES_LIST
        )
        st.markdown(f'<div class="legend-row">{legend_html}</div>', unsafe_allow_html=True)
    with cfg_col3:
        st.markdown('<div class="config-label">Session</div>', unsafe_allow_html=True)
        if st.button("Clear results", use_container_width=True,
                      disabled=not st.session_state.get("batch_results")):
            st.session_state.pop("batch_results", None)
            st.rerun()

if "batch_results" not in st.session_state:
    st.session_state.batch_results = []


# =============================================================================
# 6. RENDER SATU HASIL INSPEKSI (dipakai oleh semua mode upload)
# =============================================================================

def process_and_show(image: Image.Image, name: str, conf_threshold: float, key_prefix: str):
    t0 = time.time()
    df_det = run_inference(model, image, conf_threshold)
    elapsed = time.time() - t0
    annotated = draw_detections(image, df_det)

    is_fail = not df_det.empty
    status_word = "FAIL" if is_fail else "PASS"
    status_class = "fail" if is_fail else "pass"
    max_conf = float(df_det["confidence"].max()) if is_fail else 0.0
    detected_classes = sorted(df_det["class_name"].unique().tolist()) if is_fail else []

    with st.container(border=True):
        st.markdown(f"**{name}**")

        st.markdown(
            f"""
            <div class="status-card {status_class}">
              <div class="status-bar {status_class}"></div>
              <div>
                <div class="status-label">Inspection Status</div>
                <div class="status-value {status_class}">{status_word}</div>
              </div>
              <div class="status-meta">
                <div class="conf-num">{max_conf*100:.1f}%</div>
                <div class="conf-label">Max Confidence</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f'<div class="config-label" style="margin-bottom:8px;">Detected Defects</div>'
            f'{render_defect_chips_html(detected_classes)}',
            unsafe_allow_html=True,
        )

        img_col1, img_col2 = st.columns(2)
        with img_col1:
            st.markdown('<div class="config-label">Original Image</div>', unsafe_allow_html=True)
            st.image(image, use_container_width=True)
        with img_col2:
            st.markdown(f'<div class="config-label">Detection Result &nbsp;&middot;&nbsp; {elapsed*1000:.0f} ms</div>', unsafe_allow_html=True)
            st.image(annotated, use_container_width=True)

        if is_fail:
            st.markdown('<div class="config-label" style="margin-top:14px;">Confidence per Detection</div>', unsafe_allow_html=True)
            st.markdown(render_detection_table_html(df_det), unsafe_allow_html=True)

        dl1, dl2 = st.columns(2)
        with dl1:
            st.download_button(
                "Download annotated image", data=pil_to_bytes(annotated),
                file_name=f"inspection_{name}.png", mime="image/png",
                use_container_width=True, key=f"dl_img_{key_prefix}_{name}",
            )
        with dl2:
            st.download_button(
                "Download detections (CSV)", data=df_det.to_csv(index=False).encode("utf-8"),
                file_name=f"detections_{name}.csv", mime="text/csv",
                use_container_width=True, key=f"dl_csv_{key_prefix}_{name}",
            )

    st.session_state.batch_results.append({
        "name": name, "df_det": df_det, "annotated_bytes": pil_to_bytes(annotated),
        "elapsed_ms": elapsed * 1000, "status": status_word,
    })
    return df_det


# =============================================================================
# 7. BATCH SUMMARY
# =============================================================================

def render_batch_summary():
    results = st.session_state.batch_results
    if not results:
        return

    all_det = pd.concat(
        [r["df_det"].assign(file_name=r["name"]) for r in results if not r["df_det"].empty],
        ignore_index=True,
    ) if any(not r["df_det"].empty for r in results) else pd.DataFrame(
        columns=["class_name", "confidence", "x1", "y1", "x2", "y2", "file_name"]
    )

    n_images = len(results)
    n_fail = sum(1 for r in results if r["status"] == "FAIL")
    n_pass = n_images - n_fail
    avg_time = np.mean([r["elapsed_ms"] for r in results])

    st.markdown('<div class="section-title">Batch Summary</div>', unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    for col, val, label, css in [
        (m1, n_images, "Images Inspected", ""),
        (m2, n_pass, "Passed", "pass"),
        (m3, n_fail, "Failed", "fail"),
        (m4, f"{avg_time:.0f} ms", "Avg. Inference Time", ""),
    ]:
        col.markdown(
            f'<div class="metric-tile {css}"><div class="metric-value">{val}</div>'
            f'<div class="metric-label">{label}</div></div>',
            unsafe_allow_html=True,
        )

    if not all_det.empty:
        st.write("")
        st.markdown('<div class="config-label">Defect Distribution (This Batch)</div>', unsafe_allow_html=True)
        per_class_count = (
            all_det["class_name"].value_counts()
            .reindex(CLASS_NAMES_LIST, fill_value=0)
            .rename_axis("class_name").reset_index(name="count")
            .set_index("class_name")
        )
        st.bar_chart(per_class_count, use_container_width=True, height=220)

        with st.expander("View combined detection table"):
            st.dataframe(all_det, use_container_width=True, hide_index=True)

    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for r in results:
            zf.writestr(f"annotated/{r['name']}.png", r["annotated_bytes"])
        zf.writestr("detections_combined.csv", all_det.to_csv(index=False))
    zip_buf.seek(0)

    st.download_button(
        "Download all results (images + CSV) as .zip",
        data=zip_buf.getvalue(), file_name="inspection_batch_results.zip",
        mime="application/zip", use_container_width=True,
    )


# =============================================================================
# 8. UPLOAD TABS
# =============================================================================

st.markdown('<div class="section-title">Product Inspection</div>', unsafe_allow_html=True)
tab_upload, tab_zip, tab_examples = st.tabs(["Upload Image", "Upload ZIP Batch", "Sample Images"])

processed_this_run = False

with tab_upload:
    with st.container(border=True):
        st.markdown('<div class="config-label">Upload Product Image</div>', unsafe_allow_html=True)
        st.caption("Supports JPG, PNG, BMP. Multiple files can be uploaded and inspected at once.")
        uploaded_files = st.file_uploader(
            "Upload product image", type=["jpg", "jpeg", "png", "bmp"],
            accept_multiple_files=True, label_visibility="collapsed", key="uploader_images",
        )
    if uploaded_files:
        for uf in uploaded_files:
            image = Image.open(uf)
            process_and_show(image, os.path.splitext(uf.name)[0], conf_threshold, key_prefix="img")
        processed_this_run = True

with tab_zip:
    with st.container(border=True):
        st.markdown('<div class="config-label">Upload Inspection Batch (.zip)</div>', unsafe_allow_html=True)
        st.caption(f"Upload a .zip file containing multiple product images (up to {MAX_ZIP_IMAGES} per batch).")
        uploaded_zip = st.file_uploader(
            "Upload zip batch", type=["zip"], label_visibility="collapsed", key="uploader_zip",
        )
    if uploaded_zip is not None:
        with st.spinner("Extracting batch archive..."):
            zip_images, skipped = extract_images_from_zip(uploaded_zip, max_images=MAX_ZIP_IMAGES)

        if skipped == ["__BAD_ZIP__"]:
            st.error("The uploaded file is not a valid .zip archive.")
        else:
            if skipped:
                st.warning(f"{len(skipped)} file(s) in the archive were skipped (not a valid image).")
            if not zip_images:
                st.info("No valid images (.jpg/.jpeg/.png/.bmp) found in the archive.")
            else:
                progress = st.progress(0.0, text="Processing batch...")
                for i, (name, image) in enumerate(zip_images):
                    process_and_show(image, name, conf_threshold, key_prefix="zip")
                    progress.progress((i + 1) / len(zip_images), text=f"Processing image {i+1}/{len(zip_images)}")
                progress.empty()
                processed_this_run = True

with tab_examples:
    if os.path.isdir(EXAMPLES_DIR):
        example_files = sorted(f for f in os.listdir(EXAMPLES_DIR) if f.lower().endswith(VALID_IMAGE_EXTS))
    else:
        example_files = []

    if example_files:
        with st.container(border=True):
            st.markdown('<div class="config-label">Select Sample Image</div>', unsafe_allow_html=True)
            chosen_example = st.selectbox(
                "Select sample image", ["(none)"] + example_files, label_visibility="collapsed",
            )
        if chosen_example != "(none)":
            image_to_process = Image.open(os.path.join(EXAMPLES_DIR, chosen_example))
            process_and_show(image_to_process, os.path.splitext(chosen_example)[0], conf_threshold, key_prefix="ex")
            processed_this_run = True
    else:
        st.info("No sample images available. Add .jpg/.png files to the examples/ folder in this repository.")


render_batch_summary()

# =============================================================================
# 9. MODEL INFORMATION SECTION
# =============================================================================

st.markdown('<div class="section-title">AI Model Information</div>', unsafe_allow_html=True)
with st.container(border=True):
    map_val = metadata.get("mAP50_95", "N/A")
    dent_ap = metadata.get(f"AP_{PRIMARY_FIX_TARGET_CLASS}", "N/A")
    rows = [
        ("Model", "RF-DETR-Nano (Roboflow Detection Transformer)"),
        ("Task", "Bottle Packaging Defect Detection"),
        ("Classes", " &middot; ".join(CLASS_NAMES_LIST)),
        ("Dataset", "Primary Retail Bottle Dataset (self-annotated)"),
        ("Operational Threshold", f"{conf_threshold:.2f} confidence &middot; {IOU_MATCH_THRESHOLD:.2f} IoU match"),
        ("Validation mAP50-95", str(map_val)),
        (f"Validation AP &middot; {PRIMARY_FIX_TARGET_CLASS}", str(dent_ap)),
    ]
    table_html = "".join(
        f'<tr><td class="k">{k}</td><td class="v">{v}</td></tr>' for k, v in rows
    )
    st.markdown(f'<table class="info-table">{table_html}</table>', unsafe_allow_html=True)
