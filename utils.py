"""
=============================================================================
 utils.py — IMAGE PROCESSING & VISUALIZATION
 Automated Bottle Quality Inspection System
=============================================================================
PENTING - KONSISTENSI PIPELINE (TIDAK BOLEH BERBEDA DARI TAHAP 5 & TAHAP 6):
  - CLASS_NAMES_LIST & urutannya (menentukan category_id -> nama kelas)
  - CLASS_COLOR (skema warna per kelas, sama dgn chart evaluasi Tahap 6, agar
    warna di app konsisten dgn warna di laporan/skripsi)
  - run_inference() memanggil model.predict(image, threshold=...) lalu
    membaca .xyxy / .confidence / .class_id — PERSIS sama dgn Tahap 5 & 6.
  Fungsi di file ini HANYA menambahkan cara MENAMPILKAN hasil (bounding box
  lebih tebal, label lebih besar/jelas) — tidak ada satupun logic deteksi,
  threshold, atau urutan kelas yang diubah.
=============================================================================
"""

import os
import io
import zipfile

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

# =============================================================================
# KONFIGURASI KELAS — PERSIS SAMA DENGAN TAHAP 5 (MODELING) & TAHAP 6 (EVALUASI)
# =============================================================================

CLASS_NAMES_LIST = ["Dent", "Label Damage", "Missing Cap", "Seal Damage"]
CLASS_COLOR = {
    "Dent": "#e74c3c",
    "Label Damage": "#f39c12",
    "Missing Cap": "#8e44ad",
    "Seal Damage": "#2980b9",
}

VALID_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")

# =============================================================================
# FONT — dipakai supaya label bounding box tegas & mudah dibaca (bukan bitmap
# kecil bawaan PIL). Pillow >= 10.1 mendukung ImageFont.load_default(size=...);
# kalau versi Pillow lebih lama, fallback otomatis ke ukuran bawaan.
# =============================================================================

_FONT_CACHE = {}


def get_font(size: int = 26):
    if size in _FONT_CACHE:
        return _FONT_CACHE[size]
    font = None
    # 1) coba font TrueType umum yang sering tersedia di container Linux
    candidate_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ]
    for path in candidate_paths:
        if os.path.isfile(path):
            try:
                font = ImageFont.truetype(path, size=size)
                break
            except Exception:
                font = None
    # 2) fallback: default font Pillow versi scalable (Pillow >= 10.1)
    if font is None:
        try:
            font = ImageFont.load_default(size=size)
        except TypeError:
            font = ImageFont.load_default()
    _FONT_CACHE[size] = font
    return font


def hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


# =============================================================================
# INFERENSI — TIDAK DIUBAH DARI TAHAP 5/6 (predict only, sama urutan kelas)
# =============================================================================

def run_inference(model, image: Image.Image, conf_threshold: float) -> pd.DataFrame:
    """Sama persis dgn cara Tahap 5/6 memanggil model: model.predict(image,
    threshold=...) lalu membaca .xyxy / .confidence / .class_id."""
    detections = model.predict(image.convert("RGB"), threshold=conf_threshold)
    xyxy = np.asarray(detections.xyxy) if len(detections) else np.zeros((0, 4))
    conf = np.asarray(detections.confidence) if len(detections) else np.zeros((0,))
    cls_id = np.asarray(detections.class_id) if len(detections) else np.zeros((0,), dtype=int)

    rows = []
    for (x1, y1, x2, y2), c, k in zip(xyxy, conf, cls_id):
        rows.append({
            "class_name": CLASS_NAMES_LIST[int(k)] if int(k) < len(CLASS_NAMES_LIST) else f"class_{k}",
            "confidence": float(c),
            "x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2),
        })
    return pd.DataFrame(rows)


# =============================================================================
# VISUALISASI — bounding box lebih tebal + label lebih besar/jelas (perbaikan
# tampilan sesuai masukan; TIDAK mengubah koordinat/kelas/confidence apapun)
#
# Parameter untuk atur tampilan bounding box ada di 4 konstanta berikut:
#   - BOX_WIDTH          -> ketebalan garis kotak (px)
#   - LABEL_FONT_SIZE     -> ukuran huruf label (px)
#   - LABEL_PADDING_X/Y   -> padding kotak label di sekitar teks
# =============================================================================

BOX_WIDTH = 25
LABEL_FONT_SIZE = 120
LABEL_PADDING_X = 10
LABEL_PADDING_Y = 6


def draw_detections(image: Image.Image, df_det: pd.DataFrame) -> Image.Image:
    annotated = image.convert("RGB").copy()
    draw = ImageDraw.Draw(annotated)
    font = get_font(LABEL_FONT_SIZE)

    for _, row in df_det.iterrows():
        color = hex_to_rgb(CLASS_COLOR.get(row["class_name"], "#333333"))
        box = [row["x1"], row["y1"], row["x2"], row["y2"]]

        # Garis kotak lebih tebal + garis kontras gelap tipis di luarnya
        # supaya tetap terbaca di atas botol berwarna terang.
        draw.rectangle(box, outline=(20, 20, 20), width=BOX_WIDTH + 3)
        draw.rectangle(box, outline=color, width=BOX_WIDTH)

        # Label = nama kelas + confidence score (persis dari df_det, tidak diubah)
        label = f"{row['class_name']}  {row['confidence']*100:.0f}%"
        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]

        label_box_w = text_w + LABEL_PADDING_X * 2
        label_box_h = text_h + LABEL_PADDING_Y * 2
        label_y2 = box[1]
        label_y1 = max(0, label_y2 - label_box_h)
        # Kalau kotak deteksi terlalu dekat tepi atas gambar, taruh label DI
        # DALAM kotak (bukan di luar frame) supaya tidak terpotong.
        if label_y1 <= 0 and box[1] < label_box_h:
            label_y1 = box[1]
            label_y2 = box[1] + label_box_h

        draw.rounded_rectangle(
            [box[0], label_y1, box[0] + label_box_w, label_y2],
            radius=4, fill=color,
        )
        draw.text(
            (box[0] + LABEL_PADDING_X, label_y1 + LABEL_PADDING_Y - text_bbox[1]),
            label, fill="white", font=font,
        )

    return annotated


def pil_to_bytes(img: Image.Image, fmt="PNG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


# =============================================================================
# TABEL DETEKSI CUSTOM (HTML) — confidence ditampilkan sbg bar + persen besar,
# supaya jelas terbaca (menggantikan st.dataframe bawaan yg kurang menonjol).
# Murni presentasi - angka yg ditampilkan persis dari df_det, tidak diubah.
# =============================================================================

def render_detection_table_html(df_det: pd.DataFrame) -> str:
    if df_det.empty:
        return ""

    rows_html = []
    df_sorted = df_det.sort_values("confidence", ascending=False)
    for _, row in df_sorted.iterrows():
        color = CLASS_COLOR.get(row["class_name"], "#333333")
        pct = row["confidence"] * 100
        rows_html.append(
            f'<div class="det-row">'
            f'<div class="det-class">'
            f'<span class="det-dot" style="background:{color};"></span>'
            f'<span>{row["class_name"]}</span>'
            f'</div>'
            f'<div class="det-bar-track">'
            f'<div class="det-bar-fill" style="width:{pct:.1f}%; background:{color};"></div>'
            f'</div>'
            f'<div class="det-pct">{pct:.1f}%</div>'
            f'</div>'
        )
    return f'<div class="det-table">{"".join(rows_html)}</div>'


def render_defect_chips_html(class_names) -> str:
    if not class_names:
        return '<span class="defect-chip defect-chip-none">No defects</span>'
    chips = []
    for name in class_names:
        color = CLASS_COLOR.get(name, "#333333")
        chips.append(
            f'<span class="defect-chip" style="border-color:{color}33; '
            f'background:{color}14; color:{color};">{name}</span>'
        )
    return "".join(chips)


# =============================================================================
# HELPER — EKSTRAK GAMBAR DARI FILE .ZIP
# =============================================================================

def extract_images_from_zip(uploaded_zip_file, max_images: int = 200):
    """Baca file .zip yang di-upload, ambil semua gambar valid di dalamnya
    (termasuk yg ada di subfolder), dan kembalikan list of (nama_file, PIL.Image).
    Bukan bagian dari pipeline deteksi - murni utilitas I/O utk end-user."""
    images = []
    skipped = []
    try:
        with zipfile.ZipFile(uploaded_zip_file) as zf:
            names = [
                n for n in zf.namelist()
                if not n.endswith("/")
                and not os.path.basename(n).startswith(("__MACOSX", "."))
                and n.lower().endswith(VALID_IMAGE_EXTS)
            ]
            names = names[:max_images]
            for n in names:
                try:
                    with zf.open(n) as f:
                        img = Image.open(io.BytesIO(f.read()))
                        img.load()
                    base_name = os.path.splitext(os.path.basename(n))[0]
                    images.append((base_name, img))
                except Exception:
                    skipped.append(n)
    except zipfile.BadZipFile:
        return [], ["__BAD_ZIP__"]
    return images, skipped
