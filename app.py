# app.py
"""
Smart Doc Scanner - Enhanced:
- Improved UI
- Optional OCR via pytesseract (requires system tesseract)
- Parallel processing + caching
- Single-file app, SQLite history, outputs directory

Deploy notes:
- If you want OCR on Streamlit Cloud, add `packages.txt` with:
    tesseract-ocr
    tesseract-ocr-vie    # optional for Vietnamese
"""

import streamlit as st
from pathlib import Path
import tempfile
import time
import io
import os
import sqlite3
import json
from datetime import datetime
import logging
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Dict, Any, Optional

# Processing libs
import img2pdf
from pdf2docx import Converter
import pdfplumber
import pandas as pd
from PIL import Image, ImageOps

# Optional OCR
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except Exception:
    pytesseract = None
    TESSERACT_AVAILABLE = False

# Streamlit config
st.set_page_config(page_title="Smart Doc Scanner", layout="wide", initial_sidebar_state="expanded")

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("smart-doc-scanner-enhanced")

# Paths
ROOT = Path.cwd()
OUTPUT_DIR = ROOT / "outputs"
DB_PATH = ROOT / "app_history.db"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Init DB
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            original_filename TEXT,
            timestamp TEXT,
            conversion_type TEXT,
            output_path TEXT
        )
        """
    )
    conn.commit()
    conn.close()

init_db()

# ---------------------------
# Utilities
# ---------------------------

def save_history(original_filename: str, conversion_type: str, output_path: str):
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute(
            "INSERT INTO history (original_filename, timestamp, conversion_type, output_path) VALUES (?, ?, ?, ?)",
            (original_filename, datetime.utcnow().isoformat(), conversion_type, str(output_path)),
        )
        conn.commit()
    except Exception as e:
        logger.exception("Failed to save history: %s", e)
    finally:
        conn.close()

def query_history(search: str = "") -> List[Dict[str, Any]]:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    if search:
        q = f"%{search}%"
        c.execute("SELECT id, original_filename, timestamp, conversion_type, output_path FROM history WHERE original_filename LIKE ? ORDER BY timestamp DESC", (q,))
    else:
        c.execute("SELECT id, original_filename, timestamp, conversion_type, output_path FROM history ORDER BY timestamp DESC")
    rows = c.fetchall()
    conn.close()
    return [
        {"id": r[0], "original_filename": r[1], "timestamp": r[2], "conversion_type": r[3], "output_path": r[4]}
        for r in rows
    ]

def safe_filename(name: str) -> str:
    name = name.replace(" ", "_")
    stamp = hashlib.sha1(f"{name}{time.time()}".encode()).hexdigest()[:8]
    return f"{name}_{stamp}"

# ---------------------------
# Conversion & extraction helpers
# ---------------------------

def convert_image_bytes_to_pdf_bytes(image_bytes: bytes, filename_hint: str = "img") -> bytes:
    try:
        img = Image.open(io.BytesIO(image_bytes))
        # convert alpha -> white background
        if img.mode in ("RGBA", "LA"):
            background = Image.new("RGB", img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[-1])
            img = background
        with io.BytesIO() as out_buf:
            img_bytes_io = io.BytesIO()
            img.save(img_bytes_io, format="PNG")
            img_bytes_io.seek(0)
            pdf_bytes = img2pdf.convert(img_bytes_io)
            return pdf_bytes
    except Exception as e:
        logger.exception("Image->PDF conversion failed for %s: %s", filename_hint, e)
        raise

def convert_pdf_to_docx(pdf_path: Path, docx_path: Path) -> None:
    try:
        cv = Converter(str(pdf_path))
        cv.convert(str(docx_path), start=0, end=None)
        cv.close()
    except Exception as e:
        logger.exception("PDF->DOCX conversion failed: %s", e)
        raise

def extract_tables_and_text_from_pdf(pdf_path: Path) -> Tuple[List[pd.DataFrame], Dict[str, Any]]:
    dfs = []
    structured = {"pages": []}
    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            for p_idx, page in enumerate(pdf.pages):
                page_text = page.extract_text() or ""
                page_tables = []
                try:
                    tables = page.extract_tables()
                except Exception as e:
                    logger.warning("extract_tables failed on page %d: %s", p_idx, e)
                    tables = []
                for t_idx, table in enumerate(tables):
                    try:
                        df = pd.DataFrame(table)
                        if df.shape[0] > 0:
                            dfs.append(df)
                        page_tables.append({"table_index": t_idx, "rows": table})
                    except Exception as e:
                        logger.warning("Failed to convert table to DataFrame page %d table %d: %s", p_idx, t_idx, e)
                structured["pages"].append({"page_index": p_idx, "text": page_text, "tables": page_tables})
    except Exception as e:
        logger.exception("Failed to open/extract from PDF: %s", e)
        raise
    return dfs, structured

# ---------------------------
# OCR helpers
# ---------------------------

def is_tesseract_installed() -> bool:
    """
    Quick check: Tesseract binary should be available in PATH.
    """
    if not TESSERACT_AVAILABLE:
        return False
    try:
        # pytesseract.pytesseract.tesseract_cmd may be set; just try version
        import subprocess
        res = subprocess.run(["tesseract", "--version"], capture_output=True, text=True)
        return res.returncode == 0
    except Exception:
        return False

def ocr_image_pil(img: Image.Image, lang: str = "eng") -> str:
    """
    Run pytesseract on a PIL image. Returns extracted string.
    """
    if not TESSERACT_AVAILABLE:
        raise RuntimeError("pytesseract not available in Python environment.")
    try:
        # Simple preprocessing: convert to L and equalize
        img2 = img.convert("L")
        img2 = ImageOps.autocontrast(img2)
        txt = pytesseract.image_to_string(img2, lang=lang)
        return txt
    except Exception as e:
        logger.exception("OCR on image failed: %s", e)
        raise

def pdf_page_to_pil(page) -> Optional[Image.Image]:
    """
    Try to render a pdfplumber page to a PIL Image. Not all environments support this,
    but pdfplumber.Page.to_image() often works.
    """
    try:
        page_img_obj = page.to_image(resolution=150)
        pil_img = page_img_obj.original
        return pil_img
    except Exception as e:
        logger.warning("Could not render pdf page to image: %s", e)
        return None

# ---------------------------
# Normalization & processing pipeline (cached)
# ---------------------------

@st.cache_data(show_spinner=False)
def normalize_to_pdf_bytes(file_bytes: bytes, original_name: str) -> bytes:
    suffix = Path(original_name).suffix.lower()
    if suffix == ".pdf":
        return file_bytes
    elif suffix in [".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif"]:
        return convert_image_bytes_to_pdf_bytes(file_bytes, original_name)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")

def process_single_file_bytes(file_name: str, file_bytes: bytes, mode: str, ocr_enabled: bool, ocr_lang: str) -> Dict[str, Any]:
    """
    Process a single file (bytes) and return outputs metadata.
    This function is thread-safe (no Streamlit calls inside).
    """
    res = {"original_filename": file_name, "outputs": {}}
    tmp_pdf = None
    try:
        pdf_bytes = normalize_to_pdf_bytes(file_bytes, file_name)
        # write temp pdf
        tmp_fd = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        tmp_fd.write(pdf_bytes)
        tmp_fd.flush()
        tmp_fd.close()
        tmp_pdf = Path(tmp_fd.name)

        timestamp_str = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
        safe_name = safe_filename(Path(file_name).stem)
        run_folder = OUTPUT_DIR / f"{timestamp_str}_{safe_name}"
        run_folder.mkdir(parents=True, exist_ok=True)

        # DOCX
        if mode in ("docx", "all"):
            try:
                docx_path = run_folder / f"{safe_name}.docx"
                convert_pdf_to_docx(tmp_pdf, docx_path)
                save_history(file_name, "DOCX", docx_path)
                res["outputs"]["docx"] = str(docx_path)
            except Exception as e:
                res["outputs"]["docx_error"] = str(e)

        # Excel / tables
        structured = None
        if mode in ("excel", "all", "json"):
            try:
                dfs, structured = extract_tables_and_text_from_pdf(tmp_pdf)
                if mode in ("excel", "all"):
                    if len(dfs) == 0:
                        excel_path = run_folder / f"{safe_name}_no_tables_found.xlsx"
                        with pd.ExcelWriter(str(excel_path), engine="openpyxl") as writer:
                            pd.DataFrame({"note": ["No tables detected by pdfplumber on this document."]}).to_excel(writer, sheet_name="info", index=False)
                    else:
                        excel_path = run_folder / f"{safe_name}.xlsx"
                        with pd.ExcelWriter(str(excel_path), engine="openpyxl") as writer:
                            for idx, df in enumerate(dfs):
                                sheet_name = f"table_{idx+1}"[:31]
                                df2 = pd.DataFrame(df)
                                df2.to_excel(writer, sheet_name=sheet_name, index=False, header=False)
                    save_history(file_name, "EXCEL", excel_path)
                    res["outputs"]["excel"] = str(excel_path)
            except Exception as e:
                res["outputs"]["excel_error"] = str(e)

        # JSON
        if mode in ("json", "all"):
            try:
                if structured is None:
                    _, structured = extract_tables_and_text_from_pdf(tmp_pdf)
                json_path = run_folder / f"{safe_name}.json"
                with open(json_path, "w", encoding="utf-8") as jf:
                    json.dump(structured, jf, ensure_ascii=False, indent=2)
                save_history(file_name, "JSON", json_path)
                res["outputs"]["json"] = str(json_path)
            except Exception as e:
                res["outputs"]["json_error"] = str(e)

        # OCR: only if user enabled
        if ocr_enabled:
            try:
                ocr_texts = []
                with pdfplumber.open(str(tmp_pdf)) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text and page_text.strip():
                            # If text exists, use it (still include in OCR results as text)
                            ocr_texts.append({"page": page.page_number - 1, "text": page_text, "ocr_used": False})
                        else:
                            # Attempt to render page as image and run OCR
                            pil_img = pdf_page_to_pil(page)
                            if pil_img is None:
                                # fallback: try converting via Pillow from bytes (may not work)
                                try:
                                    # create image from single-page PDF via PIL Image.open (may require poppler; often fails)
                                    pil_img = Image.open(io.BytesIO(page.to_image(resolution=150).original.tobytes()))
                                except Exception:
                                    pil_img = None
                            if pil_img is not None:
                                try:
                                    txt = ocr_image_pil(pil_img, lang=ocr_lang)
                                    ocr_texts.append({"page": page.page_number - 1, "text": txt, "ocr_used": True})
                                except Exception as e:
                                    ocr_texts.append({"page": page.page_number - 1, "text": "", "ocr_error": str(e)})
                            else:
                                ocr_texts.append({"page": page.page_number - 1, "text": "", "ocr_error": "render_failed"})
                ocr_path = run_folder / f"{safe_name}_ocr.txt"
                with open(ocr_path, "w", encoding="utf-8") as of:
                    for p in ocr_texts:
                        of.write(f"--- PAGE {p['page']} ---\n")
                        of.write((p.get("text") or "").strip() + "\n\n")
                save_history(file_name, "OCR", ocr_path)
                res["outputs"]["ocr"] = str(ocr_path)
            except Exception as e:
                res["outputs"]["ocr_error"] = str(e)

        # cleanup tmp pdf
        try:
            if tmp_pdf and tmp_pdf.exists():
                tmp_pdf.unlink(missing_ok=True)
        except Exception:
            pass

    except Exception as e:
        logger.exception("Processing failed for %s: %s", file_name, e)
        res["error"] = str(e)
    return res

# ---------------------------
# UI / Layout / Interactions
# ---------------------------

# Small CSS for nicer look
st.markdown(
    """
    <style>
    .stApp { background-color: #f8fafc; }
    .header {padding:16px; border-radius:12px; background: linear-gradient(90deg,#4f46e5 0%,#06b6d4 100%); color: white;}
    .card { background: white; border-radius:10px; padding:12px; box-shadow: 0 2px 6px rgba(16,24,40,0.04); }
    .muted { color: #6b7280; font-size:0.9rem; }
    </style>
    """, unsafe_allow_html=True
)

with st.sidebar:
    st.markdown("<div class='header'><h2 style='margin:6px 0'>Smart Doc Scanner</h2><div class='muted'>Enhanced</div></div>", unsafe_allow_html=True)
    page = st.radio("Menu", ("Home", "History", "Settings"))
    st.markdown("---")
    st.caption("Supported: PDF, PNG, JPG, JPEG")
    st.markdown("Optimized for Streamlit Cloud — single-file app + outputs + SQLite history")

# Settings
if page == "Settings":
    st.header("Settings & Deployment notes")
    st.info("Nếu bạn muốn bật OCR trên Streamlit Cloud, hãy thêm file `packages.txt` vào repo với dòng `tesseract-ocr` (và `tesseract-ocr-vie` nếu cần tiếng Việt).")
    st.markdown("**Outputs folder:** `%s`" % OUTPUT_DIR)
    st.markdown("**DB path:** `%s`" % DB_PATH)
    st.markdown("**OCR availability:** `%s`" % ("Yes" if is_tesseract_installed() else "No (system Tesseract not found)"))
    st.markdown("---")
    st.write("Notes:")
    st.write("- OCR requires system `tesseract` binary (not only `pytesseract`).")
    st.write("- For heavy workloads, consider background workers or a cloud function / batch queue.")
    st.stop()

# HOME
if page == "Home":
    st.markdown("<div class='card'><h3>New Scan</h3><div class='muted'>Upload documents or images. Choose outputs and enable OCR if you installed Tesseract.</div></div>", unsafe_allow_html=True)
    col_left, col_right = st.columns([2,1])
    with col_left:
        uploaded_files = st.file_uploader("Upload files (multiple)", accept_multiple_
