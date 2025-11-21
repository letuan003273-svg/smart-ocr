# app.py
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

# Processing libraries
import img2pdf
from pdf2docx import Converter
import pdfplumber
import pandas as pd
from PIL import Image

# For typing
from typing import List, Tuple, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("smart-doc-scanner")

# ---------------------------
# Constants and setup
# ---------------------------
ROOT = Path.cwd()
OUTPUT_DIR = ROOT / "outputs"
DB_PATH = ROOT / "app_history.db"

# Ensure outputs folder exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# SQLite schema init
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
# Helper functions
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
    # sanitize and avoid collisions by appending short hash
    name = name.replace(" ", "_")
    stamp = hashlib.sha1(f"{name}{time.time()}".encode()).hexdigest()[:8]
    return f"{name}_{stamp}"

# ---------------------------
# Conversion utilities
# ---------------------------

def convert_image_bytes_to_pdf_bytes(image_bytes: bytes, filename_hint: str = "img") -> bytes:
    """
    Convert image bytes (png/jpg) to a single-page PDF bytes using img2pdf.
    """
    try:
        img = Image.open(io.BytesIO(image_bytes))
        # Ensure RGB for jpeg/pdf conversion
        if img.mode in ("RGBA", "LA"):
            background = Image.new("RGB", img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[-1])
            img = background
        with io.BytesIO() as out_buf:
            # img2pdf needs file-like objects or paths: use Pillow to save to bytes then feed
            img_bytes_io = io.BytesIO()
            img.save(img_bytes_io, format="PNG")
            img_bytes_io.seek(0)
            pdf_bytes = img2pdf.convert(img_bytes_io)
            return pdf_bytes
    except Exception as e:
        logger.exception("Image->PDF conversion failed for %s: %s", filename_hint, e)
        raise

def convert_pdf_to_docx(pdf_path: Path, docx_path: Path) -> None:
    """
    Use pdf2docx to convert and preserve layout as much as possible.
    """
    try:
        cv = Converter(str(pdf_path))
        # convert entire file
        cv.convert(str(docx_path), start=0, end=None)
        cv.close()
    except Exception as e:
        logger.exception("PDF->DOCX conversion failed: %s", e)
        raise

def extract_tables_and_text_from_pdf(pdf_path: Path) -> Tuple[List[pd.DataFrame], Dict[str, Any]]:
    """
    Use pdfplumber to extract tables and text.
    Returns (list_of_dataframes, structured_text_and_table_info)
    structured JSON includes per-page text and per-page tables as list of lists.
    """
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
                # convert tables to DataFrame if possible
                for t_idx, table in enumerate(tables):
                    try:
                        # Convert to DataFrame: some tables may have ragged rows => normalize
                        df = pd.DataFrame(table)
                        # drop all-NaN or none rows
                        if df.shape[0] > 0:
                            # Reset header if first row appears header-like (heuristic)
                            # Keep all tables raw; let user inspect/clean
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
# Processing pipeline
# ---------------------------

def normalize_to_pdf(file_bytes: bytes, original_name: str) -> Path:
    """
    Ensure input is a PDF. If given image bytes, convert to pdf and save temp file.
    Returns path to a saved PDF file (in temp directory).
    """
    suffix = Path(original_name).suffix.lower()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    tmp_path = Path(tmp.name)
    try:
        if suffix in [".pdf"]:
            tmp.write(file_bytes)
            tmp.flush()
            tmp.close()
            return tmp_path
        elif suffix in [".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif"]:
            pdf_bytes = convert_image_bytes_to_pdf_bytes(file_bytes, original_name)
            tmp.write(pdf_bytes)
            tmp.flush()
            tmp.close()
            return tmp_path
        else:
            raise ValueError(f"Unsupported file type for normalization: {suffix}")
    except Exception:
        # ensure file removed if failure
        try:
            tmp.close()
        except:
            pass
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise

def process_file(file_obj, mode: str) -> Dict[str, Any]:
    """
    file_obj: a Streamlit UploadedFile
    mode: 'docx' | 'excel' | 'json' (or 'all' to produce all three)
    Returns dict with status and output paths.
    """
    results = {"original_filename": file_obj.name, "outputs": {}}
    try:
        raw_bytes = file_obj.read()
        pdf_path = normalize_to_pdf(raw_bytes, file_obj.name)  # saved temp pdf
        timestamp_str = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
        safe_name = safe_filename(Path(file_obj.name).stem)
        # Create an output subfolder per run for neatness
        run_folder = OUTPUT_DIR / f"{timestamp_str}_{safe_name}"
        run_folder.mkdir(parents=True, exist_ok=True)

        # DOCX
        if mode in ("docx", "all"):
            try:
                docx_name = run_folder / f"{safe_name}.docx"
                convert_pdf_to_docx(pdf_path, docx_name)
                save_history(file_obj.name, "DOCX", docx_name)
                results["outputs"]["docx"] = str(docx_name)
            except Exception as e:
                results["outputs"]["docx_error"] = str(e)

        # Excel (tables)
        if mode in ("excel", "all"):
            try:
                dfs, structured = extract_tables_and_text_from_pdf(pdf_path)
                if len(dfs) == 0:
                    # No tables found -> create an empty workbook with note
                    excel_path = run_folder / f"{safe_name}_no_tables_found.xlsx"
                    with pd.ExcelWriter(str(excel_path), engine="openpyxl") as writer:
                        pd.DataFrame({"note": ["No tables detected by pdfplumber on this document."]}).to_excel(writer, sheet_name="info", index=False)
                else:
                    excel_path = run_folder / f"{safe_name}.xlsx"
                    with pd.ExcelWriter(str(excel_path), engine="openpyxl") as writer:
                        for idx, df in enumerate(dfs):
                            sheet_name = f"table_{idx+1}"
                            # Truncate sheet name if too long
                            sheet_name = sheet_name if len(sheet_name) <= 31 else sheet_name[:31]
                            # Attempt to coerce to dataframe (some tables may have differing row lengths)
                            df2 = pd.DataFrame(df)
                            df2.to_excel(writer, sheet_name=sheet_name, index=False, header=False)
                save_history(file_obj.name, "EXCEL", excel_path)
                results["outputs"]["excel"] = str(excel_path)
                # for JSON mode we will want structured
                results.setdefault("_structured", structured)
            except Exception as e:
                results["outputs"]["excel_error"] = str(e)

        # JSON (text + tables)
        if mode in ("json", "all"):
            try:
                # if structured not present, extract
                if "_structured" in results:
                    structured = results["_structured"]
                else:
                    _, structured = extract_tables_and_text_from_pdf(pdf_path)
                json_path = run_folder / f"{safe_name}.json"
                with open(json_path, "w", encoding="utf-8") as jf:
                    json.dump(structured, jf, ensure_ascii=False, indent=2)
                save_history(file_obj.name, "JSON", json_path)
                results["outputs"]["json"] = str(json_path)
            except Exception as e:
                results["outputs"]["json_error"] = str(e)

        # cleanup temp pdf
        try:
            if pdf_path.exists():
                pdf_path.unlink(missing_ok=True)
        except Exception:
            pass

    except Exception as e:
        logger.exception("Processing failed for file %s: %s", getattr(file_obj, "name", "unknown"), e)
        results["error"] = str(e)

    return results

# ---------------------------
# Streamlit UI
# ---------------------------

st.set_page_config(page_title="Smart Doc Scanner", layout="wide", initial_sidebar_state="expanded")

# Sidebar
with st.sidebar:
    st.title("Smart Doc Scanner")
    page = st.radio("Navigation", ("Home / New Scan", "History", "Settings"))
    st.markdown("---")
    st.caption("Supported: PDF, PNG, JPG, JPEG")
    st.markdown("Built for Streamlit Cloud — single-file app + outputs folder + SQLite history")

# Settings
if page == "Settings":
    st.header("Settings")
    st.markdown("Configuration & information")
    st.write(f"Outputs folder: `{OUTPUT_DIR}`")
    st.write(f"Database path: `{DB_PATH}`")
    st.checkbox("Keep temporary PDF files for debugging (dev)", value=False, key="keep_temp")
    st.markdown("**Note:** For OCR (image-only PDFs) you can extend this app by adding Tesseract OCR (pytesseract). This app currently extracts text via `pdfplumber` which operates on native PDF text layers.")
    st.stop()

# Home / New Scan
if page == "Home / New Scan":
    st.header("New Scan")
    col1, col2 = st.columns([2, 1])
    with col1:
        uploaded_files = st.file_uploader("Upload files (multi)", accept_multiple_files=True, type=["pdf", "png", "jpg", "jpeg"], help="Drop PDFs or images. Images will be auto-converted to PDF.", key="uploader")
        conversion_mode = st.selectbox("Conversion mode", options=["docx", "excel", "json", "all"], index=3, help="docx: full layout-preserving Word. excel: extract tables. json: structured text & tables. all: produce all outputs.")
    with col2:
        st.write("Processing options")
        show_preview = st.checkbox("Preview extracted tables/text after processing", value=True)
        st.write("Progress and status will appear below.")

    if uploaded_files:
        progress_bar = st.progress(0)
        overall_count = len(uploaded_files)
        status_area = st.empty()
        results_summary = []
        for idx, f in enumerate(uploaded_files, start=1):
            status_area.info(f"Processing {idx}/{overall_count}: {f.name}")
            try:
                with st.spinner(f"Working on {f.name}..."):
                    res = process_file(f, conversion_mode)
                results_summary.append(res)
                # update progress (simple)
                progress_bar.progress(int((idx / overall_count) * 100))
                # Display outputs links and previews
                out_col1, out_col2 = st.columns([3, 2])
                with out_col1:
                    st.subheader(f"File: {f.name}")
                    if "error" in res:
                        st.error(f"Failed: {res['error']}")
                    else:
                        outputs = res.get("outputs", {})
                        # DOCX
                        if "docx" in outputs:
                            docx_path = Path(outputs["docx"])
                            st.success(f"DOCX generated: {docx_path.name}")
                            with open(docx_path, "rb") as fh:
                                st.download_button("Download DOCX", fh.read(), file_name=docx_path.name, key=f"dl_docx_{idx}")
                        elif "docx_error" in outputs:
                            st.warning(f"DOCX error: {outputs['docx_error']}")

                        # Excel
                        if "excel" in outputs:
                            excel_path = Path(outputs["excel"])
                            st.success(f"Excel generated: {excel_path.name}")
                            with open(excel_path, "rb") as fh:
                                st.download_button("Download Excel", fh.read(), file_name=excel_path.name, key=f"dl_xlsx_{idx}")
                            if show_preview:
                                # show first sheet as preview
                                try:
                                    df_preview = pd.read_excel(str(excel_path), sheet_name=0, header=None)
                                    st.dataframe(df_preview.head(50))
                                except Exception as e:
                                    st.warning(f"Could not preview Excel: {e}")
                        elif "excel_error" in outputs:
                            st.warning(f"Excel error: {outputs['excel_error']}")

                        # JSON
                        if "json" in outputs:
                            json_path = Path(outputs["json"])
                            st.success(f"JSON generated: {json_path.name}")
                            with open(json_path, "rb") as fh:
                                st.download_button("Download JSON", fh.read(), file_name=json_path.name, key=f"dl_json_{idx}")
                            if show_preview:
                                try:
                                    with open(json_path, "r", encoding="utf-8") as jf:
                                        parsed = json.load(jf)
                                    st.json(parsed)
                                except Exception as e:
                                    st.warning(f"Could not preview JSON: {e}")
                        elif "json_error" in outputs:
                            st.warning(f"JSON error: {outputs['json_error']}")
                with out_col2:
                    st.write("Metadata & quick actions")
                    st.write(f"Filename: {res['original_filename']}")
                    # Provide a quick action to open folder (list outputs)
                    last_run_folder = None
                    for val in res.get("outputs", {}).values():
                        last_run_folder = Path(val).parent
                        break
                    if last_run_folder and last_run_folder.exists():
                        st.write(f"Outputs folder: `{last_run_folder.name}`")
                        items = list(last_run_folder.iterdir())
                        for it in items:
                            if it.is_file():
                                with open(it, "rb") as fh:
                                    st.download_button(f"Download {it.name}", fh.read(), file_name=it.name, key=f"dl_{idx}_{it.name}")
            except Exception as e:
                logger.exception("Top-level processing error for file %s: %s", getattr(f, "name", "unknown"), e)
                st.error(f"Processing failed for {f.name}: {e}")

        status_area.success("All done!")
        progress_bar.progress(100)
        st.markdown("---")
        st.write("Batch summary")
        for r in results_summary:
            st.write(r.get("original_filename"), "→", list(r.get("outputs", {}).keys()) or r.get("error"))

# History page
if page == "History":
    st.header("Scan History")
    search_q = st.text_input("Search by filename", value="", placeholder="type part of filename to filter history")
    hist = query_history(search_q)
    st.write(f"Found {len(hist)} records")
    if len(hist) == 0:
        st.info("No history records found.")
    else:
        # Display as dataframe
        df_hist = pd.DataFrame(hist)
        df_hist_display = df_hist[["id", "original_filename", "timestamp", "conversion_type"]]
        st.dataframe(df_hist_display)
        # Allow selecting rows to download
        st.markdown("### Actions")
        for record in hist:
            st.write(f"**{record['original_filename']}** — {record['conversion_type']} — {record['timestamp']}")
            out_path = Path(record["output_path"])
            if out_path.exists():
                with open(out_path, "rb") as fh:
                    st.download_button(f"Download {out_path.name}", fh.read(), file_name=out_path.name, key=f"h_{record['id']}")
                st.write(f"Stored at: `{out_path}`")
            else:
                st.warning("Output file missing from disk. (It may have been removed manually.)")

# Final footer
st.markdown("---")
st.caption("Smart Doc Scanner — built with pdf2docx, pdfplumber, img2pdf, pandas. For OCR support, integrate pytesseract separately.")
