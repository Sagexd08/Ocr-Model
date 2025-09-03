#!/usr/bin/env python3
"""
Advanced OCR Document Processing System - Streamlit Interface
Complete enterprise-grade OCR system with advanced features
"""

import os
import time
import json
import tempfile
import uuid
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from pathlib import Path
import threading
import queue

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from io import BytesIO
from PIL import Image, ImageDraw, ImageEnhance
import cv2

# Import OCR system components
try:
    from worker.model_manager import ModelManager
    from worker.storage_manager import StorageManager
    from worker.document_processor import EnhancedDocumentProcessor
    from worker.types import ProcessingMode, JobStatus
    OCR_AVAILABLE = True
except ImportError as e:
    st.error(f"OCR System not available: {e}")
    OCR_AVAILABLE = False

# Advanced page configuration
st.set_page_config(
    page_title="Advanced OCR Processing System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/your-repo/ocr-system',
        'Report a bug': 'https://github.com/your-repo/ocr-system/issues',
        'About': "Advanced OCR Document Processing System v2.0"
    }
)
st.title("CurioScan OCR – Self‑contained Demo")

with st.sidebar:
    st.header("Settings")
    api_base = st.text_input("API base URL", API_BASE)
    default_conf = st.slider("Confidence threshold", 0.0, 1.0, 0.8, 0.05)
    max_pages = st.number_input("Max pages (0=all)", min_value=0, value=3, step=1)
    mode = st.selectbox("Mode", ["BASIC", "STANDARD", "ADVANCED"], index=2)
    auto_refresh = st.toggle("Auto-refresh status", value=True, help="Poll every ~3s while processing")

def post_upload(file_name: str, file_bytes: bytes, *, confidence: float, mode: str, max_pages: int, api: str, content_type: Optional[str] = None) -> Optional[str]:
    try:
        ct = (content_type or "").strip() or "application/octet-stream"
        if ct == "application/octet-stream":
            import mimetypes
            guessed = mimetypes.guess_type(file_name)[0]
            if guessed:
                ct = guessed
        files = {"file": (file_name, file_bytes, ct)}
        data = {"confidence_threshold": confidence, "mode": mode, "max_pages": max_pages}
        r = requests.post(f"{api}/upload", files=files, data=data, timeout=60)
        if r.status_code >= 400:
            try:
                detail = r.json().get("detail")
            except Exception:
                detail = r.text
            raise RuntimeError(f"{r.status_code} {detail}")
        j = r.json()
        return j.get("job_id") or j.get("id")
    except Exception as e:
        st.error(f"Upload failed: {e}")
        return None

def get_status(job_id: str, api: str) -> Optional[dict]:
    try:
        r = requests.get(f"{api}/status/{job_id}", timeout=30)
        if r.status_code == 404:
            return {"status": "NOT_FOUND"}
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"Status error: {e}")
        return None

def get_job(job_id: str, api: str) -> Optional[dict]:
    try:
        r = requests.get(f"{api}/jobs/{job_id}", timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"Job fetch error: {e}")
        return None

def list_jobs(api: str) -> list:
    try:
        r = requests.get(f"{api}/jobs?limit=10", timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception:

# Draw overlay for tokens (expects JSON result structure)
def draw_token_overlay(img_bytes: bytes, result_json: dict, conf_min: float = 0.0) -> bytes:
    try:
        from PIL import Image, ImageDraw
        img = Image.open(BytesIO(img_bytes)).convert("RGBA")
        overlay = Image.new("RGBA", img.size, (0,0,0,0))
        draw = ImageDraw.Draw(overlay)
        pages = result_json.get("pages", [])
        if pages:
            # Assume we received the first page PNG matching pages[0]
            page = pages[0]
            tokens = page.get("tokens") or []
            for t in tokens:
                conf = float(t.get("confidence", 1.0))
                if conf < conf_min:
                    continue
                x1,y1,x2,y2 = t.get("bbox", [0,0,0,0])
                color = (255,0,0,120) if conf < 0.5 else (0,200,0,120)
                draw.rectangle([(x1,y1),(x2,y2)], outline=color, width=2)
        composed = Image.alpha_composite(img, overlay).convert("RGB")
        out = BytesIO()
        composed.save(out, format="PNG")
        return out.getvalue()
    except Exception:
        return img_bytes

def download_result(job_id: str, fmt: str, api: str) -> Optional[bytes]:
    try:
        r = requests.get(f"{api}/result/{job_id}", params={"format": fmt}, timeout=60)
        r.raise_for_status()
        return r.content
    except Exception as e:
        st.error(f"Download {fmt} failed: {e}")
        return None

# Left: upload; Right: status
c1, c2 = st.columns(2)
with c1:
    st.subheader("Upload a document")
    up = st.file_uploader("Choose a file", type=["pdf","png","jpg","jpeg","tif","tiff","docx","xlsx","txt"])
    if st.button("Start Processing", use_container_width=True) and up is not None:
        job_id = post_upload(up.name, up.getvalue(), confidence=default_conf, mode=mode, max_pages=int(max_pages), api=api_base, content_type=up.type)
        if job_id:
            st.session_state["job_id"] = job_id
            st.success(f"Job started: {job_id}")

with c2:
    st.subheader("Job status")
    job_placeholder = st.empty()
    progress_placeholder = st.empty()
    job_id_inp = st.text_input("Job ID", value=st.session_state.get("job_id", ""))
    refresh_btn = st.button("Refresh Now", use_container_width=True)

    def render_status():
        status = get_status(job_id_inp, api_base) if job_id_inp else None
        if status:
            job_placeholder.json(status)
            pct = status.get("progress", 0)
            try:
                pct_float = float(pct)
            except Exception:
                pct_float = 0.0
            progress_placeholder.progress(min(max(pct_float, 0.0), 100.0) / 100.0, text=f"Progress: {pct_float:.1f}%")
            return status.get("status") or status.get("state") or "UNKNOWN"
        return None

    st.caption("Auto-refresh checks status every ~3 seconds while processing.")

    state = None
    if refresh_btn:
        state = render_status()

    # Auto-refresh loop (3s) when enabled and job is running
    if auto_refresh and job_id_inp:
        # We run a short loop during this script execution to emulate periodic polling
        for _ in range(20):  # hard cap ~1 minute (20 * 3s)
            state = render_status()
            if state in {"COMPLETED", "FAILED", "CANCELLED", "NOT_FOUND"}:
                break
            time.sleep(3)
            try:
                # Streamlit <1.30 uses experimental_rerun; >=1.30 uses rerun
                rerun = getattr(st, "rerun", None) or getattr(st, "experimental_rerun", None)
                if callable(rerun):
                    rerun()
                else:
                    break
            except Exception:
                break

st.divider()

# Results and downloads
st.subheader("Results and Downloads")
job_id_dl = st.text_input("Job ID for downloads", value=st.session_state.get("job_id", ""), key="job_id_dl")
dl_cols = st.columns(4)

# helper to format bytes
def fmt_size(num: int) -> str:
    if num is None:
        return ""
    for unit in ["B", "KB", "MB", "GB"]:
        if num < 1024.0 or unit == "GB":
            return f"{num:.1f} {unit}" if unit != "B" else f"{num} B"
        num /= 1024.0
    return f"{num:.1f} GB"

if job_id_dl:
    with dl_cols[0]:
        if st.button("Get Job Info"):
            info = get_job(job_id_dl, api_base)
            if info:
                st.json(info)
    with dl_cols[1]:
        if st.button("Download JSON"):
            data = download_result(job_id_dl, "json", api_base)
            if data is not None:
                size = fmt_size(len(data))
                st.download_button(
                    label=f"Download JSON ({size})",
                    data=data,
                    file_name=f"{job_id_dl}_result.json",
                    mime="application/json"
                )
    with dl_cols[2]:
        if st.button("Download CSV"):
            data = download_result(job_id_dl, "csv", api_base)
            if data is not None:
                size = fmt_size(len(data))
                st.download_button(
                    label=f"Download CSV ({size})",
                    data=data,
                    file_name=f"{job_id_dl}_result.csv",
                    mime="text/csv"
                )

# Overlay controls and demo (client-side preview once preview endpoint exists)
with st.expander("Overlay Settings"):
    conf_min = st.slider("Min confidence for overlays", 0.0, 1.0, 0.0, 0.05)
    st.caption("When a preview image API becomes available, overlays will render here.")

                    file_name=f"{job_id_dl}_result.csv",
                    mime="text/csv"
                )
    with dl_cols[3]:
        if st.button("Download Excel"):
            data = download_result(job_id_dl, "xlsx", api_base)
            if data is not None:
                size = fmt_size(len(data))
                st.download_button(
                    label=f"Download Excel ({size})",
                    data=data,
                    file_name=f"{job_id_dl}_result.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

# Preview panel
st.subheader("Preview")
prev_cols = st.columns(2)
with prev_cols[0]:
    st.caption("Original Document (first page preview if supported)")
    if job_id_dl:
        try:
            # If API exposes a preview endpoint, use it; else show placeholder
            st.info("Preview rendering not yet connected to API; add endpoint for page image + overlays.")
        except Exception:
            pass
with prev_cols[1]:
    st.caption("Extracted Data Snapshot")
    if job_id_dl:
        try:
            data = download_result(job_id_dl, "json", api_base)
            if data:
                import json as _json
                st.json(_json.loads(data.decode("utf-8")))
        except Exception:
            pass

            if data is not None:
                size = fmt_size(len(data))
                st.download_button(
                    label=f"Download Excel ({size})",
                    data=data,
                    file_name=f"{job_id_dl}_result.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

# Job list
st.subheader("Recent Jobs")
jobs = list_jobs(api_base)
if jobs:
    st.dataframe(jobs, use_container_width=True)
else:
    st.info("No jobs found yet or API not reachable.")

