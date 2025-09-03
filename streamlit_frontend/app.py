"""
CurioScan OCR - Production-ready Streamlit Demo Application
"""
import streamlit as st
import time
import os
import datetime
import json
from pathlib import Path
import pandas as pd
import numpy as np
import io
import base64
import requests
from PIL import Image
import plotly.express as px
from typing import Dict, Any, List, Optional, Tuple, Union
import logging
import uuid
import tempfile
import sys
from streamlit.elements import image as st_image

# Import components
try:
    # Try relative imports
    from .components.api_client import CurioScanAPIClient
    from .components.visualization import (
        visualize_document_image,
        visualize_text_regions, 
        visualize_table_regions,
        create_form_field_visualization,
        create_analytics_dashboard
    )
except ImportError:
    # Fall back to direct imports
    from components.api_client import CurioScanAPIClient
    from components.visualization import (
        visualize_document_image,
        visualize_text_regions, 
        visualize_table_regions,
        create_form_field_visualization,
        create_analytics_dashboard
    )

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Config and environment variables
API_URL = os.environ.get("CURIOSCAN_API_URL", "http://127.0.0.1:8000")
DEMO_MODE = os.environ.get("DEMO_MODE", "false").lower() == "true"
MAX_FILE_SIZE_MB = int(os.environ.get("MAX_FILE_SIZE_MB", "50"))
ENABLE_ANALYTICS = os.environ.get("ENABLE_ANALYTICS", "true").lower() == "true"
CACHE_DIR = os.environ.get("CACHE_DIR", "./.streamlit/cache")

# Initialize API client
api_client = CurioScanAPIClient(base_url=API_URL)

# Create cache directory if it doesn't exist
os.makedirs(CACHE_DIR, exist_ok=True)

# Set page configuration
st.set_page_config(
    page_title="CurioScan OCR - Intelligent Document Processing",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': """
        # CurioScan OCR
        Intelligent Document Processing System
        
        Version: 2.0.0 (August 2025)
        """
    }
)

# Session state initialization
if 'jobs' not in st.session_state:
    st.session_state.jobs = []
if 'current_job_id' not in st.session_state:
    st.session_state.current_job_id = None
if 'api_settings' not in st.session_state:
    st.session_state.api_settings = {
        'api_url': API_URL,
        'confidence_threshold': 0.8,
        'include_provenance': True
    }
if 'analytics' not in st.session_state:
    st.session_state.analytics = {
        'processed_docs': 0,
        'successful_docs': 0,
        'failed_docs': 0,
        'total_pages': 0,
        'avg_processing_time': 0
    }

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem !important;
        font-weight: 600 !important;
    }
    .subheader {
        font-size: 1.5rem !important;
        font-weight: 500 !important;
    }
    .info-box {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0px;
    }
    .success-box {
        background-color: #d0f0d3;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0px;
    }
    .error-box {
        background-color: #f9d6d6;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0px;
    }
    .stButton>button {
        width: 100%;
    }
    /* Data table styling */
    .dataframe {
        width: 100% !important;
    }
    /* Job list styling */
    .job-list-item {
        padding: 10px;
        margin-bottom: 5px;
        border-radius: 5px;
        cursor: pointer;
    }
    .job-list-item:hover {
        background-color: #f0f2f6;
    }
    .selected-job {
        background-color: #d0f0d3 !important;
    }
</style>
""", unsafe_allow_html=True)

# App header
st.markdown('<h1 class="main-header">📄 CurioScan OCR</h1>', unsafe_allow_html=True)
st.markdown('<h2 class="subheader">Intelligent Document Processing</h2>', unsafe_allow_html=True)
    
# Create main layout with sidebar
sidebar = st.sidebar
main_area = st

# Sidebar content
with sidebar:
    st.header("📁 Upload Document")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a document to process", 
        type=["pdf", "png", "jpg", "jpeg", "tiff", "docx", "xlsx", "txt"],
        help="Supported formats: PDF, images (PNG, JPG, TIFF), MS Office (DOCX, XLSX), and plain text"
    )
    
    # Processing options
    st.header("⚙️ Processing Options")
    
    with st.expander("OCR Settings", expanded=False):
        ocr_mode = st.selectbox(
            "OCR Mode",
            options=["Auto", "Force OCR", "Native Text Only", "Legacy"],
            help="Auto: Let system decide best approach\nForce OCR: Always use OCR\nNative Text: Extract embedded text when possible\nLegacy: Use previous OCR engine"
        )

        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.8,
            step=0.05,
            help="Set minimum confidence threshold for text extraction"
        )

    with st.expander("Speed / Quality", expanded=True):
        processing_mode = st.selectbox(
            "Processing Mode",
            options=["BASIC", "STANDARD", "ADVANCED"],
            index=2,
            help="BASIC: fastest, minimal features; STANDARD: balanced; ADVANCED: full features"
        )
        max_pages = st.number_input(
            "Max Pages (0 = all)",
            min_value=0,
            max_value=100,
            value=3,
            step=1,
            help="Limit number of pages processed for speed"
        )
    
    with st.expander("Advanced Features", expanded=False):
        extract_tables = st.checkbox("Detect and extract tables", value=True)
        extract_forms = st.checkbox("Detect and extract form fields", value=True)
        analyze_layout = st.checkbox("Analyze document layout", value=True)
        detect_language = st.checkbox("Detect language", value=True)
        
    with st.expander("Export Options", expanded=False):
        output_formats = st.multiselect(
            "Output Formats",
            options=["JSON", "CSV", "Excel", "Text"],
            default=["JSON", "CSV"],
            help="Select output formats for results"
        )
        
        include_provenance = st.checkbox(
            "Include data provenance", 
            value=True,
            help="Include detailed information about text positions and confidence"
        )
    
    # Job history
    st.header("📋 Recent Jobs")
    
    if st.session_state.jobs:
        for i, job in enumerate(st.session_state.jobs):
            job_status = job.get('status', 'UNKNOWN')
            status_color = {
                'COMPLETED': '🟢',
                'PROCESSING': '🟡',
                'FAILED': '🔴',
                'UNKNOWN': '⚪'
            }.get(job_status, '⚪')
            
            # Format timestamp
            timestamp = job.get('timestamp', datetime.datetime.now())
            if isinstance(timestamp, str):
                try:
                    timestamp = datetime.datetime.fromisoformat(timestamp)
                except:
                    timestamp = datetime.datetime.now()
            
            time_str = timestamp.strftime("%H:%M:%S")
            
            # Job item with click handler
            is_selected = st.session_state.current_job_id == job.get('job_id')
            job_class = "job-list-item " + ("selected-job" if is_selected else "")
            
            job_html = f"""
            <div class="{job_class}" onclick="handleJobClick('{job.get('job_id')}')">
                {status_color} {job.get('filename', 'Unknown')} ({time_str})
            </div>
            """
            
            if st.markdown(job_html, unsafe_allow_html=True):
                st.session_state.current_job_id = job.get('job_id')
    else:
        st.info("No recent jobs")
    
    # Settings and About
    st.header("⚙️ Settings")
    
    with st.expander("API Configuration", expanded=False):
        api_url = st.text_input("API URL", value=st.session_state.api_settings['api_url'])
        
        if st.button("Save Settings"):
            st.session_state.api_settings['api_url'] = api_url
            st.success("Settings saved!")

    # About section
    with st.expander("About CurioScan", expanded=False):
        st.write("""
        **CurioScan OCR** is an advanced document processing system capable of extracting text, 
        tables, and form fields from various document types.
        
        **Version:** 2.0.0 (August 2025)
        """)

# Define the document processing function
def process_document(file, options):
    """Process a document and track progress"""
    try:
        # Start processing timer
        start_time = time.time()

        # Show processing status
        with st.status("Processing document...", expanded=True) as status:
            st.write("Uploading file...")

            # Upload file to API with mode/max_pages for speed control
            job_id = api_client.upload_file(
                file,
                confidence_threshold=options['confidence_threshold'],
                mode=options.get('processing_mode', 'STANDARD'),
                max_pages=options.get('max_pages', 5)
            )
            st.write(f"Processing job started with ID: {job_id}")
            
            # Monitor processing status
            status_data = {"status": "STARTED", "progress": 0}
            progress_bar = st.progress(0, "Starting document processing...")
            
            while status_data.get("status") not in ["COMPLETED", "FAILED"]:
                try:
                    status_data = api_client.get_job_status(job_id)
                    progress = status_data.get("progress", 0)
                    current_status = status_data.get("status", "PROCESSING")
                    
                    # Update progress bar
                    progress_bar.progress(
                        progress / 100, 
                        f"Processing: {current_status} ({progress}%)"
                    )
                    
                    # Add more detailed status if available
                    if "current_stage" in status_data:
                        st.write(f"Current stage: {status_data['current_stage']}")
                    
                    time.sleep(0.5)
                except Exception as e:
                    st.error(f"Error checking status: {str(e)}")
                    time.sleep(2)  # Longer delay on error
            
            # Process complete
            processing_time = time.time() - start_time
            
            # Update status
            if status_data.get("status") == "COMPLETED":
                status.update(label="Processing complete!", state="complete")
                progress_bar.progress(100, "Processing complete")
                
                # Update analytics
                st.session_state.analytics['processed_docs'] += 1
                st.session_state.analytics['successful_docs'] += 1
                st.session_state.analytics['avg_processing_time'] = (
                    (st.session_state.analytics['avg_processing_time'] * 
                     (st.session_state.analytics['successful_docs'] - 1) + 
                     processing_time) / 
                    st.session_state.analytics['successful_docs']
                )
                
                # Add job to history
                job_info = {
                    'job_id': job_id,
                    'filename': file.name,
                    'status': 'COMPLETED',
                    'timestamp': datetime.datetime.now(),
                    'processing_time': processing_time
                }
                st.session_state.jobs.insert(0, job_info)
                st.session_state.current_job_id = job_id
                
                return job_id, True
            else:
                status.update(label=f"Processing failed: {status_data.get('error', 'Unknown error')}", state="error")
                progress_bar.empty()
                
                # Update analytics
                st.session_state.analytics['processed_docs'] += 1
                st.session_state.analytics['failed_docs'] += 1
                
                # Add job to history
                job_info = {
                    'job_id': job_id,
                    'filename': file.name,
                    'status': 'FAILED',
                    'timestamp': datetime.datetime.now(),
                    'error': status_data.get('error', 'Unknown error')
                }
                st.session_state.jobs.insert(0, job_info)
                
                return job_id, False
    except Exception as e:
        st.error(f"Error processing document: {str(e)}")
        return None, False

# Main area content
if uploaded_file is not None:
    # Prepare processing options
    processing_options = {
        'ocr_mode': ocr_mode.upper().replace(' ', '_'),
        'confidence_threshold': confidence_threshold,
        'extract_tables': extract_tables,
        'extract_forms': extract_forms,
        'analyze_layout': analyze_layout,
        'detect_language': detect_language,
        'output_formats': [fmt.lower() for fmt in output_formats],
        'include_provenance': include_provenance,
        'processing_mode': processing_mode,
        'max_pages': int(max_pages) if max_pages else 0
    }

    # Process the document
    job_id, success = process_document(uploaded_file, processing_options)

    if success:
        st.success(f"Document processed successfully! Job ID: {job_id}")

# If we have a selected job, display results
if st.session_state.current_job_id:
    try:
        job_id = st.session_state.current_job_id
        results = api_client.get_job_results(job_id)
        
        if results:
            # Create tabs for different views
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📃 Document", "📝 Extracted Text", "📊 Tables", 
                "📋 Form Fields", "📥 Downloads"
            ])
            
            # Document view tab
            with tab1:
                st.subheader("Document Overview")
                
                # Document metadata
                if "metadata" in results:
                    metadata = results["metadata"]
                    
                    # Display key metadata
                    cols = st.columns(3)
                    with cols[0]:
                        st.metric("Document Type", metadata.get("document_type", "Unknown"))
                    with cols[1]:
                        st.metric("Pages", metadata.get("page_count", "N/A"))
                    with cols[2]:
                        confidence = metadata.get("confidence", 0) * 100
                        st.metric("Confidence", f"{confidence:.1f}%")
                    
                    # Format creation date if available
                    if "creation_date" in metadata:
                        st.write(f"**Created:** {metadata['creation_date']}")
                    
                    # Document summary if available
                    if "summary" in results:
                        st.markdown("### Summary")
                        st.info(results["summary"])
                
                # Display original document if available
                if "pages" in results:
                    for i, page in enumerate(results["pages"]):
                        if "image" in page:
                            try:
                                # Convert image data if needed
                                image_data = page["image"]
                                if isinstance(image_data, str):
                                    # Try to decode base64
                                    try:
                                        image_data = base64.b64decode(image_data)
                                    except:
                                        pass
                                
                                # Overlay toggles
                                show_ocr = st.checkbox(f"Show OCR boxes (Page {i+1})", value=False, key=f"ocr_{i}")
                                show_regions = st.checkbox(f"Show Classified Regions (Page {i+1})", value=True, key=f"regions_{i}")
                                overlay_type = st.selectbox(
                                    f"Overlay type (Page {i+1})",
                                    ["bboxes", "tokens", "confidence"],
                                    index=0,
                                    key=f"ovtype_{i}"
                                )

                                # Render image with selected overlays
                                from streamlit_demo.components.visualization import render_ocr_overlay, render_region_overlays
                                from PIL import Image
                                if isinstance(image_data, bytes):
                                    img = Image.open(io.BytesIO(image_data)).convert("RGB")
                                elif isinstance(image_data, str):
                                    img = Image.open(image_data).convert("RGB")
                                else:
                                    img = image_data

                                out_img = img
                                if show_ocr and "ocr_results" in results:
                                    out_img = render_ocr_overlay(out_img, results.get("ocr_results", {}), overlay_type)
                                if show_regions and page.get("regions"):
                                    out_img = render_region_overlays(out_img, page.get("regions", []), show_labels=True)
                                st.image(out_img, caption=f"Page {i+1}", use_column_width=True)
                            except Exception as e:
                                st.error(f"Error displaying page {i+1}: {str(e)}")
            
            # Text extraction tab
            with tab2:
                st.subheader("Extracted Text")
                
                # Text extraction settings
                show_confidence = st.checkbox("Highlight low confidence text", value=True)
                confidence_filter = st.slider("Minimum confidence", 0.0, 1.0, 0.5) if show_confidence else 0.0
                
                # Display text by page
                if "pages" in results:
                    for i, page in enumerate(results["pages"]):
                        with st.expander(f"Page {i+1}", expanded=i==0):
                            # If page has pre-formatted text
                            if "text" in page:
                                st.markdown("### Full Text")
                                st.text(page["text"])
                            
                            # If page has text regions
                            if "regions" in page:
                                st.markdown("### Text by Region")
                                visualize_text_regions(
                                    page["regions"], 
                                    highlight_confidence=show_confidence,
                                    confidence_threshold=confidence_filter
                                )
            
            # Tables tab
            with tab3:
                st.subheader("Detected Tables")
                
                # Extract all tables from all pages
                all_tables = []
                if "pages" in results:
                    for i, page in enumerate(results["pages"]):
                        if "tables" in page:
                            for table in page["tables"]:
                                table["page_number"] = i+1
                                all_tables.append(table)
                
                if all_tables:
                    visualize_table_regions(all_tables)
                else:
                    st.info("No tables detected in document")
            
            # Form fields tab
            with tab4:
                st.subheader("Form Fields")
                
                # Extract all form fields from all pages
                all_form_fields = []
                if "pages" in results:
                    for i, page in enumerate(results["pages"]):
                        if "form_fields" in page:
                            for field in page["form_fields"]:
                                field["page_number"] = i+1
                                all_form_fields.append(field)
                
                if all_form_fields:
                    create_form_field_visualization(all_form_fields)
                else:
                    st.info("No form fields detected in document")
            
            # Downloads tab
            with tab5:
                st.subheader("Download Results")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### Document with OCR")
                    ocr_pdf_url = api_client.get_download_url(job_id, "pdf_ocr")
                    st.markdown(f'''
                    <a href="{ocr_pdf_url}" target="_blank">
                        <button style="background-color:#4CAF50;color:white;padding:10px 24px;border:none;border-radius:4px;cursor:pointer;">
                            Download OCR PDF
                        </button>
                    </a>
                    ''', unsafe_allow_html=True)
                
                with col2:
                    st.markdown("#### Original Document")
                    original_url = api_client.get_download_url(job_id, "original")
                    st.markdown(f'''
                    <a href="{original_url}" target="_blank">
                        <button style="background-color:#2196F3;color:white;padding:10px 24px;border:none;border-radius:4px;cursor:pointer;">
                            Download Original
                        </button>
                    </a>
                    ''', unsafe_allow_html=True)
                
                st.markdown("### Extracted Data")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    json_url = api_client.get_download_url(job_id, "json")
                    st.markdown(f'''
                    <a href="{json_url}" target="_blank">
                        <button style="background-color:#ff9800;color:white;padding:10px 24px;border:none;border-radius:4px;cursor:pointer;">
                            Download JSON
                        </button>
                    </a>
                    ''', unsafe_allow_html=True)
                
                with col2:
                    csv_url = api_client.get_download_url(job_id, "csv")
                    st.markdown(f'''
                    <a href="{csv_url}" target="_blank">
                        <button style="background-color:#9c27b0;color:white;padding:10px 24px;border:none;border-radius:4px;cursor:pointer;">
                            Download CSV
                        </button>
                    </a>
                    ''', unsafe_allow_html=True)
                
                with col3:
                    excel_url = api_client.get_download_url(job_id, "excel")
                    st.markdown(f'''
                    <a href="{excel_url}" target="_blank">
                        <button style="background-color:#607d8b;color:white;padding:10px 24px;border:none;border-radius:4px;cursor:pointer;">
                            Download Excel
                        </button>
                    </a>
                    ''', unsafe_allow_html=True)
    
    except Exception as e:
        st.error(f"Error loading job results: {str(e)}")

# If no file is uploaded and no job is selected, show dashboard
if not uploaded_file and not st.session_state.current_job_id:
    st.markdown("## 📊 Dashboard")
    
    # Display analytics dashboard
    create_analytics_dashboard(st.session_state.analytics)
    
    # Display welcome message for new users
    if st.session_state.analytics['processed_docs'] == 0:
        st.info("""
        ### Welcome to CurioScan OCR!
        
        Get started by uploading a document using the sidebar on the left.
        
        **Supported formats:**
        - PDF documents (native and scanned)
        - Images (JPG, PNG, TIFF)
        - Office documents (DOCX, XLSX)
        - Plain text files
        
        The system will extract text, detect tables and form fields, and provide downloadable results in various formats.
        """)

# Footer
st.markdown("---")
st.markdown("&copy; 2025 CurioScan OCR - Production Version 2.0.0")

# Add JavaScript for job list item clicking
st.markdown("""
<script>
function handleJobClick(jobId) {
    // Use Streamlit's component communication mechanism
    window.parent.postMessage({
        type: "streamlit:setComponentValue",
        value: jobId,
        dataType: "string"
    }, "*");
}
</script>
""", unsafe_allow_html=True)

# Main function for running the app
if __name__ == "__main__":
    # This block executes when the script is run directly
    try:
        logging.info("Starting CurioScan OCR Streamlit Demo")
        # The app is already running via Streamlit's execution model
    except Exception as e:
        logging.error(f"Error starting application: {str(e)}")
