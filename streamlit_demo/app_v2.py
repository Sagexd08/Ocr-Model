import streamlit as st
import requests
import time
import json
import pandas as pd
import io
import base64
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px

# App configuration
st.set_page_config(
    page_title="CurioScan OCR",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #2563EB;
        margin-top: 1rem;
    }
    .api-url-input {
        max-width: 400px;
    }
    .card {
        background-color: #F9FAFB;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #EFF6FF;
        border-left: 5px solid #2563EB;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .success-box {
        background-color: #ECFDF5;
        border-left: 5px solid #10B981;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .warning-box {
        background-color: #FFFBEB;
        border-left: 5px solid #F59E0B;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .btn-primary {
        background-color: #2563EB;
        color: white;
        border-radius: 4px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        text-align: center;
        border: none;
        cursor: pointer;
    }
    .btn-primary:hover {
        background-color: #1D4ED8;
    }
    .btn-secondary {
        background-color: #6B7280;
        color: white;
        border-radius: 4px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        text-align: center;
        border: none;
        cursor: pointer;
    }
    .feature-icon {
        font-size: 1.5rem;
        margin-bottom: 0.5rem;
        color: #2563EB;
    }
    .feature-title {
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    .feature-description {
        color: #4B5563;
        font-size: 0.875rem;
    }
    .step-container {
        display: flex;
        margin-bottom: 1rem;
    }
    .step-number {
        background-color: #2563EB;
        color: white;
        border-radius: 50%;
        width: 2rem;
        height: 2rem;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 600;
        margin-right: 1rem;
    }
    .step-content {
        flex: 1;
    }
    .step-title {
        font-weight: 600;
        margin-bottom: 0.25rem;
    }
    .step-description {
        color: #4B5563;
        font-size: 0.875rem;
    }
</style>
""", unsafe_allow_html=True)

# Session state initialization
if 'api_url' not in st.session_state:
    st.session_state.api_url = "http://localhost:8000"
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []
if 'job_ids' not in st.session_state:
    st.session_state.job_ids = []
if 'results' not in st.session_state:
    st.session_state.results = {}
if 'active_tab' not in st.session_state:
    st.session_state.active_tab = "upload"
if 'debug_mode' not in st.session_state:
    st.session_state.debug_mode = False

# Sidebar
with st.sidebar:
    st.markdown('<div class="main-header">CurioScan</div>', unsafe_allow_html=True)
    st.markdown("### Document Intelligence Platform")
    
    # API Configuration
    st.markdown("### API Configuration")
    api_url = st.text_input("API URL", st.session_state.api_url, key="api_url_input", 
                           help="The URL of the CurioScan API server")
    st.session_state.api_url = api_url
    
    # Navigation
    st.markdown("### Navigation")
    nav_upload = st.button("📤 Upload & Process", 
                         key="nav_upload", 
                         help="Upload and process documents")
    nav_results = st.button("📊 View Results", 
                          key="nav_results", 
                          help="View processing results")
    nav_about = st.button("ℹ️ About & Help", 
                        key="nav_about", 
                        help="Learn about CurioScan")
    
    if nav_upload:
        st.session_state.active_tab = "upload"
    elif nav_results:
        st.session_state.active_tab = "results"
    elif nav_about:
        st.session_state.active_tab = "about"
    
    # Settings
    st.markdown("### Settings")
    st.session_state.debug_mode = st.checkbox("Debug Mode", st.session_state.debug_mode,
                                           help="Show additional debugging information")
    
    # Job history
    if st.session_state.job_ids:
        st.markdown("### Recent Jobs")
        for job_id in st.session_state.job_ids[-5:]:  # Show only the last 5 jobs
            st.code(job_id, language=None)

# Helper functions
def check_api_status():
    """Check if the API is available"""
    try:
        response = requests.get(f"{st.session_state.api_url}")
        return response.status_code == 200
    except:
        return False

def upload_document(file, options):
    """Upload a document to the API"""
    files = {'file': file.getvalue()}
    params = {
        'detect_tables': options.get('detect_tables', True),
        'enhance_accuracy': options.get('enhance_accuracy', True)
    }
    
    try:
        response = requests.post(f"{st.session_state.api_url}/upload", files=files, params=params)
        return response.json()
    except Exception as e:
        st.error(f"Error uploading document: {str(e)}")
        return None

def check_job_status(job_id):
    """Check the status of a job"""
    try:
        response = requests.get(f"{st.session_state.api_url}/status/{job_id}")
        return response.json()
    except Exception as e:
        st.error(f"Error checking job status: {str(e)}")
        return None

def get_job_result(job_id):
    """Get the results of a completed job"""
    try:
        response = requests.get(f"{st.session_state.api_url}/review/{job_id}")
        return response.json()
    except Exception as e:
        st.error(f"Error getting job results: {str(e)}")
        return None

def render_workflow_visualization():
    """Render the workflow visualization"""
    # Create a Plotly figure to visualize the workflow
    fig = go.Figure()
    
    # Define the steps in the workflow
    steps = [
        {"name": "Document Upload", "desc": "Document is uploaded and validated", "x": 0, "y": 0},
        {"name": "Pre-processing", "desc": "Document is converted, scaled, and optimized", "x": 1, "y": 0},
        {"name": "Layout Analysis", "desc": "Document structure and regions are identified", "x": 2, "y": 0},
        {"name": "OCR Processing", "desc": "Text is extracted using multiple OCR engines", "x": 3, "y": 0},
        {"name": "Table Detection", "desc": "Tables are identified and structured", "x": 3, "y": -1},
        {"name": "Post-processing", "desc": "Text is corrected and structured", "x": 4, "y": 0},
        {"name": "Export", "desc": "Results are formatted for download", "x": 5, "y": 0}
    ]
    
    # Add nodes
    for step in steps:
        fig.add_trace(go.Scatter(
            x=[step["x"]], 
            y=[step["y"]], 
            mode="markers+text",
            marker=dict(size=30, color="#2563EB"),
            text=[step["name"]],
            textposition="bottom center",
            hoverinfo="text",
            hovertext=step["desc"],
            name=step["name"]
        ))
    
    # Add edges
    edges = [
        (0, 1), (1, 2), (2, 3), (2, 4), (3, 5), (4, 5), (5, 6)
    ]
    
    for edge in edges:
        start = steps[edge[0]]
        end = steps[edge[1]]
        fig.add_trace(go.Scatter(
            x=[start["x"], end["x"]],
            y=[start["y"], end["y"]],
            mode="lines",
            line=dict(width=2, color="#94A3B8"),
            hoverinfo="none",
            showlegend=False
        ))
    
    # Set layout
    fig.update_layout(
        title="Document Processing Workflow",
        showlegend=False,
        hovermode="closest",
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor="#F9FAFB",
        height=300
    )
    
    return fig

def render_bounding_boxes(image_bytes, boxes_data):
    """Render bounding boxes on an image"""
    try:
        # Open image from bytes
        image = Image.open(io.BytesIO(image_bytes))
        draw = ImageDraw.Draw(image)
        
        # Define colors for different types
        colors = {
            "text": (0, 128, 0, 128),    # Green
            "paragraph": (0, 128, 0, 128),
            "heading": (255, 0, 0, 128),  # Red
            "table": (0, 0, 255, 128),    # Blue
            "list": (255, 165, 0, 128),   # Orange
            "figure": (128, 0, 128, 128)  # Purple
        }
        
        # Draw boxes
        for box in boxes_data:
            box_type = box.get("type", "text")
            color = colors.get(box_type, (100, 100, 100, 128))
            
            x1, y1, x2, y2 = box["bbox"]
            draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=2)
            
            # Add label
            draw.text((x1, y1 - 10), box_type, fill=color)
        
        return image
    except Exception as e:
        st.error(f"Error rendering bounding boxes: {str(e)}")
        return None

# Main content based on active tab
if st.session_state.active_tab == "upload":
    st.markdown('<div class="main-header">📤 Document Upload & Processing</div>', unsafe_allow_html=True)
    
    # API Status check
    api_status = check_api_status()
    if api_status:
        st.markdown('<div class="success-box">✅ API is connected and ready</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="warning-box">⚠️ API is not available. Please check the API URL and server status.</div>', 
                  unsafe_allow_html=True)
    
    # Document upload
    st.markdown('<div class="sub-header">Select Documents</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_files = st.file_uploader(
            "Drag and drop files here",
            type=["pdf", "png", "jpg", "jpeg", "tiff", "docx", "doc"],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            st.session_state.uploaded_files = uploaded_files
            
            # Display uploaded files
            st.markdown('<div class="info-box">📄 Uploaded Documents</div>', unsafe_allow_html=True)
            for file in uploaded_files:
                st.write(f"- {file.name} ({file.size} bytes)")
    
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Processing Options")
        
        detect_tables = st.checkbox("Detect Tables", value=True, 
                                  help="Enable table detection and extraction")
        enhance_accuracy = st.checkbox("Enhanced Accuracy", value=True, 
                                     help="Use model ensemble for higher accuracy")
        preserve_layout = st.checkbox("Preserve Layout", value=True, 
                                    help="Maintain original document layout")
        
        options = {
            "detect_tables": detect_tables,
            "enhance_accuracy": enhance_accuracy,
            "preserve_layout": preserve_layout
        }
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Process button
    if st.button("🚀 Process Documents", key="process_btn", disabled=not api_status or not uploaded_files):
        with st.spinner("Processing documents..."):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, file in enumerate(uploaded_files):
                # Update progress
                progress = int((i / len(uploaded_files)) * 100)
                progress_bar.progress(progress)
                status_text.text(f"Processing {file.name}...")
                
                # Upload and process file
                result = upload_document(file, options)
                if result and "job_id" in result:
                    job_id = result["job_id"]
                    st.session_state.job_ids.append(job_id)
                    
                    # Monitor job status
                    status = "STARTED"
                    while status not in ["SUCCESS", "FAILURE"]:
                        job_status = check_job_status(job_id)
                        if job_status:
                            status = job_status["status"]
                            status_text.text(f"Processing {file.name}: {status}")
                            
                            # If job has progress info, update progress bar
                            if "progress" in job_status:
                                inner_progress = job_status["progress"] / 100
                                progress_bar.progress(progress + (inner_progress / len(uploaded_files)))
                                
                            time.sleep(0.5)
                        else:
                            status = "FAILURE"
                    
                    # Get results if successful
                    if status == "SUCCESS":
                        result_data = get_job_result(job_id)
                        if result_data:
                            st.session_state.results[job_id] = {
                                "filename": file.name,
                                "data": result_data,
                                "timestamp": datetime.now().isoformat()
                            }
                    else:
                        st.error(f"Failed to process {file.name}")
            
            # Complete progress
            progress_bar.progress(100)
            status_text.text("Processing complete!")
            
            # Switch to results tab
            st.session_state.active_tab = "results"
            st.experimental_rerun()
    
    # Workflow visualization
    st.markdown('<div class="sub-header">How It Works</div>', unsafe_allow_html=True)
    workflow_fig = render_workflow_visualization()
    st.plotly_chart(workflow_fig, use_container_width=True)
    
    # Features
    st.markdown('<div class="sub-header">Features</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="feature-icon">🔍</div>', unsafe_allow_html=True)
        st.markdown('<div class="feature-title">Advanced OCR</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="feature-description">Multi-model ensemble for highest accuracy text extraction</div>', 
            unsafe_allow_html=True
        )
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="feature-icon">📊</div>', unsafe_allow_html=True)
        st.markdown('<div class="feature-title">Table Extraction</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="feature-description">Intelligent table detection and structured data extraction</div>', 
            unsafe_allow_html=True
        )
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="feature-icon">📱</div>', unsafe_allow_html=True)
        st.markdown('<div class="feature-title">Multiple Formats</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="feature-description">Support for PDFs, images, and Word documents</div>', 
            unsafe_allow_html=True
        )
        st.markdown("</div>", unsafe_allow_html=True)

elif st.session_state.active_tab == "results":
    st.markdown('<div class="main-header">📊 Processing Results</div>', unsafe_allow_html=True)
    
    if not st.session_state.job_ids:
        st.info("No documents have been processed yet. Please upload and process documents first.")
    else:
        # Job selection
        job_selection = st.selectbox(
            "Select a processed document", 
            options=st.session_state.job_ids,
            format_func=lambda x: f"{st.session_state.results[x]['filename']} ({x})" if x in st.session_state.results else x
        )
        
        if job_selection and job_selection in st.session_state.results:
            result = st.session_state.results[job_selection]
            filename = result["filename"]
            data = result["data"]
            
            st.markdown(f'<div class="sub-header">Results for {filename}</div>', unsafe_allow_html=True)
            
            # Result tabs
            text_tab, tables_tab, visualization_tab, download_tab = st.tabs([
                "📝 Extracted Text", "📊 Tables", "🔍 Visualization", "📥 Download"
            ])
            
            # Extracted text tab
            with text_tab:
                if "pages" in data:
                    for page in data["pages"]:
                        with st.expander(f"Page {page['page_number']}", expanded=page["page_number"]==1):
                            if "extracted_text" in page:
                                st.code(page["extracted_text"], language=None)
                            else:
                                tokens_text = "\n".join([token["text"] for token in page.get("tokens", [])])
                                st.code(tokens_text, language=None)
            
            # Tables tab
            with tables_tab:
                has_tables = False
                
                for page in data.get("pages", []):
                    tables = page.get("tables", [])
                    if tables:
                        has_tables = True
                        for i, table in enumerate(tables):
                            with st.expander(f"Table {i+1} (Page {page['page_number']})", expanded=i==0):
                                if "data" in table:
                                    # Convert to DataFrame
                                    try:
                                        df = pd.DataFrame(table["data"])
                                        if len(df) > 0:
                                            # Use first row as header if possible
                                            header = df.iloc[0]
                                            df = df.iloc[1:].reset_index(drop=True)
                                            df.columns = header
                                        st.dataframe(df, use_container_width=True)
                                    except:
                                        st.dataframe(table["data"], use_container_width=True)
                                
                                # Show table image if available
                                if "image" in table and table["image"]:
                                    try:
                                        st.image(table["image"], caption="Table Image")
                                    except:
                                        st.warning("Could not display table image")
                
                if not has_tables:
                    st.info("No tables detected in this document.")
            
            # Visualization tab
            with visualization_tab:
                st.markdown("### Document Structure Visualization")
                
                # This is a placeholder - in a production app, we would render the document with bounding boxes
                # for the detected regions, tables, etc.
                
                if st.session_state.debug_mode:
                    # Show token bounding boxes
                    for page in data.get("pages", []):
                        tokens = page.get("tokens", [])
                        if tokens:
                            # Create visualization
                            st.write(f"Page {page['page_number']} - Token Bounding Boxes")
                            
                            # In a real app, we would render this on the actual page image
                            fig = go.Figure()
                            
                            for token in tokens:
                                if "bbox" in token:
                                    x1, y1, x2, y2 = token["bbox"]
                                    fig.add_shape(
                                        type="rect",
                                        x0=x1,
                                        y0=y1,
                                        x1=x2,
                                        y1=y2,
                                        line=dict(color="blue", width=1),
                                        fillcolor="rgba(0, 0, 255, 0.1)",
                                    )
                                    
                                    # Add text label
                                    fig.add_trace(go.Scatter(
                                        x=[(x1 + x2) / 2],
                                        y=[(y1 + y2) / 2],
                                        text=[token["text"]],
                                        mode="text",
                                        textposition="middle center",
                                        hoverinfo="text",
                                        showlegend=False
                                    ))
                            
                            # Set layout
                            max_x = max([token["bbox"][2] for token in tokens if "bbox" in token], default=600)
                            max_y = max([token["bbox"][3] for token in tokens if "bbox" in token], default=800)
                            
                            fig.update_layout(
                                width=min(max_x, 800),
                                height=min(max_y, 800),
                                xaxis=dict(range=[0, max_x], showgrid=False),
                                yaxis=dict(range=[0, max_y], showgrid=False, scaleanchor="x", scaleratio=1),
                                plot_bgcolor="white"
                            )
                            
                            st.plotly_chart(fig)
            
            # Download tab
            with download_tab:
                st.markdown("### Download Extracted Data")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    # CSV download
                    csv_url = f"{st.session_state.api_url}/download/{job_selection}?format=csv"
                    st.markdown(
                        f'<a href="{csv_url}" target="_blank"><div class="btn-primary">📄 Download CSV</div></a>', 
                        unsafe_allow_html=True
                    )
                
                with col2:
                    # Excel download
                    excel_url = f"{st.session_state.api_url}/download/{job_selection}?format=excel"
                    st.markdown(
                        f'<a href="{excel_url}" target="_blank"><div class="btn-primary">📊 Download Excel</div></a>', 
                        unsafe_allow_html=True
                    )
                
                with col3:
                    # JSON download
                    json_url = f"{st.session_state.api_url}/download/{job_selection}?format=json"
                    st.markdown(
                        f'<a href="{json_url}" target="_blank"><div class="btn-primary">🔄 Download JSON</div></a>', 
                        unsafe_allow_html=True
                    )
                
                # Metadata
                if "metadata" in data:
                    st.markdown("### Document Metadata")
                    st.json(data["metadata"])
        else:
            st.warning("Job not found or processing not yet complete.")

elif st.session_state.active_tab == "about":
    st.markdown('<div class="main-header">ℹ️ About CurioScan</div>', unsafe_allow_html=True)
    
    st.markdown("""
    CurioScan is a state-of-the-art Intelligent Document Processing (IDP) system designed to extract,
    analyze, and structure information from a variety of document types including PDFs, images, and Word documents.
    
    ### Key Features
    
    - **Advanced OCR:** Multi-model ensemble with Tesseract, PaddleOCR and LayoutLMv3
    - **Layout Analysis:** Intelligent document structure understanding
    - **Table Extraction:** Automatic detection and structuring of tabular data
    - **Multi-format Support:** Works with PDFs, scanned documents, images, and Word files
    - **High Accuracy:** Enhanced processing pipeline with post-correction
    - **Flexible Outputs:** Export to CSV, Excel, or JSON formats
    
    ### How It Works
    """)
    
    # How it works steps
    st.markdown('<div class="card">', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.markdown('<div class="step-number">1</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="step-title">Document Upload</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="step-description">Documents are uploaded and validated for processing</div>', 
            unsafe_allow_html=True
        )
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.markdown('<div class="step-number">2</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="step-title">Layout Analysis</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="step-description">LayoutLMv3 analyzes document structure and identifies different regions</div>', 
            unsafe_allow_html=True
        )
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.markdown('<div class="step-number">3</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="step-title">Text & Table Extraction</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="step-description">Multi-model OCR extracts text while specialized detectors handle tables</div>', 
            unsafe_allow_html=True
        )
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.markdown('<div class="step-number">4</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="step-title">Post-processing & Validation</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="step-description">Transformer-based correction ensures high accuracy and structured data</div>', 
            unsafe_allow_html=True
        )
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.markdown('<div class="step-number">5</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="step-title">Export & Download</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="step-description">Results are available in multiple formats for integration</div>', 
            unsafe_allow_html=True
        )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Technical specs
    st.markdown('<div class="sub-header">Technical Specifications</div>', unsafe_allow_html=True)
    
    tech_col1, tech_col2 = st.columns(2)
    
    with tech_col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### OCR Engines")
        st.markdown("""
        - Tesseract OCR (v5.0)
        - PaddleOCR
        - TrOCR (transformer-based)
        - LayoutLMv3 for document understanding
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tech_col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Backend Architecture")
        st.markdown("""
        - FastAPI + Celery + Redis for async processing
        - Docker-based deployment
        - Scalable worker architecture
        - Cloud-ready design
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Contact info
    st.markdown('<div class="sub-header">Contact & Support</div>', unsafe_allow_html=True)
    
    st.markdown("""
    For questions, feature requests, or technical support, please contact our team:
    
    - **Email:** support@curioscan.ai
    - **Documentation:** https://docs.curioscan.ai
    - **GitHub:** https://github.com/curioscan/ocr-engine
    """)

# Footer
st.markdown("---")
st.markdown("© 2025 CurioScan | Intelligent Document Processing")
