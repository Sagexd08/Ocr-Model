"""
CurioScan Streamlit Demo Application

Main application file with elegant, bold UI and card-based layout.
"""

import streamlit as st
import requests
import json
import time
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import io
import base64
from typing import Dict, Any, List, Optional

# Configure page
st.set_page_config(
    page_title="CurioScan - Production OCR System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import custom components
try:
    from components.ui_components import (
        render_header, render_sidebar, render_upload_card,
        render_pipeline_status, render_results_viewer,
        render_training_dashboard, render_export_options
    )
    from components.api_client import CurioScanAPIClient
    from components.visualization import (
        render_ocr_overlay, render_table_editor,
        render_provenance_inspector, render_confidence_heatmap
    )
except ImportError:
    # Fallback if components not available
    pass

# Initialize session state
if 'api_client' not in st.session_state:
    api_base_url = st.secrets.get("API_BASE_URL", "http://localhost:8000") if hasattr(st, 'secrets') else "http://localhost:8000"
    try:
        st.session_state.api_client = CurioScanAPIClient(api_base_url)
    except:
        st.session_state.api_client = None

if 'current_job' not in st.session_state:
    st.session_state.current_job = None

if 'processing_results' not in st.session_state:
    st.session_state.processing_results = None

if 'selected_page' not in st.session_state:
    st.session_state.selected_page = 1

if 'overlay_mode' not in st.session_state:
    st.session_state.overlay_mode = "bboxes"

# Custom CSS for elegant styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    
    .card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
        border-left: 4px solid #667eea;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        margin: 0.5rem;
    }
    
    .status-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 8px;
    }
    
    .status-pending { background-color: #ffc107; }
    .status-processing { background-color: #17a2b8; }
    .status-completed { background-color: #28a745; }
    .status-failed { background-color: #dc3545; }
    
    .pipeline-step {
        background: #f8f9fa;
        border: 2px solid #e9ecef;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        transition: all 0.3s ease;
    }
    
    .pipeline-step.active {
        border-color: #667eea;
        background: #e7f3ff;
    }
    
    .pipeline-step.completed {
        border-color: #28a745;
        background: #d4edda;
    }
    
    .confidence-high { color: #28a745; font-weight: bold; }
    .confidence-medium { color: #ffc107; font-weight: bold; }
    .confidence-low { color: #dc3545; font-weight: bold; }
</style>
""", unsafe_allow_html=True)


def main():
    """Main application function."""
    
    # Render header
    render_header()
    
    # Sidebar navigation
    page = render_sidebar()
    
    # Main content area
    if page == "Upload & Process":
        render_upload_page()
    elif page == "Results Viewer":
        render_results_page()
    elif page == "Training Dashboard":
        render_training_page()
    elif page == "Settings":
        render_settings_page()


def render_upload_page():
    """Render the upload and processing page."""
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### 📤 Upload Document")
        
        # Upload card
        uploaded_file = st.file_uploader(
            "Choose a document",
            type=['pdf', 'png', 'jpg', 'jpeg', 'tiff', 'docx'],
            help="Supported formats: PDF, Images (PNG, JPG, TIFF), DOCX"
        )
        
        # Processing options
        st.markdown("### ⚙️ Processing Options")
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.8,
            step=0.05,
            help="Minimum confidence for OCR results"
        )
        
        # Process button
        if uploaded_file is not None:
            if st.button("🚀 Start Processing", type="primary", use_container_width=True):
                process_document(uploaded_file, confidence_threshold)
        
        # Pipeline status
        if st.session_state.current_job:
            render_pipeline_status_card()
    
    with col2:
        st.markdown("### 📊 Live Preview")
        
        if uploaded_file is not None:
            # Display uploaded file
            if uploaded_file.type.startswith('image/'):
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Document", use_column_width=True)
            elif uploaded_file.type == 'application/pdf':
                st.info("📄 PDF uploaded. Preview will be available after processing.")
            else:
                st.info("📄 Document uploaded. Preview will be available after processing.")
        else:
            st.info("👆 Upload a document to see preview")


def render_results_page():
    """Render the results viewing page."""
    
    if not st.session_state.processing_results:
        st.warning("No processing results available. Please upload and process a document first.")
        return
    
    # Results header
    st.markdown("### 📋 Processing Results")
    
    results = st.session_state.processing_results
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{len(results.get('rows', []))}</h3>
            <p>Total Rows</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        confidence = results.get('confidence_score', 0.0)
        st.markdown(f"""
        <div class="metric-card">
            <h3>{confidence:.1%}</h3>
            <p>Avg Confidence</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        review_count = sum(1 for row in results.get('rows', []) if row.get('needs_review', False))
        st.markdown(f"""
        <div class="metric-card">
            <h3>{review_count}</h3>
            <p>Need Review</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        render_type = results.get('render_type', 'unknown')
        st.markdown(f"""
        <div class="metric-card">
            <h3>{render_type.title()}</h3>
            <p>Document Type</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Main results area
    tab1, tab2, tab3, tab4 = st.tabs(["📄 OCR Overlay", "📊 Table View", "🔍 Provenance", "📈 Export"])
    
    with tab1:
        render_ocr_overlay_tab()
    
    with tab2:
        render_table_view_tab()
    
    with tab3:
        render_provenance_tab()
    
    with tab4:
        render_export_tab()


def render_training_page():
    """Render the training dashboard page."""
    
    st.markdown("### 🎯 Training Dashboard")
    
    # Training metrics
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Mock training data for demo
        epochs = list(range(1, 21))
        train_loss = [0.8 - 0.03 * i + 0.01 * (i % 3) for i in epochs]
        val_loss = [0.85 - 0.025 * i + 0.015 * (i % 4) for i in epochs]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=epochs, y=train_loss, name="Training Loss", line=dict(color="#667eea")))
        fig.add_trace(go.Scatter(x=epochs, y=val_loss, name="Validation Loss", line=dict(color="#764ba2")))
        
        fig.update_layout(
            title="Training Progress",
            xaxis_title="Epoch",
            yaxis_title="Loss",
            template="plotly_white"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### 🎛️ Training Controls")
        
        model_type = st.selectbox(
            "Model Type",
            ["renderer_classifier", "ocr_models", "table_detector", "layout_analyzer"]
        )
        
        dataset_path = st.text_input("Dataset Path", value="data/corrected_extractions")
        
        dry_run = st.checkbox("Dry Run", value=True)
        
        if st.button("🚀 Start Retraining", type="primary", use_container_width=True):
            trigger_retraining(model_type, dataset_path, dry_run)
        
        # Model checkpoints
        st.markdown("#### 📦 Model Checkpoints")
        
        # Mock checkpoint data
        checkpoints = [
            {"version": "v1.2.3", "f1_score": 0.95, "active": True},
            {"version": "v1.2.2", "f1_score": 0.93, "active": False},
            {"version": "v1.2.1", "f1_score": 0.91, "active": False},
        ]
        
        for checkpoint in checkpoints:
            status = "🟢 Active" if checkpoint["active"] else "⚪ Inactive"
            st.markdown(f"**{checkpoint['version']}** - F1: {checkpoint['f1_score']:.3f} - {status}")


def render_settings_page():
    """Render the settings page."""
    
    st.markdown("### ⚙️ Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🔧 Processing Settings")
        
        st.slider("Default Confidence Threshold", 0.0, 1.0, 0.8, 0.05)
        st.selectbox("Default OCR Engine", ["ensemble", "tesseract", "easyocr"])
        st.checkbox("Enable Table Detection", value=True)
        st.checkbox("Enable Layout Analysis", value=True)
        
    with col2:
        st.markdown("#### 🎨 UI Settings")
        
        st.selectbox("Theme", ["Light", "Dark"])
        st.checkbox("Show Confidence Scores", value=True)
        st.checkbox("Auto-refresh Results", value=False)
        st.slider("Refresh Interval (seconds)", 1, 30, 5)


def process_document(uploaded_file, confidence_threshold):
    """Process uploaded document."""
    
    try:
        # Upload file
        with st.spinner("Uploading document..."):
            job_id = st.session_state.api_client.upload_file(uploaded_file, confidence_threshold)
            st.session_state.current_job = job_id
            st.success(f"Document uploaded! Job ID: {job_id}")
        
        # Poll for results
        poll_for_results(job_id)
        
    except Exception as e:
        st.error(f"Failed to process document: {str(e)}")


def poll_for_results(job_id):
    """Poll for processing results."""
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    while True:
        try:
            status = st.session_state.api_client.get_job_status(job_id)
            
            progress = status.get('progress', 0.0)
            job_status = status.get('status', 'unknown')
            message = status.get('message', '')
            
            progress_bar.progress(progress)
            status_text.text(f"Status: {job_status} - {message}")
            
            if job_status == 'completed':
                # Get results
                results = st.session_state.api_client.get_job_results(job_id)
                st.session_state.processing_results = results
                st.success("Processing completed!")
                break
            elif job_status == 'failed':
                st.error("Processing failed!")
                break
            
            time.sleep(2)
            
        except Exception as e:
            st.error(f"Error polling status: {str(e)}")
            break


def render_pipeline_status_card():
    """Render pipeline status card."""
    
    if not st.session_state.current_job:
        return
    
    st.markdown("### 🔄 Pipeline Status")
    
    # Mock pipeline steps for demo
    steps = [
        {"name": "Document Classification", "status": "completed", "time": "0.5s"},
        {"name": "Preprocessing", "status": "completed", "time": "1.2s"},
        {"name": "OCR Extraction", "status": "processing", "time": "3.1s"},
        {"name": "Table Detection", "status": "pending", "time": "-"},
        {"name": "Postprocessing", "status": "pending", "time": "-"},
    ]
    
    for step in steps:
        status_class = f"status-{step['status']}"
        step_class = "pipeline-step"
        if step['status'] == 'processing':
            step_class += " active"
        elif step['status'] == 'completed':
            step_class += " completed"
        
        st.markdown(f"""
        <div class="{step_class}">
            <span class="status-indicator {status_class}"></span>
            <strong>{step['name']}</strong>
            <span style="float: right;">{step['time']}</span>
        </div>
        """, unsafe_allow_html=True)


def render_ocr_overlay_tab():
    """Render OCR overlay visualization tab."""
    
    st.markdown("#### 📄 OCR Overlay Viewer")
    
    # Controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        overlay_mode = st.selectbox(
            "Overlay Mode",
            ["bboxes", "tokens", "confidence"],
            index=0
        )
    
    with col2:
        page_num = st.number_input(
            "Page Number",
            min_value=1,
            max_value=10,
            value=1
        )
    
    with col3:
        confidence_filter = st.slider(
            "Min Confidence",
            0.0, 1.0, 0.0, 0.05
        )
    
    # Placeholder for OCR overlay
    st.info("📄 OCR overlay visualization would appear here with the selected page and overlay mode.")


def render_table_view_tab():
    """Render table view tab."""
    
    st.markdown("#### 📊 Interactive Table Editor")
    
    if not st.session_state.processing_results:
        st.warning("No results to display")
        return
    
    # Convert results to DataFrame
    rows = st.session_state.processing_results.get('rows', [])
    
    if not rows:
        st.warning("No table data found")
        return
    
    # Create DataFrame from results
    table_data = []
    for row in rows:
        row_data = {
            'Row ID': row.get('row_id', ''),
            'Page': row.get('page', 0),
            'Region': row.get('region_id', ''),
            'Confidence': row.get('provenance', {}).get('confidence', 0.0),
            'Needs Review': row.get('needs_review', False)
        }
        
        # Add column data
        columns = row.get('columns', {})
        for col_name, col_value in columns.items():
            row_data[col_name] = col_value
        
        table_data.append(row_data)
    
    df = pd.DataFrame(table_data)
    
    # Display editable table
    edited_df = st.data_editor(
        df,
        use_container_width=True,
        num_rows="dynamic",
        column_config={
            "Confidence": st.column_config.ProgressColumn(
                "Confidence",
                help="OCR confidence score",
                min_value=0.0,
                max_value=1.0,
            ),
            "Needs Review": st.column_config.CheckboxColumn(
                "Needs Review",
                help="Requires human review",
            )
        }
    )
    
    # Save changes button
    if st.button("💾 Save Changes", type="primary"):
        st.success("Changes saved successfully!")


def render_provenance_tab():
    """Render provenance inspector tab."""
    
    st.markdown("#### 🔍 Provenance Inspector")
    
    # Mock provenance data
    st.markdown("""
    **Click on any cell in the table view to see detailed provenance information:**
    
    - **Source File**: document.pdf
    - **Page**: 1
    - **Bounding Box**: [123, 456, 789, 012]
    - **Token IDs**: [45, 46, 47, 48]
    - **OCR Confidence**: 0.92
    - **Model Used**: ensemble
    - **Processing Time**: 2.3s
    """)
    
    # Placeholder for provenance visualization
    st.info("🔍 Detailed provenance information and source highlighting would appear here.")


def render_export_tab():
    """Render export options tab."""
    
    st.markdown("#### 📈 Export Options")
    
    if not st.session_state.processing_results:
        st.warning("No results to export")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("##### Export Formats")
        
        export_format = st.selectbox(
            "Format",
            ["JSON", "CSV", "XLSX", "Provenance ZIP"]
        )
        
        include_provenance = st.checkbox("Include Provenance", value=True)
        include_confidence = st.checkbox("Include Confidence Scores", value=True)
        
        if st.button("📥 Download", type="primary", use_container_width=True):
            download_results(export_format, include_provenance, include_confidence)
    
    with col2:
        st.markdown("##### Export Preview")
        
        # Show preview based on format
        if export_format == "JSON":
            st.json(st.session_state.processing_results)
        elif export_format in ["CSV", "XLSX"]:
            # Convert to DataFrame for preview
            rows = st.session_state.processing_results.get('rows', [])
            if rows:
                preview_data = []
                for row in rows[:5]:  # Show first 5 rows
                    row_data = row.get('columns', {})
                    row_data['confidence'] = row.get('provenance', {}).get('confidence', 0.0)
                    preview_data.append(row_data)
                
                df = pd.DataFrame(preview_data)
                st.dataframe(df, use_container_width=True)


def download_results(format_type, include_provenance, include_confidence):
    """Download results in specified format."""
    
    try:
        job_id = st.session_state.current_job
        if not job_id:
            st.error("No job ID available")
            return
        
        # Get download URL from API
        download_url = st.session_state.api_client.get_download_url(
            job_id, format_type.lower(), include_provenance
        )
        
        st.success(f"Download started! Format: {format_type}")
        st.markdown(f"[📥 Download Link]({download_url})")
        
    except Exception as e:
        st.error(f"Download failed: {str(e)}")


def trigger_retraining(model_type, dataset_path, dry_run):
    """Trigger model retraining."""
    
    try:
        result = st.session_state.api_client.trigger_retraining(
            model_type, dataset_path, {}, dry_run
        )
        
        if dry_run:
            st.success("Dry run completed successfully!")
        else:
            st.success(f"Retraining started! Job ID: {result.get('retrain_job_id')}")
        
    except Exception as e:
        st.error(f"Retraining failed: {str(e)}")


if __name__ == "__main__":
    main()
