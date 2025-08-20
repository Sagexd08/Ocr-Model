"""
UI components for CurioScan Streamlit demo.
"""

import streamlit as st
from typing import Dict, Any, List, Optional


def render_header():
    """Render the main application header."""
    
    st.markdown("""
    <div class="main-header">
        <h1>🔍 CurioScan</h1>
        <h3>Production-Grade OCR System</h3>
        <p>Intelligent document processing with human-in-the-loop review</p>
    </div>
    """, unsafe_allow_html=True)


def render_sidebar():
    """Render the sidebar navigation."""
    
    st.sidebar.markdown("## 🧭 Navigation")
    
    page = st.sidebar.radio(
        "Choose a page:",
        ["Upload & Process", "Results Viewer", "Training Dashboard", "Settings"],
        index=0
    )
    
    st.sidebar.markdown("---")
    
    # System status
    st.sidebar.markdown("### 📊 System Status")
    
    # Mock system metrics
    st.sidebar.metric("API Status", "🟢 Online", delta="99.9% uptime")
    st.sidebar.metric("Queue Length", "3", delta="-2 from last hour")
    st.sidebar.metric("Active Workers", "4", delta="1 worker added")
    
    st.sidebar.markdown("---")
    
    # Quick stats
    st.sidebar.markdown("### 📈 Quick Stats")
    st.sidebar.metric("Documents Processed", "1,247", delta="23 today")
    st.sidebar.metric("Avg Confidence", "94.2%", delta="2.1% improvement")
    st.sidebar.metric("Review Queue", "12", delta="-5 resolved")
    
    return page


def render_upload_card():
    """Render the upload card component."""
    
    st.markdown("""
    <div class="card">
        <h3>📤 Upload Document</h3>
        <p>Drag and drop or click to upload your document</p>
    </div>
    """, unsafe_allow_html=True)


def render_pipeline_status():
    """Render pipeline status component."""
    
    st.markdown("### 🔄 Processing Pipeline")
    
    # Pipeline steps
    steps = [
        {"name": "Document Upload", "status": "completed", "icon": "📤"},
        {"name": "Classification", "status": "completed", "icon": "🏷️"},
        {"name": "Preprocessing", "status": "processing", "icon": "⚙️"},
        {"name": "OCR Extraction", "status": "pending", "icon": "👁️"},
        {"name": "Table Detection", "status": "pending", "icon": "📊"},
        {"name": "Postprocessing", "status": "pending", "icon": "✨"},
    ]
    
    for i, step in enumerate(steps):
        status_color = {
            "completed": "#28a745",
            "processing": "#17a2b8", 
            "pending": "#6c757d"
        }.get(step["status"], "#6c757d")
        
        st.markdown(f"""
        <div style="
            display: flex; 
            align-items: center; 
            padding: 0.5rem; 
            margin: 0.25rem 0;
            border-left: 4px solid {status_color};
            background: #f8f9fa;
            border-radius: 4px;
        ">
            <span style="font-size: 1.2em; margin-right: 0.5rem;">{step["icon"]}</span>
            <span style="flex-grow: 1;">{step["name"]}</span>
            <span style="color: {status_color}; font-weight: bold;">
                {step["status"].title()}
            </span>
        </div>
        """, unsafe_allow_html=True)


def render_results_viewer():
    """Render results viewer component."""
    
    st.markdown("### 📋 Results Viewer")
    
    # Placeholder for results
    st.info("Results will appear here after processing")


def render_training_dashboard():
    """Render training dashboard component."""
    
    st.markdown("### 🎯 Training Dashboard")
    
    # Training metrics placeholder
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Model Accuracy", "94.2%", delta="1.3%")
    
    with col2:
        st.metric("Training Loss", "0.045", delta="-0.012")
    
    with col3:
        st.metric("Validation F1", "0.932", delta="0.021")


def render_export_options():
    """Render export options component."""
    
    st.markdown("### 📥 Export Options")
    
    export_formats = ["JSON", "CSV", "XLSX", "Provenance ZIP"]
    
    for format_type in export_formats:
        if st.button(f"Export as {format_type}", key=f"export_{format_type}"):
            st.success(f"Exporting as {format_type}...")


def render_confidence_indicator(confidence: float) -> str:
    """Render confidence indicator with color coding."""
    
    if confidence >= 0.8:
        color = "#28a745"
        label = "High"
    elif confidence >= 0.6:
        color = "#ffc107"
        label = "Medium"
    else:
        color = "#dc3545"
        label = "Low"
    
    return f"""
    <span style="color: {color}; font-weight: bold;">
        {confidence:.1%} ({label})
    </span>
    """


def render_status_badge(status: str) -> str:
    """Render status badge with appropriate styling."""
    
    colors = {
        "pending": "#ffc107",
        "processing": "#17a2b8",
        "completed": "#28a745",
        "failed": "#dc3545",
        "cancelled": "#6c757d"
    }
    
    color = colors.get(status, "#6c757d")
    
    return f"""
    <span style="
        background-color: {color};
        color: white;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-size: 0.875rem;
        font-weight: bold;
    ">
        {status.title()}
    </span>
    """


def render_metric_card(title: str, value: str, delta: str = None, icon: str = "📊"):
    """Render a metric card component."""
    
    delta_html = ""
    if delta:
        delta_color = "#28a745" if delta.startswith("+") or not delta.startswith("-") else "#dc3545"
        delta_html = f'<p style="color: {delta_color}; margin: 0; font-size: 0.875rem;">{delta}</p>'
    
    return f"""
    <div class="metric-card">
        <div style="font-size: 1.5em; margin-bottom: 0.5rem;">{icon}</div>
        <h3 style="margin: 0; font-size: 1.5rem;">{value}</h3>
        <p style="margin: 0; font-size: 0.875rem; opacity: 0.8;">{title}</p>
        {delta_html}
    </div>
    """


def render_progress_bar(progress: float, label: str = ""):
    """Render a custom progress bar."""
    
    progress_percent = int(progress * 100)
    
    return f"""
    <div style="margin: 1rem 0;">
        <div style="display: flex; justify-content: space-between; margin-bottom: 0.25rem;">
            <span>{label}</span>
            <span>{progress_percent}%</span>
        </div>
        <div style="
            width: 100%;
            height: 8px;
            background-color: #e9ecef;
            border-radius: 4px;
            overflow: hidden;
        ">
            <div style="
                width: {progress_percent}%;
                height: 100%;
                background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
                transition: width 0.3s ease;
            "></div>
        </div>
    </div>
    """


def render_alert(message: str, alert_type: str = "info"):
    """Render a custom alert component."""
    
    colors = {
        "info": {"bg": "#d1ecf1", "border": "#bee5eb", "text": "#0c5460"},
        "success": {"bg": "#d4edda", "border": "#c3e6cb", "text": "#155724"},
        "warning": {"bg": "#fff3cd", "border": "#ffeaa7", "text": "#856404"},
        "error": {"bg": "#f8d7da", "border": "#f5c6cb", "text": "#721c24"}
    }
    
    style = colors.get(alert_type, colors["info"])
    
    icons = {
        "info": "ℹ️",
        "success": "✅",
        "warning": "⚠️",
        "error": "❌"
    }
    
    icon = icons.get(alert_type, "ℹ️")
    
    return f"""
    <div style="
        background-color: {style['bg']};
        border: 1px solid {style['border']};
        color: {style['text']};
        padding: 0.75rem 1rem;
        border-radius: 0.375rem;
        margin: 1rem 0;
        display: flex;
        align-items: center;
    ">
        <span style="margin-right: 0.5rem; font-size: 1.2em;">{icon}</span>
        <span>{message}</span>
    </div>
    """
