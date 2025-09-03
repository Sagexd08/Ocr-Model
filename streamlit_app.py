#!/usr/bin/env python3
"""
OCR System - Streamlit Web Interface
Complete setup with document upload, processing, and results visualization
"""

import streamlit as st
import json
import time
import os
import tempfile
from pathlib import Path
import pandas as pd
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

# Import OCR system components
from worker.model_manager import ModelManager
from worker.storage_manager import StorageManager
from worker.document_processor import EnhancedDocumentProcessor
from worker.types import ProcessingMode

# Page configuration
st.set_page_config(
    page_title="OCR Document Processing System",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.25rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 0.25rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'processor' not in st.session_state:
    st.session_state.processor = None
    st.session_state.processing_history = []
    st.session_state.current_results = None

def initialize_ocr_system():
    """Initialize the OCR system components"""
    try:
        with st.spinner("Initializing OCR system..."):
            model_manager = ModelManager()
            storage_manager = StorageManager()
            processor = EnhancedDocumentProcessor(
                model_manager=model_manager, 
                storage_manager=storage_manager
            )
            st.session_state.processor = processor
            return True
    except Exception as e:
        st.error(f"Failed to initialize OCR system: {e}")
        return False

def process_document(uploaded_file, settings):
    """Process uploaded document with OCR"""
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            temp_path = tmp_file.name

        # Generate job ID
        job_id = f"streamlit_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Process document
        start_time = time.time()
        
        result = st.session_state.processor.process_document(
            job_id=job_id,
            document_path=temp_path,
            params={
                "mode": settings["mode"],
                "max_pages": settings["max_pages"],
                "profile": settings["profile"],
                "export_format": "json",
                "allow_cache": False
            }
        )
        
        processing_time = time.time() - start_time
        
        # Load detailed results
        detailed_results = None
        if result.get("result_path") and os.path.exists(result["result_path"]):
            with open(result["result_path"], 'r', encoding='utf-8') as f:
                detailed_results = json.load(f)
        
        # Clean up temp file
        os.unlink(temp_path)
        
        # Store results
        processing_record = {
            "timestamp": datetime.now(),
            "filename": uploaded_file.name,
            "job_id": job_id,
            "settings": settings,
            "processing_time": processing_time,
            "summary": result.get("summary", {}),
            "detailed_results": detailed_results,
            "status": result.get("status", "unknown")
        }
        
        st.session_state.processing_history.append(processing_record)
        st.session_state.current_results = processing_record
        
        return processing_record
        
    except Exception as e:
        st.error(f"Error processing document: {e}")
        return None

def display_results(results):
    """Display processing results with visualizations"""
    if not results:
        return
    
    st.markdown("## 📊 Processing Results")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    summary = results["summary"]
    
    with col1:
        st.metric("Pages Processed", summary.get("page_count", 0))
    
    with col2:
        st.metric("Words Extracted", summary.get("word_count", 0))
    
    with col3:
        st.metric("Processing Time", f"{results['processing_time']:.2f}s")
    
    with col4:
        st.metric("Document Type", summary.get("document_type", "Unknown").upper())
    
    # Detailed results
    if results["detailed_results"] and results["detailed_results"].get("pages"):
        st.markdown("### 📄 Extracted Text by Page")
        
        pages = results["detailed_results"]["pages"]
        
        for i, page in enumerate(pages):
            with st.expander(f"Page {page.get('page_num', i+1)} - {len(page.get('tokens', []))} tokens"):
                tokens = page.get("tokens", [])
                
                if tokens:
                    # Extract text
                    page_text = " ".join([token.get("text", "") for token in tokens])
                    st.text_area("Extracted Text", page_text, height=200, key=f"text_{i}")
                    
                    # Token analysis
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Confidence distribution
                        confidences = [token.get("confidence", 0) for token in tokens]
                        if confidences:
                            fig = px.histogram(
                                x=confidences,
                                title="OCR Confidence Distribution",
                                labels={"x": "Confidence Score", "y": "Number of Tokens"}
                            )
                            st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Token statistics
                        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
                        st.metric("Average Confidence", f"{avg_confidence:.1%}")
                        st.metric("High Confidence Tokens", f"{sum(1 for c in confidences if c > 0.9)}")
                        st.metric("Low Confidence Tokens", f"{sum(1 for c in confidences if c < 0.7)}")
                    
                    # Token details table
                    if st.checkbox(f"Show token details for page {page.get('page_num', i+1)}", key=f"details_{i}"):
                        token_df = pd.DataFrame([
                            {
                                "Text": token.get("text", ""),
                                "Confidence": f"{token.get('confidence', 0):.1%}",
                                "X1": token.get("bbox", [0,0,0,0])[0],
                                "Y1": token.get("bbox", [0,0,0,0])[1],
                                "X2": token.get("bbox", [0,0,0,0])[2],
                                "Y2": token.get("bbox", [0,0,0,0])[3]
                            }
                            for token in tokens[:50]  # Limit to first 50 tokens
                        ])
                        st.dataframe(token_df, use_container_width=True)
                else:
                    st.warning("No tokens extracted from this page")

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">🔍 OCR Document Processing System</h1>', unsafe_allow_html=True)
    
    # Sidebar for settings
    st.sidebar.markdown("## ⚙️ Processing Settings")
    
    # Initialize system if needed
    if st.session_state.processor is None:
        if st.sidebar.button("🚀 Initialize OCR System"):
            if initialize_ocr_system():
                st.sidebar.success("✅ OCR System Ready!")
                st.rerun()
        else:
            st.warning("Please initialize the OCR system first using the sidebar.")
            return
    
    # Processing settings
    profile = st.sidebar.selectbox(
        "Processing Profile",
        ["performance", "quality", "default"],
        index=0,
        help="Performance: Fast processing, Quality: High accuracy, Default: Balanced"
    )
    
    mode = st.sidebar.selectbox(
        "Processing Mode",
        ["advanced", "standard", "fast"],
        index=0,
        help="Advanced: Full OCR + analysis, Standard: Basic OCR, Fast: Quick extraction"
    )
    
    max_pages = st.sidebar.number_input(
        "Max Pages to Process",
        min_value=1,
        max_value=50,
        value=1,
        help="Limit processing to first N pages for faster results"
    )
    
    settings = {
        "profile": profile,
        "mode": mode,
        "max_pages": max_pages
    }
    
    # Main content area
    tab1, tab2, tab3 = st.tabs(["📤 Upload & Process", "📊 Results", "📈 History"])
    
    with tab1:
        st.markdown("## 📤 Document Upload")
        
        uploaded_file = st.file_uploader(
            "Choose a document to process",
            type=["pdf", "png", "jpg", "jpeg", "tiff", "bmp"],
            help="Supported formats: PDF, PNG, JPG, JPEG, TIFF, BMP"
        )
        
        if uploaded_file is not None:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.info(f"📄 **File:** {uploaded_file.name}")
                st.info(f"📏 **Size:** {uploaded_file.size / 1024:.1f} KB")
                st.info(f"🔧 **Settings:** {profile.title()} profile, {mode.title()} mode, {max_pages} page(s)")
            
            with col2:
                if st.button("🚀 Process Document", type="primary"):
                    with st.spinner("Processing document... This may take a few moments."):
                        results = process_document(uploaded_file, settings)
                        if results:
                            st.success("✅ Document processed successfully!")
                            st.rerun()
    
    with tab2:
        st.markdown("## 📊 Current Results")
        if st.session_state.current_results:
            display_results(st.session_state.current_results)
        else:
            st.info("No results to display. Please process a document first.")
    
    with tab3:
        st.markdown("## 📈 Processing History")
        
        if st.session_state.processing_history:
            # Summary statistics
            total_docs = len(st.session_state.processing_history)
            total_pages = sum(r["summary"].get("page_count", 0) for r in st.session_state.processing_history)
            total_words = sum(r["summary"].get("word_count", 0) for r in st.session_state.processing_history)
            avg_time = sum(r["processing_time"] for r in st.session_state.processing_history) / total_docs
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Documents Processed", total_docs)
            with col2:
                st.metric("Total Pages", total_pages)
            with col3:
                st.metric("Total Words", total_words)
            with col4:
                st.metric("Avg Processing Time", f"{avg_time:.2f}s")
            
            # Processing history table
            history_df = pd.DataFrame([
                {
                    "Timestamp": r["timestamp"].strftime("%Y-%m-%d %H:%M:%S"),
                    "Filename": r["filename"],
                    "Profile": r["settings"]["profile"],
                    "Mode": r["settings"]["mode"],
                    "Pages": r["summary"].get("page_count", 0),
                    "Words": r["summary"].get("word_count", 0),
                    "Time (s)": f"{r['processing_time']:.2f}",
                    "Status": r["status"]
                }
                for r in reversed(st.session_state.processing_history)
            ])
            
            st.dataframe(history_df, use_container_width=True)
            
            # Processing time chart
            if len(st.session_state.processing_history) > 1:
                fig = px.line(
                    x=[r["timestamp"] for r in st.session_state.processing_history],
                    y=[r["processing_time"] for r in st.session_state.processing_history],
                    title="Processing Time Over Time",
                    labels={"x": "Timestamp", "y": "Processing Time (seconds)"}
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No processing history available.")
    
    # Footer
    st.markdown("---")
    st.markdown("**OCR System Status:** ✅ Ready | **Version:** 1.0.0 | **Last Updated:** 2025-09-03")

if __name__ == "__main__":
    main()
