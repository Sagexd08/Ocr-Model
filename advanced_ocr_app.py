#!/usr/bin/env python3
"""
Advanced OCR Document Processing System - Enterprise Streamlit Interface
Complete production-ready OCR system with advanced analytics and processing
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
import io
import base64

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
        'Get Help': 'https://docs.ocr-system.com',
        'Report a bug': 'https://github.com/ocr-system/issues',
        'About': "Advanced OCR Document Processing System v2.0 - Enterprise Edition"
    }
)

# Advanced CSS styling
st.markdown("""
<style>
    .main-header {
        font-size: 3.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #1f77b4, #ff7f0e, #2ca02c);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
        animation: gradient 3s ease-in-out infinite;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        color: white;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    .success-box {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        border-radius: 1rem;
        padding: 1.5rem;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(79, 172, 254, 0.3);
    }
    .error-box {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        border-radius: 1rem;
        padding: 1.5rem;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(250, 112, 154, 0.3);
    }
    .processing-card {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        border-radius: 1rem;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(168, 237, 234, 0.3);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 3rem;
        padding: 0 2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 1rem;
        color: white;
        border: none;
    }
    .upload-area {
        border: 3px dashed #1f77b4;
        border-radius: 1rem;
        padding: 2rem;
        text-align: center;
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        margin: 1rem 0;
    }
    .stats-container {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        border-radius: 1rem;
        padding: 1.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state with comprehensive tracking
if 'processor' not in st.session_state:
    st.session_state.processor = None
    st.session_state.processing_history = []
    st.session_state.current_results = None
    st.session_state.processing_queue = queue.Queue()
    st.session_state.system_stats = {
        'total_documents': 0,
        'total_pages': 0,
        'total_words': 0,
        'avg_confidence': 0.0,
        'processing_time_total': 0.0,
        'success_rate': 100.0,
        'error_count': 0
    }
    st.session_state.performance_metrics = []
    st.session_state.confidence_history = []
    st.session_state.processing_times = []

def initialize_ocr_system():
    """Initialize the advanced OCR system with comprehensive error handling"""
    try:
        with st.spinner("🚀 Initializing Advanced OCR System..."):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Step 1: Initialize Model Manager
            status_text.text("Loading AI models...")
            progress_bar.progress(25)
            model_manager = ModelManager()
            
            # Step 2: Initialize Storage Manager
            status_text.text("Setting up storage systems...")
            progress_bar.progress(50)
            storage_manager = StorageManager()
            
            # Step 3: Initialize Document Processor
            status_text.text("Configuring document processor...")
            progress_bar.progress(75)
            processor = EnhancedDocumentProcessor(
                model_manager=model_manager, 
                storage_manager=storage_manager
            )
            
            # Step 4: System validation
            status_text.text("Validating system components...")
            progress_bar.progress(100)
            
            st.session_state.processor = processor
            status_text.text("✅ OCR System Ready!")
            time.sleep(1)
            progress_bar.empty()
            status_text.empty()
            
            return True
    except Exception as e:
        st.error(f"❌ Failed to initialize OCR system: {e}")
        return False

def process_document_advanced(uploaded_file, settings):
    """Advanced document processing with comprehensive analytics"""
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            temp_path = tmp_file.name

        # Generate comprehensive job ID
        job_id = f"adv_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        
        # Advanced processing with real-time updates
        start_time = time.time()
        
        # Create processing status container
        status_container = st.container()
        with status_container:
            st.markdown("### 🔄 Processing Status")
            progress_bar = st.progress(0)
            status_text = st.empty()
            metrics_container = st.container()
        
        # Update status
        status_text.text("📄 Analyzing document structure...")
        progress_bar.progress(20)
        
        # Process document
        result = st.session_state.processor.process_document(
            job_id=job_id,
            document_path=temp_path,
            params={
                "mode": settings["mode"],
                "max_pages": settings["max_pages"],
                "profile": settings["profile"],
                "export_format": "json",
                "allow_cache": settings.get("allow_cache", False),
                "extract_tables": settings.get("extract_tables", True),
                "classify_document": settings.get("classify_document", True),
                "extract_metadata": settings.get("extract_metadata", True)
            }
        )
        
        status_text.text("🔍 Performing OCR analysis...")
        progress_bar.progress(60)
        
        processing_time = time.time() - start_time
        
        status_text.text("📊 Generating analytics...")
        progress_bar.progress(80)
        
        # Load detailed results
        detailed_results = None
        if result.get("result_path") and os.path.exists(result["result_path"]):
            with open(result["result_path"], 'r', encoding='utf-8') as f:
                detailed_results = json.load(f)
        
        status_text.text("✅ Processing complete!")
        progress_bar.progress(100)
        
        # Clean up
        os.unlink(temp_path)
        
        # Create comprehensive processing record
        processing_record = {
            "timestamp": datetime.now(),
            "filename": uploaded_file.name,
            "file_size": uploaded_file.size,
            "job_id": job_id,
            "settings": settings,
            "processing_time": processing_time,
            "summary": result.get("summary", {}),
            "detailed_results": detailed_results,
            "status": result.get("status", "unknown"),
            "confidence_scores": [],
            "word_count": 0,
            "page_count": 0,
            "error_count": 0
        }
        
        # Extract advanced metrics
        if detailed_results and detailed_results.get("pages"):
            pages = detailed_results["pages"]
            all_confidences = []
            total_words = 0
            
            for page in pages:
                tokens = page.get("tokens", [])
                page_confidences = [token.get("confidence", 0) for token in tokens]
                all_confidences.extend(page_confidences)
                total_words += len(tokens)
            
            processing_record["confidence_scores"] = all_confidences
            processing_record["word_count"] = total_words
            processing_record["page_count"] = len(pages)
            processing_record["avg_confidence"] = np.mean(all_confidences) if all_confidences else 0
        
        # Update session state
        st.session_state.processing_history.append(processing_record)
        st.session_state.current_results = processing_record
        
        # Update system statistics
        stats = st.session_state.system_stats
        stats['total_documents'] += 1
        stats['total_pages'] += processing_record["page_count"]
        stats['total_words'] += processing_record["word_count"]
        stats['processing_time_total'] += processing_time
        
        if processing_record["confidence_scores"]:
            stats['avg_confidence'] = np.mean([
                np.mean(record["confidence_scores"]) 
                for record in st.session_state.processing_history 
                if record["confidence_scores"]
            ])
        
        # Clear status
        time.sleep(1)
        status_container.empty()
        
        return processing_record

    except Exception as e:
        st.error(f"❌ Error processing document: {e}")
        return None

def display_advanced_results(results):
    """Display comprehensive results with advanced analytics and visualizations"""
    if not results:
        return

    st.markdown("## 📊 Advanced Processing Results")

    # Key Performance Indicators
    col1, col2, col3, col4, col5 = st.columns(5)

    summary = results["summary"]

    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>📄 Pages</h3>
            <h2>{}</h2>
        </div>
        """.format(summary.get("page_count", 0)), unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>📝 Words</h3>
            <h2>{:,}</h2>
        </div>
        """.format(summary.get("word_count", 0)), unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>⚡ Time</h3>
            <h2>{:.2f}s</h2>
        </div>
        """.format(results["processing_time"]), unsafe_allow_html=True)

    with col4:
        avg_conf = results.get("avg_confidence", 0)
        st.markdown("""
        <div class="metric-card">
            <h3>🎯 Confidence</h3>
            <h2>{:.1%}</h2>
        </div>
        """.format(avg_conf), unsafe_allow_html=True)

    with col5:
        doc_type = summary.get("document_type", "Unknown").upper()
        st.markdown("""
        <div class="metric-card">
            <h3>📋 Type</h3>
            <h2>{}</h2>
        </div>
        """.format(doc_type), unsafe_allow_html=True)

    # Advanced Analytics Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📄 Extracted Text",
        "📊 Analytics",
        "🔍 Token Analysis",
        "📈 Performance",
        "💾 Export"
    ])

    with tab1:
        st.markdown("### 📄 Extracted Text by Page")

        if results["detailed_results"] and results["detailed_results"].get("pages"):
            pages = results["detailed_results"]["pages"]

            for i, page in enumerate(pages):
                with st.expander(f"📄 Page {page.get('page_num', i+1)} - {len(page.get('tokens', []))} tokens", expanded=i==0):
                    tokens = page.get("tokens", [])

                    if tokens:
                        # Extract and display text
                        page_text = " ".join([token.get("text", "") for token in tokens])

                        # Text statistics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Characters", len(page_text))
                        with col2:
                            st.metric("Words", len(page_text.split()))
                        with col3:
                            avg_conf = np.mean([token.get("confidence", 0) for token in tokens])
                            st.metric("Avg Confidence", f"{avg_conf:.1%}")

                        # Display text with formatting
                        st.markdown("**Extracted Text:**")
                        st.text_area(
                            "Text Content",
                            page_text,
                            height=300,
                            key=f"text_{i}",
                            help="Raw extracted text from OCR"
                        )

                        # Word cloud option
                        if st.checkbox(f"Generate Word Cloud for Page {i+1}", key=f"wordcloud_{i}"):
                            try:
                                from wordcloud import WordCloud
                                import matplotlib.pyplot as plt

                                if page_text.strip():
                                    wordcloud = WordCloud(
                                        width=800,
                                        height=400,
                                        background_color='white',
                                        colormap='viridis'
                                    ).generate(page_text)

                                    fig, ax = plt.subplots(figsize=(10, 5))
                                    ax.imshow(wordcloud, interpolation='bilinear')
                                    ax.axis('off')
                                    st.pyplot(fig)
                                else:
                                    st.warning("No text available for word cloud")
                            except ImportError:
                                st.info("Install wordcloud package for word cloud visualization")
                    else:
                        st.warning("No tokens extracted from this page")
        else:
            st.info("No detailed results available")

    with tab2:
        st.markdown("### 📊 Advanced Analytics")

        if results["confidence_scores"]:
            # Confidence distribution
            fig_conf = px.histogram(
                x=results["confidence_scores"],
                nbins=20,
                title="OCR Confidence Score Distribution",
                labels={"x": "Confidence Score", "y": "Number of Tokens"},
                color_discrete_sequence=['#1f77b4']
            )
            fig_conf.update_layout(
                showlegend=False,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_conf, use_container_width=True)

            # Confidence statistics
            conf_stats = pd.DataFrame({
                'Metric': ['Mean', 'Median', 'Std Dev', 'Min', 'Max'],
                'Value': [
                    np.mean(results["confidence_scores"]),
                    np.median(results["confidence_scores"]),
                    np.std(results["confidence_scores"]),
                    np.min(results["confidence_scores"]),
                    np.max(results["confidence_scores"])
                ]
            })

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Confidence Statistics:**")
                st.dataframe(conf_stats, use_container_width=True)

            with col2:
                # Quality assessment
                high_conf = sum(1 for c in results["confidence_scores"] if c > 0.9)
                med_conf = sum(1 for c in results["confidence_scores"] if 0.7 <= c <= 0.9)
                low_conf = sum(1 for c in results["confidence_scores"] if c < 0.7)

                quality_data = pd.DataFrame({
                    'Quality': ['High (>90%)', 'Medium (70-90%)', 'Low (<70%)'],
                    'Count': [high_conf, med_conf, low_conf],
                    'Percentage': [
                        high_conf / len(results["confidence_scores"]) * 100,
                        med_conf / len(results["confidence_scores"]) * 100,
                        low_conf / len(results["confidence_scores"]) * 100
                    ]
                })

                fig_quality = px.pie(
                    quality_data,
                    values='Count',
                    names='Quality',
                    title="OCR Quality Distribution",
                    color_discrete_sequence=['#2ca02c', '#ff7f0e', '#d62728']
                )
                st.plotly_chart(fig_quality, use_container_width=True)
        else:
            st.info("No confidence data available for analysis")

    with tab3:
        st.markdown("### 🔍 Detailed Token Analysis")

        if results["detailed_results"] and results["detailed_results"].get("pages"):
            # Token analysis controls
            col1, col2, col3 = st.columns(3)
            with col1:
                min_confidence = st.slider("Minimum Confidence", 0.0, 1.0, 0.0, 0.05)
            with col2:
                page_filter = st.selectbox(
                    "Page Filter",
                    ["All Pages"] + [f"Page {i+1}" for i in range(len(results["detailed_results"]["pages"]))]
                )
            with col3:
                max_tokens = st.number_input("Max Tokens to Display", 1, 1000, 100)

            # Collect and filter tokens
            all_tokens = []
            pages = results["detailed_results"]["pages"]

            for page_idx, page in enumerate(pages):
                if page_filter == "All Pages" or page_filter == f"Page {page_idx + 1}":
                    for token in page.get("tokens", []):
                        if token.get("confidence", 0) >= min_confidence:
                            token_data = {
                                "Page": page_idx + 1,
                                "Text": token.get("text", ""),
                                "Confidence": token.get("confidence", 0),
                                "X1": token.get("bbox", [0,0,0,0])[0],
                                "Y1": token.get("bbox", [0,0,0,0])[1],
                                "X2": token.get("bbox", [0,0,0,0])[2],
                                "Y2": token.get("bbox", [0,0,0,0])[3],
                                "Width": token.get("bbox", [0,0,0,0])[2] - token.get("bbox", [0,0,0,0])[0],
                                "Height": token.get("bbox", [0,0,0,0])[3] - token.get("bbox", [0,0,0,0])[1]
                            }
                            all_tokens.append(token_data)

            if all_tokens:
                # Display tokens table
                tokens_df = pd.DataFrame(all_tokens[:max_tokens])

                # Format confidence as percentage
                tokens_df["Confidence"] = tokens_df["Confidence"].apply(lambda x: f"{x:.1%}")

                st.markdown(f"**Showing {len(tokens_df)} tokens (filtered from {len(all_tokens)} total)**")
                st.dataframe(
                    tokens_df,
                    use_container_width=True,
                    column_config={
                        "Text": st.column_config.TextColumn("Text", width="large"),
                        "Confidence": st.column_config.TextColumn("Confidence", width="small"),
                        "Page": st.column_config.NumberColumn("Page", width="small")
                    }
                )

                # Token length analysis
                if st.checkbox("Show Token Length Analysis"):
                    token_lengths = [len(token["Text"]) for token in all_tokens]
                    fig_lengths = px.histogram(
                        x=token_lengths,
                        nbins=20,
                        title="Token Length Distribution",
                        labels={"x": "Token Length (characters)", "y": "Count"}
                    )
                    st.plotly_chart(fig_lengths, use_container_width=True)
            else:
                st.warning("No tokens match the current filters")
        else:
            st.info("No token data available")

    with tab4:
        st.markdown("### 📈 Performance Metrics")

        # Current document performance
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Current Document Performance:**")
            perf_metrics = {
                "Processing Time": f"{results['processing_time']:.2f} seconds",
                "Pages per Second": f"{results['page_count'] / results['processing_time']:.2f}" if results['processing_time'] > 0 else "N/A",
                "Words per Second": f"{results['word_count'] / results['processing_time']:.0f}" if results['processing_time'] > 0 else "N/A",
                "File Size": f"{results['file_size'] / 1024:.1f} KB",
                "Processing Rate": f"{results['file_size'] / 1024 / results['processing_time']:.1f} KB/s" if results['processing_time'] > 0 else "N/A"
            }

            for metric, value in perf_metrics.items():
                st.metric(metric, value)

        with col2:
            # System performance over time
            if len(st.session_state.processing_history) > 1:
                history_df = pd.DataFrame([
                    {
                        "Timestamp": record["timestamp"],
                        "Processing Time": record["processing_time"],
                        "Pages": record["page_count"],
                        "Words": record["word_count"],
                        "Confidence": record.get("avg_confidence", 0)
                    }
                    for record in st.session_state.processing_history
                ])

                fig_perf = px.line(
                    history_df,
                    x="Timestamp",
                    y="Processing Time",
                    title="Processing Time Trend",
                    markers=True
                )
                st.plotly_chart(fig_perf, use_container_width=True)
            else:
                st.info("Process more documents to see performance trends")

        # System statistics
        st.markdown("**System Statistics:**")
        stats = st.session_state.system_stats

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Documents", stats['total_documents'])
        with col2:
            st.metric("Total Pages", stats['total_pages'])
        with col3:
            st.metric("Total Words", f"{stats['total_words']:,}")
        with col4:
            avg_time = stats['processing_time_total'] / stats['total_documents'] if stats['total_documents'] > 0 else 0
            st.metric("Avg Processing Time", f"{avg_time:.2f}s")

    with tab5:
        st.markdown("### 💾 Export Options")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**📄 Text Export**")
            if results["detailed_results"] and results["detailed_results"].get("pages"):
                # Extract all text
                all_text = []
                for page in results["detailed_results"]["pages"]:
                    page_text = " ".join([token.get("text", "") for token in page.get("tokens", [])])
                    all_text.append(f"=== Page {page.get('page_num', 1)} ===\n{page_text}\n")

                full_text = "\n".join(all_text)

                st.download_button(
                    label="📄 Download as TXT",
                    data=full_text,
                    file_name=f"{results['job_id']}_extracted_text.txt",
                    mime="text/plain"
                )

        with col2:
            st.markdown("**📊 Data Export**")
            if results["detailed_results"]:
                # Create comprehensive CSV
                csv_data = []
                for page in results["detailed_results"].get("pages", []):
                    for token in page.get("tokens", []):
                        csv_data.append({
                            "Page": page.get("page_num", 1),
                            "Text": token.get("text", ""),
                            "Confidence": token.get("confidence", 0),
                            "X1": token.get("bbox", [0,0,0,0])[0],
                            "Y1": token.get("bbox", [0,0,0,0])[1],
                            "X2": token.get("bbox", [0,0,0,0])[2],
                            "Y2": token.get("bbox", [0,0,0,0])[3]
                        })

                if csv_data:
                    csv_df = pd.DataFrame(csv_data)
                    csv_string = csv_df.to_csv(index=False)

                    st.download_button(
                        label="📊 Download as CSV",
                        data=csv_string,
                        file_name=f"{results['job_id']}_tokens.csv",
                        mime="text/csv"
                    )

        with col3:
            st.markdown("**🔧 JSON Export**")
            if results["detailed_results"]:
                json_string = json.dumps(results["detailed_results"], indent=2, default=str)

                st.download_button(
                    label="🔧 Download as JSON",
                    data=json_string,
                    file_name=f"{results['job_id']}_full_results.json",
                    mime="application/json"
                )

        # Processing report
        st.markdown("**📋 Processing Report**")
        report = f"""
# OCR Processing Report

## Document Information
- **Filename:** {results['filename']}
- **File Size:** {results['file_size'] / 1024:.1f} KB
- **Processing Date:** {results['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}
- **Job ID:** {results['job_id']}

## Processing Settings
- **Mode:** {results['settings']['mode']}
- **Profile:** {results['settings']['profile']}
- **Max Pages:** {results['settings']['max_pages']}

## Results Summary
- **Pages Processed:** {results['page_count']}
- **Words Extracted:** {results['word_count']:,}
- **Average Confidence:** {results.get('avg_confidence', 0):.1%}
- **Processing Time:** {results['processing_time']:.2f} seconds
- **Processing Rate:** {results['word_count'] / results['processing_time']:.0f} words/second

## Quality Metrics
- **High Confidence Tokens (>90%):** {sum(1 for c in results.get('confidence_scores', []) if c > 0.9)}
- **Medium Confidence Tokens (70-90%):** {sum(1 for c in results.get('confidence_scores', []) if 0.7 <= c <= 0.9)}
- **Low Confidence Tokens (<70%):** {sum(1 for c in results.get('confidence_scores', []) if c < 0.7)}

Generated by Advanced OCR Processing System v2.0
        """

        st.download_button(
            label="📋 Download Processing Report",
            data=report,
            file_name=f"{results['job_id']}_report.md",
            mime="text/markdown"
        )

def display_system_dashboard():
    """Display comprehensive system dashboard"""
    st.markdown("## 🎛️ System Dashboard")

    if st.session_state.processing_history:
        # System overview metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        stats = st.session_state.system_stats

        with col1:
            st.markdown("""
            <div class="metric-card">
                <h3>📚 Documents</h3>
                <h2>{}</h2>
            </div>
            """.format(stats['total_documents']), unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div class="metric-card">
                <h3>📄 Pages</h3>
                <h2>{}</h2>
            </div>
            """.format(stats['total_pages']), unsafe_allow_html=True)

        with col3:
            st.markdown("""
            <div class="metric-card">
                <h3>📝 Words</h3>
                <h2>{:,}</h2>
            </div>
            """.format(stats['total_words']), unsafe_allow_html=True)

        with col4:
            st.markdown("""
            <div class="metric-card">
                <h3>🎯 Avg Confidence</h3>
                <h2>{:.1%}</h2>
            </div>
            """.format(stats['avg_confidence']), unsafe_allow_html=True)

        with col5:
            avg_time = stats['processing_time_total'] / stats['total_documents'] if stats['total_documents'] > 0 else 0
            st.markdown("""
            <div class="metric-card">
                <h3>⚡ Avg Time</h3>
                <h2>{:.1f}s</h2>
            </div>
            """.format(avg_time), unsafe_allow_html=True)

        # Processing history table
        st.markdown("### 📊 Processing History")

        history_data = []
        for record in reversed(st.session_state.processing_history[-20:]):  # Last 20 records
            history_data.append({
                "Timestamp": record["timestamp"].strftime("%Y-%m-%d %H:%M:%S"),
                "Filename": record["filename"][:30] + "..." if len(record["filename"]) > 30 else record["filename"],
                "Pages": record["page_count"],
                "Words": record["word_count"],
                "Confidence": f"{record.get('avg_confidence', 0):.1%}",
                "Time (s)": f"{record['processing_time']:.2f}",
                "Status": "✅" if record["status"] == "completed" else "❌",
                "Job ID": record["job_id"]
            })

        if history_data:
            history_df = pd.DataFrame(history_data)
            st.dataframe(
                history_df,
                use_container_width=True,
                column_config={
                    "Filename": st.column_config.TextColumn("Filename", width="large"),
                    "Job ID": st.column_config.TextColumn("Job ID", width="medium")
                }
            )

            # Performance analytics
            col1, col2 = st.columns(2)

            with col1:
                # Processing time trend
                time_data = [record["processing_time"] for record in st.session_state.processing_history]
                timestamps = [record["timestamp"] for record in st.session_state.processing_history]

                fig_time = px.line(
                    x=timestamps,
                    y=time_data,
                    title="Processing Time Trend",
                    labels={"x": "Time", "y": "Processing Time (seconds)"},
                    markers=True
                )
                fig_time.update_layout(showlegend=False)
                st.plotly_chart(fig_time, use_container_width=True)

            with col2:
                # Confidence trend
                conf_data = [record.get("avg_confidence", 0) for record in st.session_state.processing_history]

                fig_conf = px.line(
                    x=timestamps,
                    y=conf_data,
                    title="Average Confidence Trend",
                    labels={"x": "Time", "y": "Average Confidence"},
                    markers=True
                )
                fig_conf.update_layout(showlegend=False)
                st.plotly_chart(fig_conf, use_container_width=True)

        # System controls
        st.markdown("### ⚙️ System Controls")
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("🗑️ Clear History", type="secondary"):
                st.session_state.processing_history = []
                st.session_state.system_stats = {
                    'total_documents': 0,
                    'total_pages': 0,
                    'total_words': 0,
                    'avg_confidence': 0.0,
                    'processing_time_total': 0.0,
                    'success_rate': 100.0,
                    'error_count': 0
                }
                st.rerun()

        with col2:
            if st.button("📊 Export Analytics", type="secondary"):
                analytics_data = {
                    "system_stats": st.session_state.system_stats,
                    "processing_history": [
                        {
                            **record,
                            "timestamp": record["timestamp"].isoformat()
                        }
                        for record in st.session_state.processing_history
                    ]
                }

                st.download_button(
                    label="📊 Download Analytics JSON",
                    data=json.dumps(analytics_data, indent=2),
                    file_name=f"ocr_analytics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )

        with col3:
            if st.button("🔄 Refresh Dashboard", type="secondary"):
                st.rerun()
    else:
        st.info("No processing history available. Process some documents to see analytics.")

# Main Application Interface
def main():
    """Main application interface with advanced features"""

    # Header with gradient styling
    st.markdown('<h1 class="main-header">🔍 Advanced OCR Processing System</h1>', unsafe_allow_html=True)
    st.markdown("### Enterprise-Grade Document Processing with AI-Powered Analytics")

    # Sidebar with advanced settings
    with st.sidebar:
        st.markdown("## ⚙️ Advanced Configuration")

        # System initialization
        if st.session_state.processor is None:
            st.markdown("### 🚀 System Initialization")
            if st.button("🚀 Initialize OCR System", type="primary", use_container_width=True):
                if OCR_AVAILABLE and initialize_ocr_system():
                    st.success("✅ OCR System Ready!")
                    st.rerun()
                else:
                    st.error("❌ Failed to initialize OCR system")
        else:
            st.success("✅ OCR System Active")

        st.divider()

        # Processing settings
        st.markdown("### 🔧 Processing Settings")

        profile = st.selectbox(
            "Processing Profile",
            ["performance", "quality", "balanced"],
            index=0,
            help="Performance: Fast processing, Quality: High accuracy, Balanced: Optimal speed/accuracy"
        )

        mode = st.selectbox(
            "Processing Mode",
            ["advanced", "standard", "fast"],
            index=0,
            help="Advanced: Full OCR + analysis, Standard: Basic OCR, Fast: Quick extraction"
        )

        max_pages = st.number_input(
            "Max Pages to Process",
            min_value=1,
            max_value=100,
            value=5,
            help="Limit processing to first N pages for faster results"
        )

        st.divider()

        # Advanced options
        st.markdown("### 🎛️ Advanced Options")

        extract_tables = st.checkbox("Extract Tables", value=True)
        classify_document = st.checkbox("Document Classification", value=True)
        extract_metadata = st.checkbox("Extract Metadata", value=True)
        allow_cache = st.checkbox("Enable Caching", value=False)

        # Confidence threshold
        confidence_threshold = st.slider(
            "Confidence Threshold",
            0.0, 1.0, 0.7, 0.05,
            help="Minimum confidence for token acceptance"
        )

        st.divider()

        # System information
        st.markdown("### ℹ️ System Information")
        st.info(f"""
        **Status:** {'🟢 Active' if st.session_state.processor else '🔴 Inactive'}
        **Documents Processed:** {st.session_state.system_stats['total_documents']}
        **Total Pages:** {st.session_state.system_stats['total_pages']}
        **System Uptime:** {datetime.now().strftime('%H:%M:%S')}
        """)

    # Main content area with tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📤 Document Processing",
        "📊 Current Results",
        "🎛️ System Dashboard",
        "📚 Help & Documentation"
    ])

    with tab1:
        st.markdown("## 📤 Advanced Document Processing")

        if st.session_state.processor is None:
            st.warning("⚠️ Please initialize the OCR system first using the sidebar.")
            return

        # File upload with advanced options
        st.markdown("### 📁 Document Upload")

        uploaded_file = st.file_uploader(
            "Choose a document to process",
            type=["pdf", "png", "jpg", "jpeg", "tiff", "bmp", "webp"],
            help="Supported formats: PDF, PNG, JPG, JPEG, TIFF, BMP, WebP (Max size: 200MB)"
        )

        if uploaded_file is not None:
            # File information
            col1, col2 = st.columns([2, 1])

            with col1:
                st.markdown(f"""
                <div class="upload-area">
                    <h4>📄 {uploaded_file.name}</h4>
                    <p><strong>Size:</strong> {uploaded_file.size / 1024:.1f} KB</p>
                    <p><strong>Type:</strong> {uploaded_file.type}</p>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                # Processing settings summary
                settings = {
                    "profile": profile,
                    "mode": mode,
                    "max_pages": max_pages,
                    "extract_tables": extract_tables,
                    "classify_document": classify_document,
                    "extract_metadata": extract_metadata,
                    "allow_cache": allow_cache,
                    "confidence_threshold": confidence_threshold
                }

                st.markdown("**Processing Configuration:**")
                st.json(settings)

            # Process button
            if st.button("🚀 Process Document", type="primary", use_container_width=True):
                with st.spinner("🔄 Processing document with advanced analytics..."):
                    results = process_document_advanced(uploaded_file, settings)
                    if results:
                        st.success("✅ Document processed successfully!")
                        st.balloons()
                        st.rerun()

    with tab2:
        st.markdown("## 📊 Current Processing Results")
        if st.session_state.current_results:
            display_advanced_results(st.session_state.current_results)
        else:
            st.info("No results to display. Please process a document first.")

    with tab3:
        display_system_dashboard()

    with tab4:
        st.markdown("## 📚 Help & Documentation")

        # Quick start guide
        with st.expander("🚀 Quick Start Guide", expanded=True):
            st.markdown("""
            ### Getting Started with Advanced OCR System

            1. **Initialize System**: Click "🚀 Initialize OCR System" in the sidebar
            2. **Configure Settings**: Choose processing profile and options
            3. **Upload Document**: Drag & drop or browse for your file
            4. **Process**: Click "🚀 Process Document" and wait for results
            5. **Analyze**: Review extracted text, analytics, and performance metrics
            6. **Export**: Download results in various formats

            ### Processing Profiles
            - **Performance**: Optimized for speed (~5-10 seconds per page)
            - **Quality**: Maximum accuracy with detailed analysis
            - **Balanced**: Optimal balance of speed and accuracy

            ### Supported Features
            - ✅ Multi-format document support (PDF, images)
            - ✅ Advanced OCR with confidence scoring
            - ✅ Table extraction and analysis
            - ✅ Document classification
            - ✅ Metadata extraction
            - ✅ Real-time analytics and visualization
            - ✅ Comprehensive export options
            - ✅ Performance monitoring and optimization
            """)

        # Advanced features
        with st.expander("🎛️ Advanced Features"):
            st.markdown("""
            ### Advanced Analytics
            - **Confidence Distribution**: Analyze OCR quality across documents
            - **Performance Metrics**: Track processing speed and efficiency
            - **Token Analysis**: Detailed examination of extracted text elements
            - **System Dashboard**: Comprehensive overview of processing history

            ### Export Options
            - **Text Files**: Plain text extraction
            - **CSV Data**: Structured token data with coordinates
            - **JSON**: Complete processing results with metadata
            - **Processing Reports**: Detailed analysis summaries

            ### System Monitoring
            - **Real-time Performance**: Live processing metrics
            - **Historical Analytics**: Trend analysis over time
            - **Quality Assessment**: Confidence scoring and validation
            - **Error Tracking**: Comprehensive error handling and reporting
            """)

        # API documentation
        with st.expander("🔧 API Integration"):
            st.markdown("""
            ### Python API Usage
            ```python
            from worker.document_processor import EnhancedDocumentProcessor
            from worker.model_manager import ModelManager
            from worker.storage_manager import StorageManager

            # Initialize system
            model_manager = ModelManager()
            storage_manager = StorageManager()
            processor = EnhancedDocumentProcessor(model_manager, storage_manager)

            # Process document
            result = processor.process_document(
                job_id="my_job",
                document_path="document.pdf",
                params={
                    "mode": "advanced",
                    "profile": "performance",
                    "max_pages": 5
                }
            )
            ```

            ### Command Line Usage
            ```bash
            python -m cli.process_pdf "document.pdf" --profile performance --mode advanced --max-pages 5
            ```
            """)

        # System requirements
        with st.expander("💻 System Requirements"):
            st.markdown("""
            ### Minimum Requirements
            - **RAM**: 8GB (16GB recommended)
            - **Storage**: 10GB free space for models and cache
            - **CPU**: Multi-core processor (4+ cores recommended)
            - **GPU**: Optional, CUDA-compatible for acceleration

            ### Dependencies
            - Python 3.8+
            - PaddleOCR
            - Streamlit
            - OpenCV
            - Pillow
            - Pandas
            - Plotly

            ### Performance Tips
            - Use SSD storage for better I/O performance
            - Ensure sufficient RAM for large documents
            - Enable GPU acceleration when available
            - Use performance profile for batch processing
            """)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p><strong>Advanced OCR Processing System v2.0</strong> | Enterprise Edition</p>
        <p>Powered by PaddleOCR • Built with Streamlit • Enhanced with AI Analytics</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
