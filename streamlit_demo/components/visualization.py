"""
Visualization components for CurioScan Streamlit demo.
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import Dict, Any, List, Optional, Tuple
import io
import base64


def render_ocr_overlay(image: Image.Image, ocr_results: Dict[str, Any], 
                      overlay_mode: str = "bboxes", confidence_threshold: float = 0.0) -> Image.Image:
    """
    Render OCR overlay on image.
    
    Args:
        image: Source image
        ocr_results: OCR results with tokens and bboxes
        overlay_mode: Type of overlay ("bboxes", "tokens", "confidence")
        confidence_threshold: Minimum confidence to display
        
    Returns:
        Image with overlay
    """
    
    overlay_image = image.copy()
    draw = ImageDraw.Draw(overlay_image)
    
    # Try to load a font
    try:
        font = ImageFont.truetype("arial.ttf", 12)
    except:
        font = ImageFont.load_default()
    
    tokens = ocr_results.get('tokens', [])
    
    for token in tokens:
        confidence = token.get('confidence', 0.0)
        
        # Skip low confidence tokens
        if confidence < confidence_threshold:
            continue
        
        bbox = token.get('bbox', [0, 0, 0, 0])
        text = token.get('text', '')
        
        # Convert bbox to integers
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        
        if overlay_mode == "bboxes":
            # Draw bounding boxes
            color = _get_confidence_color(confidence)
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            
        elif overlay_mode == "tokens":
            # Draw bounding boxes with text
            color = _get_confidence_color(confidence)
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            
            # Draw text above bbox
            text_y = max(0, y1 - 15)
            draw.text((x1, text_y), text, fill=color, font=font)
            
        elif overlay_mode == "confidence":
            # Color-coded confidence heatmap
            alpha = int(confidence * 255)
            color = _get_confidence_color(confidence) + (alpha,)
            
            # Create a semi-transparent overlay
            overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
            overlay_draw = ImageDraw.Draw(overlay)
            overlay_draw.rectangle([x1, y1, x2, y2], fill=color)
            
            # Composite with main image
            overlay_image = Image.alpha_composite(
                overlay_image.convert('RGBA'), 
                overlay
            ).convert('RGB')
    
    return overlay_image


def render_table_editor(table_data: List[Dict[str, Any]], 
                       editable: bool = True) -> pd.DataFrame:
    """
    Render interactive table editor.
    
    Args:
        table_data: List of table rows
        editable: Whether table should be editable
        
    Returns:
        Edited DataFrame
    """
    
    if not table_data:
        st.warning("No table data to display")
        return pd.DataFrame()
    
    # Convert to DataFrame
    df = pd.DataFrame(table_data)
    
    # Add confidence styling
    if 'confidence' in df.columns:
        def style_confidence(val):
            if val >= 0.8:
                return 'background-color: #d4edda'
            elif val >= 0.6:
                return 'background-color: #fff3cd'
            else:
                return 'background-color: #f8d7da'
        
        styled_df = df.style.applymap(style_confidence, subset=['confidence'])
        st.dataframe(styled_df, use_container_width=True)
    
    if editable:
        # Editable version
        edited_df = st.data_editor(
            df,
            use_container_width=True,
            num_rows="dynamic",
            column_config={
                "confidence": st.column_config.ProgressColumn(
                    "Confidence",
                    min_value=0.0,
                    max_value=1.0,
                ),
                "needs_review": st.column_config.CheckboxColumn(
                    "Needs Review"
                )
            }
        )
        return edited_df
    else:
        st.dataframe(df, use_container_width=True)
        return df


def render_provenance_inspector(row_data: Dict[str, Any], 
                               source_image: Optional[Image.Image] = None):
    """
    Render provenance inspector for a selected row.
    
    Args:
        row_data: Row data with provenance information
        source_image: Optional source image to highlight region
    """
    
    st.markdown("#### 🔍 Provenance Details")
    
    provenance = row_data.get('provenance', {})
    
    # Provenance information
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Source Information:**")
        st.write(f"📄 File: {provenance.get('file', 'Unknown')}")
        st.write(f"📃 Page: {provenance.get('page', 'Unknown')}")
        st.write(f"📊 Confidence: {provenance.get('confidence', 0.0):.1%}")
        
    with col2:
        st.markdown("**Location:**")
        bbox = provenance.get('bbox', [0, 0, 0, 0])
        st.write(f"📍 Bounding Box: [{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]")
        
        token_ids = provenance.get('token_ids', [])
        st.write(f"🔤 Token Count: {len(token_ids)}")
    
    # Highlight region on source image
    if source_image and bbox != [0, 0, 0, 0]:
        highlighted_image = _highlight_region(source_image, bbox)
        st.image(highlighted_image, caption="Source Region", use_column_width=True)


def render_confidence_heatmap(ocr_results: Dict[str, Any], 
                             image_size: Tuple[int, int]) -> go.Figure:
    """
    Render confidence heatmap visualization.
    
    Args:
        ocr_results: OCR results with confidence scores
        image_size: Size of the source image (width, height)
        
    Returns:
        Plotly heatmap figure
    """
    
    tokens = ocr_results.get('tokens', [])
    
    if not tokens:
        # Empty heatmap
        fig = go.Figure()
        fig.add_annotation(
            text="No OCR data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig
    
    # Create confidence grid
    grid_size = 50
    width, height = image_size
    
    x_step = width / grid_size
    y_step = height / grid_size
    
    confidence_grid = np.zeros((grid_size, grid_size))
    count_grid = np.zeros((grid_size, grid_size))
    
    # Map tokens to grid
    for token in tokens:
        bbox = token.get('bbox', [0, 0, 0, 0])
        confidence = token.get('confidence', 0.0)
        
        # Calculate grid position
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2
        
        grid_x = int(center_x / x_step)
        grid_y = int(center_y / y_step)
        
        # Ensure within bounds
        grid_x = max(0, min(grid_size - 1, grid_x))
        grid_y = max(0, min(grid_size - 1, grid_y))
        
        confidence_grid[grid_y, grid_x] += confidence
        count_grid[grid_y, grid_x] += 1
    
    # Average confidence per grid cell
    with np.errstate(divide='ignore', invalid='ignore'):
        avg_confidence = np.divide(confidence_grid, count_grid)
        avg_confidence[count_grid == 0] = 0
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=avg_confidence,
        colorscale='RdYlGn',
        zmin=0,
        zmax=1,
        colorbar=dict(title="Confidence")
    ))
    
    fig.update_layout(
        title="OCR Confidence Heatmap",
        xaxis_title="X Position",
        yaxis_title="Y Position",
        height=400
    )
    
    return fig


def render_processing_timeline(job_data: Dict[str, Any]) -> go.Figure:
    """
    Render processing timeline visualization.
    
    Args:
        job_data: Job data with timing information
        
    Returns:
        Plotly timeline figure
    """
    
    # Mock timeline data for demo
    stages = [
        {"stage": "Upload", "start": 0, "duration": 0.5, "status": "completed"},
        {"stage": "Classification", "start": 0.5, "duration": 1.0, "status": "completed"},
        {"stage": "Preprocessing", "start": 1.5, "duration": 2.0, "status": "completed"},
        {"stage": "OCR", "start": 3.5, "duration": 5.0, "status": "processing"},
        {"stage": "Table Detection", "start": 8.5, "duration": 2.0, "status": "pending"},
        {"stage": "Postprocessing", "start": 10.5, "duration": 1.0, "status": "pending"},
    ]
    
    fig = go.Figure()
    
    colors = {
        "completed": "#28a745",
        "processing": "#17a2b8",
        "pending": "#6c757d"
    }
    
    for stage in stages:
        fig.add_trace(go.Scatter(
            x=[stage["start"], stage["start"] + stage["duration"]],
            y=[stage["stage"], stage["stage"]],
            mode='lines',
            line=dict(color=colors[stage["status"]], width=10),
            name=stage["stage"],
            showlegend=False
        ))
    
    fig.update_layout(
        title="Processing Timeline",
        xaxis_title="Time (seconds)",
        yaxis_title="Processing Stage",
        height=300,
        template="plotly_white"
    )
    
    return fig


def render_accuracy_metrics(metrics_data: Dict[str, Any]) -> go.Figure:
    """
    Render accuracy metrics visualization.
    
    Args:
        metrics_data: Metrics data
        
    Returns:
        Plotly metrics figure
    """
    
    # Mock metrics data
    metrics = {
        "Token F1": 0.94,
        "Table Detection": 0.89,
        "Layout Analysis": 0.92,
        "Overall Confidence": 0.91
    }
    
    fig = go.Figure(data=[
        go.Bar(
            x=list(metrics.keys()),
            y=list(metrics.values()),
            marker_color=['#667eea', '#764ba2', '#667eea', '#764ba2']
        )
    ])
    
    fig.update_layout(
        title="Model Performance Metrics",
        yaxis_title="Score",
        yaxis=dict(range=[0, 1]),
        template="plotly_white",
        height=300
    )
    
    return fig


# Helper functions

def _get_confidence_color(confidence: float) -> str:
    """Get color based on confidence score."""
    if confidence >= 0.8:
        return "#28a745"  # Green
    elif confidence >= 0.6:
        return "#ffc107"  # Yellow
    else:
        return "#dc3545"  # Red


def _highlight_region(image: Image.Image, bbox: List[int], 
                     color: str = "#ff0000", width: int = 3) -> Image.Image:
    """Highlight a region on an image."""
    highlighted = image.copy()
    draw = ImageDraw.Draw(highlighted)
    
    x1, y1, x2, y2 = bbox
    draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
    
    return highlighted


def image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string."""
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"


def create_download_link(data: bytes, filename: str, mime_type: str = "application/octet-stream") -> str:
    """Create a download link for data."""
    b64_data = base64.b64encode(data).decode()
    return f'<a href="data:{mime_type};base64,{b64_data}" download="{filename}">📥 Download {filename}</a>'
