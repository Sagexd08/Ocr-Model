import uuid
from typing import Dict, Any, Optional, List

from ..types import Document, Page, Token, Bbox
from ..model_manager import ModelManager
from ..utils.logging import get_logger

logger = get_logger(__name__)

def analyze_layout(
    document: Document, 
    model_manager: ModelManager,
    render_info: Dict[str, Any]
) -> Document:
    """
    Analyze document layout to detect regions, tables, and structure.
    
    Uses LayoutLMv3/DocTR-based detection head or fallbacks based on document type.
    
    Args:
        document: Document to analyze
        model_manager: Model provider for layout analysis
        render_info: Document renderer classification info
        
    Returns:
        Document with detected regions
    """
    render_type = render_info.get("render_type", "unknown")
    logger.info(f"Analyzing layout for {render_type} document")
    
    for i, page in enumerate(document.pages):
        if not page.image:
            logger.warning(f"Skipping layout analysis for page {i}: No image available")
            continue
            
        # Get layout analysis from model
        layout_results = model_manager.analyze_layout(page.image)
        
        # Add regions and table detections to page
        if "regions" in layout_results:
            for region_data in layout_results["regions"]:
                region_id = f"region_{uuid.uuid4().hex[:8]}"
                bbox = Bbox(
                    x1=region_data["bbox"][0],
                    y1=region_data["bbox"][1],
                    x2=region_data["bbox"][2],
                    y2=region_data["bbox"][3]
                )
                
                # Create region with appropriate type
                region = {
                    "id": region_id,
                    "bbox": bbox,
                    "type": region_data["type"],
                    "confidence": region_data["confidence"],
                    "tokens": []  # Will be populated during OCR
                }
                
                page.regions.append(region)
                
        # Add table detections
        if "tables" in layout_results:
            for table_data in layout_results["tables"]:
                table_id = f"table_{uuid.uuid4().hex[:8]}"
                bbox = Bbox(
                    x1=table_data["bbox"][0],
                    y1=table_data["bbox"][1],
                    x2=table_data["bbox"][2],
                    y2=table_data["bbox"][3]
                )
                
                # Create table placeholder, will be processed by tables module
                table = {
                    "id": table_id,
                    "bbox": bbox,
                    "confidence": table_data["confidence"],
                    "rows": [],
                    "columns": []
                }
                
                page.tables.append(table)
    
    return document
