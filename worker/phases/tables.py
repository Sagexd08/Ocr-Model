from typing import Dict, Any, List, Optional, Tuple
import uuid

from ..types import Document, Bbox, Token
from ..model_manager import ModelManager
from ..pipeline.tables import extract_table_content
from ..utils.logging import get_logger

logger = get_logger(__name__)

def process_tables(document: Document, model_manager: ModelManager) -> Document:
    """
    Process tables in the document.
    
    Performs table detection and reconstruction:
    1. If tables weren't detected in layout phase, detect tables
    2. For each detected table, extract cell structure
    3. Reconstruct table with row/column structure
    4. Apply cell extraction and OCR to table cells
    
    Args:
        document: Document with OCR tokens and possibly tables
        model_manager: Model provider for table detection
        
    Returns:
        Document with processed tables
    """
    for page_idx, page in enumerate(document.pages):
        # Skip if no image available
        if not page.image:
            logger.warning(f"Skipping table detection for page {page_idx}: No image available")
            continue
            
        # Detect tables if none were detected during layout phase
        if not page.tables and model_manager.is_model_loaded("table_detector"):
            logger.info(f"Running table detection for page {page_idx}")
            table_results = model_manager.detect_tables(page.image)
            
            # Add detected tables
            for table_data in table_results.get("tables", []):
                table_id = f"table_{uuid.uuid4().hex[:8]}"
                bbox = Bbox(
                    x1=table_data["bbox"][0],
                    y1=table_data["bbox"][1],
                    x2=table_data["bbox"][2],
                    y2=table_data["bbox"][3]
                )
                
                # Create table placeholder
                table = {
                    "id": table_id,
                    "bbox": bbox,
                    "confidence": table_data["confidence"],
                    "rows": [],
                    "columns": []
                }
                
                page.tables.append(table)
        
        # Process each detected table
        for table in page.tables:
            logger.info(f"Processing table {table.id} on page {page_idx}")
            
            # Extract tokens that intersect with this table
            table_bbox = (table.bbox.x1, table.bbox.y1, table.bbox.x2, table.bbox.y2)
            
            # Extract table content structure
            table_content = extract_table_content(
                table_bbox=table_bbox,
                ocr_tokens=page.tokens,
                page_num=page_idx,
                region_id=table.id
            )
            
            # Update table with extracted structure
            table.rows = table_content.get("rows", [])
            table.columns = table_content.get("columns", [])
            table.needs_review = table_content.get("needs_review", False)
            
            # Apply post-processing to improve table quality
            table = _improve_table_structure(table)
            
            # Ensure every cell has tokens and text
            _process_table_cells(table)
            
    return document

def _improve_table_structure(table: Dict[str, Any]) -> Dict[str, Any]:
    """
    Improve table structure by detecting and fixing common issues:
    - Merging split cells
    - Detecting row/column spans
    - Adjusting column widths
    - Handling empty cells
    
    Args:
        table: Raw table data
        
    Returns:
        Improved table structure
    """
    # Implement additional table structure improvements here
    # For example, detecting row/column spans using IoU heuristics
    
    # This is where advanced table structure analysis would happen
    # For now, we'll return the table as-is
    return table

def _process_table_cells(table: Dict[str, Any]) -> None:
    """
    Process each table cell to ensure it has text and token references
    
    Args:
        table: Table to process
    """
    for row in table.rows:
        # Ensure each cell has text derived from its tokens
        for cell in row:
            if not hasattr(cell, "text") or not cell.text:
                # Derive text from tokens
                text = " ".join([t.text for t in cell.tokens]) if cell.tokens else ""
                cell.text = text
                
            # Calculate confidence
            if cell.tokens:
                confidences = [t.confidence for t in cell.tokens]
                cell.confidence = sum(confidences) / len(confidences)
            else:
                cell.confidence = 0.0
