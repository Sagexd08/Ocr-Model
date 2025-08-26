from typing import List, Tuple, Dict, Any, Optional, Set, Union
import uuid
import numpy as np
import cv2
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from worker.types import OCRToken, TableContentTD
from worker.utils.logging import get_logger, log_execution_time

logger = get_logger(__name__)

@log_execution_time
def extract_table_content(
    table_bbox: Tuple[int, int, int, int], 
    ocr_tokens: List[OCRToken], 
    page_image: Optional[np.ndarray] = None,
    page_num: int = 0,
    region_id: str = None,
    confidence_threshold: float = 0.8,
    adaptive_clustering: bool = True,
    use_line_detection: bool = True
) -> TableContentTD:
    """
    Extract structured table content from OCR tokens within a table region.
    
    Args:
        table_bbox: Bounding box of the table region (x1, y1, x2, y2)
        ocr_tokens: List of OCR tokens from the page
        page_image: Optional page image for line-based table analysis
        page_num: Page number
        region_id: Table region identifier
        confidence_threshold: Minimum confidence threshold for valid data
        adaptive_clustering: Use adaptive clustering for columns
        use_line_detection: Use line detection from image when available
        
    Returns:
        Structured table data with rows, columns and metadata
    """
    region_id = region_id or f"table_{uuid.uuid4().hex[:8]}"
    
    # Filter tokens inside table bbox and sort by reading order
    in_table = [t for t in ocr_tokens if _inside(t.get("bbox", [0, 0, 0, 0]), table_bbox)]
    in_table.sort(key=lambda t: (t["bbox"][1], t["bbox"][0]))
    
    if not in_table:
        logger.warning(f"No tokens found in table region {region_id}")
        return {"rows": [], "columns": [], "region_id": region_id, "needs_review": True}

    # Get table structure from lines if image is available
    row_lines, col_lines = [], []
    if use_line_detection and page_image is not None:
        row_lines, col_lines = _detect_table_lines(
            page_image, 
            (table_bbox[0], table_bbox[1], table_bbox[2]-table_bbox[0], table_bbox[3]-table_bbox[1])
        )

    # Adaptive row grouping using median token height for robustness
    rows: List[List[OCRToken]] = []
    current_row_y: Optional[float] = None
    row: List[OCRToken] = []
    
    # Calculate median height for adaptive thresholding
    heights = [t["bbox"][3] - t["bbox"][1] for t in in_table] or [12]
    median_h = np.median(heights) if heights else 12
    row_threshold = max(8, int(0.6 * median_h))

    # First pass: group tokens into rows
    for tok in in_table:
        y1, y2 = tok["bbox"][1], tok["bbox"][3]
        ty = (y1 + y2) / 2
        
        if current_row_y is None:
            current_row_y = ty
            row.append(tok)
            continue
            
        if abs(ty - current_row_y) > row_threshold:
            if row:
                rows.append(row)
            row = [tok]
            current_row_y = ty
        else:
            row.append(tok)
            
    if row:
        rows.append(row)

    # Advanced column clustering using 1D density-based clustering
    centers: List[Tuple[float, float]] = []  # (center_x, width)
    for r in rows:
        for t in r:
            x1, _, x2, _ = t["bbox"]
            centers.append(((x1 + x2) / 2, (x2 - x1)))

    if not centers:
        return {"rows": rows, "columns": [], "region_id": region_id, "needs_review": True}

    # Get column positions - either from lines or clustering
    col_positions = []
    
    # If we have column lines from image detection, use those
    if col_lines:
        # Sort and filter column lines
        col_positions = sorted(set(x for x, _, _, _ in col_lines))
        logger.debug(f"Using {len(col_positions)} column positions from line detection")
    else:
        # Use advanced clustering for column detection
        if adaptive_clustering and len(centers) >= 8:
            # Use hierarchical clustering for better column detection
            try:
                col_positions = _cluster_columns_hierarchical([c[0] for c in centers])
                logger.debug(f"Used hierarchical clustering to find {len(col_positions)} columns")
            except Exception as e:
                logger.warning(f"Hierarchical clustering failed: {str(e)}. Using fallback.")
                col_positions = []
        
        # Fallback to traditional gap-based clustering
        if not col_positions:
            # Sort centers by x-coordinate
            centers.sort(key=lambda cw: cw[0])
            
            # Advanced clustering algorithm
            clustered: List[List[float]] = [[centers[0][0]]]
            total_span = max(1.0, centers[-1][0] - centers[0][0])
            min_gap = max(20, 0.04 * total_span)
            
            # Use gap-based detection to find column boundaries
            for i in range(1, len(centers)):
                prev_c = clustered[-1][-1]
                c = centers[i][0]
                w = centers[i][1]
                
                # Larger gap detection creates a new column
                if abs(c - prev_c) > min_gap:
                    clustered.append([c])
                else:
                    clustered[-1].append(c)
                    
            # Calculate column centers
            col_positions = [sum(cluster)/len(cluster) for cluster in clustered]

    # Generate column headers with unique identifiers
    columns = [f"col_{i}" for i in range(len(col_positions))]
    
    # Assign tokens to cells based on their position
    cells = _assign_tokens_to_cells(in_table, rows, col_positions)
    
    # Create final table structure with provenance
    table_data = {
        "rows": rows,
        "cells": cells,
        "columns": columns,
        "region_id": region_id,
        "page": page_num,
        "bbox": table_bbox,
        "col_positions": col_positions,
        "needs_review": any(t.get("confidence", 1.0) < confidence_threshold for t in in_table)
    }
    
    return table_data


def _detect_table_lines(image: np.ndarray, table_rect: Tuple[int, int, int, int]) -> Tuple[List[Tuple[int, int, int, int]], List[Tuple[int, int, int, int]]]:
    """
    Detect horizontal and vertical lines in a table region.
    
    Args:
        image: Page image as numpy array
        table_rect: Table rectangle as (x, y, w, h)
        
    Returns:
        Tuple of (horizontal_lines, vertical_lines) where each line is (x1, y1, x2, y2)
    """
    # Extract table region
    x, y, w, h = table_rect
    if x < 0 or y < 0 or x + w > image.shape[1] or y + h > image.shape[0]:
        # Table is outside image bounds
        return [], []
        
    table_img = image[y:y+h, x:x+w]
    
    # Convert to grayscale if needed
    if len(table_img.shape) == 3:
        gray = cv2.cvtColor(table_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = table_img
        
    # Apply adaptive thresholding to handle varying lighting
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # Define kernels for morphological operations
    kernel_len = max(w // 50, 5)  # Scale kernel with table size
    
    # Detect horizontal lines
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len, 1))
    h_detected = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)
    h_lines = cv2.HoughLinesP(
        h_detected, 1, np.pi/180, threshold=50,
        minLineLength=w*0.25, maxLineGap=20
    )
    
    # Detect vertical lines
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_len))
    v_detected = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)
    v_lines = cv2.HoughLinesP(
        v_detected, 1, np.pi/180, threshold=50,
        minLineLength=h*0.25, maxLineGap=20
    )
    
    # Convert to absolute coordinates
    h_coords = []
    if h_lines is not None:
        for line in h_lines:
            x1, y1, x2, y2 = line[0]
            # Only include mostly horizontal lines
            if abs(y2 - y1) < 10:
                h_coords.append((x + x1, y + y1, x + x2, y + y2))
                
    v_coords = []
    if v_lines is not None:
        for line in v_lines:
            x1, y1, x2, y2 = line[0]
            # Only include mostly vertical lines
            if abs(x2 - x1) < 10:
                v_coords.append((x + x1, y + y1, x + x2, y + y2))
                
    return h_coords, v_coords


def _cluster_columns_hierarchical(centers: List[float], distance_threshold: float = 20.0) -> List[float]:
    """
    Use hierarchical clustering to identify column boundaries.
    
    Args:
        centers: X-coordinates of token centers
        distance_threshold: Distance threshold for clustering
        
    Returns:
        List of column center positions
    """
    if len(centers) <= 1:
        return centers
    
    # Reshape for scikit-learn
    X = np.array(centers).reshape(-1, 1)
    
    # Compute linkage
    Z = linkage(X, 'ward')
    
    # Form flat clusters
    clusters = fcluster(Z, t=distance_threshold, criterion='distance')
    
    # Compute cluster centers
    unique_clusters = np.unique(clusters)
    column_centers = []
    
    for cluster_id in unique_clusters:
        cluster_points = X[clusters == cluster_id].flatten()
        column_centers.append(np.mean(cluster_points))
    
    # Sort the column centers
    return sorted(column_centers)


def _assign_tokens_to_cells(tokens: List[OCRToken], rows: List[List[OCRToken]], col_positions: List[float]) -> List[List[List[OCRToken]]]:
    """
    Assign tokens to cells in a grid based on row and column positions.
    
    Args:
        tokens: List of OCR tokens
        rows: List of rows (each row is a list of tokens)
        col_positions: List of column center positions
        
    Returns:
        Grid of tokens arranged by row and column
    """
    # Initialize grid
    grid = [[[] for _ in range(len(col_positions))] for _ in range(len(rows))]
    
    # Add column bounds to make assignment easier
    col_bounds = [-float('inf')] + [(a + b)/2 for a, b in zip(col_positions[:-1], col_positions[1:])] + [float('inf')]
    
    # Assign tokens to cells
    for row_idx, row_tokens in enumerate(rows):
        for token in row_tokens:
            # Find the column this token belongs to
            token_center_x = (token["bbox"][0] + token["bbox"][2]) / 2
            for col_idx in range(len(col_bounds) - 1):
                if col_bounds[col_idx] <= token_center_x < col_bounds[col_idx + 1]:
                    grid[row_idx][col_idx].append(token)
                    break
                    
    return grid


def _inside(b: List[int], box: Tuple[int, int, int, int], overlap_threshold: float = 0.5) -> bool:
    """
    Determine if a bounding box is inside another with configurable overlap threshold.
    
    Args:
        b: Token bounding box [x1, y1, x2, y2]
        box: Table bounding box (x1, y1, x2, y2)
        overlap_threshold: Minimum IoU for a token to be considered part of the table
        
    Returns:
        True if token is inside or overlaps significantly with the table
    """
    # Fast path: complete containment
    if b[0] >= box[0] and b[1] >= box[1] and b[2] <= box[2] and b[3] <= box[3]:
        return True
        
    # Calculate IoU for partial overlaps
    # Intersection
    ix1 = max(b[0], box[0])
    iy1 = max(b[1], box[1])
    ix2 = min(b[2], box[2])
    iy2 = min(b[3], box[3])
    
    # No overlap case
    if ix1 >= ix2 or iy1 >= iy2:
        return False
    
    intersection = (ix2 - ix1) * (iy2 - iy1)
    
    # Union
    b_area = (b[2] - b[0]) * (b[3] - b[1])
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    union = b_area + box_area - intersection
    
    # Return true if IoU exceeds threshold
    return (intersection / union) >= overlap_threshold

