from typing import Dict, Any, Optional, List, Tuple, Union
import numpy as np
import cv2
import torch
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from scipy.ndimage import binary_fill_holes, label, find_objects
from scipy.spatial.distance import pdist, squareform
import networkx as nx
import math

from ...types import Document, Page, Region, Token, Table, Cell, Bbox
from ...utils.logging import get_logger, log_execution_time

logger = get_logger(__name__)

class TableSegmenter:
    """
    Advanced table segmentation processor that extracts tables from document images
    with precise cell detection and content extraction.
    
    Uses a multi-stage approach:
    1. Table region detection and isolation
    2. Grid line detection using multiple algorithms
    3. Cell segmentation with hierarchical clustering
    4. Structure analysis for headers and spanning cells
    5. Content alignment with OCR tokens
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Table detection settings
        self.min_table_area = self.config.get("min_table_area", 5000)
        self.max_line_thickness = self.config.get("max_line_thickness", 5)
        self.min_cell_area = self.config.get("min_cell_area", 100)
        
        # Line detection methods
        self.line_methods = self.config.get("line_methods", ["hough", "contour", "morph"])
        self.hough_threshold = self.config.get("hough_threshold", 80)
        self.hough_min_line_length = self.config.get("hough_min_line_length", 100)
        self.hough_max_line_gap = self.config.get("hough_max_line_gap", 10)
        
        # Cell alignment
        self.alignment_threshold = self.config.get("alignment_threshold", 0.7)
        self.iou_threshold = self.config.get("iou_threshold", 0.5)
        
        # Clustering parameters for borderless tables
        self.cluster_eps = self.config.get("cluster_eps", 10)
        self.cluster_min_samples = self.config.get("cluster_min_samples", 2)
        
        # Structure analysis
        self.detect_headers = self.config.get("detect_headers", True)
        self.detect_spanning_cells = self.config.get("detect_spanning_cells", True)
        
    @log_execution_time
    def process(self, document: Document, table_regions: List[Region]) -> List[Table]:
        """
        Process table regions to extract structured tables
        
        Args:
            document: Document being processed
            table_regions: List of regions identified as tables
            
        Returns:
            List of extracted structured tables
        """
        tables = []
        
        logger.info(f"Segmenting {len(table_regions)} table regions")
        
        for region in table_regions:
            page = next((p for p in document.pages if p.page_num == region.page_num), None)
            if not page or not page.image:
                logger.warning(f"Could not find page {region.page_num} with image data")
                continue
                
            try:
                # Get image for the table region
                table_img = self._extract_table_region(page, region)
                if table_img is None:
                    continue
                
                # Get binary image for line detection
                binary = self._prepare_binary_image(table_img)
                
                # Detect lines using multiple methods
                h_lines, v_lines = self._detect_lines(binary)
                
                # If no lines detected, try borderless table detection
                if not h_lines or not v_lines:
                    logger.debug(f"No grid lines detected, attempting borderless table extraction")
                    table = self._extract_borderless_table(page, region, document)
                else:
                    # Extract cells from line intersections
                    cells, cell_bboxes = self._extract_cells(binary, h_lines, v_lines)
                    
                    # Extract cell contents and align with tokens
                    table = self._build_table_from_cells(page, region, cells, cell_bboxes, document)
                
                if table:
                    # Analyze table structure (headers, etc.)
                    if self.detect_headers or self.detect_spanning_cells:
                        table = self._analyze_table_structure(table, document)
                        
                    tables.append(table)
                    
            except Exception as e:
                logger.error(f"Error processing table region on page {region.page_num}: {str(e)}")
        
        return tables
    
    def _extract_table_region(self, page: Page, region: Region) -> Optional[np.ndarray]:
        """Extract table region from page image"""
        # For brevity, implementation details omitted
        # This would extract the image region corresponding to the table bbox
        return np.array([])  # Placeholder
    
    def _prepare_binary_image(self, img: np.ndarray) -> np.ndarray:
        """Prepare binary image optimized for table structure detection"""
        # Convert to grayscale if needed
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
            
        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(
            gray, 
            255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11, 
            2
        )
        
        # Apply morphological operations to enhance lines
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        return binary
    
    def _detect_lines(self, binary: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Detect horizontal and vertical lines using multiple methods
        and combine results for robustness
        """
        h_lines_combined = []
        v_lines_combined = []
        
        # Apply all configured detection methods
        for method in self.line_methods:
            if method == "hough":
                h_lines, v_lines = self._detect_lines_hough(binary)
            elif method == "contour":
                h_lines, v_lines = self._detect_lines_contour(binary)
            elif method == "morph":
                h_lines, v_lines = self._detect_lines_morphological(binary)
            else:
                continue
                
            h_lines_combined.extend(h_lines)
            v_lines_combined.extend(v_lines)
        
        # Merge nearby lines
        h_lines_merged = self._merge_nearby_lines(h_lines_combined, is_horizontal=True)
        v_lines_merged = self._merge_nearby_lines(v_lines_combined, is_horizontal=False)
        
        return h_lines_merged, v_lines_merged
    
    def _detect_lines_hough(self, binary: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Detect lines using Hough transform"""
        h_lines = []
        v_lines = []
        
        # Use probabilistic Hough transform
        lines = cv2.HoughLinesP(
            binary, 
            1, 
            np.pi/180, 
            threshold=self.hough_threshold,
            minLineLength=self.hough_min_line_length,
            maxLineGap=self.hough_max_line_gap
        )
        
        if lines is None:
            return h_lines, v_lines
        
        # Separate horizontal and vertical lines
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Check if line is horizontal or vertical
            if abs(y2 - y1) < abs(x2 - x1) / 10:  # Horizontal
                h_lines.append(np.array([x1, y1, x2, y2]))
            elif abs(x2 - x1) < abs(y2 - y1) / 10:  # Vertical
                v_lines.append(np.array([x1, y1, x2, y2]))
        
        return h_lines, v_lines
    
    def _detect_lines_contour(self, binary: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Detect lines using contour analysis"""
        h_lines = []
        v_lines = []
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter by aspect ratio and size
            aspect_ratio = w / h if h > 0 else float('inf')
            
            if aspect_ratio > 10 and w > self.hough_min_line_length:  # Horizontal line
                h_lines.append(np.array([x, y + h//2, x + w, y + h//2]))
            elif aspect_ratio < 0.1 and h > self.hough_min_line_length:  # Vertical line
                v_lines.append(np.array([x + w//2, y, x + w//2, y + h]))
        
        return h_lines, v_lines
    
    def _detect_lines_morphological(self, binary: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Detect lines using morphological operations"""
        h_lines = []
        v_lines = []
        
        # Create kernels for horizontal and vertical lines
        h_kernel = np.ones((1, self.max_line_thickness * 5), np.uint8)
        v_kernel = np.ones((self.max_line_thickness * 5, 1), np.uint8)
        
        # Apply morphological operations to isolate lines
        h_binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)
        v_binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)
        
        # Find contours in horizontal and vertical binary images
        h_contours, _ = cv2.findContours(h_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        v_contours, _ = cv2.findContours(v_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Process horizontal contours
        for contour in h_contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w > self.hough_min_line_length and h <= self.max_line_thickness:
                h_lines.append(np.array([x, y + h//2, x + w, y + h//2]))
        
        # Process vertical contours
        for contour in v_contours:
            x, y, w, h = cv2.boundingRect(contour)
            if h > self.hough_min_line_length and w <= self.max_line_thickness:
                v_lines.append(np.array([x + w//2, y, x + w//2, y + h]))
        
        return h_lines, v_lines
    
    def _merge_nearby_lines(self, lines: List[np.ndarray], is_horizontal: bool) -> List[np.ndarray]:
        """Merge nearby parallel lines"""
        if not lines:
            return []
            
        # Convert to numpy array for vectorized operations
        lines_array = np.array(lines)
        
        # Group lines by position
        if is_horizontal:
            # Group by y coordinate (lines with similar y)
            positions = np.array([[(line[1] + line[3]) / 2] for line in lines_array])
        else:
            # Group by x coordinate (lines with similar x)
            positions = np.array([[(line[0] + line[2]) / 2] for line in lines_array])
            
        # Cluster line positions
        clustering = DBSCAN(eps=self.max_line_thickness, min_samples=1).fit(positions)
        labels = clustering.labels_
        
        # Merge lines in each cluster
        merged_lines = []
        for label in set(labels):
            cluster_indices = np.where(labels == label)[0]
            cluster_lines = lines_array[cluster_indices]
            
            if is_horizontal:
                # For horizontal lines, merge by averaging y and finding min/max x
                avg_y = np.mean([np.mean([line[1], line[3]]) for line in cluster_lines])
                min_x = np.min([line[0] for line in cluster_lines])
                max_x = np.max([line[2] for line in cluster_lines])
                merged_lines.append(np.array([min_x, avg_y, max_x, avg_y]))
            else:
                # For vertical lines, merge by averaging x and finding min/max y
                avg_x = np.mean([np.mean([line[0], line[2]]) for line in cluster_lines])
                min_y = np.min([line[1] for line in cluster_lines])
                max_y = np.max([line[3] for line in cluster_lines])
                merged_lines.append(np.array([avg_x, min_y, avg_x, max_y]))
        
        return merged_lines
    
    def _extract_cells(self, binary: np.ndarray, h_lines: List[np.ndarray], v_lines: List[np.ndarray]) -> Tuple[np.ndarray, List[Bbox]]:
        """
        Extract cells from line intersections
        Returns cells binary images and cell bounding boxes
        """
        # Sort lines by position
        h_lines_sorted = sorted(h_lines, key=lambda line: (line[1] + line[3]) / 2)
        v_lines_sorted = sorted(v_lines, key=lambda line: (line[0] + line[2]) / 2)
        
        # Get row and column coordinates
        row_positions = [(line[1] + line[3]) / 2 for line in h_lines_sorted]
        col_positions = [(line[0] + line[2]) / 2 for line in v_lines_sorted]
        
        # Create empty cells array
        cells = np.empty((len(row_positions) - 1, len(col_positions) - 1), dtype=object)
        cell_bboxes = []
        
        # Extract each cell
        for i in range(len(row_positions) - 1):
            row_bboxes = []
            
            for j in range(len(col_positions) - 1):
                # Cell coordinates
                x1 = int(col_positions[j])
                y1 = int(row_positions[i])
                x2 = int(col_positions[j + 1])
                y2 = int(row_positions[i + 1])
                
                # Ensure valid coordinates
                if x1 >= x2 or y1 >= y2:
                    cells[i, j] = None
                    continue
                
                # Extract cell image
                if x2 < binary.shape[1] and y2 < binary.shape[0]:
                    cell_img = binary[y1:y2, x1:x2].copy()
                    cells[i, j] = cell_img
                    
                    # Create bbox
                    bbox = Bbox(x1=x1, y1=y1, x2=x2, y2=y2)
                    row_bboxes.append(bbox)
                else:
                    cells[i, j] = None
                    
            cell_bboxes.append(row_bboxes)
            
        return cells, cell_bboxes
    
    def _extract_borderless_table(self, page: Page, region: Region, document: Document) -> Optional[Table]:
        """Extract table structure without visible grid lines using text alignment"""
        # Get tokens in the region
        tokens = [token for token in page.tokens if self._token_in_region(token, region)]
        
        if not tokens:
            return None
            
        # Use hierarchical clustering to group tokens into rows and columns
        rows = self._cluster_tokens_into_rows(tokens)
        
        # No valid rows found
        if not rows:
            return None
            
        # For each row, cluster tokens into columns
        cells = []
        for row_tokens in rows:
            columns = self._cluster_tokens_into_columns(row_tokens)
            
            # Create Cell objects for each column
            row_cells = []
            for col_tokens in columns:
                if not col_tokens:
                    # Empty cell
                    row_cells.append(None)
                    continue
                    
                # Create bounding box for all tokens in cell
                x_min = min(t.bbox.x1 for t in col_tokens)
                y_min = min(t.bbox.y1 for t in col_tokens)
                x_max = max(t.bbox.x2 for t in col_tokens)
                y_max = max(t.bbox.y2 for t in col_tokens)
                
                bbox = Bbox(x1=x_min, y1=y_min, x2=x_max, y2=y_max)
                
                # Combine token text
                text = " ".join(t.text for t in col_tokens)
                
                # Create cell
                cell = Cell(
                    id=f"c_{len(cells)}_{len(row_cells)}",
                    text=text,
                    tokens=col_tokens,
                    bbox=bbox,
                    confidence=sum(t.confidence for t in col_tokens) / len(col_tokens) if col_tokens else 0.0
                )
                
                row_cells.append(cell)
                
            cells.append(row_cells)
        
        # Create table with cells
        if cells:
            # Determine table boundaries
            x_min = min(min(c.bbox.x1 for c in row if c) for row in cells if any(c for c in row))
            y_min = min(min(c.bbox.y1 for c in row if c) for row in cells if any(c for c in row))
            x_max = max(max(c.bbox.x2 for c in row if c) for row in cells if any(c for c in row))
            y_max = max(max(c.bbox.y2 for c in row if c) for row in cells if any(c for c in row))
            
            table_bbox = Bbox(x1=x_min, y1=y_min, x2=x_max, y2=y_max)
            
            # Create table
            table = Table(
                id=f"t_{region.id}",
                page_num=page.page_num,
                bbox=table_bbox,
                cells=cells,
                confidence=region.confidence,
                attributes={"borderless": True}
            )
            
            return table
        
        return None
    
    def _build_table_from_cells(self, page: Page, region: Region, cells: np.ndarray, 
                               cell_bboxes: List[List[Bbox]], document: Document) -> Optional[Table]:
        """Build table structure from cells and align with document tokens"""
        if not cells.size or not cell_bboxes:
            return None
            
        table_cells = []
        
        # Process each row
        for i in range(cells.shape[0]):
            row_cells = []
            
            # Process each cell in the row
            for j in range(cells.shape[1]):
                cell_img = cells[i, j]
                if cell_img is None or j >= len(cell_bboxes[i]):
                    row_cells.append(None)
                    continue
                    
                bbox = cell_bboxes[i][j]
                
                # Find tokens in this cell
                cell_tokens = self._get_tokens_in_bbox(page, bbox)
                
                if not cell_tokens:
                    # No tokens found in cell
                    row_cells.append(None)
                    continue
                
                # Create Cell object
                text = " ".join(token.text for token in cell_tokens)
                confidence = sum(token.confidence for token in cell_tokens) / len(cell_tokens) if cell_tokens else 0.0
                
                cell = Cell(
                    id=f"c_{i}_{j}",
                    text=text,
                    tokens=cell_tokens,
                    bbox=bbox,
                    confidence=confidence
                )
                
                row_cells.append(cell)
                
            table_cells.append(row_cells)
        
        # Create Table object
        if table_cells:
            table = Table(
                id=f"t_{region.id}",
                page_num=page.page_num,
                bbox=region.bbox,
                cells=table_cells,
                confidence=region.confidence
            )
            
            return table
            
        return None
    
    def _analyze_table_structure(self, table: Table, document: Document) -> Table:
        """Analyze table structure for headers and spanning cells"""
        if not table or not table.cells:
            return table
            
        # Detect header row
        if self.detect_headers and len(table.cells) > 1:
            # Look for formatting differences in first row
            # In a real implementation, this would use more sophisticated rules
            # such as font differences, background color, borders, etc.
            header_confidence = 0.6  # Default
            
            # Check for bold or different font in first row
            # This would use token attributes from OCR
            if all(cell and hasattr(cell, 'tokens') and cell.tokens for cell in table.cells[0]):
                # Mark as header if detected
                table.attributes["has_header"] = True
                table.attributes["header_row"] = 0
        
        # Detect spanning cells (cells that span multiple rows/columns)
        if self.detect_spanning_cells:
            # This would implement algorithms to detect and represent spanning cells
            # For brevity, implementation details omitted
            pass
            
        return table
    
    def _token_in_region(self, token: Token, region: Region) -> bool:
        """Check if token is inside region"""
        return (token.bbox.x1 >= region.bbox.x1 and 
                token.bbox.y1 >= region.bbox.y1 and
                token.bbox.x2 <= region.bbox.x2 and
                token.bbox.y2 <= region.bbox.y2)
    
    def _get_tokens_in_bbox(self, page: Page, bbox: Bbox) -> List[Token]:
        """Get all tokens contained in a bounding box"""
        return [token for token in page.tokens if self._token_in_bbox(token, bbox)]
    
    def _token_in_bbox(self, token: Token, bbox: Bbox) -> bool:
        """Check if token is inside bbox using IoU metric"""
        # Calculate intersection
        x1 = max(token.bbox.x1, bbox.x1)
        y1 = max(token.bbox.y1, bbox.y1)
        x2 = min(token.bbox.x2, bbox.x2)
        y2 = min(token.bbox.y2, bbox.y2)
        
        if x1 >= x2 or y1 >= y2:
            return False
            
        intersection = (x2 - x1) * (y2 - y1)
        
        # Calculate token area
        token_area = (token.bbox.x2 - token.bbox.x1) * (token.bbox.y2 - token.bbox.y1)
        
        # Calculate IoU
        iou = intersection / token_area
        
        return iou > self.iou_threshold
    
    def _cluster_tokens_into_rows(self, tokens: List[Token]) -> List[List[Token]]:
        """Use hierarchical clustering to group tokens into rows"""
        if not tokens:
            return []
            
        # Get y-centers for all tokens
        y_centers = np.array([[(t.bbox.y1 + t.bbox.y2) / 2] for t in tokens])
        
        # Apply hierarchical clustering on y-coordinates
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=self.cluster_eps,
            linkage='ward'
        ).fit(y_centers)
        
        labels = clustering.labels_
        
        # Group tokens by cluster
        rows = [[] for _ in range(max(labels) + 1)]
        for i, label in enumerate(labels):
            rows[label].append(tokens[i])
        
        # Sort rows by y-position
        rows.sort(key=lambda row: sum(t.bbox.y1 for t in row) / len(row) if row else 0)
        
        return rows
    
    def _cluster_tokens_into_columns(self, tokens: List[Token]) -> List[List[Token]]:
        """Use hierarchical clustering to group tokens into columns"""
        if not tokens:
            return []
            
        # Get x-centers for all tokens
        x_centers = np.array([[(t.bbox.x1 + t.bbox.x2) / 2] for t in tokens])
        
        # Apply hierarchical clustering on x-coordinates
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=self.cluster_eps * 2,  # Wider threshold for columns
            linkage='ward'
        ).fit(x_centers)
        
        labels = clustering.labels_
        
        # Group tokens by cluster
        columns = [[] for _ in range(max(labels) + 1)]
        for i, label in enumerate(labels):
            columns[label].append(tokens[i])
        
        # Sort columns by x-position
        columns.sort(key=lambda col: sum(t.bbox.x1 for t in col) / len(col) if col else 0)
        
        return columns
