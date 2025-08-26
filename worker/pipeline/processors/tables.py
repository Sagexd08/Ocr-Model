from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import cv2
import math
from ...types import Document, Page, Table, Cell, Bbox, Token, TableContentTD, RowTD
from ...utils.logging import get_logger, log_execution_time

logger = get_logger(__name__)

class TableExtractor:
    """
    Extract tables from documents and reconstruct their structure.
    Handles detection, structure extraction, and cell content mapping.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        # Load configuration parameters
        self.min_table_rows = self.config.get("min_table_rows", 2)
        self.min_table_cols = self.config.get("min_table_cols", 2)
        self.line_thickness_threshold = self.config.get("line_thickness_threshold", 3)
        self.max_line_gap = self.config.get("max_line_gap", 10)
        self.cell_content_padding = self.config.get("cell_content_padding", 5)
        self.iou_threshold = self.config.get("iou_threshold", 0.5)
        self.use_external_detector = self.config.get("use_external_detector", False)
        
    @log_execution_time
    def process(self, document: Document) -> Document:
        """
        Process a document to extract tables.
        
        Args:
            document: Document object with pages and tokens
            
        Returns:
            Document with extracted tables
        """
        logger.info(f"Extracting tables from document with {len(document.pages)} pages")
        
        for i, page in enumerate(document.pages):
            logger.debug(f"Extracting tables for page {page.page_num}")
            
            if page.image is None:
                logger.warning(f"No image found for page {page.page_num}, skipping table extraction")
                continue
            
            try:
                # Detect tables in the page
                table_regions = self._detect_tables(page)
                
                if not table_regions:
                    logger.info(f"No tables detected on page {page.page_num}")
                    continue
                
                # Extract and structure each table
                tables = []
                for j, table_region in enumerate(table_regions):
                    logger.debug(f"Processing table {j+1} on page {page.page_num}")
                    
                    # Extract table grid structure
                    rows, cols = self._extract_table_structure(page.image, table_region)
                    
                    if len(rows) < self.min_table_rows or len(cols) < self.min_table_cols:
                        logger.debug(f"Table {j+1} on page {page.page_num} too small, skipping")
                        continue
                    
                    # Map tokens to cells
                    table = self._map_tokens_to_cells(page, rows, cols, table_region)
                    tables.append(table)
                
                # Update the page with extracted tables
                page.tables = tables
                document.pages[i] = page
                
                logger.info(f"Extracted {len(tables)} tables from page {page.page_num}")
            except Exception as e:
                logger.error(f"Error extracting tables from page {page.page_num}: {str(e)}")
        
        return document
    
    def _detect_tables(self, page: Page) -> List[Bbox]:
        """Detect table regions in a page image"""
        if self.use_external_detector:
            # Use external table detector if configured
            return self._external_table_detection(page)
        else:
            # Use built-in table detection based on lines
            return self._line_based_table_detection(page)
    
    def _external_table_detection(self, page: Page) -> List[Bbox]:
        """Use an external table detector model"""
        # This would integrate with an external ML model for table detection
        # For now, we're returning an empty list as a placeholder
        logger.warning("External table detection not implemented, using line-based detection")
        return self._line_based_table_detection(page)
    
    def _line_based_table_detection(self, page: Page) -> List[Bbox]:
        """Detect tables based on line patterns in the image"""
        image = page.image
        if isinstance(image, str):
            # Load image if it's a file path
            image = cv2.imread(image)
        
        if image is None:
            return []
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Apply adaptive thresholding to get binary image
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Detect horizontal and vertical lines
        horizontal = self._detect_lines(binary, True)
        vertical = self._detect_lines(binary, False)
        
        # Combine horizontal and vertical lines
        combined = cv2.bitwise_or(horizontal, vertical)
        
        # Find contours to identify potential tables
        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours to identify tables
        table_regions = []
        img_height, img_width = gray.shape
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter out small regions
            if w < img_width * 0.1 or h < img_height * 0.05:
                continue
            
            # Check if region has sufficient line intersections
            region_mask = np.zeros_like(combined)
            cv2.drawContours(region_mask, [contour], 0, 255, -1)
            
            horizontal_in_region = cv2.bitwise_and(horizontal, horizontal, mask=region_mask)
            vertical_in_region = cv2.bitwise_and(vertical, vertical, mask=region_mask)
            
            h_lines = cv2.HoughLinesP(horizontal_in_region, 1, np.pi/180, 50, 
                                      minLineLength=w*0.3, maxLineGap=self.max_line_gap)
            v_lines = cv2.HoughLinesP(vertical_in_region, 1, np.pi/180, 50, 
                                     minLineLength=h*0.3, maxLineGap=self.max_line_gap)
            
            if h_lines is not None and v_lines is not None:
                if len(h_lines) >= self.min_table_rows and len(v_lines) >= self.min_table_cols:
                    # Normalize coordinates to [0,1]
                    table_regions.append(Bbox(
                        x1=x/img_width, 
                        y1=y/img_height, 
                        x2=(x+w)/img_width, 
                        y2=(y+h)/img_height
                    ))
        
        return table_regions
    
    def _detect_lines(self, binary: np.ndarray, horizontal: bool) -> np.ndarray:
        """Detect horizontal or vertical lines in the image"""
        # Define the structuring element
        if horizontal:
            structure_size = int(binary.shape[1] / 30)
            structure = cv2.getStructuringElement(cv2.MORPH_RECT, (structure_size, 1))
        else:
            structure_size = int(binary.shape[0] / 30)
            structure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, structure_size))
        
        # Apply morphological operations
        eroded = cv2.erode(binary, structure)
        dilated = cv2.dilate(eroded, structure)
        
        return dilated
    
    def _extract_table_structure(self, image: np.ndarray, table_region: Bbox) -> Tuple[List[int], List[int]]:
        """Extract the row and column positions for a table region"""
        # Convert normalized coordinates to pixel values
        img_height, img_width = image.shape[:2]
        x1 = int(table_region.x1 * img_width)
        y1 = int(table_region.y1 * img_height)
        x2 = int(table_region.x2 * img_width)
        y2 = int(table_region.y2 * img_height)
        
        # Crop the table region
        table_img = image[y1:y2, x1:x2]
        if len(table_img.shape) == 3:
            table_gray = cv2.cvtColor(table_img, cv2.COLOR_BGR2GRAY)
        else:
            table_gray = table_img
        
        # Apply thresholding
        _, binary = cv2.threshold(table_gray, 150, 255, cv2.THRESH_BINARY_INV)
        
        # Detect horizontal and vertical lines
        horizontal = self._detect_lines(binary, True)
        vertical = self._detect_lines(binary, False)
        
        # Extract row positions
        row_positions = self._extract_line_positions(horizontal, is_horizontal=True)
        
        # Extract column positions
        col_positions = self._extract_line_positions(vertical, is_horizontal=False)
        
        # Add table boundaries if needed
        if not row_positions or row_positions[0] > self.max_line_gap:
            row_positions.insert(0, 0)
        if not row_positions or row_positions[-1] < binary.shape[0] - self.max_line_gap:
            row_positions.append(binary.shape[0])
            
        if not col_positions or col_positions[0] > self.max_line_gap:
            col_positions.insert(0, 0)
        if not col_positions or col_positions[-1] < binary.shape[1] - self.max_line_gap:
            col_positions.append(binary.shape[1])
        
        # Convert back to original image coordinates
        row_positions = [y1 + y for y in row_positions]
        col_positions = [x1 + x for x in col_positions]
        
        return row_positions, col_positions
    
    def _extract_line_positions(self, line_image: np.ndarray, is_horizontal: bool) -> List[int]:
        """Extract positions of lines from a binary image of lines"""
        if is_horizontal:
            # Sum along columns to get row profile
            profile = np.sum(line_image, axis=1)
        else:
            # Sum along rows to get column profile
            profile = np.sum(line_image, axis=0)
        
        # Find peaks in the profile
        threshold = 0.5 * np.max(profile)
        line_positions = []
        
        i = 0
        while i < len(profile):
            if profile[i] > threshold:
                # Found a line, calculate its center position
                start = i
                while i < len(profile) and profile[i] > threshold:
                    i += 1
                end = i - 1
                center = (start + end) // 2
                line_positions.append(center)
            else:
                i += 1
        
        # Filter to remove duplicates and sort
        if line_positions:
            filtered_positions = [line_positions[0]]
            for pos in line_positions[1:]:
                if pos - filtered_positions[-1] > self.max_line_gap:
                    filtered_positions.append(pos)
            
            return sorted(filtered_positions)
        else:
            return []
    
    def _map_tokens_to_cells(self, page: Page, rows: List[int], cols: List[int], table_region: Bbox) -> Table:
        """Map OCR tokens to table cells based on their positions"""
        if not page.tokens or len(rows) < 2 or len(cols) < 2:
            # Not enough information to build a table
            return Table(bbox=table_region)
        
        img_height = page.height or 1.0
        img_width = page.width or 1.0
        
        # Create empty cells matrix
        num_rows = len(rows) - 1
        num_cols = len(cols) - 1
        table_cells = [[Cell() for _ in range(num_cols)] for _ in range(num_rows)]
        
        # Map tokens to cells
        for token in page.tokens:
            # Check if token is inside the table region
            if not self._is_inside(token.bbox, table_region):
                continue
                
            # Find which cell the token belongs to
            row_idx = -1
            for i in range(len(rows) - 1):
                y1_norm = rows[i] / img_height
                y2_norm = rows[i+1] / img_height
                if y1_norm <= token.bbox.y1 < y2_norm or y1_norm < token.bbox.y2 <= y2_norm:
                    row_idx = i
                    break
            
            col_idx = -1
            for j in range(len(cols) - 1):
                x1_norm = cols[j] / img_width
                x2_norm = cols[j+1] / img_width
                if x1_norm <= token.bbox.x1 < x2_norm or x1_norm < token.bbox.x2 <= x2_norm:
                    col_idx = j
                    break
            
            # Add token to cell if valid position found
            if row_idx >= 0 and col_idx >= 0:
                cell = table_cells[row_idx][col_idx]
                cell.tokens.append(token)
                if cell.text:
                    cell.text += " " + token.text
                else:
                    cell.text = token.text
                
                # Update confidence as average of token confidences
                cell.confidence = sum(t.confidence for t in cell.tokens) / len(cell.tokens)
                
                # Update the cell
                table_cells[row_idx][col_idx] = cell
        
        # Create table with extracted cells
        table = Table(
            bbox=table_region,
            rows=table_cells,
            confidence=self._calculate_table_confidence(table_cells)
        )
        
        # Try to extract column headers
        table.columns = self._extract_column_headers(table)
        
        return table
    
    def _is_inside(self, token_bbox: Bbox, region_bbox: Bbox) -> bool:
        """Check if a token is inside a region with some tolerance"""
        # Allow for a small tolerance since bounding boxes may not align perfectly
        tolerance = 0.01
        return (
            token_bbox.x1 >= region_bbox.x1 - tolerance and
            token_bbox.y1 >= region_bbox.y1 - tolerance and
            token_bbox.x2 <= region_bbox.x2 + tolerance and
            token_bbox.y2 <= region_bbox.y2 + tolerance
        )
    
    def _calculate_table_confidence(self, cells: List[List[Cell]]) -> float:
        """Calculate overall confidence score for a table"""
        # Count cells with content
        total_cells = sum(len(row) for row in cells)
        non_empty_cells = sum(1 for row in cells for cell in row if cell.tokens)
        
        # Calculate proportion of filled cells
        fill_ratio = non_empty_cells / total_cells if total_cells > 0 else 0
        
        # Calculate average confidence of non-empty cells
        confidences = [cell.confidence for row in cells for cell in row if cell.tokens]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        # Combine metrics (higher fill ratio and confidence is better)
        return (fill_ratio * 0.5 + avg_confidence * 0.5)
    
    def _extract_column_headers(self, table: Table) -> List[str]:
        """Extract column headers from the first row of the table"""
        if not table.rows:
            return []
        
        # Assume the first row contains headers
        header_row = table.rows[0]
        
        # Extract header texts
        headers = [cell.text for cell in header_row]
        
        # Check if these are likely headers (non-empty, relatively short)
        if all(len(h) > 0 and len(h) < 50 for h in headers):
            return headers
        else:
            # Generate generic column names
            return [f"Column {i+1}" for i in range(len(header_row))]
    
    def extract_table_content(self, page_image: np.ndarray, tokens: List[dict]) -> TableContentTD:
        """
        Legacy method for extracting table content from an image and OCR tokens.
        Used for backward compatibility with existing code.
        
        Args:
            page_image: Image containing the table
            tokens: List of OCR tokens
            
        Returns:
            Dictionary with table content
        """
        try:
            # Convert the grayscale image
            if len(page_image.shape) == 3:
                gray = cv2.cvtColor(page_image, cv2.COLOR_BGR2GRAY)
            else:
                gray = page_image
            
            # Detect lines
            horizontal_lines, vertical_lines = self._detect_table_lines(gray)
            
            # Extract row and column positions
            row_positions = self._cluster_line_positions(horizontal_lines, True, gray.shape)
            col_positions = self._cluster_line_positions(vertical_lines, False, gray.shape)
            
            # Create grid cells
            cells = self._create_grid_cells(row_positions, col_positions)
            
            # Map tokens to cells
            grid = self._map_tokens_to_grid(tokens, cells)
            
            # Extract table content
            rows, columns = self._structure_table_data(grid, row_positions, col_positions)
            
            return {"rows": rows, "columns": columns}
        
        except Exception as e:
            logger.error(f"Error extracting table content: {str(e)}")
            return {"rows": [], "columns": []}
    
    def _detect_table_lines(self, gray_image: np.ndarray) -> Tuple[List[Tuple[int, int, int, int]], List[Tuple[int, int, int, int]]]:
        """Detect horizontal and vertical lines in a grayscale image"""
        # Apply binary threshold
        _, binary = cv2.threshold(gray_image, 150, 255, cv2.THRESH_BINARY_INV)
        
        # Get image dimensions
        height, width = binary.shape
        
        # Detect horizontal lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (width // 30, 1))
        horizontal_mask = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
        horizontal_lines = cv2.HoughLinesP(horizontal_mask, 1, np.pi/180, 50, 
                                         minLineLength=width//3, maxLineGap=20)
        
        # Detect vertical lines
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, height // 30))
        vertical_mask = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)
        vertical_lines = cv2.HoughLinesP(vertical_mask, 1, np.pi/180, 50, 
                                       minLineLength=height//3, maxLineGap=20)
        
        # Convert to list of line tuples (x1, y1, x2, y2)
        h_lines = []
        if horizontal_lines is not None:
            h_lines = [(line[0][0], line[0][1], line[0][2], line[0][1]) for line in horizontal_lines]
        
        v_lines = []
        if vertical_lines is not None:
            v_lines = [(line[0][0], line[0][1], line[0][0], line[0][3]) for line in vertical_lines]
        
        return h_lines, v_lines
    
    def _cluster_line_positions(self, lines: List[Tuple[int, int, int, int]], 
                              is_horizontal: bool, img_shape: Tuple[int, int]) -> List[int]:
        """Cluster lines to identify consistent row/column positions"""
        if not lines:
            return [0, img_shape[0 if is_horizontal else 1]]
        
        # Extract coordinates
        positions = []
        for line in lines:
            y = (line[1] + line[3]) // 2 if is_horizontal else (line[0] + line[2]) // 2
            positions.append(y)
        
        # Sort positions
        positions.sort()
        
        # Cluster nearby lines
        clusters = []
        current_cluster = [positions[0]]
        
        for pos in positions[1:]:
            if pos - current_cluster[-1] <= self.max_line_gap:
                current_cluster.append(pos)
            else:
                # Calculate average position for cluster
                clusters.append(sum(current_cluster) // len(current_cluster))
                current_cluster = [pos]
        
        # Add last cluster
        if current_cluster:
            clusters.append(sum(current_cluster) // len(current_cluster))
        
        # Add boundaries
        if clusters and clusters[0] > 10:
            clusters.insert(0, 0)
        
        img_dim = img_shape[0] if is_horizontal else img_shape[1]
        if clusters and clusters[-1] < img_dim - 10:
            clusters.append(img_dim)
        
        return clusters
    
    def _create_grid_cells(self, row_positions: List[int], col_positions: List[int]) -> List[List[Tuple[int, int, int, int]]]:
        """Create grid cell coordinates from row and column positions"""
        cells = []
        for i in range(len(row_positions) - 1):
            row = []
            for j in range(len(col_positions) - 1):
                cell = (
                    col_positions[j],         # x1
                    row_positions[i],         # y1
                    col_positions[j + 1],     # x2
                    row_positions[i + 1]      # y2
                )
                row.append(cell)
            cells.append(row)
        
        return cells
    
    def _map_tokens_to_grid(self, tokens: List[dict], cells: List[List[Tuple[int, int, int, int]]]) -> List[List[List[dict]]]:
        """Map OCR tokens to grid cells"""
        # Initialize empty grid
        grid = [[[] for _ in range(len(row))] for row in cells]
        
        # Calculate IoU (Intersection over Union) for each token-cell pair
        for token in tokens:
            token_bbox = token["bbox"]  # [x1, y1, x2, y2]
            
            # Find best matching cell
            best_iou = 0
            best_cell = (-1, -1)
            
            for i, row in enumerate(cells):
                for j, cell in enumerate(row):
                    iou = self._calculate_iou(token_bbox, cell)
                    if iou > best_iou:
                        best_iou = iou
                        best_cell = (i, j)
            
            # Add token to best matching cell if IoU is above threshold
            if best_iou > self.iou_threshold:
                i, j = best_cell
                grid[i][j].append(token)
        
        # Sort tokens within cells by their x-coordinate
        for i in range(len(grid)):
            for j in range(len(grid[i])):
                grid[i][j].sort(key=lambda t: t["bbox"][0])
        
        return grid
    
    def _calculate_iou(self, box1: List[int], box2: Tuple[int, int, int, int]) -> float:
        """Calculate Intersection over Union between two boxes"""
        # Extract coordinates
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate area of each box
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        
        # Calculate intersection coordinates
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        # Check if boxes overlap
        if x1_i >= x2_i or y1_i >= y2_i:
            return 0.0
        
        # Calculate intersection area
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union area
        union = area1 + area2 - intersection
        
        # Calculate IoU
        return intersection / union if union > 0 else 0.0
    
    def _structure_table_data(self, grid: List[List[List[dict]]], 
                            row_positions: List[int], col_positions: List[int]) -> Tuple[List[RowTD], List[str]]:
        """Structure table data into rows and columns"""
        rows = []
        
        # Process each row
        for i, grid_row in enumerate(grid):
            row_data = []
            
            # Process each cell in the row
            for j, cell_tokens in enumerate(grid_row):
                if not cell_tokens:
                    continue
                
                # Combine tokens into a single string
                cell_text = " ".join([token["text"] for token in cell_tokens])
                
                # Create cell data
                token_data = {
                    "text": cell_text,
                    "bbox": [
                        col_positions[j],
                        row_positions[i],
                        col_positions[j+1],
                        row_positions[i+1]
                    ],
                    "confidence": sum([t.get("confidence", 1.0) for t in cell_tokens]) / len(cell_tokens) if cell_tokens else 1.0
                }
                
                row_data.append(token_data)
            
            # Add row if not empty
            if row_data:
                rows.append(row_data)
        
        # Extract column names from first row (if available)
        columns = []
        if rows:
            columns = [token["text"] for token in rows[0]]
            
            # If empty headers, generate generic names
            if not all(columns):
                columns = [f"Column {i+1}" for i in range(len(rows[0]))]
        
        return rows, columns
