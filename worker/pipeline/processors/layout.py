from typing import Dict, Any, List, Optional
import numpy as np
import cv2
from ...types import Document, Page, Region, Token, Bbox
from ...utils.logging import get_logger, log_execution_time

logger = get_logger(__name__)

class LayoutProcessor:
    """
    Extract document layout elements like paragraphs, headings, lists, etc.
    from OCR tokens based on their spatial arrangement and formatting.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.min_region_tokens = self.config.get("min_region_tokens", 3)
        self.line_height_threshold = self.config.get("line_height_threshold", 1.5)
        self.paragraph_distance_threshold = self.config.get("paragraph_distance_threshold", 2.0)
        self.heading_size_threshold = self.config.get("heading_size_threshold", 1.2)
        self.max_line_gap = self.config.get("max_line_gap", 0.5)
        
    @log_execution_time
    def process(self, document: Document) -> Document:
        """
        Process a document to extract layout regions.
        
        Args:
            document: Document object with tokens
            
        Returns:
            Document with extracted regions
        """
        logger.info(f"Processing document layout with {len(document.pages)} pages")
        
        for i, page in enumerate(document.pages):
            logger.debug(f"Processing layout for page {page.page_num}")
            if not page.tokens:
                logger.warning(f"No tokens found on page {page.page_num}, skipping layout analysis")
                continue
            
            try:
                # Group tokens into lines
                lines = self._group_tokens_into_lines(page.tokens)
                
                # Group lines into regions
                regions = self._group_lines_into_regions(lines, page)
                
                # Classify regions (paragraph, heading, list, etc.)
                regions = self._classify_regions(regions, page)
                
                # Update the page with extracted regions
                page.regions = regions
                document.pages[i] = page
                
                logger.info(f"Extracted {len(regions)} regions from page {page.page_num}")
            except Exception as e:
                logger.error(f"Error extracting layout from page {page.page_num}: {str(e)}")
                
        return document
    
    def _group_tokens_into_lines(self, tokens: List[Token]) -> List[List[Token]]:
        """Group tokens into lines based on y-coordinate proximity"""
        if not tokens:
            return []
        
        # Sort tokens by y-coordinate and then by x-coordinate
        sorted_tokens = sorted(tokens, key=lambda t: (t.bbox.y1, t.bbox.x1))
        
        lines = []
        current_line = [sorted_tokens[0]]
        avg_height = np.mean([t.bbox.y2 - t.bbox.y1 for t in tokens])
        
        for token in sorted_tokens[1:]:
            # Check if token is on the same line as current_line
            last_token = current_line[-1]
            y_diff = abs(token.bbox.y1 - last_token.bbox.y1)
            
            if y_diff <= self.max_line_gap * avg_height:
                current_line.append(token)
            else:
                # Sort the current line by x-coordinate
                current_line = sorted(current_line, key=lambda t: t.bbox.x1)
                lines.append(current_line)
                current_line = [token]
        
        # Add the last line
        if current_line:
            current_line = sorted(current_line, key=lambda t: t.bbox.x1)
            lines.append(current_line)
        
        return lines
    
    def _group_lines_into_regions(self, lines: List[List[Token]], page: Page) -> List[Region]:
        """Group lines into regions based on spacing and formatting"""
        regions = []
        
        if not lines:
            return regions
        
        current_region_lines = [lines[0]]
        avg_line_height = self._calculate_avg_line_height(lines)
        
        for i in range(1, len(lines)):
            current_line = lines[i]
            prev_line = lines[i-1]
            
            # Calculate vertical distance between lines
            prev_line_bottom = max(t.bbox.y2 for t in prev_line)
            curr_line_top = min(t.bbox.x1 for t in current_line)
            distance = curr_line_top - prev_line_bottom
            
            if distance <= self.paragraph_distance_threshold * avg_line_height:
                # Lines are part of the same region
                current_region_lines.append(current_line)
            else:
                # Create a new region from the collected lines
                if len(current_region_lines) >= self.min_region_tokens:
                    region = self._create_region_from_lines(current_region_lines)
                    regions.append(region)
                
                # Start a new region
                current_region_lines = [current_line]
        
        # Add the last region
        if current_region_lines and len(current_region_lines) >= self.min_region_tokens:
            region = self._create_region_from_lines(current_region_lines)
            regions.append(region)
        
        return regions
    
    def _calculate_avg_line_height(self, lines: List[List[Token]]) -> float:
        """Calculate the average line height"""
        if not lines:
            return 0.0
        
        heights = []
        for line in lines:
            if line:
                min_y = min(t.bbox.y1 for t in line)
                max_y = max(t.bbox.y2 for t in line)
                heights.append(max_y - min_y)
        
        return np.mean(heights) if heights else 0.0
    
    def _create_region_from_lines(self, lines: List[List[Token]]) -> Region:
        """Create a region from lines of tokens"""
        # Flatten all tokens
        all_tokens = [token for line in lines for token in line]
        
        # Calculate bounding box for the region
        x1 = min(token.bbox.x1 for token in all_tokens)
        y1 = min(token.bbox.y1 for token in all_tokens)
        x2 = max(token.bbox.x2 for token in all_tokens)
        y2 = max(token.bbox.y2 for token in all_tokens)
        
        # Extract text
        text = " ".join([" ".join([token.text for token in line]) for line in lines])
        
        # Calculate confidence as average of token confidences
        confidence = sum(token.confidence for token in all_tokens) / len(all_tokens) if all_tokens else 1.0
        
        # Create the region
        return Region(
            type="paragraph",  # Default type, will be classified later
            bbox=Bbox(x1=x1, y1=y1, x2=x2, y2=y2),
            text=text,
            tokens=all_tokens,
            confidence=confidence
        )
    
    def _classify_regions(self, regions: List[Region], page: Page) -> List[Region]:
        """Classify regions as paragraphs, headings, lists, etc."""
        if not regions:
            return []
        
        for region in regions:
            # Extract features for classification
            region_tokens = region.tokens
            if not region_tokens:
                continue
                
            # Check for heading based on font size and position
            is_heading = self._is_heading(region, regions, page)
            
            # Check for list based on formatting
            is_list = self._is_list(region)
            
            # Assign region type
            if is_heading:
                region.type = "heading"
            elif is_list:
                region.type = "list"
            else:
                region.type = "paragraph"
        
        return regions
    
    def _is_heading(self, region: Region, regions: List[Region], page: Page) -> bool:
        """Determine if a region is a heading based on font size and position"""
        # Simple implementation - can be expanded with more sophisticated logic
        region_height = region.bbox.y2 - region.bbox.y1
        avg_region_height = np.mean([r.bbox.y2 - r.bbox.y1 for r in regions])
        
        # Check if region is significantly taller (suggesting larger font)
        is_larger_font = region_height > self.heading_size_threshold * avg_region_height
        
        # Check if region is near the top of the page
        page_height = 1.0  # Assuming normalized coordinates
        is_top_position = region.bbox.y1 < page_height * 0.2
        
        # Check if text is short (headings are typically shorter than paragraphs)
        is_short_text = len(region.text.split()) <= 20
        
        return (is_larger_font or is_top_position) and is_short_text
    
    def _is_list(self, region: Region) -> bool:
        """Determine if a region is a list based on formatting"""
        # Simple implementation - can be expanded with more sophisticated logic
        text = region.text.strip()
        lines = text.split("\n")
        
        # Check for bullet points or numbering
        bullet_patterns = ["•", "·", "-", "*"]
        numbered_pattern = any(line.strip() and line.strip()[0].isdigit() and "." in line[:5] for line in lines)
        bullet_pattern = any(line.strip() and any(line.strip().startswith(bullet) for bullet in bullet_patterns) for line in lines)
        
        return bullet_pattern or numbered_pattern
