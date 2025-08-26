from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import cv2
import regex as re
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json

from ...types import Document, Page, Region, Token, Table, Cell
from ...utils.logging import get_logger, log_execution_time

logger = get_logger(__name__)

class PostProcessor:
    """
    Performs document post-processing to refine and enhance extraction results.
    Includes content validation, text consolidation, and relationship extraction.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.pii_redaction = self.config.get("pii_redaction", False)
        self.consolidate_regions = self.config.get("consolidate_regions", True)
        self.extract_relationships = self.config.get("extract_relationships", True)
        self.confidence_threshold = self.config.get("confidence_threshold", 0.7)
        
        # Load PII detection model if needed
        self.pii_model = None
        self.pii_tokenizer = None
        if self.pii_redaction:
            self._load_pii_model()
        
    @log_execution_time
    def process(self, document: Document) -> Document:
        """
        Post-process document extraction results.
        
        Args:
            document: Document object with extracted content
            
        Returns:
            Document with post-processed content
        """
        logger.info(f"Post-processing document with {len(document.pages)} pages")
        
        # Process each page
        for i, page in enumerate(document.pages):
            logger.debug(f"Post-processing page {page.page_num}")
            
            try:
                # Redact PII if configured
                if self.pii_redaction:
                    page = self._redact_pii(page)
                
                # Consolidate similar adjacent regions
                if self.consolidate_regions:
                    page.regions = self._consolidate_regions(page.regions)
                
                # Extract relationships between regions (headers, lists, etc.)
                if self.extract_relationships:
                    self._extract_relationships(page)
                
                # Update the page
                document.pages[i] = page
                
            except Exception as e:
                logger.error(f"Error post-processing page {page.page_num}: {str(e)}")
        
        # Perform document-level post-processing
        document = self._document_level_processing(document)
        
        return document
    
    def _load_pii_model(self):
        """Load PII detection model"""
        try:
            # This would load a proper PII detection model in production
            # For demonstration purposes, we're just loading placeholder elements
            model_name = self.config.get("pii_model_name", "bert-base-uncased")
            self.pii_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.pii_model = AutoModelForSequenceClassification.from_pretrained(model_name)
            logger.info("PII detection model loaded")
        except Exception as e:
            logger.error(f"Error loading PII model: {str(e)}")
    
    def _redact_pii(self, page: Page) -> Page:
        """Redact personally identifiable information"""
        if not self.pii_model or not self.pii_tokenizer:
            return page
        
        # For demonstration, use regex patterns for common PII
        pii_patterns = {
            "email": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
            "phone": r"\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}",
            "ssn": r"\d{3}[-]?\d{2}[-]?\d{4}",
            "credit_card": r"\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}"
        }
        
        # Process each region
        for region_idx, region in enumerate(page.regions):
            redacted_text = region.text
            
            for pii_type, pattern in pii_patterns.items():
                # Find and redact PII
                matches = re.finditer(pattern, redacted_text)
                for match in matches:
                    start, end = match.span()
                    redaction = f"[REDACTED:{pii_type}]"
                    redacted_text = redacted_text[:start] + redaction + redacted_text[end:]
            
            # Update region if text was redacted
            if redacted_text != region.text:
                region.text = redacted_text
                region.attributes["contains_pii"] = True
                page.regions[region_idx] = region
        
        return page
    
    def _consolidate_regions(self, regions: List[Region]) -> List[Region]:
        """Consolidate similar adjacent regions"""
        if not regions or len(regions) <= 1:
            return regions
        
        consolidated = []
        current_region = regions[0]
        
        for next_region in regions[1:]:
            # Check if regions should be merged
            if self._should_merge_regions(current_region, next_region):
                # Merge regions
                current_region = self._merge_regions(current_region, next_region)
            else:
                # Add current region to result and move to next
                consolidated.append(current_region)
                current_region = next_region
        
        # Add the last region
        consolidated.append(current_region)
        
        return consolidated
    
    def _should_merge_regions(self, region1: Region, region2: Region) -> bool:
        """Determine if two regions should be merged"""
        # Check if regions are of the same type
        if region1.type != region2.type:
            return False
        
        # Check if regions are adjacent
        y1_max = region1.bbox.y2
        y2_min = region2.bbox.y1
        vertical_gap = y2_min - y1_max
        
        # Calculate average line height
        r1_height = region1.bbox.y2 - region1.bbox.y1
        r2_height = region2.bbox.y2 - region2.bbox.y1
        avg_height = (r1_height + r2_height) / 2
        
        # Regions should be merged if they're close enough vertically
        # and have similar horizontal positions
        horizontal_overlap = min(region1.bbox.x2, region2.bbox.x2) - max(region1.bbox.x1, region2.bbox.x1)
        horizontal_overlap_ratio = horizontal_overlap / min(
            region1.bbox.x2 - region1.bbox.x1, 
            region2.bbox.x2 - region2.bbox.x1
        )
        
        return (vertical_gap < 0.5 * avg_height and horizontal_overlap_ratio > 0.5)
    
    def _merge_regions(self, region1: Region, region2: Region) -> Region:
        """Merge two regions into one"""
        # Create a new region with combined properties
        merged = Region(
            id=f"{region1.id}_{region2.id}",
            type=region1.type,
            bbox=Bbox(
                x1=min(region1.bbox.x1, region2.bbox.x1),
                y1=min(region1.bbox.y1, region2.bbox.y1),
                x2=max(region1.bbox.x2, region2.bbox.x2),
                y2=max(region1.bbox.y2, region2.bbox.y2)
            ),
            text=f"{region1.text}\n{region2.text}",
            tokens=region1.tokens + region2.tokens,
            confidence=(region1.confidence + region2.confidence) / 2
        )
        
        # Merge attributes
        merged.attributes = {**region1.attributes, **region2.attributes}
        
        return merged
    
    def _extract_relationships(self, page: Page):
        """Extract relationships between regions (headers, paragraphs, etc.)"""
        if not page.regions or len(page.regions) <= 1:
            return
        
        # Sort regions by vertical position
        sorted_regions = sorted(page.regions, key=lambda r: r.bbox.y1)
        
        # Look for header-content relationships
        for i, region in enumerate(sorted_regions):
            if region.type == "heading" and i < len(sorted_regions) - 1:
                # The next region might be related content
                next_region = sorted_regions[i + 1]
                
                # Check if the next region is close enough to be related
                vertical_gap = next_region.bbox.y1 - region.bbox.y2
                avg_height = (region.bbox.y2 - region.bbox.y1 + next_region.bbox.y2 - next_region.bbox.y1) / 2
                
                if vertical_gap < avg_height and next_region.type in ["paragraph", "list"]:
                    # Establish relationship
                    next_region.attributes["header_id"] = region.id
                    region.attributes["content_ids"] = region.attributes.get("content_ids", []) + [next_region.id]
    
    def _document_level_processing(self, document: Document) -> Document:
        """Perform document-level post-processing"""
        # Extract document metadata from content
        document.metadata["extracted_metadata"] = self._extract_document_metadata(document)
        
        # Add processing summary
        document.metadata["processing_summary"] = {
            "page_count": len(document.pages),
            "region_count": sum(len(page.regions) for page in document.pages),
            "table_count": sum(len(page.tables) for page in document.pages),
            "confidence": document.confidence
        }
        
        return document
    
    def _extract_document_metadata(self, document: Document) -> Dict[str, Any]:
        """Extract metadata from document content"""
        metadata = {}
        
        # Look for common metadata patterns
        date_pattern = r"(?:Date|DATE)[:\s]+([A-Za-z0-9\s,/.-]+)"
        invoice_pattern = r"(?:Invoice|INVOICE)[#\s]*[:\s]*([A-Za-z0-9-]+)"
        amount_pattern = r"(?:Total|TOTAL)[:\s]*[$€£]?([0-9,.]+)"
        
        # Concatenate all text for searching
        all_text = ""
        for page in document.pages:
            for region in page.regions:
                all_text += region.text + "\n"
        
        # Extract metadata using patterns
        date_match = re.search(date_pattern, all_text)
        if date_match:
            metadata["date"] = date_match.group(1).strip()
            
        invoice_match = re.search(invoice_pattern, all_text)
        if invoice_match:
            metadata["invoice_number"] = invoice_match.group(1).strip()
            
        amount_match = re.search(amount_pattern, all_text)
        if amount_match:
            metadata["total_amount"] = amount_match.group(1).strip()
        
        return metadata
