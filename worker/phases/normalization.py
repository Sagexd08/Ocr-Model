from typing import Dict, Any, List, Optional
import re
import datetime

from ..types import Document
from ..model_manager import ModelManager
from ..utils.logging import get_logger

logger = get_logger(__name__)

# Regex patterns for deterministic normalization
PATTERNS = {
    "date_dmy": r"(\d{1,2})[/.-](\d{1,2})[/.-](\d{2,4})",
    "date_mdy": r"(\d{1,2})[/.-](\d{1,2})[/.-](\d{2,4})",
    "currency": r"[\$€£¥][\s]*([0-9,]+\.?[0-9]*)",
    "number": r"([0-9,]+\.?[0-9]*)",
    "phone": r"(\+?[0-9]{1,3}[- ]?)?(\([0-9]{3}\)|[0-9]{3})[- ]?[0-9]{3}[- ]?[0-9]{4}",
    "email": r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+",
    "ssn": r"\b(?!000|666|9\d{2})([0-8]\d{2}|7([0-6]\d))-(?!00)\d{2}-(?!0000)\d{4}\b",
}

# LLM system prompt for normalization (required verbatim)
LLM_SYSTEM_PROMPT = """You are a strict normalizer. Input: OCR tokens (text, bbox, confidence). Output: JSON matching the provided schema only. Do NOT fabricate missing values. If unsure about any field, set needs_review=true. Output JSON only."""

def normalize(
    document: Document, 
    model_manager: ModelManager,
    confidence_threshold: float = 0.8,
    use_llm: bool = False
) -> Document:
    """
    Apply normalization rules to document text.
    
    Args:
        document: Document to normalize
        model_manager: Model provider for optional LLM normalization
        confidence_threshold: Confidence threshold for review flagging
        use_llm: Whether to use LLM for complex normalization
    
    Returns:
        Normalized document
    """
    # Apply deterministic rule-based normalization
    document = _apply_deterministic_rules(document, confidence_threshold)
    
    # Optionally use LLM for more complex normalization
    if use_llm and model_manager.is_model_loaded("llm_normalizer"):
        document = _apply_llm_normalization(document, model_manager, confidence_threshold)
    
    return document

def _apply_deterministic_rules(document: Document, confidence_threshold: float) -> Document:
    """Apply deterministic normalization rules"""
    for page in document.pages:
        # Normalize text regions
        for region in page.regions:
            text = region.text
            tokens = region.tokens
            
            # Track if any token has low confidence
            has_low_confidence = any(token.confidence < confidence_threshold for token in tokens)
            
            # Only apply normalizations to high-confidence text unless configured otherwise
            if not has_low_confidence or getattr(document, "normalize_all", False):
                # Number normalization (remove commas)
                if re.search(PATTERNS["number"], text):
                    normalized = re.sub(r"(\d),(\d)", r"\1\2", text)
                    if normalized != text:
                        region.normalized_text = normalized
                
                # Date normalization (to ISO format)
                date_match = re.search(PATTERNS["date_dmy"], text)
                if date_match:
                    try:
                        d, m, y = map(int, date_match.groups())
                        # Handle 2-digit year
                        if y < 100:
                            y = 2000 + y if y < 50 else 1900 + y
                        iso_date = f"{y:04d}-{m:02d}-{d:02d}"
                        region.normalized_text = iso_date
                    except (ValueError, IndexError):
                        # If date parsing fails, flag for review
                        region.needs_review = True
            
            # Always flag low confidence for review
            if has_low_confidence:
                region.needs_review = True
        
        # Normalize table cells
        for table in page.tables:
            for row in table.rows:
                for cell in row:
                    text = cell.text
                    tokens = cell.tokens
                    
                    # Track if any token has low confidence
                    has_low_confidence = any(token.confidence < confidence_threshold for token in tokens)
                    
                    # Only apply normalizations to high-confidence text
                    if not has_low_confidence or getattr(document, "normalize_all", False):
                        # Number normalization (remove commas)
                        if re.search(PATTERNS["number"], text):
                            normalized = re.sub(r"(\d),(\d)", r"\1\2", text)
                            if normalized != text:
                                cell.normalized_text = normalized
                    
                    # Always flag low confidence for review
                    if has_low_confidence:
                        cell.needs_review = True
                    
    return document

def _apply_llm_normalization(
    document: Document, 
    model_manager: ModelManager,
    confidence_threshold: float
) -> Document:
    """
    Apply LLM-based normalization for complex cases
    
    This is an optional enhancement that uses LLMs for structural
    disambiguation and normalization.
    """
    # Skip if LLM normalization is not available
    if not model_manager.is_model_loaded("llm_normalizer"):
        logger.info("LLM normalization requested but model not loaded")
        return document
        
    logger.info("Applying LLM normalization")
    
    # Process complex regions that need structural disambiguation
    for page in document.pages:
        # Find regions that might benefit from LLM normalization
        complex_regions = []
        
        for region in page.regions:
            # Only consider regions that might be structured data
            if region.type in ["key_value", "form_field"] or len(region.tokens) > 5:
                complex_regions.append(region)
        
        if not complex_regions:
            continue
            
        # Batch regions for efficiency
        for i in range(0, len(complex_regions), 5):
            batch = complex_regions[i:i+5]
            
            # Prepare input context for the LLM
            context = []
            for region in batch:
                tokens = [
                    {
                        "text": t.text,
                        "bbox": [t.bbox.x1, t.bbox.y1, t.bbox.x2, t.bbox.y2],
                        "confidence": t.confidence
                    } 
                    for t in region.tokens
                ]
                context.append({
                    "region_id": region.id,
                    "text": region.text,
                    "tokens": tokens
                })
                
            # Send to LLM normalizer with strict system prompt
            try:
                normalized = model_manager.normalize_with_llm(
                    context=context,
                    system_prompt=LLM_SYSTEM_PROMPT
                )
                
                # Apply normalizations from LLM
                for item in normalized:
                    region_id = item.get("region_id")
                    normalized_text = item.get("normalized_text")
                    needs_review = item.get("needs_review", False)
                    
                    # Find matching region
                    for region in batch:
                        if region.id == region_id and normalized_text:
                            region.normalized_text = normalized_text
                            if needs_review:
                                region.needs_review = True
                            break
                            
            except Exception as e:
                logger.error(f"LLM normalization failed: {str(e)}")
                # Flag these regions for review on failure
                for region in batch:
                    region.needs_review = True
                
    return document
