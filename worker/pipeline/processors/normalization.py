from typing import Dict, Any, List, Optional, Tuple
import re
import string
import unicodedata
from ...types import Document, Page, Region, Token
from ...utils.logging import get_logger, log_execution_time

logger = get_logger(__name__)

class TextNormalizer:
    """
    Normalize text content from OCR tokens to improve extraction quality.
    Handles common OCR errors, whitespace issues, and text normalization.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        # Load configuration for text normalization
        self.preserve_case = self.config.get("preserve_case", True)
        self.normalize_whitespace = self.config.get("normalize_whitespace", True)
        self.expand_abbreviations = self.config.get("expand_abbreviations", False)
        self.fix_common_errors = self.config.get("fix_common_errors", True)
        self.normalization_level = self.config.get("normalization_level", "basic")
        
        # Common OCR error patterns
        self.error_patterns = {
            r'(\d)l': r'\1I',  # Replace '1' looking like 'l' with 'I'
            r'O(\d)': r'0\1',  # Replace 'O' looking like '0' with '0' in numbers
            r'l(\d)': r'1\1',  # Replace 'l' looking like '1' with '1' in numbers
            r'(\w)\.(\w)': r'\1\2',  # Remove period in the middle of a word (OCR artifact)
            r'rn': 'm',  # Common 'rn' vs 'm' confusion
            r'(\w),(\w)': r'\1.\2',  # Comma vs period in decimal numbers
        }
        
        # Common abbreviations
        self.abbreviations = {
            "corp.": "Corporation",
            "inc.": "Incorporated",
            "ltd.": "Limited",
            "dept.": "Department",
            "dr.": "Doctor",
            "mr.": "Mister",
            "mrs.": "Misses",
            "st.": "Street",
            "co.": "Company",
        }
        
        # Load any additional patterns from config
        if "custom_error_patterns" in self.config:
            self.error_patterns.update(self.config["custom_error_patterns"])
        
        if "custom_abbreviations" in self.config:
            self.abbreviations.update(self.config["custom_abbreviations"])
    
    @log_execution_time
    def process(self, document: Document) -> Document:
        """
        Apply text normalization to document content.
        
        Args:
            document: Document object with tokens and regions
            
        Returns:
            Document with normalized text
        """
        logger.info(f"Normalizing text in document with {len(document.pages)} pages")
        
        for i, page in enumerate(document.pages):
            logger.debug(f"Normalizing text for page {page.page_num}")
            
            try:
                # Normalize token text
                for j, token in enumerate(page.tokens):
                    normalized_text = self._normalize_text(token.text)
                    token.attributes["normalized_text"] = normalized_text
                    page.tokens[j] = token
                
                # Normalize region text
                for j, region in enumerate(page.regions):
                    region.normalized_text = self._normalize_text(region.text)
                    page.regions[j] = region
                
                # Update the page
                document.pages[i] = page
                
            except Exception as e:
                logger.error(f"Error normalizing text on page {page.page_num}: {str(e)}")
        
        return document
    
    def _normalize_text(self, text: str) -> str:
        """Apply various text normalization steps to improve text quality"""
        if not text:
            return ""
        
        # Basic normalization for all levels
        normalized = text
        
        # Normalize unicode characters (decompose then compose)
        normalized = unicodedata.normalize('NFKC', normalized)
        
        if self.normalize_whitespace:
            # Remove excess whitespace
            normalized = self._normalize_whitespace(normalized)
        
        if self.fix_common_errors:
            # Fix common OCR errors
            normalized = self._fix_common_errors(normalized)
        
        # Apply normalization based on configured level
        if self.normalization_level == "basic":
            pass  # Basic normalization already applied
        
        elif self.normalization_level == "medium":
            # Standardize punctuation
            normalized = self._standardize_punctuation(normalized)
        
        elif self.normalization_level == "aggressive":
            # Standardize punctuation
            normalized = self._standardize_punctuation(normalized)
            
            # Convert to lowercase if not preserving case
            if not self.preserve_case:
                normalized = normalized.lower()
            
            # Expand abbreviations
            if self.expand_abbreviations:
                normalized = self._expand_abbreviations(normalized)
        
        return normalized
    
    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace in text"""
        # Replace multiple spaces with a single space
        text = re.sub(r'\s+', ' ', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        # Fix line break issues (OCR sometimes adds line breaks inappropriately)
        text = re.sub(r'(\w)-\s+(\w)', r'\1\2', text)
        
        return text
    
    def _fix_common_errors(self, text: str) -> str:
        """Fix common OCR errors using regular expressions"""
        normalized = text
        
        for pattern, replacement in self.error_patterns.items():
            normalized = re.sub(pattern, replacement, normalized)
        
        return normalized
    
    def _standardize_punctuation(self, text: str) -> str:
        """Standardize punctuation marks"""
        # Standardize quotes
        text = re.sub(r'[''‚‛]', "'", text)
        text = re.sub(r'[""„‟]', '"', text)
        
        # Standardize dashes
        text = re.sub(r'[‒–—―]', '-', text)
        
        # Standardize ellipsis
        text = re.sub(r'\.\.\.', '…', text)
        
        # Handle special cases for decimal points
        text = re.sub(r'(\d),(\d)', r'\1.\2', text)
        
        return text
    
    def _expand_abbreviations(self, text: str) -> str:
        """Expand common abbreviations"""
        normalized = text
        
        # Case insensitive expansion
        for abbr, expansion in self.abbreviations.items():
            pattern = re.compile(re.escape(abbr), re.IGNORECASE)
            normalized = pattern.sub(expansion, normalized)
        
        return normalized
