from typing import Dict, Any, Optional, List, Tuple, Set, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import math
from PIL import Image
import os
import uuid
import json
import re
import difflib
import string
from collections import Counter
from datetime import datetime

from ...types import Document, Page, Region, Token, Bbox, Table
from ...utils.logging import get_logger, log_execution_time

logger = get_logger(__name__)

class QAProcessor:
    """
    Advanced quality analysis and validation on OCR results.
    Performs deep analysis to detect errors, calculate confidence scores,
    and flag items for review with high precision.
    
    Features:
    - Multi-level confidence scoring (token, region, page, document)
    - Linguistic pattern validation and anomaly detection
    - Semantic consistency checking
    - Advanced table structure validation
    - Document type-specific validation rules
    - Context-aware spell checking
    - Numeric and date format validation
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Confidence thresholds
        self.low_confidence_threshold = self.config.get("low_confidence_threshold", 0.7)
        self.high_confidence_threshold = self.config.get("high_confidence_threshold", 0.9)
        self.critical_confidence_threshold = self.config.get("critical_confidence_threshold", 0.5)
        
        # Validation features
        self.spell_check = self.config.get("spell_check", True)
        self.validate_dates = self.config.get("validate_dates", True)
        self.validate_amounts = self.config.get("validate_amounts", True)
        self.validate_tables = self.config.get("validate_tables", True)
        self.validate_document_structure = self.config.get("validate_document_structure", True)
        self.validate_language = self.config.get("validate_language", True)
        self.detect_context_anomalies = self.config.get("detect_context_anomalies", True)
        self.detect_numeric_anomalies = self.config.get("detect_numeric_anomalies", True)
        
        # Document type-specific settings
        self.document_type_validators = self.config.get("document_type_validators", {})
        
        # Critical fields detection
        self.critical_fields = self.config.get("critical_fields", [
            "total", "amount", "date", "invoice number", "account", "ssn", "tax"
        ])
        
        # Load dictionaries and validation resources
        self._load_validation_resources()
        
    @log_execution_time
    @log_execution_time
    def process(self, document: Document) -> Document:
        """
        Process a document to validate OCR results and flag potential issues with high precision.
        
        Args:
            document: Document object with OCR results
            
        Returns:
            Document with comprehensive quality validation metadata
        """
        logger.info(f"Performing advanced quality analysis on document with {len(document.pages)} pages")
        
        # Initialize document type if available
        document_type = document.metadata.get("document_type", "unknown")
        logger.info(f"Document type: {document_type}")
        
        # Store comprehensive validation results
        document.metadata["qa_results"] = {
            "overall_confidence": 0.0,
            "needs_review": False,
            "quality_score": 0.0,  # 0-100 quality score
            "error_counts": {
                "critical": 0,     # Critical errors requiring immediate attention
                "major": 0,        # Major issues that likely need correction
                "minor": 0,        # Minor issues that might be acceptable
                "spelling": 0,
                "low_confidence": 0,
                "critical_field_issues": 0,
                "invalid_dates": 0,
                "invalid_amounts": 0,
                "table_errors": 0,
                "structure_issues": 0,
                "numeric_anomalies": 0,
                "context_anomalies": 0
            },
            "critical_field_issues": [],   # Details on critical field issues
            "document_structure_issues": [],
            "confidence_distribution": {}, # Histogram of confidence scores
            "review_areas": []             # Areas that need human review
        }
        
        # Initialize metrics
        total_confidence = 0.0
        total_tokens = 0
        confidence_scores = []
        
        # Extract document context for semantic validation
        document_context = self._extract_document_context(document)
        
        # Process each page
        for i, page in enumerate(document.pages):
            logger.debug(f"Validating page {page.page_num}")
            
            # Initialize page quality metrics
            if not hasattr(page, 'attributes'):
                page.attributes = {}
                
            page.attributes["qa_results"] = {
                "confidence": 0.0,
                "quality_score": 0.0,
                "issues": [],
                "error_counts": {
                    "critical": 0,
                    "major": 0,
                    "minor": 0
                }
            }
            
            page_confidence_scores = []
            
            # Validate tokens
            self._validate_tokens(page, document, document_context)
            
            # Calculate token-level metrics
            for token in page.tokens:
                confidence_scores.append(token.confidence)
                page_confidence_scores.append(token.confidence)
                total_confidence += token.confidence
                total_tokens += 1
            
            # Validate regions
            self._validate_regions(page, document, document_context, document_type)
            
            # Validate tables
            if self.validate_tables:
                self._validate_tables(page, document)
            
            # Calculate page-level confidence and quality metrics
            if page_confidence_scores:
                page.attributes["qa_results"]["confidence"] = np.mean(page_confidence_scores)
                
                # Calculate page quality score (more sophisticated than just confidence)
                total_issues = sum(page.attributes["qa_results"]["error_counts"].values())
                quality_penalty = min(50, total_issues * 5)  # More issues = lower quality
                page.attributes["qa_results"]["quality_score"] = max(0, 
                    min(100, 100 * page.attributes["qa_results"]["confidence"] - quality_penalty))
            
            # Update page
            document.pages[i] = page
        
        # Perform document-level validation
        if self.validate_document_structure:
            self._validate_document_structure(document, document_type, document_context)
        
        if self.detect_context_anomalies:
            self._detect_context_anomalies(document, document_context)
        
        # Calculate overall document metrics
        if confidence_scores:
            # Calculate overall confidence
            document.metadata["qa_results"]["overall_confidence"] = np.mean(confidence_scores)
            
            # Calculate confidence distribution
            bins = np.linspace(0, 1, 11)  # 0.0-1.0 in 0.1 increments
            hist, edges = np.histogram(confidence_scores, bins=bins)
            document.metadata["qa_results"]["confidence_distribution"] = {
                f"{edges[i]:.1f}-{edges[i+1]:.1f}": int(hist[i]) 
                for i in range(len(hist))
            }
        
        # Calculate document quality score
        error_weights = {
            "critical": 10,
            "major": 5,
            "minor": 1
        }
        
        error_penalty = 0
        for err_type, weight in error_weights.items():
            error_penalty += document.metadata["qa_results"]["error_counts"][err_type] * weight
            
        base_score = 100 * document.metadata["qa_results"]["overall_confidence"]
        document.metadata["qa_results"]["quality_score"] = max(0, 
            min(100, base_score - min(75, error_penalty)))
        
        # Determine document review status
        document.metadata["qa_results"]["needs_review"] = (
            document.metadata["qa_results"]["error_counts"]["critical"] > 0 or
            document.metadata["qa_results"]["error_counts"]["major"] > 2 or
            document.metadata["qa_results"]["overall_confidence"] < self.low_confidence_threshold or
            document.metadata["qa_results"]["quality_score"] < 70
        )
        
        # Generate review areas
        document.metadata["qa_results"]["review_areas"] = self._generate_review_areas(document)
        
        # Add quality assessment
        if document.metadata["qa_results"]["quality_score"] >= 90:
            document.metadata["qa_results"]["assessment"] = "excellent"
        elif document.metadata["qa_results"]["quality_score"] >= 75:
            document.metadata["qa_results"]["assessment"] = "good"
        elif document.metadata["qa_results"]["quality_score"] >= 50:
            document.metadata["qa_results"]["assessment"] = "fair"
        else:
            document.metadata["qa_results"]["assessment"] = "poor"
        
        logger.info(f"Advanced quality analysis complete. "
                   f"Confidence: {document.metadata['qa_results']['overall_confidence']:.2f}, "
                   f"Quality Score: {document.metadata['qa_results']['quality_score']:.1f}, "
                   f"Critical Errors: {document.metadata['qa_results']['error_counts']['critical']}, "
                   f"Major Errors: {document.metadata['qa_results']['error_counts']['major']}")
        
        return document
        
    def _validate_tokens(self, page: Page, document: Document, document_context: Dict[str, Any]) -> None:
        """Validate tokens with comprehensive checks"""
        for j, token in enumerate(page.tokens):
            # Initialize token attributes if needed
            if not hasattr(token, 'attributes') or token.attributes is None:
                token.attributes = {}
            
            # Check token confidence with tiered thresholds
            if token.confidence < self.critical_confidence_threshold:
                token.attributes["needs_review"] = True
                token.attributes["review_reason"] = "critical_low_confidence"
                token.attributes["severity"] = "critical"
                document.metadata["qa_results"]["error_counts"]["critical"] += 1
                document.metadata["qa_results"]["error_counts"]["low_confidence"] += 1
                page.attributes["qa_results"]["error_counts"]["critical"] += 1
                
            elif token.confidence < self.low_confidence_threshold:
                token.attributes["needs_review"] = True
                token.attributes["review_reason"] = "low_confidence"
                token.attributes["severity"] = "major"
                document.metadata["qa_results"]["error_counts"]["major"] += 1
                document.metadata["qa_results"]["error_counts"]["low_confidence"] += 1
                page.attributes["qa_results"]["error_counts"]["major"] += 1
            
            # Check if token might be part of a critical field
            if self._is_critical_field_token(token, document_context):
                # Apply stricter standards to critical fields
                if token.confidence < self.high_confidence_threshold:
                    token.attributes["needs_review"] = True
                    token.attributes["review_reason"] = "critical_field_low_confidence"
                    token.attributes["severity"] = "major"
                    document.metadata["qa_results"]["error_counts"]["major"] += 1
                    document.metadata["qa_results"]["error_counts"]["critical_field_issues"] += 1
                    page.attributes["qa_results"]["error_counts"]["major"] += 1
                    
                    document.metadata["qa_results"]["critical_field_issues"].append({
                        "field": token.attributes.get("critical_field_type", "unknown"),
                        "text": token.text,
                        "confidence": token.confidence,
                        "page": page.page_num,
                        "bbox": token.bbox.to_dict() if hasattr(token, 'bbox') else None
                    })
            
            # Perform context-aware spell checking
            if self.spell_check and len(token.text) > 2 and token.text.isalpha():
                # Skip potential proper nouns (capitalized words not at start of sentence)
                is_likely_proper_noun = token.text[0].isupper() and token.position.get('is_first_in_line', False) == False
                
                # Skip numbers and codes
                is_numeric_or_code = any(c.isdigit() for c in token.text) or '_' in token.text
                
                if not (is_likely_proper_noun or is_numeric_or_code):
                    if not self._is_valid_word(token.text, document_context):
                        token.attributes["needs_review"] = True
                        token.attributes["review_reason"] = "spelling"
                        token.attributes["severity"] = "minor"
                        token.attributes["spelling_suggestions"] = self._get_spelling_suggestions(token.text, document_context)
                        document.metadata["qa_results"]["error_counts"]["minor"] += 1
                        document.metadata["qa_results"]["error_counts"]["spelling"] += 1
                        page.attributes["qa_results"]["error_counts"]["minor"] += 1
            
            # Update token in page
            page.tokens[j] = token
            
    def _validate_regions(self, page: Page, document: Document, 
                         document_context: Dict[str, Any], document_type: str) -> None:
        """Validate regions with comprehensive checks"""
        for j, region in enumerate(page.regions):
            # Skip non-text regions
            if region.type in ["image", "figure", "chart"]:
                continue
                
            # Initialize region attributes if needed
            if not hasattr(region, 'attributes') or region.attributes is None:
                region.attributes = {}
            
            # Skip empty regions
            if not region.text or len(region.text.strip()) == 0:
                continue
                
            region_issues = []
            
            # Check for critical fields in this region
            critical_field_matches = self._identify_critical_fields(region.text, document_type)
            if critical_field_matches:
                region.attributes["critical_fields"] = critical_field_matches
                
                # Validate each critical field value
                for field in critical_field_matches:
                    is_valid = self._validate_critical_field(field["field_type"], field["value"], document_context)
                    if not is_valid:
                        region_issues.append({
                            "type": "invalid_critical_field",
                            "severity": "critical",
                            "field": field["field_type"],
                            "value": field["value"]
                        })
                        document.metadata["qa_results"]["error_counts"]["critical"] += 1
                        document.metadata["qa_results"]["error_counts"]["critical_field_issues"] += 1
                        page.attributes["qa_results"]["error_counts"]["critical"] += 1
                        
                        # Add to critical field issues
                        document.metadata["qa_results"]["critical_field_issues"].append({
                            "field": field["field_type"],
                            "text": field["value"],
                            "page": page.page_num,
                            "reason": "invalid_format"
                        })
            
            # Validate dates in region
            if self.validate_dates:
                dates, issues = self._extract_and_validate_dates(region.text)
                if issues:
                    region.attributes["invalid_dates"] = dates
                    region_issues.extend(issues)
                    document.metadata["qa_results"]["error_counts"]["major"] += len(issues)
                    document.metadata["qa_results"]["error_counts"]["invalid_dates"] += len(issues)
                    page.attributes["qa_results"]["error_counts"]["major"] += len(issues)
            
            # Validate monetary amounts
            if self.validate_amounts:
                amounts, issues = self._extract_and_validate_amounts(region.text, document_context)
                if issues:
                    region.attributes["invalid_amounts"] = amounts
                    region_issues.extend(issues)
                    document.metadata["qa_results"]["error_counts"]["major"] += len(issues)
                    document.metadata["qa_results"]["error_counts"]["invalid_amounts"] += len(issues)
                    page.attributes["qa_results"]["error_counts"]["major"] += len(issues)
            
            # Check for numeric anomalies
            if self.detect_numeric_anomalies:
                anomalies = self._detect_numeric_anomalies(region.text, document_context)
                if anomalies:
                    region.attributes["numeric_anomalies"] = anomalies
                    region_issues.extend(anomalies)
                    document.metadata["qa_results"]["error_counts"]["major"] += len(anomalies)
                    document.metadata["qa_results"]["error_counts"]["numeric_anomalies"] += len(anomalies)
                    page.attributes["qa_results"]["error_counts"]["major"] += len(anomalies)
            
            # Flag region if it has issues
            if region_issues:
                region.needs_review = True
                region.attributes["validation_issues"] = region_issues
                page.attributes["qa_results"]["issues"].append({
                    "region_id": region.id,
                    "issues": region_issues
                })
            
            # Update region
            page.regions[j] = region
            
    def _validate_tables(self, page: Page, document: Document) -> None:
        """Perform comprehensive table validation"""
        for j, table in enumerate(page.tables):
            # Skip tables without cells
            if not hasattr(table, 'cells') or not table.cells:
                continue
                
            # Initialize table attributes if needed
            if not hasattr(table, 'attributes') or table.attributes is None:
                table.attributes = {}
            
            is_valid, issues = self._validate_table_structure(table)
            
            # Check for data inconsistencies
            data_issues = self._validate_table_data(table)
            if data_issues:
                issues.extend(data_issues)
            
            if issues:
                table.needs_review = True
                table.attributes["validation_issues"] = issues
                document.metadata["qa_results"]["error_counts"]["table_errors"] += len(issues)
                
                # Categorize issues by severity
                for issue in issues:
                    severity = issue.get("severity", "minor")
                    if severity == "critical":
                        document.metadata["qa_results"]["error_counts"]["critical"] += 1
                        page.attributes["qa_results"]["error_counts"]["critical"] += 1
                    elif severity == "major":
                        document.metadata["qa_results"]["error_counts"]["major"] += 1
                        page.attributes["qa_results"]["error_counts"]["major"] += 1
                    else:
                        document.metadata["qa_results"]["error_counts"]["minor"] += 1
                        page.attributes["qa_results"]["error_counts"]["minor"] += 1
                
                # Add to page issues
                page.attributes["qa_results"]["issues"].append({
                    "table_id": table.id,
                    "issues": issues
                })
            
            # Update table
            page.tables[j] = table
    
    def _load_validation_resources(self):
        """Load dictionaries and validation resources"""
        # Common English words (would use a proper dictionary in production)
        self.word_dict = set([
            "the", "be", "to", "of", "and", "a", "in", "that", "have", "i", "it", "for", "not", 
            "on", "with", "he", "as", "you", "do", "at", "this", "but", "his", "by", "from", "they",
            "we", "say", "her", "she", "or", "an", "will", "my", "one", "all", "would", "there",
            "their", "what", "so", "up", "out", "if", "about", "who", "get", "which", "go", "me"
        ])
        
        # Date patterns for different formats
        self.date_patterns = [
            # US format
            r'(?:\b|\D)(?:(0?[1-9]|1[0-2])[\-/\.](0?[1-9]|[12]\d|3[01])[\-/\.]((?:19|20)\d{2}))(?:\b|\D)',  # MM/DD/YYYY
            # European format
            r'(?:\b|\D)(?:(0?[1-9]|[12]\d|3[01])[\-/\.](0?[1-9]|1[0-2])[\-/\.]((?:19|20)\d{2}))(?:\b|\D)',  # DD/MM/YYYY
            # ISO format
            r'(?:\b|\D)(?:((?:19|20)\d{2})[\-/\.](0?[1-9]|1[0-2])[\-/\.](0?[1-9]|[12]\d|3[01]))(?:\b|\D)',  # YYYY/MM/DD
            # Written format
            r'(?:\b)((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[\s,]+(?:[0-9]{1,2})(?:st|nd|rd|th)?[\s,]+(?:19|20)[0-9]{2})(?:\b)',  # Month DD, YYYY
            r'(?:\b)((?:[0-9]{1,2})(?:st|nd|rd|th)?[\s,]+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[\s,]+(?:19|20)[0-9]{2})(?:\b)',  # DD Month, YYYY
            r'(?:\b)((?:January|February|March|April|May|June|July|August|September|October|November|December)[\s,]+(?:[0-9]{1,2})(?:st|nd|rd|th)?[\s,]+(?:19|20)[0-9]{2})(?:\b)',  # Month DD, YYYY full
            r'(?:\b)((?:[0-9]{1,2})(?:st|nd|rd|th)?[\s,]+(?:January|February|March|April|May|June|July|August|September|October|November|December)[\s,]+(?:19|20)[0-9]{2})(?:\b)'   # DD Month, YYYY full
        ]
        
        # Amount patterns for different formats and currencies
        self.amount_patterns = [
            # Dollar formats
            r'\$\s*(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d{2})?)(?!\d)',  # $1,234.56 or $1234.56
            r'(?:USD|US\$)\s*(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d{2})?)(?!\d)',  # USD 1,234.56
            # Euro formats
            r'€\s*(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:,\d{2})?)(?!\d)',  # €1.234,56
            r'(?:EUR)\s*(?:(?:\d{1,3}(?:\.\d{3})+|\d+)(?:,\d{2})?)(?!\d)',  # EUR 1.234,56
            # Pound formats
            r'£\s*(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d{2})?)(?!\d)',  # £1,234.56
            r'(?:GBP)\s*(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d{2})?)(?!\d)',  # GBP 1,234.56
            # Generic formats
            r'(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d{2})?)\s*(?:dollars|USD|euros|EUR|pounds|GBP)(?!\w)',  # 1,234.56 dollars
            r'(?:(?:\d{1,3}(?:\.\d{3})+|\d+)(?:,\d{2})?)\s*(?:euros|EUR)(?!\w)'  # 1.234,56 euros
        ]
        
        # Document structure patterns
        self.document_structure_patterns = {
            "invoice": {
                "required_fields": [
                    r'(?:\b)invoice(?:\s+number|#|no\.?)?(?:\b)',
                    r'(?:\b)date(?:\b)',
                    r'(?:\b)(?:total|amount|sum)(?:\b)',
                    r'(?:\b)(?:bill\s+to|sold\s+to|customer)(?:\b)'
                ],
                "optional_fields": [
                    r'(?:\b)tax(?:\b)',
                    r'(?:\b)(?:payment\s+terms|due\s+date)(?:\b)',
                    r'(?:\b)(?:p\.?o\.?|purchase\s+order)(?:\b)'
                ]
            },
            "receipt": {
                "required_fields": [
                    r'(?:\b)(?:receipt|sale)(?:\b)',
                    r'(?:\b)date(?:\b)',
                    r'(?:\b)(?:total|amount|sum)(?:\b)'
                ],
                "optional_fields": [
                    r'(?:\b)tax(?:\b)',
                    r'(?:\b)(?:cashier|store|branch)(?:\b)',
                    r'(?:\b)(?:payment|paid\s+by)(?:\b)'
                ]
            },
            "contract": {
                "required_fields": [
                    r'(?:\b)(?:agreement|contract)(?:\b)',
                    r'(?:\b)(?:parties|between)(?:\b)',
                    r'(?:\b)(?:dated|effective\s+date)(?:\b)',
                    r'(?:\b)signature(?:\b)'
                ],
                "optional_fields": [
                    r'(?:\b)(?:term|period|duration)(?:\b)',
                    r'(?:\b)(?:governing\s+law|jurisdiction)(?:\b)',
                    r'(?:\b)(?:termination|cancel)(?:\b)'
                ]
            }
        }
        
        # Critical field patterns
        self.critical_field_patterns = {
            "invoice_number": [
                r'(?:invoice|inv|bill)(?:\s+|-|#|no\.?|number|num)?\s*[:=]?\s*([A-Z0-9][-A-Z0-9]*(?:[A-Z0-9]|\b))',
                r'(?:\b)(?:invoice|inv)(?:\s+|-|#|no\.?|number|num)?\s*([A-Z0-9][-A-Z0-9]*(?:[A-Z0-9]|\b))'
            ],
            "date": self.date_patterns,
            "total_amount": [
                r'(?:total|amount|sum|balance|due)(?:\s+|-|:)(?:due|payable|amount)?\s*[:=]?\s*([$€£]?\s*\d+(?:,\d{3})*(?:\.\d{2})?)',
                r'(?:total|amount|sum|balance|due)(?:\s+|-|:)(?:due|payable|amount)?\s*[:=]?\s*(\d+(?:,\d{3})*(?:\.\d{2})?\s*[$€£]?)'
            ],
            "tax_amount": [
                r'(?:tax|vat|gst|hst|sales\s+tax)(?:\s+|-|:)(?:amount)?\s*[:=]?\s*([$€£]?\s*\d+(?:,\d{3})*(?:\.\d{2})?)',
                r'(?:tax|vat|gst|hst|sales\s+tax)(?:\s+|-|:)(?:amount)?\s*[:=]?\s*(\d+(?:,\d{3})*(?:\.\d{2})?\s*[$€£]?)'
            ],
            "account_number": [
                r'(?:account|acct)(?:\s+|-|#|no\.?|number|num)?\s*[:=]?\s*([A-Z0-9][-A-Z0-9]*(?:[A-Z0-9]|\b))'
            ],
            "ssn": [
                r'(?:ssn|social\s+security|social\s+security\s+number)(?:\s+|-|#|no\.?|number|num)?\s*[:=]?\s*(\d{3}[-\s]?\d{2}[-\s]?\d{4})'
            ]
        }
        
        # Document type specific validations
        self.document_type_validators = {
            "invoice": self._validate_invoice,
            "receipt": self._validate_receipt,
            "contract": self._validate_contract,
            "report": self._validate_report,
            "form": self._validate_form
        }
        
        # Numeric patterns for anomaly detection
        self.numeric_patterns = {
            "percentage": r'(\d{1,3}(?:\.\d{1,2})?\s*%)',
            "large_number": r'(\d{7,}(?:\.\d+)?)',
            "phone_number": r'((?:\+\d{1,3}[-\s]?)?\(?\d{3}\)?[-\s]?\d{3}[-\s]?\d{4})'
        }
    
    def _extract_document_context(self, document: Document) -> Dict[str, Any]:
        """Extract document context for semantic validation"""
        # Extract all text
        all_text = " ".join(region.text for page in document.pages 
                          for region in page.regions
                          if hasattr(region, "text") and region.text)
        
        # Count word frequencies
        words = re.findall(r'\b([a-zA-Z]{2,})\b', all_text.lower())
        word_counts = Counter(words)
        
        # Extract numbers
        numbers = re.findall(r'\b(\d+(?:\.\d+)?)\b', all_text)
        number_values = [float(n) for n in numbers if self._is_valid_float(n)]
        
        # Extract potential amounts
        amounts = []
        for pattern in self.amount_patterns:
            matches = re.findall(pattern, all_text)
            amounts.extend(matches)
        
        # Clean amounts and convert to float
        amount_values = []
        for amount in amounts:
            # Remove currency symbols and commas
            clean_amount = re.sub(r'[$€£¥,\s]', '', amount)
            # Replace European decimal comma with dot
            clean_amount = clean_amount.replace(',', '.')
            try:
                amount_values.append(float(clean_amount))
            except ValueError:
                continue
        
        # Extract dates
        dates = []
        for pattern in self.date_patterns:
            matches = re.findall(pattern, all_text)
            if matches:
                dates.extend([m if isinstance(m, str) else m[0] for m in matches])
        
        return {
            "common_words": {word for word, count in word_counts.most_common(100)},
            "word_counts": word_counts,
            "number_values": number_values,
            "amount_values": amount_values,
            "dates": dates,
            "statistics": {
                "avg_amount": np.mean(amount_values) if amount_values else 0,
                "max_amount": max(amount_values) if amount_values else 0,
                "min_amount": min(amount_values) if amount_values else 0,
                "avg_number": np.mean(number_values) if number_values else 0
            }
        }

    def _is_valid_word(self, word: str, document_context: Dict[str, Any]) -> bool:
        """
        Check if a word is valid using context-aware spell checking
        
        Args:
            word: The word to check
            document_context: Document context for domain-specific terms
        
        Returns:
            bool: True if the word is valid, False otherwise
        """
        # Remove punctuation and convert to lowercase
        clean_word = ''.join(c for c in word.lower() if c.isalnum())
        
        # Skip short words, numbers, and codes
        if len(clean_word) <= 2 or clean_word.isdigit() or any(c.isdigit() for c in clean_word):
            return True
        
        # Check if it's a common word in the document (domain-specific term)
        if clean_word in document_context["common_words"]:
            return True
            
        # Check if it's in our dictionary
        if clean_word in self.word_dict:
            return True
        
        # Check if it's a compound word
        if len(clean_word) > 6:  # Only check longer words for compounds
            for i in range(2, len(clean_word) - 1):
                if (clean_word[:i] in self.word_dict or 
                    clean_word[:i] in document_context["common_words"]) and \
                   (clean_word[i:] in self.word_dict or 
                    clean_word[i:] in document_context["common_words"]):
                    return True
        
        # Check for plurals and common suffixes
        if len(clean_word) > 3 and clean_word.endswith('s'):
            if clean_word[:-1] in self.word_dict or clean_word[:-1] in document_context["common_words"]:
                return True
                
        if len(clean_word) > 4 and clean_word.endswith('ed'):
            if clean_word[:-2] in self.word_dict or clean_word[:-2] in document_context["common_words"]:
                return True
                
        if len(clean_word) > 4 and clean_word.endswith('ing'):
            if clean_word[:-3] in self.word_dict or clean_word[:-3] in document_context["common_words"]:
                return True
        
        # Not a valid word
        return False
    
    def _get_spelling_suggestions(self, word: str, document_context: Dict[str, Any]) -> List[str]:
        """
        Get spelling suggestions for a potentially misspelled word
        
        Args:
            word: The word to check
            document_context: Document context for domain-specific suggestions
            
        Returns:
            List of spelling suggestions
        """
        # Simple implementation using difflib
        # In production, would use a proper spelling correction library
        
        # Clean the word
        clean_word = ''.join(c for c in word.lower() if c.isalpha())
        if len(clean_word) <= 2:
            return []
            
        # Combine dictionary with common document words
        candidate_words = list(self.word_dict) + list(document_context["common_words"])
        
        # Get close matches
        suggestions = difflib.get_close_matches(clean_word, candidate_words, n=3, cutoff=0.7)
        
        # Add potential corrections based on common typos
        if not suggestions:
            # Check for doubled letters
            for i in range(len(clean_word) - 1):
                if clean_word[i] == clean_word[i+1]:
                    candidate = clean_word[:i] + clean_word[i+1:]
                    if candidate in self.word_dict or candidate in document_context["common_words"]:
                        suggestions.append(candidate)
            
            # Check for missing letters
            for i in range(len(clean_word) + 1):
                for char in string.ascii_lowercase:
                    candidate = clean_word[:i] + char + clean_word[i:]
                    if candidate in self.word_dict or candidate in document_context["common_words"]:
                        suggestions.append(candidate)
        
        return suggestions
    
    def _extract_and_validate_dates(self, text: str) -> Tuple[List[str], List[Dict[str, Any]]]:
        """
        Extract and validate dates from text
        
        Args:
            text: Text to extract dates from
            
        Returns:
            Tuple of (list of date strings, list of validation issues)
        """
        dates = []
        issues = []
        
        for pattern in self.date_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                date_str = match.group(0)
                dates.append(date_str)
                
                # Validate date
                is_valid, reason = self._is_valid_date(date_str)
                if not is_valid:
                    issues.append({
                        "type": "invalid_date",
                        "severity": "major",
                        "value": date_str,
                        "reason": reason,
                        "position": match.span()
                    })
        
        return dates, issues
    
    def _is_valid_date(self, date_str: str) -> Tuple[bool, Optional[str]]:
        """
        Check if a date string is valid
        
        Args:
            date_str: Date string to validate
            
        Returns:
            Tuple of (is valid, reason if invalid)
        """
        # Remove any surrounding whitespace or punctuation
        date_str = date_str.strip()
        
        # Extract components based on common formats
        components = re.split(r'[-/\s,\.]+', date_str)
        components = [c for c in components if c and c.strip()]
        
        # Filter out ordinal indicators
        components = [re.sub(r'(?:st|nd|rd|th)$', '', c) for c in components]
        
        # Handle month names
        month_names = {
            'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
            'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12,
            'january': 1, 'february': 2, 'march': 3, 'april': 4, 'june': 6,
            'july': 7, 'august': 8, 'september': 9, 'october': 10, 'november': 11, 'december': 12
        }
        
        month = None
        day = None
        year = None
        
        # Extract numeric components and month names
        for component in components:
            # Skip empty components
            if not component:
                continue
                
            # Check if it's a month name
            month_match = re.match(r'([a-zA-Z]+)', component)
            if month_match:
                month_name = month_match.group(1).lower()
                if month_name in month_names:
                    month = month_names[month_name]
                    continue
            
            # Check if it's a number
            if component.isdigit():
                num = int(component)
                
                # Year (4 digits, or 2 digits > 50 assume 19xx, else 20xx)
                if len(component) == 4:
                    year = num
                    continue
                elif len(component) == 2:
                    if num > 50:
                        year = 1900 + num
                    else:
                        year = 2000 + num
                    continue
                
                # Day (1-31)
                if 1 <= num <= 31 and day is None:
                    day = num
                    continue
                
                # Month (1-12)
                if 1 <= num <= 12 and month is None:
                    month = num
                    continue
        
        # If we have day/month/year, validate the date
        if day and month and year:
            # Validate month
            if month < 1 or month > 12:
                return False, "Invalid month"
                
            # Validate day based on month
            days_in_month = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
            
            # Adjust for leap year
            if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0):
                days_in_month[2] = 29
                
            if day < 1 or day > days_in_month[month]:
                return False, f"Invalid day for month {month}"
                
            # Validate year
            current_year = datetime.now().year
            if year < 1900 or year > current_year + 5:  # Allow dates up to 5 years in the future
                return False, "Unlikely year"
                
            return True, None
        
        # Missing components
        missing = []
        if day is None: missing.append("day")
        if month is None: missing.append("month")
        if year is None: missing.append("year")
        
        if missing:
            return False, f"Missing date components: {', '.join(missing)}"
            
        return False, "Could not parse date format"
    
    def _extract_and_validate_amounts(self, text: str, document_context: Dict[str, Any]) -> Tuple[List[str], List[Dict[str, Any]]]:
        """
        Extract and validate monetary amounts from text
        
        Args:
            text: Text to extract amounts from
            document_context: Document context for validating amounts
            
        Returns:
            Tuple of (list of amount strings, list of validation issues)
        """
        amounts = []
        issues = []
        
        for pattern in self.amount_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                amount_str = match.group(0)
                amounts.append(amount_str)
                
                # Validate amount
                is_valid, reason = self._is_valid_amount(amount_str, document_context)
                if not is_valid:
                    issues.append({
                        "type": "invalid_amount",
                        "severity": "major",
                        "value": amount_str,
                        "reason": reason,
                        "position": match.span()
                    })
        
        return amounts, issues
    
    def _is_valid_amount(self, amount_str: str, document_context: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Check if a monetary amount string is valid
        
        Args:
            amount_str: Amount string to validate
            document_context: Document context for validating amounts
            
        Returns:
            Tuple of (is valid, reason if invalid)
        """
        # Extract numeric part from the amount string
        amount_value = None
        
        # Remove currency symbols and whitespace
        numeric_str = re.sub(r'[$€£¥\s]', '', amount_str)
        
        # Handle different decimal separators
        if ',' in numeric_str and '.' in numeric_str:
            # Both commas and periods - determine which is decimal separator
            if numeric_str.rindex(',') > numeric_str.rindex('.'):
                # Comma is decimal separator (e.g., 1.234,56)
                numeric_str = numeric_str.replace('.', '')
                numeric_str = numeric_str.replace(',', '.')
            else:
                # Period is decimal separator (e.g., 1,234.56)
                numeric_str = numeric_str.replace(',', '')
        elif ',' in numeric_str:
            # Only commas - determine if thousand separator or decimal separator
            parts = numeric_str.split(',')
            if len(parts[-1]) == 2:
                # Likely decimal separator
                numeric_str = numeric_str.replace(',', '.')
            else:
                # Likely thousand separator
                numeric_str = numeric_str.replace(',', '')
        
        # Try to convert to float
        try:
            amount_value = float(numeric_str)
        except ValueError:
            return False, "Could not parse amount"
        
        # Check if amount is reasonable for the document
        if document_context["statistics"]["avg_amount"] > 0:
            # Check if the amount is an extreme outlier
            max_amount = document_context["statistics"]["max_amount"]
            avg_amount = document_context["statistics"]["avg_amount"]
            
            # If amount is > 100x the average and > 10x the max, it's suspicious
            if amount_value > avg_amount * 100 and amount_value > max_amount * 10:
                return False, "Amount is an extreme outlier"
        
        # Check for invalid precision
        decimal_part = numeric_str.split('.')[-1] if '.' in numeric_str else ''
        if len(decimal_part) > 2 and not (decimal_part.endswith('0') * (len(decimal_part) - 2)):
            return False, "Unusual decimal precision for currency"
        
        # Amount seems valid
        return True, None
        
    def _validate_table_structure(self, table: Table) -> Tuple[bool, List[Dict[str, Any]]]:
        """
        Validate table structure with detailed analysis
        
        Args:
            table: Table to validate
            
        Returns:
            Tuple of (is valid, list of issues)
        """
        issues = []
        
        # Check if there are enough rows and columns
        if not hasattr(table, 'rows') or not table.rows:
            if hasattr(table, 'cells') and table.cells:
                # We have cells but not rows, use cells
                rows = table.cells
            else:
                issues.append({
                    "type": "empty_table",
                    "severity": "major",
                    "details": "Table has no rows or cells"
                })
                return False, issues
        else:
            rows = table.rows
        
        # Check if there are enough rows
        if len(rows) < 2:
            issues.append({
                "type": "insufficient_rows",
                "severity": "major",
                "details": f"Table has only {len(rows)} rows"
            })
        
        # Check if all rows have the same number of cells
        row_lengths = [len(row) for row in rows]
        if len(set(row_lengths)) > 1:
            issues.append({
                "type": "irregular_structure",
                "severity": "major",
                "details": f"Inconsistent row lengths: {row_lengths}"
            })
        
        # Check for empty cells in critical positions
        empty_cells = 0
        total_cells = 0
        
        for i, row in enumerate(rows):
            for j, cell in enumerate(row):
                total_cells += 1
                if not cell or not hasattr(cell, 'text') or not cell.text.strip():
                    empty_cells += 1
                    if i == 0 or j == 0:  # Header row or first column
                        issues.append({
                            "type": "empty_header",
                            "severity": "major",
                            "details": f"Empty {'header' if i == 0 else 'key cell'} at row {i+1}, column {j+1}"
                        })
        
        # Check for too many empty cells
        if total_cells > 0:
            empty_ratio = empty_cells / total_cells
            if empty_ratio > 0.33:  # More than 1/3 cells empty
                issues.append({
                    "type": "sparse_table",
                    "severity": "minor",
                    "details": f"Table is {empty_ratio:.1%} empty cells"
                })
        
        return len(issues) == 0, issues
        
    def _validate_table_data(self, table: Table) -> List[Dict[str, Any]]:
        """
        Validate table data for consistency and correctness
        
        Args:
            table: Table to validate
            
        Returns:
            List of validation issues
        """
        issues = []
        
        # Skip tables without proper structure
        if not hasattr(table, 'rows') or not table.rows:
            if not hasattr(table, 'cells') or not table.cells:
                return issues
            rows = table.cells
        else:
            rows = table.rows
            
        if len(rows) < 2 or len(rows[0]) < 1:
            return issues
        
        # Identify potential numeric columns
        numeric_columns = []
        amount_columns = []
        
        # Check header row for amount/total indicators
        header_row = rows[0]
        for j, cell in enumerate(header_row):
            if not cell or not hasattr(cell, 'text'):
                continue
                
            cell_text = cell.text.strip().lower()
            
            # Check for amount-related headers
            if any(term in cell_text for term in ['amount', 'price', 'total', 'cost', 'sum', 'value', 'fee']):
                amount_columns.append(j)
                numeric_columns.append(j)
            # Check for quantity-related headers
            elif any(term in cell_text for term in ['quantity', 'qty', 'count', 'number', 'units']):
                numeric_columns.append(j)
        
        # Analyze data columns
        for j in range(len(rows[0])):
            # Skip columns already identified
            if j in numeric_columns:
                continue
                
            # Check data in column to see if it's numeric
            numeric_count = 0
            total_count = 0
            
            for i in range(1, len(rows)):
                if j >= len(rows[i]):
                    continue
                    
                cell = rows[i][j]
                if not cell or not hasattr(cell, 'text'):
                    continue
                    
                cell_text = cell.text.strip()
                if not cell_text:
                    continue
                    
                total_count += 1
                
                # Check if cell content is numeric
                numeric_match = re.search(r'^\s*[$€£¥]?\s*\d+(?:[.,]\d+)?\s*%?\s*$', cell_text)
                if numeric_match:
                    numeric_count += 1
            
            # If most non-empty cells are numeric, consider it a numeric column
            if total_count > 0 and numeric_count / total_count > 0.8:
                numeric_columns.append(j)
                
                # If it has currency symbols, consider it an amount column
                currency_count = 0
                for i in range(1, len(rows)):
                    if j >= len(rows[i]):
                        continue
                        
                    cell = rows[i][j]
                    if not cell or not hasattr(cell, 'text'):
                        continue
                        
                    cell_text = cell.text.strip()
                    if not cell_text:
                        continue
                        
                    if re.search(r'[$€£¥]', cell_text):
                        currency_count += 1
                
                if currency_count > 0:
                    amount_columns.append(j)
        
        # Validate numeric columns for consistency
        for j in numeric_columns:
            values = []
            
            for i in range(1, len(rows)):
                if j >= len(rows[i]):
                    continue
                    
                cell = rows[i][j]
                if not cell or not hasattr(cell, 'text'):
                    continue
                    
                cell_text = cell.text.strip()
                if not cell_text:
                    continue
                
                # Extract numeric value
                numeric_str = re.sub(r'[$€£¥%\s,]', '', cell_text)
                try:
                    value = float(numeric_str)
                    values.append(value)
                except ValueError:
                    # Not a valid number
                    if j in amount_columns:
                        issues.append({
                            "type": "invalid_amount_format",
                            "severity": "major",
                            "details": f"Non-numeric value '{cell_text}' in amount column {j+1}, row {i+1}"
                        })
                    continue
            
            # Skip columns with too few values
            if len(values) < 3:
                continue
                
            # Check for outliers using standard deviation
            mean = np.mean(values)
            std = np.std(values)
            
            if std > 0:
                for i, value in enumerate(values, 1):
                    # Check if value is an outlier (> 3 standard deviations from mean)
                    if abs(value - mean) > 3 * std:
                        issues.append({
                            "type": "numeric_outlier",
                            "severity": "minor",
                            "details": f"Outlier value {value} in column {j+1}, row {i+1}"
                        })
        
        # Validate amount columns for mathematical consistency
        if len(amount_columns) >= 2 and any("total" in cell.text.lower() 
                                          for cell in header_row if hasattr(cell, 'text')):
            # Try to identify total column and component columns
            total_col = None
            component_cols = []
            
            for j in amount_columns:
                if j >= len(header_row):
                    continue
                    
                cell = header_row[j]
                if not hasattr(cell, 'text'):
                    continue
                    
                if "total" in cell.text.lower():
                    total_col = j
                else:
                    component_cols.append(j)
            
            # If we found a total column and component columns, check if totals match
            if total_col is not None and component_cols:
                for i in range(1, len(rows)):
                    if total_col >= len(rows[i]):
                        continue
                        
                    # Extract total value
                    total_cell = rows[i][total_col]
                    if not total_cell or not hasattr(total_cell, 'text'):
                        continue
                        
                    total_text = total_cell.text.strip()
                    if not total_text:
                        continue
                        
                    numeric_str = re.sub(r'[$€£¥%\s,]', '', total_text)
                    try:
                        total_value = float(numeric_str)
                    except ValueError:
                        continue
                    
                    # Sum component values
                    component_sum = 0.0
                    component_count = 0
                    
                    for j in component_cols:
                        if j >= len(rows[i]):
                            continue
                            
                        cell = rows[i][j]
                        if not cell or not hasattr(cell, 'text'):
                            continue
                            
                        cell_text = cell.text.strip()
                        if not cell_text:
                            continue
                            
                        numeric_str = re.sub(r'[$€£¥%\s,]', '', cell_text)
                        try:
                            value = float(numeric_str)
                            component_sum += value
                            component_count += 1
                        except ValueError:
                            continue
                    
                    # Check if sum matches total
                    if component_count > 0:
                        tolerance = 0.01 * abs(total_value)  # 1% tolerance
                        if abs(total_value - component_sum) > tolerance:
                            issues.append({
                                "type": "inconsistent_total",
                                "severity": "critical",
                                "details": f"Row {i+1}: Total ({total_value}) does not match sum of components ({component_sum})"
                            })
        
        return issues
        
    def _is_critical_field_token(self, token, document_context: Dict[str, Any]) -> bool:
        """Check if token might be part of a critical field"""
        if not hasattr(token, 'text') or not token.text:
            return False
            
        # Check for currency symbols or common critical field indicators
        if any(symbol in token.text for symbol in ['$', '€', '£', '¥']):
            token.attributes["critical_field_type"] = "amount"
            return True
            
        # Check for date formats
        date_pattern = r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}'
        if re.search(date_pattern, token.text):
            token.attributes["critical_field_type"] = "date"
            return True
            
        # Check for invoice number formats
        inv_pattern = r'(?:INV|INVOICE|inv)[- ]?#?\d+'
        if re.search(inv_pattern, token.text, re.IGNORECASE):
            token.attributes["critical_field_type"] = "invoice_number"
            return True
            
        # Check for "total" with nearby amount
        if token.text.lower() in ["total", "amount", "balance", "due"]:
            token.attributes["critical_field_type"] = "total"
            return True
            
        return False
        
    def _identify_critical_fields(self, text: str, document_type: str) -> List[Dict[str, Any]]:
        """Identify critical fields in text based on document type"""
        critical_fields = []
        
        # Skip empty text
        if not text:
            return critical_fields
            
        # Check for each critical field pattern
        for field_type, patterns in self.critical_field_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    # The full match includes the label, group(1) has just the value
                    if match.groups():
                        value = match.group(1)
                    else:
                        value = match.group(0)
                    
                    critical_fields.append({
                        "field_type": field_type,
                        "value": value,
                        "position": match.span()
                    })
        
        return critical_fields
        
    def _validate_critical_field(self, field_type: str, value: str, document_context: Dict[str, Any]) -> bool:
        """Validate a critical field value"""
        if field_type == "date":
            is_valid, _ = self._is_valid_date(value)
            return is_valid
            
        elif field_type in ["total_amount", "tax_amount"]:
            is_valid, _ = self._is_valid_amount(value, document_context)
            return is_valid
            
        elif field_type == "invoice_number":
            # Basic validation of invoice number format
            clean_value = value.strip()
            if len(clean_value) < 3:  # Too short to be a valid invoice number
                return False
                
            # Should have at least one digit
            if not any(c.isdigit() for c in clean_value):
                return False
                
            return True
            
        elif field_type == "ssn":
            # Validate SSN format (XXX-XX-XXXX)
            clean_value = re.sub(r'[^0-9]', '', value)
            return len(clean_value) == 9 and not clean_value == '000000000'
            
        # Default to valid if no specific validation
        return True
        
    def _detect_numeric_anomalies(self, text: str, document_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect numeric anomalies in text"""
        issues = []
        
        # Extract all numbers
        number_matches = re.finditer(r'\b(\d+(?:\.\d+)?)\b', text)
        for match in number_matches:
            number_str = match.group(1)
            
            try:
                number = float(number_str)
                
                # Check for extremely large numbers that aren't likely amounts
                if number > 1000000:  # 1 million
                    # Make sure it's not a currency amount
                    if not re.search(r'[$€£¥]\s*' + re.escape(number_str), text):
                        # Check nearby text for indications this is intended to be large
                        context_text = text[max(0, match.start() - 20):min(len(text), match.end() + 20)]
                        large_indicators = ["million", "billion", "trillion", "population", "total", "count"]
                        
                        if not any(indicator in context_text.lower() for indicator in large_indicators):
                            issues.append({
                                "type": "unusual_large_number",
                                "severity": "minor",
                                "value": number_str,
                                "position": match.span()
                            })
                
                # Check for numbers with unusual precision
                if '.' in number_str:
                    integer_part, decimal_part = number_str.split('.')
                    if len(decimal_part) > 3 and not decimal_part.endswith('0' * (len(decimal_part) - 3)):
                        issues.append({
                            "type": "unusual_precision",
                            "severity": "minor",
                            "value": number_str,
                            "position": match.span()
                        })
            except ValueError:
                pass  # Not a valid number
        
        return issues
        
    def _validate_document_structure(self, document: Document, document_type: str, document_context: Dict[str, Any]) -> None:
        """Validate document structure based on document type"""
        if document_type == "unknown":
            # Try to infer document type
            document_type = self._infer_document_type(document)
            document.metadata["inferred_document_type"] = document_type
        
        # If we have a validator for this document type, use it
        if document_type in self.document_type_validators:
            issues = self.document_type_validators[document_type](document, document_context)
            if issues:
                document.metadata["qa_results"]["document_structure_issues"] = issues
                document.metadata["qa_results"]["error_counts"]["structure_issues"] += len(issues)
                
                # Categorize issues by severity
                for issue in issues:
                    severity = issue.get("severity", "minor")
                    if severity == "critical":
                        document.metadata["qa_results"]["error_counts"]["critical"] += 1
                    elif severity == "major":
                        document.metadata["qa_results"]["error_counts"]["major"] += 1
                    else:
                        document.metadata["qa_results"]["error_counts"]["minor"] += 1
        
    def _infer_document_type(self, document: Document) -> str:
        """Infer document type based on content"""
        # Extract all text
        all_text = " ".join(region.text for page in document.pages 
                          for region in page.regions
                          if hasattr(region, "text") and region.text)
        
        all_text_lower = all_text.lower()
        
        # Count occurrences of type-specific keywords
        type_scores = {
            "invoice": 0,
            "receipt": 0,
            "contract": 0,
            "report": 0,
            "form": 0
        }
        
        # Invoice indicators
        invoice_keywords = ["invoice", "bill to", "payment terms", "due date", "subtotal", "tax"]
        for keyword in invoice_keywords:
            if keyword in all_text_lower:
                type_scores["invoice"] += 1
        
        # Receipt indicators
        receipt_keywords = ["receipt", "transaction", "cashier", "store", "change", "cash", "card"]
        for keyword in receipt_keywords:
            if keyword in all_text_lower:
                type_scores["receipt"] += 1
        
        # Contract indicators
        contract_keywords = ["agreement", "contract", "parties", "terms", "conditions", "signed", "signature"]
        for keyword in contract_keywords:
            if keyword in all_text_lower:
                type_scores["contract"] += 1
        
        # Report indicators
        report_keywords = ["report", "analysis", "summary", "findings", "conclusion", "results", "data"]
        for keyword in report_keywords:
            if keyword in all_text_lower:
                type_scores["report"] += 1
        
        # Form indicators
        form_keywords = ["form", "please fill", "please complete", "applicant", "application", "submit"]
        for keyword in form_keywords:
            if keyword in all_text_lower:
                type_scores["form"] += 1
        
        # Return the document type with the highest score
        max_score = max(type_scores.values())
        if max_score > 0:
            for doc_type, score in type_scores.items():
                if score == max_score:
                    return doc_type
        
        return "unknown"
        
    def _validate_invoice(self, document: Document, document_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Validate invoice structure"""
        issues = []
        
        # Extract all text
        all_text = " ".join(region.text for page in document.pages 
                          for region in page.regions
                          if hasattr(region, "text") and region.text)
        
        # Check for required invoice fields
        required_fields = self.document_structure_patterns["invoice"]["required_fields"]
        missing_fields = []
        
        for field_pattern in required_fields:
            if not re.search(field_pattern, all_text, re.IGNORECASE):
                # Extract field name from pattern for user-friendly message
                field_name = re.sub(r'[\\(\\)\\?\\|]', '', field_pattern)
                field_name = re.sub(r'\\s+', ' ', field_name)
                field_name = re.sub(r'\\b', '', field_name)
                
                missing_fields.append(field_name.strip())
        
        if missing_fields:
            issues.append({
                "type": "missing_required_fields",
                "severity": "major",
                "details": f"Invoice missing required fields: {', '.join(missing_fields)}"
            })
        
        # Check for invoice number
        invoice_number_found = False
        for pattern in self.critical_field_patterns["invoice_number"]:
            if re.search(pattern, all_text, re.IGNORECASE):
                invoice_number_found = True
                break
                
        if not invoice_number_found:
            issues.append({
                "type": "missing_invoice_number",
                "severity": "critical",
                "details": "Invoice number not found"
            })
        
        # Check for amount consistency
        total_amounts = []
        subtotal_amounts = []
        tax_amounts = []
        
        # Extract total amounts
        total_patterns = [
            r'(?:total|amount due|balance due|grand total)(?:\s*:)?\s*([$€£]\s*\d+(?:,\d{3})*(?:\.\d{2})?)',
            r'(?:total|amount due|balance due|grand total)(?:\s*:)?\s*(\d+(?:,\d{3})*(?:\.\d{2})?\s*[$€£])'
        ]
        
        for pattern in total_patterns:
            matches = re.findall(pattern, all_text, re.IGNORECASE)
            for match in matches:
                # Clean amount and convert to float
                clean_amount = re.sub(r'[$€£,\s]', '', match)
                try:
                    total_amounts.append(float(clean_amount))
                except ValueError:
                    continue
        
        # Extract subtotal amounts
        subtotal_patterns = [
            r'(?:subtotal|sub-total|sub total)(?:\s*:)?\s*([$€£]\s*\d+(?:,\d{3})*(?:\.\d{2})?)',
            r'(?:subtotal|sub-total|sub total)(?:\s*:)?\s*(\d+(?:,\d{3})*(?:\.\d{2})?\s*[$€£])'
        ]
        
        for pattern in subtotal_patterns:
            matches = re.findall(pattern, all_text, re.IGNORECASE)
            for match in matches:
                # Clean amount and convert to float
                clean_amount = re.sub(r'[$€£,\s]', '', match)
                try:
                    subtotal_amounts.append(float(clean_amount))
                except ValueError:
                    continue
        
        # Extract tax amounts
        tax_patterns = [
            r'(?:tax|vat|gst|hst|sales tax)(?:\s*:)?\s*([$€£]\s*\d+(?:,\d{3})*(?:\.\d{2})?)',
            r'(?:tax|vat|gst|hst|sales tax)(?:\s*:)?\s*(\d+(?:,\d{3})*(?:\.\d{2})?\s*[$€£])'
        ]
        
        for pattern in tax_patterns:
            matches = re.findall(pattern, all_text, re.IGNORECASE)
            for match in matches:
                # Clean amount and convert to float
                clean_amount = re.sub(r'[$€£,\s]', '', match)
                try:
                    tax_amounts.append(float(clean_amount))
                except ValueError:
                    continue
        
        # Check for amount consistency
        if total_amounts and subtotal_amounts and tax_amounts:
            total_val = max(total_amounts)  # Use highest total if multiple found
            subtotal_val = max(subtotal_amounts)
            tax_val = max(tax_amounts)
            
            # Check if total = subtotal + tax
            expected_total = subtotal_val + tax_val
            tolerance = 0.01 * total_val  # 1% tolerance
            
            if abs(total_val - expected_total) > tolerance:
                issues.append({
                    "type": "inconsistent_amounts",
                    "severity": "critical",
                    "details": f"Total amount ({total_val}) does not match subtotal ({subtotal_val}) + tax ({tax_val})"
                })
        
        return issues
    
    def _validate_receipt(self, document: Document, document_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Validate receipt structure"""
        # Implementation would check for receipt-specific requirements
        # Similar to invoice validation but with receipt-specific patterns
        return []
    
    def _validate_contract(self, document: Document, document_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Validate contract structure"""
        # Implementation would check for contract-specific requirements
        # E.g., signatures, dates, key sections
        return []
    
    def _validate_report(self, document: Document, document_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Validate report structure"""
        # Implementation would check for report-specific requirements
        # E.g., executive summary, conclusion sections
        return []
    
    def _validate_form(self, document: Document, document_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Validate form structure"""
        # Implementation would check for form-specific requirements
        # E.g., input fields, checkboxes, signatures
        return []
    
    def _detect_context_anomalies(self, document: Document, document_context: Dict[str, Any]) -> None:
        """Detect contextual anomalies in document content"""
        issues = []
        
        # Analyze each page and region for contextual consistency
        for i, page in enumerate(document.pages):
            page_text = " ".join(region.text for region in page.regions
                              if hasattr(region, "text") and region.text)
            
            if not page_text:
                continue
                
            # Detect language inconsistencies
            if self.validate_language:
                lang_issues = self._detect_language_inconsistencies(page_text, document_context)
                if lang_issues:
                    issues.extend(lang_issues)
                    
                    for issue in lang_issues:
                        issue["page"] = page.page_num
                        
                        severity = issue.get("severity", "minor")
                        if severity == "critical":
                            document.metadata["qa_results"]["error_counts"]["critical"] += 1
                        elif severity == "major":
                            document.metadata["qa_results"]["error_counts"]["major"] += 1
                        else:
                            document.metadata["qa_results"]["error_counts"]["minor"] += 1
                            
                    document.metadata["qa_results"]["error_counts"]["context_anomalies"] += len(lang_issues)
        
        # Add context anomalies to document metadata
        if issues:
            if "context_anomalies" not in document.metadata["qa_results"]:
                document.metadata["qa_results"]["context_anomalies"] = []
                
            document.metadata["qa_results"]["context_anomalies"].extend(issues)
    
    def _detect_language_inconsistencies(self, text: str, document_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect language inconsistencies in text"""
        # This would ideally use language detection libraries
        # For brevity, just implement basic checks
        issues = []
        
        # Check for mixed case issues
        sentences = re.split(r'[.!?]+', text)
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence or len(sentence) < 10:
                continue
                
            # Check if sentence starts with lowercase
            first_word = sentence.split()[0] if sentence.split() else ""
            if first_word and len(first_word) > 1 and first_word[0].islower() and not first_word.isupper():
                issues.append({
                    "type": "case_inconsistency",
                    "severity": "minor",
                    "details": f"Sentence may start with lowercase: '{sentence[:20]}...'"
                })
        
        # Check for repeated words
        repeated_words = re.findall(r'\b(\w+)\s+\1\b', text.lower())
        if repeated_words:
            issues.append({
                "type": "repeated_words",
                "severity": "minor",
                "details": f"Text contains repeated words: {', '.join(set(repeated_words))}"
            })
        
        return issues
    
    def _generate_review_areas(self, document: Document) -> List[Dict[str, Any]]:
        """Generate areas that need human review based on detected issues"""
        review_areas = []
        
        # First, collect critical field issues
        for field_issue in document.metadata["qa_results"]["critical_field_issues"]:
            if "page" in field_issue and "bbox" in field_issue:
                review_areas.append({
                    "type": "critical_field",
                    "page": field_issue["page"],
                    "field_type": field_issue["field"],
                    "bbox": field_issue["bbox"],
                    "reason": field_issue.get("reason", "validation_error")
                })
        
        # Then collect other high-priority review areas
        for page in document.pages:
            # Check regions with issues
            for region in page.regions:
                if hasattr(region, "needs_review") and region.needs_review:
                    # Only add regions with high-severity issues
                    has_major_issues = False
                    if hasattr(region, "attributes") and "validation_issues" in region.attributes:
                        for issue in region.attributes["validation_issues"]:
                            if issue.get("severity") in ["critical", "major"]:
                                has_major_issues = True
                                break
                    
                    if has_major_issues:
                        review_areas.append({
                            "type": "region",
                            "page": page.page_num,
                            "region_id": region.id,
                            "bbox": region.bbox.to_dict() if hasattr(region, "bbox") else None,
                            "reason": "validation_issues"
                        })
            
            # Check tables with issues
            for table in page.tables:
                if hasattr(table, "needs_review") and table.needs_review:
                    # Only add tables with high-severity issues
                    has_major_issues = False
                    if hasattr(table, "attributes") and "validation_issues" in table.attributes:
                        for issue in table.attributes["validation_issues"]:
                            if issue.get("severity") in ["critical", "major"]:
                                has_major_issues = True
                                break
                    
                    if has_major_issues:
                        review_areas.append({
                            "type": "table",
                            "page": page.page_num,
                            "table_id": table.id,
                            "bbox": table.bbox.to_dict() if hasattr(table, "bbox") else None,
                            "reason": "validation_issues"
                        })
        
        return review_areas
        
    def _is_valid_float(self, value: str) -> bool:
        """Check if string can be converted to float"""
        try:
            float(value)
            return True
        except ValueError:
            return False
