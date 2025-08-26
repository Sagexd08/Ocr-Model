from typing import Dict, Any, Optional, List, Union
import numpy as np
import re
from spacy import Language
import spacy
from transformers import pipeline
import torch

from ...types import Document, Page, Region, Token
from ...utils.logging import get_logger, log_execution_time

logger = get_logger(__name__)

class TextProcessor:
    """
    Advanced text processing component for improving OCR text quality.
    Performs text cleaning, correction, and normalization.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.language = self.config.get("language", "en")
        self.enable_spelling_correction = self.config.get("spelling_correction", True)
        self.enable_normalization = self.config.get("normalization", True)
        self.enable_entity_extraction = self.config.get("entity_extraction", True)
        self.confidence_threshold = self.config.get("confidence_threshold", 0.5)
        
        # Load language models
        self.nlp = None
        self.spell_corrector = None
        self.ner_pipeline = None
        
        self._load_models()
        
        # Common OCR error patterns
        self.ocr_error_patterns = [
            (r"[lI]", "I"),  # lowercase l to uppercase I when it appears alone
            (r"rn", "m"),    # 'rn' is often misrecognized as 'm'
            (r"c1", "d"),    # 'c1' is often misrecognized as 'd'
            (r"S", "5"),     # 'S' is often misrecognized as '5' in numbers
            (r"O", "0"),     # 'O' is often misrecognized as '0' in numbers
            (r"I", "1"),     # 'I' is often misrecognized as '1' in numbers
            (r"l", "1"),     # 'l' is often misrecognized as '1' in numbers
            (r"a", "o"),     # 'a' is often misrecognized as 'o'
            (r"[.,;]", "")   # Remove punctuation that's often misrecognized
        ]
    
    @log_execution_time
    def process(self, document: Document) -> Document:
        """
        Process and improve OCR text quality.
        
        Args:
            document: Document with extracted text
            
        Returns:
            Document with improved text
        """
        logger.info(f"Processing text for document with {len(document.pages)} pages")
        
        for i, page in enumerate(document.pages):
            logger.debug(f"Processing text for page {page.page_num}")
            
            try:
                # Process each region
                for j, region in enumerate(page.regions):
                    # Skip tables and figures
                    if region.type in ["table", "figure", "image"]:
                        continue
                    
                    # Clean and normalize text
                    processed_text = region.text
                    
                    if self.enable_spelling_correction:
                        processed_text = self._correct_spelling(processed_text)
                    
                    if self.enable_normalization:
                        processed_text = self._normalize_text(processed_text, region.type)
                    
                    # Extract entities if enabled
                    if self.enable_entity_extraction and self.ner_pipeline:
                        entities = self._extract_entities(processed_text)
                        if entities:
                            region.attributes["entities"] = entities
                    
                    # Update the region with processed text
                    region.text = processed_text
                    page.regions[j] = region
                
                # Update the page
                document.pages[i] = page
                
            except Exception as e:
                logger.error(f"Error processing text for page {page.page_num}: {str(e)}")
        
        return document
    
    def _load_models(self):
        """Load required NLP models"""
        try:
            # Load spaCy model for general NLP tasks
            self.nlp = spacy.load(f"{self.language}_core_web_sm")
            logger.info(f"Loaded spaCy model for language: {self.language}")
            
            # Initialize spelling correction
            if self.enable_spelling_correction:
                # Here we'd load a proper spelling correction model
                # For now, we'll just use spaCy
                logger.info("Using spaCy for spelling correction")
            
            # Load NER pipeline if entity extraction is enabled
            if self.enable_entity_extraction:
                try:
                    self.ner_pipeline = pipeline("ner", model=f"dbmdz/bert-large-cased-finetuned-conll03-english")
                    logger.info("Loaded NER pipeline")
                except Exception as e:
                    logger.warning(f"Failed to load NER pipeline: {str(e)}")
                    self.ner_pipeline = None
                
        except Exception as e:
            logger.error(f"Error loading NLP models: {str(e)}")
    
    def _correct_spelling(self, text: str) -> str:
        """Correct spelling errors in text"""
        if not text or not self.nlp:
            return text
        
        # First, apply common OCR error corrections
        for pattern, replacement in self.ocr_error_patterns:
            # Only apply in appropriate contexts
            if pattern in "O0" and re.search(r'\d[' + pattern + r']\d', text):  # Only in number contexts
                text = re.sub(pattern, replacement, text)
            elif pattern in "Il1" and re.search(r'\b[' + pattern + r']\b', text):  # Only for single characters
                text = re.sub(r'\b[' + pattern + r']\b', replacement, text)
        
        # Use spaCy for more complex corrections
        try:
            doc = self.nlp(text)
            corrected_tokens = []
            
            for token in doc:
                if token.is_alpha and len(token.text) > 1:
                    # Here we would use a proper spelling correction model
                    # For now, we just keep the original token
                    corrected_tokens.append(token.text)
                else:
                    corrected_tokens.append(token.text)
            
            return " ".join(corrected_tokens)
        except Exception as e:
            logger.warning(f"Error in spelling correction: {str(e)}")
            return text
    
    def _normalize_text(self, text: str, region_type: str) -> str:
        """Normalize text based on region type"""
        if not text:
            return text
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Apply specific normalization based on region type
        if region_type == "heading":
            # Capitalize headings properly
            text = self._capitalize_heading(text)
        elif region_type == "paragraph":
            # Ensure proper sentence casing and fix periods
            text = self._normalize_paragraph(text)
        elif region_type == "list":
            # Fix list formatting
            text = self._normalize_list(text)
        elif region_type == "caption":
            # Capitalize first letter only
            if text and len(text) > 0:
                text = text[0].upper() + text[1:] if len(text) > 1 else text.upper()
        
        return text
    
    def _capitalize_heading(self, text: str) -> str:
        """Properly capitalize a heading"""
        # Don't capitalize certain words (articles, conjunctions, prepositions)
        dont_capitalize = {"a", "an", "the", "and", "but", "or", "nor", "for", "so", "yet", 
                          "to", "of", "by", "in", "on", "at", "from", "with", "about", "as"}
        
        words = text.split()
        if not words:
            return text
        
        # Always capitalize the first and last word
        result = []
        for i, word in enumerate(words):
            if i == 0 or i == len(words) - 1 or word.lower() not in dont_capitalize:
                result.append(word.capitalize())
            else:
                result.append(word.lower())
        
        return " ".join(result)
    
    def _normalize_paragraph(self, text: str) -> str:
        """Normalize paragraph text"""
        if not text:
            return text
            
        # Fix spacing after periods
        text = re.sub(r'\.(\S)', '. \\1', text)
        
        # Ensure proper sentence casing
        sentences = re.split(r'(?<=[.!?])\s+', text)
        normalized_sentences = []
        
        for sentence in sentences:
            if sentence and len(sentence) > 0:
                sentence = sentence[0].upper() + sentence[1:] if len(sentence) > 1 else sentence.upper()
                normalized_sentences.append(sentence)
        
        return " ".join(normalized_sentences)
    
    def _normalize_list(self, text: str) -> str:
        """Normalize list text"""
        # Split into list items if they're not already separated
        items = re.split(r'(?:\r?\n)|(?:\r)|(?:•|\*|-)(?=\s)', text)
        normalized_items = []
        
        for item in items:
            item = item.strip()
            if item:
                # Remove list markers if present
                item = re.sub(r'^[•\*\-]\s*', '', item)
                # Capitalize first letter
                if len(item) > 0:
                    item = item[0].upper() + item[1:] if len(item) > 1 else item.upper()
                normalized_items.append(f"• {item}")
        
        return "\n".join(normalized_items)
    
    def _extract_entities(self, text: str) -> Dict[str, List[Dict[str, Any]]]:
        """Extract named entities from text"""
        if not text or not self.ner_pipeline:
            return {}
        
        try:
            entities = self.ner_pipeline(text)
            
            # Group entities by type
            entity_dict = {}
            for entity in entities:
                entity_type = entity['entity'].split('-')[-1]  # Get type after BIO prefix
                if entity_type not in entity_dict:
                    entity_dict[entity_type] = []
                
                entity_dict[entity_type].append({
                    'text': entity['word'],
                    'score': entity['score'],
                    'start': entity['start'],
                    'end': entity['end']
                })
            
            # Only keep entities with sufficient confidence
            filtered_entities = {
                entity_type: [e for e in entities if e['score'] >= self.confidence_threshold]
                for entity_type, entities in entity_dict.items()
            }
            
            # Remove empty entity types
            return {k: v for k, v in filtered_entities.items() if v}
            
        except Exception as e:
            logger.warning(f"Error extracting entities: {str(e)}")
            return {}
