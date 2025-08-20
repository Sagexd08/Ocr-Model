"""
Document processor for CurioScan workers.

Handles the complete document processing pipeline.
"""

import os
import logging
import json
import uuid
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image
import pdfplumber
import pdf2image
from docx import Document as DocxDocument

from worker.model_manager import ModelManager
from worker.storage_manager import StorageManager

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Handles document processing pipeline."""
    
    def __init__(self, model_manager: ModelManager, storage_manager: StorageManager, 
                 confidence_threshold: float = 0.8):
        self.model_manager = model_manager
        self.storage_manager = storage_manager
        self.confidence_threshold = confidence_threshold
    
    def classify_document(self, input_path: str) -> Dict[str, Any]:
        """
        Classify document render type.
        
        Returns classification result with render type and confidence.
        """
        try:
            logger.info(f"Classifying document: {input_path}")
            
            # Download file from storage
            file_content = self.storage_manager.download_file(input_path)
            
            # Extract metadata
            metadata = self._extract_metadata(file_content, input_path)
            
            # Convert to image for classification
            image = self._convert_to_image(file_content, input_path)
            
            # Classify using model
            classification = self.model_manager.classify_document(image, metadata)
            
            logger.info(f"Document classified as: {classification['render_type']} "
                       f"(confidence: {classification['confidence']:.3f})")
            
            return classification
            
        except Exception as e:
            logger.error(f"Document classification failed: {str(e)}")
            raise
    
    def preprocess_document(self, input_path: str, classification_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Preprocess document based on classification.
        
        Returns preprocessed data ready for OCR.
        """
        try:
            logger.info(f"Preprocessing document: {input_path}")
            
            render_type = classification_result["render_type"]
            
            # Download file
            file_content = self.storage_manager.download_file(input_path)
            
            # Convert to images
            images = self._convert_to_images(file_content, input_path, render_type)
            
            # Preprocess each image
            preprocessed_images = []
            for i, image in enumerate(images):
                preprocessed_image = self._preprocess_image(image, render_type)
                preprocessed_images.append({
                    "page": i + 1,
                    "image": preprocessed_image,
                    "original_size": image.size
                })
            
            logger.info(f"Preprocessed {len(preprocessed_images)} pages")
            
            return {
                "pages": preprocessed_images,
                "render_type": render_type,
                "total_pages": len(preprocessed_images)
            }
            
        except Exception as e:
            logger.error(f"Document preprocessing failed: {str(e)}")
            raise
    
    def extract_text(self, preprocessed_data: Dict[str, Any], 
                    classification_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract text from preprocessed document.
        
        Returns OCR results with tokens and confidence scores.
        """
        try:
            logger.info("Extracting text from document")
            
            render_type = classification_result["render_type"]
            pages = preprocessed_data["pages"]
            
            all_results = []
            
            for page_data in pages:
                page_num = page_data["page"]
                image = page_data["image"]
                
                logger.info(f"Processing page {page_num}")
                
                # Extract text using appropriate method
                if render_type == "digital_pdf":
                    # Try native text extraction first
                    page_result = self._extract_native_text(page_num)
                    if not page_result or len(page_result.get("tokens", [])) == 0:
                        # Fallback to OCR
                        page_result = self.model_manager.extract_text_ocr(image, render_type)
                else:
                    # Use OCR for scanned/image documents
                    page_result = self.model_manager.extract_text_ocr(image, render_type)
                
                page_result["page"] = page_num
                all_results.append(page_result)
            
            logger.info(f"Text extraction completed for {len(all_results)} pages")
            
            return {
                "pages": all_results,
                "total_tokens": sum(len(page.get("tokens", [])) for page in all_results),
                "render_type": render_type
            }
            
        except Exception as e:
            logger.error(f"Text extraction failed: {str(e)}")
            raise
    
    def detect_tables(self, preprocessed_data: Dict[str, Any], 
                     ocr_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detect and extract tables from document.
        
        Returns table detection and extraction results.
        """
        try:
            logger.info("Detecting tables in document")
            
            pages = preprocessed_data["pages"]
            ocr_pages = ocr_results["pages"]
            
            all_table_results = []
            
            for page_data, ocr_page in zip(pages, ocr_pages):
                page_num = page_data["page"]
                image = page_data["image"]
                tokens = ocr_page.get("tokens", [])
                
                logger.info(f"Detecting tables on page {page_num}")
                
                # Detect tables using model
                table_detection = self.model_manager.detect_tables(image)
                
                # Extract table content
                table_extractions = []
                for table in table_detection.get("tables", []):
                    table_content = self._extract_table_content(table, tokens, image)
                    table_extractions.append(table_content)
                
                all_table_results.append({
                    "page": page_num,
                    "tables": table_extractions,
                    "detection_method": table_detection.get("method", "unknown")
                })
            
            logger.info(f"Table detection completed for {len(all_table_results)} pages")
            
            return {
                "pages": all_table_results,
                "total_tables": sum(len(page.get("tables", [])) for page in all_table_results)
            }
            
        except Exception as e:
            logger.error(f"Table detection failed: {str(e)}")
            raise
    
    def postprocess_results(self, ocr_results: Dict[str, Any], table_results: Dict[str, Any],
                           classification_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Postprocess and normalize extraction results.
        
        Returns final structured results in the required schema.
        """
        try:
            logger.info("Postprocessing extraction results")
            
            render_type = classification_result["render_type"]
            
            # Combine OCR and table results
            final_rows = []
            row_counter = 0
            
            # Process each page
            for page_idx, (ocr_page, table_page) in enumerate(zip(
                ocr_results["pages"], table_results["pages"]
            )):
                page_num = page_idx + 1
                
                # Add table rows first (higher priority)
                for table_idx, table in enumerate(table_page.get("tables", [])):
                    for row_idx, row in enumerate(table.get("rows", [])):
                        row_id = f"table_{page_num}_{table_idx}_{row_idx}"
                        region_id = f"table_{table_idx}"
                        
                        # Calculate confidence and review flag
                        avg_confidence = np.mean([cell.get("confidence", 0.0) 
                                                for cell in row.get("cells", [])])
                        needs_review = avg_confidence < self.confidence_threshold
                        
                        final_row = {
                            "row_id": row_id,
                            "page": page_num,
                            "region_id": region_id,
                            "bbox": row.get("bbox", [0, 0, 0, 0]),
                            "columns": row.get("data", {}),
                            "provenance": {
                                "file": "input_file",  # TODO: Get actual filename
                                "page": page_num,
                                "bbox": row.get("bbox", [0, 0, 0, 0]),
                                "token_ids": row.get("token_ids", []),
                                "confidence": avg_confidence
                            },
                            "needs_review": needs_review
                        }
                        
                        final_rows.append(final_row)
                        row_counter += 1
                
                # Add non-table text regions
                text_regions = self._extract_text_regions(
                    ocr_page.get("tokens", []), 
                    table_page.get("tables", [])
                )
                
                for region_idx, region in enumerate(text_regions):
                    row_id = f"text_{page_num}_{region_idx}"
                    region_id = f"text_{region_idx}"
                    
                    # Calculate confidence
                    avg_confidence = np.mean([token.get("confidence", 0.0) 
                                            for token in region.get("tokens", [])])
                    needs_review = avg_confidence < self.confidence_threshold
                    
                    final_row = {
                        "row_id": row_id,
                        "page": page_num,
                        "region_id": region_id,
                        "bbox": region.get("bbox", [0, 0, 0, 0]),
                        "columns": {"text": region.get("text", "")},
                        "provenance": {
                            "file": "input_file",  # TODO: Get actual filename
                            "page": page_num,
                            "bbox": region.get("bbox", [0, 0, 0, 0]),
                            "token_ids": [token.get("token_id", 0) 
                                        for token in region.get("tokens", [])],
                            "confidence": avg_confidence
                        },
                        "needs_review": needs_review
                    }
                    
                    final_rows.append(final_row)
                    row_counter += 1
            
            # Calculate overall confidence
            if final_rows:
                overall_confidence = np.mean([
                    row["provenance"]["confidence"] for row in final_rows
                ])
            else:
                overall_confidence = 0.0
            
            logger.info(f"Postprocessing completed: {len(final_rows)} rows extracted")
            
            return {
                "rows": final_rows,
                "metadata": {
                    "render_type": render_type,
                    "total_pages": len(ocr_results["pages"]),
                    "total_tokens": ocr_results.get("total_tokens", 0),
                    "total_tables": table_results.get("total_tables", 0),
                    "confidence_threshold": self.confidence_threshold
                },
                "confidence_score": overall_confidence,
                "render_type": render_type
            }
            
        except Exception as e:
            logger.error(f"Postprocessing failed: {str(e)}")
            raise
    
    def store_results(self, job_id: str, results: Dict[str, Any]) -> str:
        """
        Store processing results.
        
        Returns the output path where results are stored.
        """
        try:
            logger.info(f"Storing results for job {job_id}")
            
            # Convert results to JSON
            results_json = json.dumps(results, indent=2, default=str)
            
            # Store in storage backend
            output_path = f"output/{job_id}/results.json"
            self.storage_manager.upload_file(results_json.encode(), output_path)
            
            logger.info(f"Results stored at: {output_path}")
            
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to store results: {str(e)}")
            raise
    
    # Helper methods
    
    def _extract_metadata(self, file_content: bytes, file_path: str) -> Dict[str, Any]:
        """Extract metadata from file."""
        metadata = {
            "file_size": len(file_content),
            "mime_type": self._detect_mime_type(file_content, file_path),
            "has_embedded_text": False,
            "page_count": 1
        }
        
        # Try to extract more metadata based on file type
        try:
            if metadata["mime_type"] == "application/pdf":
                # Use pdfplumber to extract PDF metadata
                import io
                with pdfplumber.open(io.BytesIO(file_content)) as pdf:
                    metadata["page_count"] = len(pdf.pages)
                    # Check for embedded text
                    for page in pdf.pages[:3]:  # Check first 3 pages
                        if page.extract_text().strip():
                            metadata["has_embedded_text"] = True
                            break
        except Exception as e:
            logger.warning(f"Failed to extract detailed metadata: {str(e)}")
        
        return metadata
    
    def _detect_mime_type(self, file_content: bytes, file_path: str) -> str:
        """Detect MIME type from file content and extension."""
        import mimetypes
        
        # Try to detect from extension first
        mime_type, _ = mimetypes.guess_type(file_path)
        
        if mime_type:
            return mime_type
        
        # Fallback to magic bytes
        if file_content.startswith(b'%PDF'):
            return "application/pdf"
        elif file_content.startswith(b'\xff\xd8\xff'):
            return "image/jpeg"
        elif file_content.startswith(b'\x89PNG'):
            return "image/png"
        elif file_content.startswith(b'PK'):
            return "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        else:
            return "application/octet-stream"
    
    def _convert_to_image(self, file_content: bytes, file_path: str) -> Image.Image:
        """Convert file to single image for classification."""
        images = self._convert_to_images(file_content, file_path)
        return images[0] if images else Image.new('RGB', (100, 100), 'white')
    
    def _convert_to_images(self, file_content: bytes, file_path: str, 
                          render_type: str = None) -> List[Image.Image]:
        """Convert file to list of images."""
        import io
        
        mime_type = self._detect_mime_type(file_content, file_path)
        
        try:
            if mime_type == "application/pdf":
                # Convert PDF to images
                images = pdf2image.convert_from_bytes(file_content, dpi=300)
                return images
            
            elif mime_type.startswith("image/"):
                # Load image directly
                image = Image.open(io.BytesIO(file_content))
                return [image.convert('RGB')]
            
            elif "word" in mime_type:
                # For DOCX, we'd need to convert to PDF first or extract images
                # For now, create a placeholder
                return [Image.new('RGB', (800, 1000), 'white')]
            
            else:
                # Unknown format, create placeholder
                return [Image.new('RGB', (800, 1000), 'white')]
                
        except Exception as e:
            logger.error(f"Failed to convert file to images: {str(e)}")
            return [Image.new('RGB', (800, 1000), 'white')]
    
    def _preprocess_image(self, image: Image.Image, render_type: str) -> Image.Image:
        """Preprocess image based on render type."""
        try:
            # Convert to numpy array
            img_array = np.array(image)
            
            if render_type in ["scanned_image", "photograph"]:
                # Apply preprocessing for scanned/photographed documents
                img_array = self._deskew_image(img_array)
                img_array = self._denoise_image(img_array)
                img_array = self._enhance_contrast(img_array)
            
            # Convert back to PIL Image
            return Image.fromarray(img_array)
            
        except Exception as e:
            logger.warning(f"Image preprocessing failed: {str(e)}")
            return image
    
    def _deskew_image(self, img_array: np.ndarray) -> np.ndarray:
        """Deskew image to correct rotation."""
        # Simple deskewing implementation
        # In production, you'd use more sophisticated methods
        return img_array
    
    def _denoise_image(self, img_array: np.ndarray) -> np.ndarray:
        """Remove noise from image."""
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        # Apply median filter to remove noise
        denoised = cv2.medianBlur(gray, 3)
        
        # Convert back to RGB if needed
        if len(img_array.shape) == 3:
            return cv2.cvtColor(denoised, cv2.COLOR_GRAY2RGB)
        else:
            return denoised
    
    def _enhance_contrast(self, img_array: np.ndarray) -> np.ndarray:
        """Enhance image contrast."""
        if len(img_array.shape) == 3:
            # Convert to LAB color space for better contrast enhancement
            lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            
            # Merge channels and convert back
            enhanced = cv2.merge([l, a, b])
            return cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
        else:
            # Grayscale image
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            return clahe.apply(img_array)
    
    def _extract_native_text(self, page_num: int) -> Optional[Dict[str, Any]]:
        """Extract native text from PDF (placeholder)."""
        # TODO: Implement native PDF text extraction
        return None
    
    def _extract_table_content(self, table: Dict[str, Any], tokens: List[Dict[str, Any]], 
                              image: Image.Image) -> Dict[str, Any]:
        """Extract content from detected table."""
        # TODO: Implement table content extraction
        return {
            "bbox": table.get("bbox", [0, 0, 0, 0]),
            "rows": [],
            "confidence": table.get("confidence", 0.0)
        }
    
    def _extract_text_regions(self, tokens: List[Dict[str, Any]], 
                             tables: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract text regions that are not part of tables."""
        # TODO: Implement text region extraction
        return []
