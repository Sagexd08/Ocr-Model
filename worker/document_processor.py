import os
import time
import uuid
from typing import Dict, Any, Optional, List, Union
import traceback
import logging
from pathlib import Path

from .types import Document, ProcessingResult
from .pipeline.pipeline_builder import PipelineBuilder
from .utils.logging import get_logger, log_execution_time, log_context
from .model_manager import ModelManager
from .storage_manager import StorageManager

logger = get_logger(__name__)

class DocumentProcessor:
    """
    Main document processing pipeline that implements the full OCR workflow.
    
    The pipeline consists of these stages:
    1. Preprocessing (document-type specific preparation)
    2. Image enhancement (deskewing, contrast improvement)
    3. Renderer classification (determine document type for specialized handling)
    4. Layout analysis (segment document into regions)
    5. OCR (extract text from regions)
    6. Table detection and reconstruction
    7. Text processing and normalization
    8. Quality analysis
    9. Post-processing and metadata extraction
    10. Result packaging with provenance and export
    
    Provides robust error handling, telemetry, and progress tracking.
    """
    def __init__(
        self,
        document: Optional[Document] = None,
        file_data: Optional[bytes] = None,
        job_id: Optional[str] = None,
        model_manager: Optional[ModelManager] = None,
        storage_manager: Optional[StorageManager] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        self.document = document
        self.file_data = file_data
        self.job_id = job_id or str(uuid.uuid4())
        self.model_manager = model_manager or ModelManager()
        self.storage_manager = storage_manager or StorageManager()
        self.config = config or {}
        
        # Load pipeline configuration
        pipeline_config = self.config.get("pipeline", {})
        pipeline_config_path = pipeline_config.get("config_path")
        self.pipeline_name = pipeline_config.get("name", "default")
        
        # Initialize pipeline builder and build processor pipeline
        self.pipeline_builder = PipelineBuilder(pipeline_config_path)
        self.processors = []
        
        # Performance metrics
        self.metrics: Dict[str, Any] = {
            "processing_start": time.time(),
            "stages": {},
            "token_count": 0,
            "region_count": 0,
            "table_count": 0,
            "page_count": 0,
            "confidence_avg": 0.0,
        }
        self.progress = 0.0
        
        # Configurable settings with defaults
        self.confidence_threshold = self.config.get("confidence_threshold", 0.8)
        self.webhook_enabled = self.config.get("webhook_enabled", True)
        self.pii_redaction = self.config.get("pii_redaction", False)
        
    @log_execution_time
    def process(self) -> ProcessingResult:
        """
        Process the document through the entire OCR pipeline.
        
        Returns:
            ProcessingResult object containing extracted data and metadata
        """
        try:
            # Build processing pipeline if not already built
            if not self.processors:
                self.processors = self.pipeline_builder.build_pipeline(self.pipeline_name)
                processor_names = [p[0] for p in self.processors]
                logger.info(
                    f"Built processing pipeline with {len(self.processors)} processors: {', '.join(processor_names)}",
                    extra={"job_id": self.job_id, "pipeline": self.pipeline_name}
                )
            
            # Initialize progress tracking
            total_processors = len(self.processors)
            progress_increment = 100.0 / total_processors if total_processors > 0 else 100.0
            
            # Process document through each processor in pipeline
            for idx, (processor_name, processor) in enumerate(self.processors):
                stage_name = processor_name.split('.')[-1]
                self._start_stage(stage_name)
                
                # Process the document
                with log_context(f"{stage_name} processing", logger=logger):
                    try:
                        self.document = processor.process(self.document)
                        
                        # Save intermediate result if configured
                        if self.config.get("save_intermediates", False):
                            self._save_intermediate(stage_name, idx)
                            
                    except Exception as e:
                        logger.error(
                            f"Error in {stage_name} processor: {str(e)}",
                            extra={
                                "job_id": self.job_id,
                                "processor": processor_name,
                                "traceback": traceback.format_exc()
                            }
                        )
                        if not self.config.get("continue_on_error", False):
                            raise
                
                # Update progress
                self.progress = (idx + 1) * progress_increment
                self._complete_stage(stage_name, progress=self.progress)
                
                # Update metrics
                self._update_metrics()
            
            # Package results
            self._start_stage("packaging")
            result = self._package_results()
            self._complete_stage("packaging", progress=100)
            
            # Log completion metrics
            self.metrics["processing_end"] = time.time()
            self.metrics["processing_duration"] = (
                self.metrics["processing_end"] - self.metrics["processing_start"]
            )
            
            logger.info(
                f"Processing completed for job {self.job_id}",
                extra={
                    "job_id": self.job_id,
                    "duration": self.metrics["processing_duration"],
                    "page_count": self.metrics["page_count"],
                    "token_count": self.metrics["token_count"],
                    "confidence_avg": self.metrics["confidence_avg"]
                }
            )
            
            return result
            
        except Exception as e:
            logger.error(
                f"Processing failed for job {self.job_id}: {str(e)}",
                extra={
                    "job_id": self.job_id,
                    "error": str(e),
                    "traceback": traceback.format_exc()
                }
            )
            # Re-raise with detailed error
            raise DocumentProcessingError(
                f"Failed to process document: {str(e)}",
                job_id=self.job_id,
                stage=self.metrics.get("current_stage", "unknown"),
                details=str(e)
            )
    
    def _update_metrics(self):
        """Update document processing metrics"""
        if not self.document:
            return
            
        # Update page count
        self.metrics["page_count"] = len(self.document.pages)
        
        # Update token count and confidence
        token_count = 0
        confidence_sum = 0.0
        region_count = 0
        table_count = 0
        
        for page in self.document.pages:
            # Count tokens and get confidence from regions
            for region in page.regions:
                region_count += 1
                token_count += len(region.tokens)
                confidence_sum += sum(token.confidence for token in region.tokens) if region.tokens else 0
            
            # Count tables
            table_count += len(page.tables)
            
            # Count tokens in tables
            for table in page.tables:
                for row in table.cells:
                    for cell in row:
                        if cell:
                            token_count += len(cell.tokens)
                            confidence_sum += sum(token.confidence for token in cell.tokens) if cell.tokens else 0
        
        # Update metrics
        self.metrics["token_count"] = token_count
        self.metrics["region_count"] = region_count
        self.metrics["table_count"] = table_count
        self.metrics["confidence_avg"] = (
            confidence_sum / token_count if token_count > 0 else 0.0
        )
    
    def _save_intermediate(self, stage_name: str, idx: int):
        """Save intermediate processing result"""
        if not self.document:
            return
            
        try:
            # Create intermediate filename
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = f"{self.job_id}_{idx:02d}_{stage_name}_{timestamp}.json"
            intermediates_path = self.config.get("intermediates_path", "intermediates")
            os.makedirs(intermediates_path, exist_ok=True)
            filepath = os.path.join(intermediates_path, filename)
            
            # Save document as JSON
            self.storage_manager.save_document_json(self.document, filepath)
            
            logger.debug(
                f"Saved intermediate result after {stage_name}",
                extra={
                    "job_id": self.job_id,
                    "stage": stage_name,
                    "filepath": filepath
                }
            )
        except Exception as e:
            logger.warning(
                f"Failed to save intermediate result: {str(e)}",
                extra={
                    "job_id": self.job_id,
                    "stage": stage_name
                }
            )
    
    def _start_stage(self, stage_name: str):
        """Mark the start of a processing stage"""
        self.metrics["current_stage"] = stage_name
        self.metrics["stages"][stage_name] = {
            "start": time.time()
        }
        logger.info(
            f"Starting {stage_name} stage for job {self.job_id}",
            extra={
                "job_id": self.job_id,
                "stage": stage_name
            }
        )
    
    def _complete_stage(self, stage_name: str, progress: float = None):
        """Mark completion of a processing stage"""
        if stage_name in self.metrics["stages"]:
            self.metrics["stages"][stage_name]["end"] = time.time()
            self.metrics["stages"][stage_name]["duration"] = (
                self.metrics["stages"][stage_name]["end"] - 
                self.metrics["stages"][stage_name]["start"]
            )
        
        if progress is not None:
            self.progress = progress
            
        logger.info(
            f"Completed {stage_name} stage for job {self.job_id}",
            extra={
                "job_id": self.job_id,
                "stage": stage_name,
                "duration": self.metrics["stages"].get(stage_name, {}).get("duration"),
                "progress": self.progress
            }
        )
    
    def _package_results(self) -> ProcessingResult:
        """Package processing results with full provenance"""
        rows = []
        
        # Generate rows from document data with full provenance
        for page_idx, page in enumerate(self.document.pages):
            # Handle text regions
            for region in page.regions:
                row = {
                    "row_id": f"r_{uuid.uuid4().hex[:8]}",
                    "page": page_idx,
                    "region_id": region.id,
                    "bbox": region.bbox.dict(),
                    "text": region.text,
                    "type": region.type,
                    "columns": {"text": region.text},
                    "provenance": {
                        "file": self.document.metadata.get("filename", ""),
                        "page": page_idx,
                        "bbox": region.bbox.dict(),
                        "token_ids": [t.id for t in region.tokens],
                        "confidence": region.confidence
                    },
                    "needs_review": region.confidence < self.confidence_threshold,
                    "attributes": region.attributes
                }
                rows.append(row)
                
            # Handle tables with special column structure
            for table in page.tables:
                for row_idx, row_data in enumerate(table.cells):
                    # Convert row to standard output format
                    column_values = {}
                    token_ids = []
                    min_confidence = 1.0
                    
                    # Extract column values and track provenance
                    for col_idx, cell in enumerate(row_data):
                        if cell:
                            column_name = f"col_{col_idx}" if not hasattr(table, 'columns') or not table.columns or col_idx >= len(table.columns) else table.columns[col_idx]
                            column_values[column_name] = cell.text
                            token_ids.extend([t.id for t in cell.tokens])
                            cell_confidence = sum(t.confidence for t in cell.tokens) / len(cell.tokens) if cell.tokens else 0
                            min_confidence = min(min_confidence, cell_confidence)
                    
                    table_row = {
                        "row_id": f"t{table.id}_r{row_idx}",
                        "page": page_idx,
                        "region_id": table.id,
                        "bbox": table.bbox.dict(),
                        "type": "table_row",
                        "columns": column_values,
                        "provenance": {
                            "file": self.document.metadata.get("filename", ""),
                            "page": page_idx,
                            "bbox": table.bbox.dict(),
                            "token_ids": token_ids,
                            "confidence": min_confidence,
                            "table_id": table.id
                        },
                        "needs_review": min_confidence < self.confidence_threshold,
                        "attributes": table.attributes
                    }
                    rows.append(table_row)
        
        # Create the final result structure with improved metadata
        metadata = {
            **self.document.metadata,
            "processing": {
                "job_id": self.job_id,
                "pipeline": self.pipeline_name,
                "processors": [p[0] for p in self.processors],
                "duration": self.metrics["processing_duration"] if "processing_duration" in self.metrics else None,
                "confidence": self.metrics["confidence_avg"],
                "page_count": self.metrics["page_count"],
                "token_count": self.metrics["token_count"],
                "region_count": self.metrics["region_count"],
                "table_count": self.metrics["table_count"]
            }
        }
        
        result = ProcessingResult(
            job_id=self.job_id,
            document_id=self.document.id,
            filename=self.document.metadata.get("filename", "unknown"),
            rows=rows,
            metrics=self.metrics,
            metadata=metadata
        )
        
        return result

    def get_available_processors(self) -> Dict[str, str]:
        """
        Get list of available processors
        
        Returns:
            Dictionary of processor_id -> description
        """
        return self.pipeline_builder.list_available_processors()
    
    def get_processor_details(self, processor_id: str) -> Dict[str, Any]:
        """
        Get detailed information about a processor
        
        Args:
            processor_id: ID of the processor
            
        Returns:
            Dictionary with processor details
        """
        return self.pipeline_builder.get_processor_details(processor_id)


class DocumentProcessingError(Exception):
    """Exception raised for document processing errors with context"""
    
    def __init__(
        self, 
        message: str, 
        job_id: str = None, 
        stage: str = None,
        details: str = None
    ):
        self.message = message
        self.job_id = job_id
        self.stage = stage
        self.details = details
        super().__init__(self.message)
