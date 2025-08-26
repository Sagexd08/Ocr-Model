# Advanced Pipeline Processors

CurioScan implements a flexible processing pipeline architecture with specialized processors for each stage of document processing. This guide covers the advanced processors available in the system.

## Pipeline Overview

The CurioScan pipeline is modular and configurable, allowing for custom processing flows based on document type and specific requirements. The general processing flow follows these stages:

```
Document → Classification → Preprocessing → Layout Analysis → OCR → Table Processing → Postprocessing → Validation → Export
```

## Available Processors

### Document Classification

#### `RendererClassifier`

Determines how the document was created (digital PDF, scanned image, etc.) to optimize downstream processing.

```python
from worker.pipeline.processors.classifier import RendererClassifier

classifier = RendererClassifier(config={
    "confidence_threshold": 0.85,
    "use_gpu": True
})

document = classifier.process(document)
```

Key features:
- Digital vs scanned classification
- PDF text layer detection
- Image quality assessment
- Document type prediction

### Preprocessing

#### `ImageEnhancer`

Advanced image enhancement for optimal OCR performance.

```python
from worker.pipeline.processors.image_enhancer import ImageEnhancer

enhancer = ImageEnhancer(config={
    "denoise": True,
    "deskew": True,
    "contrast_enhancement": "adaptive",
    "binarization_method": "adaptive_otsu"
})

document = enhancer.process(document)
```

Key features:
- Multi-scale retinex algorithm for shadow removal
- Adaptive thresholding with multiple techniques
- Fourier transform-based deskewing
- Document border detection and cleaning
- Advanced denoising with bilateral filtering

#### `Binarizer`

Specialized binarization techniques for different document types.

```python
from worker.pipeline.processors.binarizer import Binarizer

binarizer = Binarizer(config={
    "method": "sauvola",
    "window_size": 25,
    "k_value": 0.2
})

document = binarizer.process(document)
```

Supported binarization methods:
- Otsu global thresholding
- Sauvola local adaptive thresholding
- Niblack adaptive thresholding
- Wolf adaptive thresholding
- Contrast-limited adaptive histogram equalization (CLAHE)

### Layout Analysis

#### `LayoutAnalyzer`

Extracts document structure, regions, and reading order.

```python
from worker.pipeline.processors.layout_analyzer import LayoutAnalyzer

analyzer = LayoutAnalyzer(config={
    "detect_columns": True,
    "detect_headers_footers": True,
    "region_types": ["text", "table", "figure", "list", "heading"]
})

document = analyzer.process(document)
```

Key features:
- Page segmentation into structural regions
- Logical section identification (headers, body, footnotes)
- Reading order determination
- Content classification by type

### OCR Processing

#### `AdvancedOCR`

High-precision text recognition using advanced models.

```python
from worker.pipeline.processors.advanced_ocr import AdvancedOCR

ocr = AdvancedOCR(config={
    "model": "transformer",
    "languages": ["eng", "fra", "deu"],
    "batch_size": 16,
    "use_gpu": True,
    "text_confidence_threshold": 0.75
})

document = ocr.process(document)
```

Key features:
- Transformer-based model (TrOCR) integration
- Multi-language support
- Context-aware text recognition
- Automatic language detection
- Token-level confidence scores

### Table Processing

#### `TableDetector`

Identifies and extracts tables from document pages.

```python
from worker.pipeline.processors.table_detector import TableDetector

detector = TableDetector(config={
    "detection_method": "hybrid",
    "min_confidence": 0.75,
    "detect_borderless": True
})

document = detector.process(document)
```

Key features:
- Rule-based and ML-based detection methods
- Borderless table detection
- Nested table support
- Header row identification

#### `TableSegmenter`

Analyzes table structure and extracts cell content.

```python
from worker.pipeline.processors.table_segmenter import TableSegmenter

segmenter = TableSegmenter(config={
    "segmentation_method": "grid",
    "merge_cells": True,
    "detect_spanning_cells": True
})

document = segmenter.process(document)
```

Key features:
- Cell structure extraction
- Row and column span detection
- Header identification
- Cell content extraction and normalization
- Hierarchical structure analysis

### Postprocessing

#### `TextProcessor`

Normalizes and enhances extracted text.

```python
from worker.pipeline.processors.text_processor import TextProcessor

processor = TextProcessor(config={
    "normalize_whitespace": True,
    "remove_hyphenation": True,
    "normalize_characters": True,
    "detect_languages": True
})

document = processor.process(document)
```

Key features:
- Whitespace normalization
- Hyphenation removal
- Character normalization
- Special character handling
- Case correction

#### `Postprocessor`

Performs entity extraction and content enrichment.

```python
from worker.pipeline.processors.postprocessing import Postprocessor

post = Postprocessor(config={
    "extract_entities": True,
    "entity_types": ["person", "organization", "date", "amount"],
    "redact_pii": False
})

document = processor.process(document)
```

Key features:
- Named entity recognition
- PII detection and optional redaction
- Date normalization
- Amount and currency detection
- Key-value pair extraction

### Quality Analysis

#### `QAProcessor`

Validates results and flags items for review.

```python
from worker.pipeline.processors.qa import QAProcessor

qa = QAProcessor(config={
    "low_confidence_threshold": 0.7,
    "high_confidence_threshold": 0.9,
    "validate_dates": True,
    "validate_amounts": True,
    "validate_tables": True,
    "detect_context_anomalies": True
})

document = qa.process(document)
```

Key features:
- Multi-level confidence scoring
- Date format validation
- Numeric amount validation
- Table structure validation
- Context-aware anomaly detection
- Document structure validation

### Export

#### `Exporter`

Converts processed document to various output formats.

```python
from worker.pipeline.processors.exporter import Exporter

exporter = Exporter(config={
    "formats": ["json", "csv", "xlsx"],
    "include_confidence": True,
    "include_metadata": True,
    "include_bbox": True
})

outputs = exporter.process(document)
```

Key features:
- Multiple export formats
- Confidence score inclusion
- BBox coordinate inclusion
- Hierarchical JSON output
- Tabular output for spreadsheets
- Document metadata preservation

## Custom Pipeline Configuration

You can configure custom pipelines using the `PipelineBuilder`:

```python
from worker.pipeline.pipeline_builder import PipelineBuilder

# Create a pipeline optimized for invoices
pipeline = PipelineBuilder.build(
    profile="invoice_processing",
    config={
        "use_gpu": True,
        "languages": ["eng"],
        "table_detection": True,
        "confidence_threshold": 0.7
    }
)

# Process a document through the pipeline
result = pipeline.run(document)
```

Predefined profiles include:
- `default` - Balanced processing for general documents
- `invoice_processing` - Optimized for invoices and receipts
- `form_processing` - Optimized for forms with fields
- `table_extraction` - Focused on table detection and extraction
- `high_precision` - Maximum accuracy at the cost of speed
- `fast` - Optimized for speed over accuracy

## Pipeline Metrics

The system collects detailed metrics for each pipeline run:

```python
from worker.pipeline.metrics import get_pipeline_metrics

metrics = get_pipeline_metrics(job_id)
print(f"Total processing time: {metrics['total_time']}s")
print(f"OCR confidence: {metrics['avg_confidence']}")
```

Available metrics include:
- Processing time per stage
- Memory usage
- CPU/GPU utilization
- Confidence scores
- Error rates
- Document complexity metrics
