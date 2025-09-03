from typing import Dict, Any, Optional, List
import yaml
from pathlib import Path

# Configuration for CurioScan OCR pipeline
pipeline_config = {
    # Default pipeline configuration
    "default": {
        "processors": [
            {"id": "pdf_processor.PDFProcessor", "params": {}},
            {"id": "image_ingestion.ImageIngestion", "params": {}},
            {"id": "advanced_ocr.AdvancedOCRProcessor", "params": {}},
            {"id": "table_detector.TableDetector", "params": {}},
            {"id": "exporter.Exporter", "params": {"default_format": "json"}}
        ]
    },
    
    # High performance pipeline with minimal processing
    "performance": {
        "processors": [
            {"id": "pdf_processor.PDFProcessor", "params": {}},
            {"id": "advanced_ocr.AdvancedOCRProcessor", "params": {}},
            {"id": "exporter.Exporter", "params": {"default_format": "json"}}
        ]
    },
    
    # Quality-focused pipeline with all processors
    "quality": {
        "processors": [
            {"id": "pdf_processor.PDFProcessor", "params": {}},
            {"id": "image_ingestion.ImageIngestion", "params": {}},
            {"id": "advanced_ocr.AdvancedOCRProcessor", "params": {}},
            {"id": "table_detector.TableDetector", "params": {}},
            {"id": "exporter.Exporter", "params": {"default_format": "json"}}
        ]
    }
}

# Export config to YAML
def export_config():
    config = {
        "pipelines": pipeline_config
    }
    
    output_path = Path("configs/pipeline.yaml")
    output_path.parent.mkdir(exist_ok=True)
    
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"Exported pipeline configuration to {output_path}")

if __name__ == "__main__":
    export_config()
