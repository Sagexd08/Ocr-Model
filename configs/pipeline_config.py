from typing import Dict, Any, Optional, List
import yaml
from pathlib import Path

# Configuration for CurioScan OCR pipeline
pipeline_config = {
    # Default pipeline configuration
    "default": {
        "processors": [
            {
                "id": "pdf_processor.PDFProcessor",
                "params": {}
            },
            {
                "id": "image_ingestion.ImageIngestion",
                "params": {}
            },
            {
                "id": "advanced_ocr.AdvancedOCRProcessor",
                "params": {}
            },
            {
                "id": "image_enhancer.ImageEnhancer",
                "params": {
                    "deskew_enabled": True,
                    "denoise_enabled": True,
                    "enhance_contrast": True,
                    "upscale_resolution": False,
                    "target_dpi": 300
                }
            },
            {
                "id": "classifier.DocumentClassifier",
                "params": {
                    "model_type": "mobilenet",
                    "confidence_threshold": 0.7,
                    "enable_renderer_detection": True
                }
            },
            {
                "id": "layout.LayoutAnalyzer",
                "params": {
                    "region_types": ["paragraph", "heading", "list", "table", "image", "footer", "header"],
                    "min_region_size": 50,
                    "max_vertical_gap": 20
                }
            },
            {
                "id": "region_classifier.RegionClassifierProcessor",
                "params": {
                    "score_threshold": 0.4
                }
            },
            {
                "id": "tables.TableExtractor",
                "params": {
                    "line_detection_method": "hough",
                    "table_detection_confidence": 0.75,
                    "max_cell_merge_distance": 5,
                    "use_hierarchical_clustering": True
                }
            },
            {
                "id": "text_processor.TextProcessor",
                "params": {
                    "language": "en",
                    "spelling_correction": True,
                    "normalization": True,
                    "entity_extraction": True
                }
            },
            {
                "id": "qa.QualityAnalyzer",
                "params": {
                    "validation_level": "strict",
                    "confidence_threshold": 0.75,
                    "validate_tables": True,
                    "validate_dates": True,
                    "validate_amounts": True
                }
            },
            {
                "id": "postprocessing.PostProcessor",
                "params": {
                    "pii_redaction": False,
                    "consolidate_regions": True,
                    "extract_relationships": True,
                    "confidence_threshold": 0.7
                }
            },
            {
                "id": "exporter.Exporter",
                "params": {
                    "default_format": "json",
                    "formats": {
                        "json": {
                            "format_type": "json",
                            "include_confidence": True,
                            "include_bbox": True,
                            "include_metadata": True,
                            "structure_preserving": True
                        },
                        "csv": {
                            "format_type": "csv",
                            "include_confidence": True,
                            "include_bbox": True,
                            "include_metadata": True
                        },
                        "txt": {
                            "format_type": "txt",
                            "include_confidence": False,
                            "include_bbox": False,
                            "structure_preserving": True
                        }
                    }
                }
            }
        ]
    },
    
    # High performance pipeline with minimal processing
    "performance": {
        "processors": [
            {
                "id": "pdf_processor.PDFProcessor",
                "params": {}
            },
            {
                "id": "image_ingestion.ImageIngestion",
                "params": {}
            },
            {
                "id": "advanced_ocr.AdvancedOCRProcessor",
                "params": {}
            },
            {
                "id": "image_enhancer.ImageEnhancer",
                "params": {
                    "deskew_enabled": True,
                    "denoise_enabled": False,
                    "enhance_contrast": True,
                    "upscale_resolution": False
                }
            },
            {
                "id": "layout.LayoutAnalyzer",
                "params": {
                    "region_types": ["paragraph", "heading", "table"],
                    "min_region_size": 100,
                    "max_vertical_gap": 30
                }
            },
            {
                "id": "region_classifier.RegionClassifierProcessor",
                "params": {
                    "score_threshold": 0.5
                }
            },
            {
                "id": "tables.TableExtractor",
                "params": {
                    "line_detection_method": "contour",
                    "use_hierarchical_clustering": False
                }
            },
            {
                "id": "text_processor.TextProcessor",
                "params": {
                    "spelling_correction": False,
                    "normalization": True,
                    "entity_extraction": False
                }
            },
            {
                "id": "exporter.Exporter",
                "params": {
                    "default_format": "json"
                }
            }
        ]
    },
    
    # Quality-focused pipeline with all processors
    "quality": {
        "processors": [
            {
                "id": "pdf_processor.PDFProcessor",
                "params": {}
            },
            {
                "id": "image_ingestion.ImageIngestion",
                "params": {}
            },
            {
                "id": "advanced_ocr.AdvancedOCRProcessor",
                "params": {}
            },
            {
                "id": "image_enhancer.ImageEnhancer",
                "params": {
                    "deskew_enabled": True,
                    "denoise_enabled": True,
                    "enhance_contrast": True,
                    "upscale_resolution": True,
                    "upscale_factor": 2.0,
                    "target_dpi": 400
                }
            },
            {
                "id": "classifier.DocumentClassifier",
                "params": {
                    "model_type": "densenet",
                    "confidence_threshold": 0.8,
                    "enable_renderer_detection": True
                }
            },
            {
                "id": "layout.LayoutAnalyzer",
                "params": {
                    "region_types": ["paragraph", "heading", "subheading", "list", "table", "image", "footer", "header", "caption"],
                    "min_region_size": 30,
                    "max_vertical_gap": 15
                }
            },
            {
                "id": "region_classifier.RegionClassifierProcessor",
                "params": {
                    "score_threshold": 0.4
                }
            },
            {
                "id": "tables.TableExtractor",
                "params": {
                    "line_detection_method": "combined",
                    "table_detection_confidence": 0.85,
                    "max_cell_merge_distance": 3,
                    "use_hierarchical_clustering": True
                }
            },
            {
                "id": "text_processor.TextProcessor",
                "params": {
                    "language": "en",
                    "spelling_correction": True,
                    "normalization": True,
                    "entity_extraction": True
                }
            },
            {
                "id": "qa.QualityAnalyzer",
                "params": {
                    "validation_level": "strict",
                    "confidence_threshold": 0.85,
                    "validate_tables": True,
                    "validate_dates": True,
                    "validate_amounts": True,
                    "spell_check": True,
                    "grammar_check": True
                }
            },
            {
                "id": "postprocessing.PostProcessor",
                "params": {
                    "pii_redaction": True,
                    "consolidate_regions": True,
                    "extract_relationships": True,
                    "confidence_threshold": 0.8
                }
            },
            {
                "id": "exporter.Exporter",
                "params": {
                    "default_format": "json",
                    "formats": {
                        "json": {
                            "format_type": "json",
                            "include_confidence": True,
                            "include_bbox": True,
                            "include_metadata": True,
                            "structure_preserving": True
                        },
                        "xml": {
                            "format_type": "xml",
                            "include_confidence": True,
                            "include_bbox": True,
                            "include_metadata": True,
                            "structure_preserving": True
                        },
                        "pdf": {
                            "format_type": "pdf",
                            "include_confidence": True,
                            "include_bbox": True
                        }
                    }
                }
            }
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
