"""
Unit tests for CurioScan models.
"""

import pytest
import torch
import numpy as np
from PIL import Image

from models.renderer_classifier import RendererClassifier
from models.ocr_models import TesseractOCR, OCRModelEnsemble
from models.table_detector import TableDetector
from models.layout_analyzer import LayoutAnalyzer


class TestRendererClassifier:
    """Test renderer classifier model."""
    
    def test_model_creation(self):
        """Test model can be created."""
        model = RendererClassifier(num_classes=8)
        assert model is not None
        assert hasattr(model, 'backbone')
        assert hasattr(model, 'classifier')
    
    def test_forward_pass(self):
        """Test forward pass works."""
        model = RendererClassifier(num_classes=8)
        model.eval()
        
        # Test with batch
        batch_size = 4
        x = torch.randn(batch_size, 3, 224, 224)
        
        with torch.no_grad():
            output = model(x)
        
        assert output.shape == (batch_size, 8)
        assert not torch.isnan(output).any()
    
    def test_predict_method(self, sample_image):
        """Test predict method."""
        model = RendererClassifier(num_classes=8)
        model.eval()
        
        metadata = {
            "file_size": 1024,
            "has_embedded_text": True,
            "page_count": 1
        }
        
        render_type, confidence = model.predict(sample_image, metadata)
        
        assert isinstance(render_type, str)
        assert 0.0 <= confidence <= 1.0
        assert render_type in model.class_names
    
    def test_different_input_sizes(self):
        """Test model handles different input sizes."""
        model = RendererClassifier(num_classes=8)
        model.eval()
        
        # Test different sizes
        sizes = [(224, 224), (256, 256), (512, 512)]
        
        for size in sizes:
            x = torch.randn(1, 3, *size)
            with torch.no_grad():
                output = model(x)
            assert output.shape == (1, 8)


class TestOCRModels:
    """Test OCR models."""
    
    def test_tesseract_ocr_creation(self):
        """Test TesseractOCR can be created."""
        ocr = TesseractOCR()
        assert ocr is not None
        assert hasattr(ocr, 'extract_text')
    
    def test_tesseract_extract_text(self, sample_image):
        """Test Tesseract text extraction."""
        ocr = TesseractOCR()
        
        result = ocr.extract_text(sample_image)
        
        assert hasattr(result, 'tokens')
        assert hasattr(result, 'page_bbox')
        assert hasattr(result, 'processing_time')
        assert isinstance(result.tokens, list)
    
    def test_ocr_ensemble_creation(self):
        """Test OCR ensemble can be created."""
        tesseract = TesseractOCR()
        ensemble = OCRModelEnsemble([tesseract])
        
        assert ensemble is not None
        assert len(ensemble.models) == 1
    
    def test_ocr_ensemble_extract_text(self, sample_image):
        """Test OCR ensemble text extraction."""
        tesseract = TesseractOCR()
        ensemble = OCRModelEnsemble([tesseract])
        
        result = ensemble.extract_text(sample_image)
        
        assert hasattr(result, 'tokens')
        assert isinstance(result.tokens, list)
    
    @pytest.mark.slow
    def test_ocr_confidence_filtering(self, sample_image):
        """Test OCR confidence filtering."""
        ocr = TesseractOCR(confidence_threshold=0.8)
        
        result = ocr.extract_text(sample_image)
        
        # All tokens should have confidence >= 0.8
        for token in result.tokens:
            assert token.confidence >= 0.8


class TestTableDetector:
    """Test table detector model."""
    
    def test_table_detector_creation(self):
        """Test table detector can be created."""
        detector = TableDetector()
        assert detector is not None
        assert hasattr(detector, 'detect_tables')
    
    def test_detect_tables(self, sample_image):
        """Test table detection."""
        detector = TableDetector()
        
        tables = detector.detect_tables(sample_image)
        
        assert isinstance(tables, list)
        # Each table should have required fields
        for table in tables:
            assert 'bbox' in table
            assert 'confidence' in table
            assert len(table['bbox']) == 4
            assert 0.0 <= table['confidence'] <= 1.0
    
    def test_extract_table_content(self, sample_image, sample_ocr_results):
        """Test table content extraction."""
        detector = TableDetector()
        
        # Mock table detection
        table_bbox = [20, 20, 200, 100]
        
        content = detector.extract_table_content(
            sample_image, 
            table_bbox, 
            sample_ocr_results['tokens']
        )
        
        assert 'rows' in content
        assert 'columns' in content
        assert isinstance(content['rows'], list)
        assert isinstance(content['columns'], list)
    
    def test_table_structure_analysis(self, sample_table_data):
        """Test table structure analysis."""
        detector = TableDetector()
        
        structure = detector.analyze_table_structure(sample_table_data['tables'][0])
        
        assert 'num_rows' in structure
        assert 'num_columns' in structure
        assert 'has_header' in structure
        assert structure['num_rows'] > 0
        assert structure['num_columns'] > 0


class TestLayoutAnalyzer:
    """Test layout analyzer model."""
    
    def test_layout_analyzer_creation(self):
        """Test layout analyzer can be created."""
        analyzer = LayoutAnalyzer()
        assert analyzer is not None
        assert hasattr(analyzer, 'analyze_layout')
    
    def test_analyze_layout(self, sample_image):
        """Test layout analysis."""
        analyzer = LayoutAnalyzer()
        
        layout = analyzer.analyze_layout(sample_image)
        
        assert 'regions' in layout
        assert isinstance(layout['regions'], list)
        
        # Each region should have required fields
        for region in layout['regions']:
            assert 'type' in region
            assert 'bbox' in region
            assert 'confidence' in region
            assert len(region['bbox']) == 4
            assert 0.0 <= region['confidence'] <= 1.0
    
    def test_segment_regions(self, sample_image):
        """Test region segmentation."""
        analyzer = LayoutAnalyzer()
        
        regions = analyzer.segment_regions(sample_image)
        
        assert isinstance(regions, list)
        # Should find at least one region
        assert len(regions) >= 1
    
    def test_classify_regions(self, sample_image):
        """Test region classification."""
        analyzer = LayoutAnalyzer()
        
        # Mock regions
        regions = [
            {'bbox': [10, 10, 100, 50], 'type': 'unknown'},
            {'bbox': [10, 60, 200, 150], 'type': 'unknown'}
        ]
        
        classified_regions = analyzer.classify_regions(sample_image, regions)
        
        assert len(classified_regions) == len(regions)
        for region in classified_regions:
            assert region['type'] in ['text', 'table', 'image', 'header', 'footer']
    
    def test_reading_order_detection(self, sample_image):
        """Test reading order detection."""
        analyzer = LayoutAnalyzer()
        
        # Mock regions
        regions = [
            {'bbox': [10, 10, 100, 30], 'type': 'text'},
            {'bbox': [10, 40, 100, 60], 'type': 'text'},
            {'bbox': [10, 70, 200, 150], 'type': 'table'}
        ]
        
        ordered_regions = analyzer.detect_reading_order(regions)
        
        assert len(ordered_regions) == len(regions)
        # Should have reading_order field
        for region in ordered_regions:
            assert 'reading_order' in region
            assert isinstance(region['reading_order'], int)


class TestModelIntegration:
    """Test model integration and compatibility."""
    
    def test_model_device_compatibility(self):
        """Test models work on different devices."""
        model = RendererClassifier(num_classes=8)
        
        # Test CPU
        model.to('cpu')
        x = torch.randn(1, 3, 224, 224)
        output = model(x)
        assert output.device.type == 'cpu'
        
        # Test CUDA if available
        if torch.cuda.is_available():
            model.to('cuda')
            x = x.to('cuda')
            output = model(x)
            assert output.device.type == 'cuda'
    
    def test_model_serialization(self, temp_dir):
        """Test model can be saved and loaded."""
        model = RendererClassifier(num_classes=8)
        
        # Save model
        model_path = temp_dir / "test_model.pth"
        torch.save(model.state_dict(), model_path)
        
        # Load model
        new_model = RendererClassifier(num_classes=8)
        new_model.load_state_dict(torch.load(model_path, map_location='cpu'))
        
        # Test they produce same output
        x = torch.randn(1, 3, 224, 224)
        
        model.eval()
        new_model.eval()
        
        with torch.no_grad():
            output1 = model(x)
            output2 = new_model(x)
        
        assert torch.allclose(output1, output2, atol=1e-6)
    
    def test_model_memory_usage(self):
        """Test model memory usage is reasonable."""
        model = RendererClassifier(num_classes=8)
        
        # Count parameters
        param_count = sum(p.numel() for p in model.parameters())
        
        # Should be reasonable for a classification model
        assert param_count < 50_000_000  # Less than 50M parameters
        
        # Test memory usage with batch
        x = torch.randn(8, 3, 224, 224)
        
        with torch.no_grad():
            output = model(x)
        
        # Should not crash or use excessive memory
        assert output.shape == (8, 8)
