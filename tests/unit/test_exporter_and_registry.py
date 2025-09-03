import os
import importlib
from worker.pipeline.pipeline_builder import PipelineBuilder
from worker.pipeline.processors.exporter import Exporter


def test_exporter_config_merging(tmp_path):
    exp = Exporter({"output_dir": str(tmp_path), "default_format": "json"}, default_format="pdf")
    assert exp.default_format == "pdf"
    assert exp.output_dir == str(tmp_path)


def test_registry_maps_core_ids():
    pb = PipelineBuilder()
    reg = pb.processor_registry
    assert "table_detector.TableDetector" in reg
    assert "exporter.Exporter" in reg

