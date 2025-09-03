from typing import Dict, Any, List, Optional, Tuple, Union
import yaml
import importlib
import inspect
import os
from pathlib import Path

from worker.utils.logging import get_logger

# Pre-import core processors to ensure registry contains them
try:
    import worker.pipeline.processors.table_detector  # noqa: F401
    import worker.pipeline.processors.exporter  # noqa: F401
except Exception:
    pass

logger = get_logger(__name__)

class PipelineBuilder:
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self.processor_registry = {}
        self.pipeline_config = {}
        if config_path:
            self._load_config(config_path)
        self._discover_processors()

    def _load_config(self, config_path: str):
        try:
            with open(config_path, 'r') as f:
                self.pipeline_config = yaml.safe_load(f)
            logger.info(f"Loaded pipeline configuration from {config_path}")
        except Exception as e:
            logger.error(f"Error loading pipeline configuration: {str(e)}")
            self.pipeline_config = {}

    def _discover_processors(self):
        try:
            processors_path = Path(__file__).parent / "processors"

            if not processors_path.exists():
                logger.warning(f"Processors directory not found at {processors_path}")
                return

            processor_files = [f for f in processors_path.glob("*.py") if f.is_file() and not f.name.startswith("__")]
            for file_path in processor_files:
                module_name = file_path.stem

                try:
                    # Import the module
                    module = importlib.import_module(f"worker.pipeline.processors.{module_name}")

                    # Find processor classes in the module
                    for name, obj in inspect.getmembers(module):
                        # Look for classes with a process method
                        if (inspect.isclass(obj) and
                            hasattr(obj, 'process') and
                            inspect.isfunction(getattr(obj, 'process'))):

                            # Register the processor
                            processor_id = f"{module_name}.{name}"
                            self.processor_registry[processor_id] = obj
                            logger.debug(f"Registered processor: {processor_id}")

                except Exception as e:
                    logger.warning(f"Error loading processor module {module_name}: {str(e)}")
            try:
                from worker.pipeline.processors.table_detector import TableDetector as _TD
                self.processor_registry["table_detector.TableDetector"] = _TD
            except Exception:
                pass
            try:
                from worker.pipeline.processors.exporter import Exporter as _EX
                self.processor_registry["exporter.Exporter"] = _EX
            except Exception:
                pass


            logger.info(f"Discovered {len(self.processor_registry)} processors")

        except Exception as e:
            logger.error(f"Error discovering processors: {str(e)}")

    def build_pipeline(self, pipeline_name: Optional[str] = None) -> List[Tuple[str, Any]]:
        pipeline = []
        config_root = self.pipeline_config.get('pipelines') or self.pipeline_config
        if pipeline_name and pipeline_name in config_root:
            raw = config_root[pipeline_name]
            processors_list = raw.get('processors') if isinstance(raw, dict) else raw
            if isinstance(processors_list, list):
                logger.info(f"Building pipeline '{pipeline_name}' with {len(processors_list)} processors")
                for processor_config in processors_list:
                    proc_id = processor_config.get('id') if isinstance(processor_config, dict) else None
                    params = processor_config.get('params', {}) if isinstance(processor_config, dict) else {}
                    if not proc_id:
                        continue
                    if proc_id in self.processor_registry:
                        cls = self.processor_registry[proc_id]
                        instance = cls(**params)
                        pipeline.append((proc_id, instance))
                    else:
                        logger.warning(f"Processor '{proc_id}' not found in registry")
        if not pipeline:
            core_default = [
                ("pdf_processor.PDFProcessor", {}),
                ("image_ingestion.ImageIngestion", {}),
                ("advanced_ocr.AdvancedOCRProcessor", {}),
                ("table_detector.TableDetector", {}),
                ("exporter.Exporter", {}),
            ]
            logger.info("Building default core pipeline")
            for proc_id, params in core_default:
                if proc_id in self.processor_registry:
                    cls = self.processor_registry[proc_id]
                    instance = cls(**params)
                    pipeline.append((proc_id, instance))
                else:
                    logger.warning(f"Default processor '{proc_id}' not found in registry")
        logger.info(f"Built pipeline with {len(pipeline)} processors")
        return pipeline

    def list_available_processors(self) -> Dict[str, str]:
        processors = {}
        for processor_id, processor_class in self.processor_registry.items():
            description = inspect.getdoc(processor_class) or ""
            description = description.split('\n')[0]
            processors[processor_id] = description
        return processors

    def get_processor_details(self, processor_id: str) -> Dict[str, Any]:
        if processor_id not in self.processor_registry:
            logger.warning(f"Processor '{processor_id}' not found in registry")
            return {}
        processor_class = self.processor_registry[processor_id]
        description = inspect.getdoc(processor_class) or ""
        signature = inspect.signature(processor_class.__init__)
        parameters = {}
        for name, param in signature.parameters.items():
            if name == 'self':
                continue
            param_info = {
                'required': param.default is inspect.Parameter.empty,
                'default': None if param.default is inspect.Parameter.empty else param.default,
                'annotation': str(param.annotation) if param.annotation is not inspect.Parameter.empty else 'Any'
            }
            parameters[name] = param_info
        methods = {}
        process_method = getattr(processor_class, 'process', None)
        if process_method:
            process_signature = inspect.signature(process_method)
            methods['process'] = {
                'description': inspect.getdoc(process_method) or "",
                'parameters': {},
                'returns': str(process_signature.return_annotation) if process_signature.return_annotation is not inspect.Parameter.empty else 'Any'
            }
            for name, param in process_signature.parameters.items():
                if name == 'self':
                    continue
                param_info = {
                    'required': param.default is inspect.Parameter.empty,
                    'default': None if param.default is inspect.Parameter.empty else param.default,
                    'annotation': str(param.annotation) if param.annotation is not inspect.Parameter.empty else 'Any'
                }
                methods['process']['parameters'][name] = param_info
        return {
            'id': processor_id,
            'description': description,
            'parameters': parameters,
            'methods': methods
        }
