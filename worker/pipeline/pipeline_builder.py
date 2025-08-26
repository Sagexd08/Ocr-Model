from typing import Dict, Any, List, Optional, Tuple, Union
import yaml
import importlib
import inspect
import os
from pathlib import Path

from ...utils.logging import get_logger

logger = get_logger(__name__)

class PipelineBuilder:
    """
    Dynamic pipeline builder that discovers and configures processors
    from configuration files or programmatically.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self.processor_registry = {}
        self.pipeline_config = {}
        
        # Load configuration if path is provided
        if config_path:
            self._load_config(config_path)
            
        # Discover and register available processors
        self._discover_processors()
    
    def _load_config(self, config_path: str):
        """Load pipeline configuration from a YAML file"""
        try:
            with open(config_path, 'r') as f:
                self.pipeline_config = yaml.safe_load(f)
            logger.info(f"Loaded pipeline configuration from {config_path}")
        except Exception as e:
            logger.error(f"Error loading pipeline configuration: {str(e)}")
            self.pipeline_config = {}
    
    def _discover_processors(self):
        """Discover available processors in the processors package"""
        try:
            # Get the directory where processors are located
            processors_path = Path(__file__).parent.parent / "processors"
            
            if not processors_path.exists():
                logger.warning(f"Processors directory not found at {processors_path}")
                return
                
            # Find Python files (potential processors)
            processor_files = [f for f in processors_path.glob("*.py") 
                             if f.is_file() and not f.name.startswith("__")]
            
            for file_path in processor_files:
                module_name = file_path.stem
                
                try:
                    # Import the module
                    module = importlib.import_module(f"...pipeline.processors.{module_name}", package=__package__)
                    
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
            
            logger.info(f"Discovered {len(self.processor_registry)} processors")
            
        except Exception as e:
            logger.error(f"Error discovering processors: {str(e)}")
    
    def build_pipeline(self, pipeline_name: Optional[str] = None) -> List[Tuple[str, Any]]:
        """
        Build a processing pipeline based on configuration or default settings
        
        Args:
            pipeline_name: Optional name of the pipeline configuration to use
            
        Returns:
            List of tuples with (processor_name, processor_instance)
        """
        pipeline = []
        
        # Use configuration if provided
        if pipeline_name and pipeline_name in self.pipeline_config.get('pipelines', {}):
            pipeline_config = self.pipeline_config['pipelines'][pipeline_name]
            logger.info(f"Building pipeline '{pipeline_name}' with {len(pipeline_config)} processors")
            
            for processor_config in pipeline_config:
                processor_id = processor_config['id']
                processor_params = processor_config.get('params', {})
                
                if processor_id in self.processor_registry:
                    processor_class = self.processor_registry[processor_id]
                    processor_instance = processor_class(**processor_params)
                    pipeline.append((processor_id, processor_instance))
                else:
                    logger.warning(f"Processor '{processor_id}' not found in registry")
        
        # Use default pipeline if no configuration is provided
        elif not pipeline:
            logger.info("Building default pipeline")
            
            # Define default processor order and configuration
            default_pipeline = [
                # Basic image processing
                ("image_enhancer.ImageEnhancer", {}),
                
                # Classification
                ("classifier.DocumentClassifier", {}),
                
                # Layout analysis
                ("layout.LayoutAnalyzer", {}),
                
                # Table extraction
                ("tables.TableExtractor", {}),
                
                # Text processing
                ("text_processor.TextProcessor", {}),
                
                # QA and validation
                ("qa.QualityAnalyzer", {}),
                
                # Post-processing
                ("postprocessing.PostProcessor", {})
            ]
            
            # Instantiate processors for default pipeline
            for processor_id, params in default_pipeline:
                if processor_id in self.processor_registry:
                    processor_class = self.processor_registry[processor_id]
                    processor_instance = processor_class(**params)
                    pipeline.append((processor_id, processor_instance))
                else:
                    logger.warning(f"Default processor '{processor_id}' not found in registry")
        
        logger.info(f"Built pipeline with {len(pipeline)} processors")
        return pipeline
    
    def list_available_processors(self) -> Dict[str, str]:
        """
        List all available processors with their descriptions
        
        Returns:
            Dictionary with processor_id -> description
        """
        processors = {}
        
        for processor_id, processor_class in self.processor_registry.items():
            # Get the class docstring as description
            description = inspect.getdoc(processor_class) or "No description available"
            # Take just the first line for brevity
            description = description.split('\n')[0]
            processors[processor_id] = description
        
        return processors
    
    def get_processor_details(self, processor_id: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific processor
        
        Args:
            processor_id: ID of the processor to get details for
            
        Returns:
            Dictionary with processor details
        """
        if processor_id not in self.processor_registry:
            logger.warning(f"Processor '{processor_id}' not found in registry")
            return {}
            
        processor_class = self.processor_registry[processor_id]
        
        # Get processor documentation
        description = inspect.getdoc(processor_class) or "No description available"
        
        # Get constructor parameters
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
        
        # Get methods (especially the process method)
        methods = {}
        process_method = getattr(processor_class, 'process', None)
        if process_method:
            process_signature = inspect.signature(process_method)
            methods['process'] = {
                'description': inspect.getdoc(process_method) or "No description available",
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
