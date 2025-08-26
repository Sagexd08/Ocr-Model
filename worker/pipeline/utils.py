from typing import Dict, Any
import importlib
import inspect
import pkgutil
from pathlib import Path

def get_available_processors() -> Dict[str, Any]:
    """
    Discover and return all available document processors in the pipeline.
    
    Returns:
        Dictionary mapping processor names to processor classes
    """
    from .. import pipeline
    
    processor_classes = {}
    
    # Get the directory containing processors
    processors_dir = Path(pipeline.__file__).parent / "processors"
    
    # Iterate through Python modules in the processors directory
    for _, module_name, _ in pkgutil.iter_modules([str(processors_dir)]):
        # Skip __init__.py
        if module_name == "__init__":
            continue
            
        # Import the module
        module = importlib.import_module(f"worker.pipeline.processors.{module_name}")
        
        # Find processor classes in the module
        for name, obj in inspect.getmembers(module):
            # Look for classes that end with 'Processor'
            if (inspect.isclass(obj) and name.endswith("Processor") 
                    and obj.__module__ == module.__name__):
                processor_classes[name] = obj
    
    return processor_classes


def create_processor_pipeline(config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Create a pipeline of document processors based on configuration.
    
    Args:
        config: Pipeline configuration dictionary
        
    Returns:
        Dictionary of processor instances
    """
    config = config or {}
    processor_classes = get_available_processors()
    processors = {}
    
    # Create processor instances for each enabled processor
    for processor_name, processor_class in processor_classes.items():
        # Check if processor is enabled in config
        processor_config = config.get(processor_name.lower().replace("processor", ""), {})
        enabled = processor_config.get("enabled", True)
        
        if enabled:
            # Create processor instance with its config
            processors[processor_name] = processor_class(processor_config)
    
    return processors
