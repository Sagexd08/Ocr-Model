import logging
import json
import os
import sys
import time
import contextlib
from typing import Dict, Any, Optional, Union, Callable
from functools import wraps

# Store loggers to avoid creating multiple instances
_loggers: Dict[str, logging.Logger] = {}

def get_logger(name: str, level: Optional[int] = None) -> logging.Logger:
    """
    Get a configured logger with consistent formatting.
    
    Args:
        name: Logger name (typically __name__ of the calling module)
        level: Optional logging level override
        
    Returns:
        Configured logger instance
    """
    if name in _loggers:
        return _loggers[name]
        
    logger = logging.getLogger(name)
    
    # Use env var or default to INFO
    if level is None:
        level = getattr(logging, os.environ.get("CURIOSCAN_LOG_LEVEL", "INFO"))
    
    logger.setLevel(level)
    
    # Skip if handlers already configured
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        
        # Use JSON formatting in production
        if os.environ.get("CURIOSCAN_ENV") == "production":
            formatter = JsonFormatter()
        else:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    # Cache logger instance    
    _loggers[name] = logger
    
    return logger

def log_execution_time(func=None, *, logger_name: Optional[str] = None, level: int = logging.DEBUG):
    """
    Decorator to log execution time of a function.
    
    Args:
        func: Function to decorate
        logger_name: Logger name to use, defaults to function's module
        level: Logging level for the execution time message
    
    Returns:
        Decorated function
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get logger
            _logger_name = logger_name or func.__module__
            logger = get_logger(_logger_name)
            
            # Log start
            start_time = time.time()
            logger.log(level, f"Starting {func.__name__}")
            
            # Execute function
            result = func(*args, **kwargs)
            
            # Log end with execution time
            end_time = time.time()
            execution_time = end_time - start_time
            logger.log(level, f"Finished {func.__name__} in {execution_time:.4f} seconds")
            
            return result
        
        return wrapper
    
    # Allow using as @log_execution_time or @log_execution_time(logger_name="x")
    if func is None:
        return decorator
    return decorator(func)

@contextlib.contextmanager
def log_context(message: str, logger: Union[logging.Logger, str], 
                level: int = logging.INFO, include_execution_time: bool = True):
    """
    Context manager to log execution time of a block of code.
    
    Args:
        message: Message to log
        logger: Logger instance or name
        level: Logging level
        include_execution_time: Whether to include execution time in the log
        
    Yields:
        None
    """
    # Get logger instance if string provided
    if isinstance(logger, str):
        logger = get_logger(logger)
    
    # Log start
    start_time = time.time()
    logger.log(level, f"Start: {message}")
    
    try:
        # Execute the context block
        yield
        
        # Log end with execution time
        if include_execution_time:
            end_time = time.time()
            execution_time = end_time - start_time
            logger.log(level, f"End: {message} (took {execution_time:.4f} seconds)")
        else:
            logger.log(level, f"End: {message}")
            
    except Exception as e:
        # Log exception
        logger.error(f"Error in {message}: {str(e)}", exc_info=True)
        raise

class JsonFormatter(logging.Formatter):
    """
    Format logs as JSON for structured logging and observability.
    Includes standard fields: timestamp, level, name, message
    Also includes any extra fields provided in the log record.
    """
    def format(self, record: logging.LogRecord) -> str:
        log_data: Dict[str, Any] = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage()
        }
        
        # Add any extra attributes
        for key, value in record.__dict__.items():
            if key not in {
                "args", "asctime", "created", "exc_info", "exc_text", "filename",
                "funcName", "id", "levelname", "levelno", "lineno", "module",
                "msecs", "message", "msg", "name", "pathname", "process",
                "processName", "relativeCreated", "stack_info", "thread", "threadName"
            }:
                log_data[key] = value
                
        # Add job context if available
        if hasattr(record, "job_id"):
            log_data["job_id"] = record.job_id
        if hasattr(record, "stage"):
            log_data["stage"] = record.stage
        if hasattr(record, "file_id"):
            log_data["file_id"] = record.file_id
        if hasattr(record, "page"):
            log_data["page"] = record.page
        if hasattr(record, "region_id"):
            log_data["region_id"] = record.region_id
            
        return json.dumps(log_data)

# Configure root logger with basic settings
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
