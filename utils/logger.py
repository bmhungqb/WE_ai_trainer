import logging
import logging.handlers
import sys
from pathlib import Path
from datetime import datetime

def setup_logger(name: str = "TextileDefectDetection", log_dir: str = "./logs") -> logging.Logger:
    """
    Configure and return a logger with both file and console handlers.
    
    Args:
        name: Logger name
        log_dir: Directory to store log files
        
    Returns:
        Configured logger instance
    """
    # Create logs directory if it doesn't exist
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    # Remove existing handlers to prevent duplicates
    logger.handlers.clear()
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    simple_formatter = logging.Formatter(
        fmt='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # File handler - logs everything
    log_file = log_path / f"textile_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    logger.addHandler(file_handler)
    
    # Console handler - logs INFO and above
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    logger.addHandler(console_handler)
    
    # Log initial setup message
    logger.info(f"Logger initialized. Log file: {log_file}")
    
    return logger


def get_logger(name: str = "TextileDefectDetection") -> logging.Logger:
    """Get or create a logger with the given name."""
    return logging.getLogger(name)
