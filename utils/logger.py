import logging
import logging.handlers
import sys
from pathlib import Path

def setup_logger(log_dir: str = "./logs") -> logging.Logger:
    """
    Configure the ROOT logger with both file and console handlers.
    This ensures ALL loggers throughout the application use the same handlers.
    
    Args:
        log_dir: Directory to store log files
        
    Returns:
        Configured root logger instance
    """
    # Create logs directory if it doesn't exist
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Configure ROOT logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    
    # Check if handlers already exist (prevents duplicate handlers)
    if root_logger.handlers:
        return root_logger
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    simple_formatter = logging.Formatter(
        fmt='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # File handler - logs everything to a SINGLE file (no timestamp)
    log_file = log_path / "textile_detection.log"
    file_handler = logging.FileHandler(log_file, mode='a')  # 'a' for append mode
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    root_logger.addHandler(file_handler)
    
    # Console handler - logs INFO and above
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    root_logger.addHandler(console_handler)
    
    # Log initial setup message
    root_logger.info(f"Logger initialized. Log file: {log_file}")
    
    return root_logger


def get_logger(name: str = None) -> logging.Logger:
    """
    Get a logger instance. If name is provided, returns a named logger
    (which will inherit handlers from root logger). Otherwise returns root logger.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)
