import sys
import logging
from pathlib import Path

from utils.logger import setup_logger, get_logger
from src.agentic_pipeline import AgenticAIPipeline

# Setup logging
setup_logger()
logger = get_logger(__name__)


def main():
    """Main function to run the agentic AI pipeline."""
    try:
        # Initialize and run the pipeline
        logger.info("Initializing agentic AI pipeline...")
        pipeline = AgenticAIPipeline()
        
        # Execute the complete pipeline
        logger.info("Executing complete pipeline...")
        result = pipeline.run_prepare_data_pipeline()
        
    except FileNotFoundError as e:
        logger.error(f"File not found error: {str(e)}", exc_info=True)
        return 1
    except ValueError as e:
        logger.error(f"Value error: {str(e)}", exc_info=True)
        return 1
    except Exception as e:
        logger.error(f"Unexpected error occurred: {str(e)}", exc_info=True)
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

