"""
Main entry point for Agentic AI Textile Defect Detection system.
Orchestrates the complete pipeline from data processing to model evaluation.
"""

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
    logger.info("=" * 80)
    logger.info("AGENTIC AI TEXTILE DEFECT DETECTION SYSTEM")
    logger.info("=" * 80)
    
    try:
        # Initialize and run the pipeline
        logger.info("Initializing agentic AI pipeline...")
        pipeline = AgenticAIPipeline()
        
        # Execute the complete pipeline
        logger.info("Executing complete pipeline...")
        result = pipeline.run_prepare_data_pipeline()
        
        # Check result
        if result["status"] == "success":
            logger.info("=" * 80)
            logger.info("PIPELINE EXECUTION COMPLETED SUCCESSFULLY")
            logger.info("=" * 80)
            return 0
        else:
            logger.error(f"Pipeline execution failed: {result.get('error', 'Unknown error')}")
            logger.info("=" * 80)
            logger.info("PIPELINE EXECUTION FAILED")
            logger.info("=" * 80)
            return 1
        
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

