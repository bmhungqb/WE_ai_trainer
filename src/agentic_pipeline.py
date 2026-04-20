"""
Agentic AI Pipeline orchestrator.
Coordinates the entire workflow from data processing to model deployment.
"""

import logging
import pickle
import json 
from pathlib import Path
from typing import Dict, Any, List
from utils.logger import get_logger
from src.data_processor import DataProcessor
from src.ai_verify import AIVerify
from utils.evaluator import Evaluator
from utils.ai_trainer import EnhancedAITrainer
from utils.dataset_manager import DatasetManager
from utils.models import ImageRecord, VerifyResult, TrainedModel
from utils.gcs_utils import push_file_to_gcs

logger = get_logger(__name__)


class AgenticAIPipeline:
    """Main orchestrator for the agentic AI pipeline."""
    
    def __init__(self, config_path: str = "./configs/pipeline_config.json"):
        logger.info("=" * 80)
        logger.info("INITIALIZING AGENTIC AI TEXTILE DEFECT DETECTION PIPELINE")
        logger.info("=" * 80)
        self._load_config(config_path)
        self.data_processor = DataProcessor()
        self.ai_verify = AIVerify()
        self.evaluator = Evaluator()
        self.dataset_manager = DatasetManager()
        self.ai_trainer = EnhancedAITrainer(use_optuna=True, use_distillation=True)
        self.pipeline_state = {}
        logger.info("Pipeline components initialized successfully")
    
    def _load_config(self, config_path: str):
        """Load configuration from JSON file."""
        logger.info(f"Loading configuration from {config_path}")
        try:
            with open(config_path, "r") as f:
                self.config = json.load(f)
            logger.info("Configuration loaded successfully")
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}", exc_info=True)
            raise
        
    def run_prepare_data_pipeline(self, config: dict) -> bool:
        '''Prepare new data for human review.'''
        logger.info("\n" + "=" * 80)
        logger.info("PREPARING NEW DATA")
        logger.info("=" * 80)
        try:
            # pull new data from gcs
            sample_records = self.data_processor.download_data_from_gcs(config["data_pipeline"]['gcs_buckets'])
            # Using larger AI models to verify and voting mechanism.
            verified_records = self.ai_verify.predict_with_models(sample_records)
            # Push new data to label studio for human review.
            json_tmp_file_path = self.data_processor.get_label_studio_format_json(verified_records)

            push_file_to_gcs(json_tmp_file_path, config["data_pipeline"]['gcs_destination'])
        except Exception as e:
            logger.error(f"Phase Prepare Data failed: {str(e)}", exc_info=True)
            raise   
        
    def run_training_pipeline(self) -> bool:
        '''Run experiment to find best model'''
        logger.info("\n" + "=" * 80)
        logger.info("TRAINING PIPELINE")
        logger.info("=" * 80)
        try:
            # Pull reviewed data from label studio
            human_verified_data = self.data_processor.pull_from_label_studio()
            # Mix new data and old dataset.
            
            # Train multiple model versions
            
            # Evaluate models

            # Return best model based on defined criteria.
        except Exception as e:
            logger.error(f"Phase Training failed: {str(e)}", exc_info=True)
            raise
    
    def run_complete_pipeline(self) -> Dict[str, Any]:
        """Run the complete agentic AI pipeline."""
        logger.info("\n" + "=" * 80)
        logger.info("STARTING COMPLETE AGENTIC AI PIPELINE")
        logger.info("=" * 80)
        
        try:
            # Phase 1: Data Processing
            processed_data = self.phase_1_data_processing()
            
            # Phase 2: AI Verify
            verify_result = self.phase_2_ai_verify(processed_data)
            
            # Phase 3: Human Review
            verified_data = self.phase_3_human_review(verify_result)
            
            # Phase 4: Dataset Management
            dataset_result = self.phase_4_dataset_management(verified_data)
            
            # Phase 5: AI Training
            training_result = self.phase_5_ai_training(
                dataset_result["train_set"],
                dataset_result["val_set"]
            )
            
            # Phase 6: Evaluation
            evaluation_result = self.phase_6_evaluation(training_result["best_model"])
            
            # Summary
            logger.info("\n" + "=" * 80)
            logger.info("PIPELINE EXECUTION SUMMARY")
            logger.info("=" * 80)
            logger.info(f"✓ Phase 1: Data Processing - COMPLETED")
            logger.info(f"✓ Phase 2: AI Verify - COMPLETED")
            logger.info(f"✓ Phase 3: Human Review - COMPLETED")
            logger.info(f"✓ Phase 4: Dataset Management - COMPLETED")
            logger.info(f"✓ Phase 5: AI Training - COMPLETED")
            logger.info(f"✓ Phase 6: Evaluation - {'PASSED' if evaluation_result['criteria_met'] else 'NEEDS REVIEW'}")
            logger.info(f"\nBest Model: {evaluation_result['model_id']}")
            logger.info(f"Accuracy: {evaluation_result['metrics'].accuracy:.4f}")
            logger.info("=" * 80)
            
            return {
                "status": "success",
                "evaluation": evaluation_result
            }
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {str(e)}", exc_info=True)
            return {
                "status": "failed",
                "error": str(e)
            }
