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
from utils.ai_trainer import EnhancedAITrainer
from src.dataset_manager import DatasetManager
from utils.models import ImageRecord, VerifyResult, TrainedModel
from utils.gcs_utils import push_file_to_gcs
from utils.label_studio_utils import pull_from_label_studio 
from src.ai_trainer import AITrainer
logger = get_logger(__name__)


class AgenticAIPipeline:
    """Main orchestrator for the agentic AI pipeline."""
    
    def __init__(self, config_path: str = "./configs/pipeline_config.json"):
        logger.info("=" * 80)
        logger.info("INITIALIZING AGENTIC AI TEXTILE DEFECT DETECTION PIPELINE")
        logger.info("=" * 80)
        self._load_config(config_path)
        self.data_processor = DataProcessor()
        self.ai_verify = AIVerify(self.config)
        self.ai_trainer = AITrainer(self.config['training_pipeline']['ai_trainer_configs'])
        self.data_manager = DatasetManager(self.config['data_management'])
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
        
    def run_prepare_data_pipeline(self) -> bool:
        '''Prepare new data for human review.'''
        logger.info("\n" + "=" * 80)
        logger.info("PREPARING NEW DATA")
        logger.info("=" * 80)
        try:
            # pull new data from gcs
            sample_records = self.data_processor.download_data_from_gcs(self.config["data_pipeline"]['gcs_buckets'])
            # Using larger AI models to verify and voting mechanism.
            verified_records = self.ai_verify.predict_with_models(sample_records)
            # Push new data to label studio for human review.
            json_tmp_file_path = self.data_processor.get_label_studio_format_json(verified_records)
            push_file_to_gcs(json_tmp_file_path, self.config["data_pipeline"]['output_configs']['gcs_destination'])
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
            path_new_annotation_file = pull_from_label_studio()
            path_old_annotation_file = pull_from_label_studio(is_pull_old_dataset=True)
            # Mix new data and old dataset.
            self.data_manager.merge_old_new_data(path_old_annotation_file, path_new_annotation_file)
            # Optuna hyperparameter optimization.
            self.ai_trainer.train_with_optuna()
        except Exception as e:
            logger.error(f"Phase Training failed: {str(e)}", exc_info=True)
            raise
    
    def run_complete_pipeline(self) -> Dict[str, Any]:
        """Run the complete agentic AI pipeline."""
        logger.info("\n" + "=" * 80)
        logger.info("STARTING COMPLETE AGENTIC AI PIPELINE")
        logger.info("=" * 80)