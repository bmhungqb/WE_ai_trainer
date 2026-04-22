"""
Agentic AI Pipeline orchestrator.
Coordinates the entire workflow from data processing to model deployment.
"""

import logging
import json 
from pathlib import Path
from typing import Dict, Any, List
from utils.logger import get_logger
from src.data_processor import DataProcessor
from src.ai_verify import AIVerify
from src.data_manager import DatasetManager
from utils.schemas import ImageRecord, VerifyResult, TrainedModel
from utils.gcs_utils import push_file_to_gcs, push_folder_to_gcs
from utils.label_studio_utils import pull_data_from_label_studio, push_new_samples_to_label_studio 
from src.ai_trainer import AITrainer
from utils.config import config as app_config

logger = get_logger(__name__)


class AgenticAIPipeline:
    """Main orchestrator for the agentic AI pipeline."""
    
    def __init__(self):
        logger.info("=" * 80)
        logger.info("INITIALIZING AGENTIC AI TEXTILE DEFECT DETECTION PIPELINE")
        logger.info("=" * 80)
        self.config = app_config
        self.data_processor = DataProcessor()
        self.ai_verify = AIVerify(self.config.data_pipeline)
        self.ai_trainer = AITrainer(app_config.ai_trainer_configs)
        self.data_manager = DatasetManager(app_config.training_pipeline.get('data_merge_config'))
        logger.info("Pipeline components initialized successfully")
        
    def run_prepare_data_pipeline(self) -> Dict[str, Any]:
        '''Prepare new data for human review.'''
        logger.info("\n" + "=" * 80)
        logger.info("PREPARING NEW DATA")
        logger.info("=" * 80)
        try:
            # pull new data from gcs
            sample_records = self.data_processor.download_data_from_gcs(self.config.data_pipeline['gcs_buckets'])
            # Using larger AI models to verify and voting mechanism.
            verified_records = self.ai_verify.predict_with_models(sample_records)
            # Push new data to label studio for human review.
            json_tmp_file_path = self.data_processor.get_label_studio_format_json(verified_records)
            push_file_to_gcs(json_tmp_file_path, self.config.data_pipeline['output_configs']['gcs_destination'])
            push_new_samples_to_label_studio(url=self.config.LABEL_STUDIO_URL, api_key=self.config.LABEL_STUDIO_API_KEY, project_id=self.config.label_studio_configs['project_id'], path_to_new_samples_json=json_tmp_file_path)
            return {"status": "success"}
        except Exception as e:
            logger.error(f"Phase Prepare Data failed: {str(e)}", exc_info=True)
            return {"status": "failed", "error": str(e)}

    def run_training_pipeline(self) -> Dict[str, Any]:
        '''Run experiment to find best model'''
        logger.info("\n" + "=" * 80)
        logger.info("TRAINING PIPELINE")    
        logger.info("=" * 80)
        try:
            # Pull reviewed data from label studio
            date_cfg = self.config.pipeline_config.get('date', {})
            path_new_annotation_file = pull_data_from_label_studio(url=self.config.LABEL_STUDIO_URL, api_key=self.config.LABEL_STUDIO_API_KEY, start=date_cfg.get('start'), end=date_cfg.get('end'), project_id=self.config.label_studio_configs['project_id'])
            path_old_annotation_file = pull_data_from_label_studio(url=self.config.LABEL_STUDIO_URL, api_key=self.config.LABEL_STUDIO_API_KEY, start=date_cfg.get('start'), end=date_cfg.get('end'), project_id=self.config.label_studio_configs['project_id'], is_pull_old_dataset=True)
            # Mix new data and old dataset
            dataset_path = self.data_manager.merge_old_new_data(path_old_annotation_file, path_new_annotation_file)
            # Optuna hyperparameter optimization
            self.ai_trainer.train_with_optuna(dataset_path=dataset_path)
            # push full output to gcs
            push_folder_to_gcs(Path("tmp/rfdetr_tuning"), self.config.training_pipeline.get('output_gcs_models'))
            return {"status": "success"}
        except Exception as e:
            logger.error(f"Phase Training failed: {str(e)}", exc_info=True)
            return {"status": "failed", "error": str(e)}
    
    def run_complete_pipeline(self) -> Dict[str, Any]:
        """Run the complete agentic AI pipeline."""
        logger.info("\n" + "=" * 80)
        logger.info("STARTING COMPLETE AGENTIC AI PIPELINE")
        logger.info("=" * 80)
        
        prep_res = self.run_prepare_data_pipeline()
        if prep_res.get("status") != "success":
            return prep_res
            
        train_res = self.run_training_pipeline()
        if train_res.get("status") != "success":
            return train_res
            
        return {"status": "success"}