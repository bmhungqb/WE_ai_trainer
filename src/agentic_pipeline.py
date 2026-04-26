"""
Agentic AI Pipeline orchestrator.

Coordinates the entire workflow from data processing to model deployment:
1. Prepare Data Pipeline: Download data, run predictions, format for human review
2. Training Pipeline: Pull reviewed data, merge with old data, train models
3. Complete Pipeline: Run both pipelines in sequence
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from utils.logger import get_logger
from src.data_processor import DataProcessor
from src.ai_verify import AIVerify
from src.data_manager import DatasetManager
from utils.schemas import SampleInfo
from utils.gcs_utils import push_file_to_gcs, push_folder_to_gcs
from utils.label_studio_utils import pull_data_from_label_studio, push_new_samples_to_label_studio 
from src.ai_trainer import AITrainer
from utils.config import config as app_config

logger = get_logger(__name__)


class PipelinePhaseError(Exception):
    """Exception raised when a pipeline phase fails."""
    pass


class AgenticAIPipeline:
    """Main orchestrator for the complete agentic AI pipeline.
    
    Manages three main phases:
    1. Data Preparation: Data download, verification, and formatting
    2. Model Training: Data merging, training, and evaluation
    3. Complete Pipeline: Orchestration of all phases
    
    Returns results as dictionaries with status and error information.
    """
    
    def __init__(self):
        """Initialize pipeline components.
        
        Raises:
            Exception: If configuration is invalid or components fail to initialize
        """
        logger.info("=" * 80)
        logger.info("INITIALIZING AGENTIC AI TEXTILE DEFECT DETECTION PIPELINE")
        logger.info("=" * 80)
        try:
            self.config = app_config
            self.data_processor = DataProcessor()
            self.ai_verify = None
            self.ai_trainer = None
            self.data_manager = DatasetManager(app_config.training_pipeline.get('data_merge_config'))
            logger.info("Pipeline components initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize pipeline components: {str(e)}", exc_info=True)
            raise
        
    def run_prepare_data_pipeline(self) -> Dict[str, Any]:
        """Prepare new data for human review in Label Studio.
        
        Pipeline steps:
        1. Download data from GCS
        2. Run AI verification (predictions)
        3. Format predictions as Label Studio pre-annotations
        4. Push to GCS and Label Studio
        
        Returns:
            Dict with status ('success' or 'failed') and error message if failed
        """
        phase_name = "Prepare Data"
        logger.info(f"\n{'=' * 80}\nPHASE: {phase_name}\n{'=' * 80}")
        
        try:
            # Step 1: Download data from GCS
            logger.info("Step 1: Downloading data from GCS...")
            sample_records = self.data_processor.download_data_from_gcs(
                self.config.data_pipeline['gcs_buckets'],
                start_date_str=self.config.date.get('start'),
                end_date_str=self.config.date.get('end')
            )
            logger.info(f"Downloaded {len(sample_records)} samples")
            
            if not sample_records:
                raise PipelinePhaseError("No samples downloaded from GCS")
            
            # Step 2: Run AI verification
            logger.info("Step 2: Running AI verification...")
            output_save_sliced_images_path = self._build_sliced_images_output_local_path()
            output_sliced_images_gcs_path = f"{self.config.data_pipeline['output_configs']['gcs_bucket_name']}/{self.config.data_pipeline['output_configs']['gcs_sliced_images_folder_path']}"
            self.ai_verify = AIVerify(self.config.data_pipeline['verify_configs'])
            verified_records = self.ai_verify.predict_with_models(
                sample_records, 
                output_save_sliced_images_path,
                output_sliced_images_gcs_path
            )
            logger.info(f"Verified {len(verified_records)} records")
            
            # Step 3: Push sliced images to GCS
            logger.info("Step 3: Pushing sliced images to GCS...")
            status, msg = push_folder_to_gcs(
                output_save_sliced_images_path, 
                output_sliced_images_gcs_path
            )
            if not status:
                raise PipelinePhaseError(f"Failed to push sliced images: {msg}")
            
            # Step 4: Format for Label Studio
            logger.info("Step 4: Formatting predictions for Label Studio...")
            json_tmp_local_path = Path("tmp") / f"tasks_{self.config.date.get('start')}_{self.config.date.get('end')}.json"
            self.data_processor.get_label_studio_format_json(verified_records, json_tmp_local_path)
            
            # Step 5: Push JSON to GCS
            logger.info("Step 5: Pushing annotation JSON to GCS...")
            status, msg = push_file_to_gcs(
                json_tmp_local_path, 
                gcs_name = self.config.data_pipeline['output_configs']['gcs_bucket_name'],
                gcs_destination = self.config.data_pipeline['output_configs']['gcs_tasks_folder_path']
            )
            if not status:
                raise PipelinePhaseError(f"Failed to push annotations to GCS: {msg}")
            
            # Step 6: Push to Label Studio
            logger.info("Step 6: Pushing samples to Label Studio...")
            status, msg = push_new_samples_to_label_studio(
                url=self.config.LABEL_STUDIO_URL, 
                api_key=self.config.LABEL_STUDIO_API_KEY, 
                project_id=self.config.label_studio_configs['project_id'], 
                path_to_new_samples_json=json_tmp_local_path
            )
            if not status:
                raise PipelinePhaseError(f"Failed to push to Label Studio: {msg}")
            
            logger.info(f"✓ {phase_name} completed successfully")
            return {"status": "success"}
            
        except PipelinePhaseError as e:
            logger.error(f"✗ {phase_name} failed: {str(e)}")
            return {"status": "failed", "error": str(e)}
        except Exception as e:
            logger.error(f"✗ {phase_name} failed with unexpected error: {str(e)}", exc_info=True)
            return {"status": "failed", "error": f"Unexpected error: {str(e)}"}

    def _build_sliced_images_output_local_path(self) -> Path:
        """Build the output path for sliced images directory.
        
        Returns:
            Path object for sliced images directory
        """
        date_start = self.config.date['start']
        date_end = self.config.date['end']
        img_size = self.config.data_pipeline['verify_configs']['image_size']
        size_str = f"{img_size[0]}x{img_size[1]}"
        
        return Path(f"./tmp/slice_images_{date_start}_{date_end}_{size_str}")

    def run_training_pipeline(self) -> Dict[str, Any]:
        """Run training pipeline with data merging and model optimization.
        
        Pipeline steps:
        1. Pull reviewed data from Label Studio (new data)
        2. Pull historical data (old data)
        3. Merge datasets with configured ratios
        4. Train models with Optuna hyperparameter optimization
        5. Push results to GCS
        
        Returns:
            Dict with status ('success' or 'failed') and error message if failed
        """
        phase_name = "Training"
        logger.info(f"\n{'=' * 80}\nPHASE: {phase_name}\n{'=' * 80}")
        
        try:
            # Step 1: Pull reviewed data from Label Studio
            logger.info("Step 1: Pulling reviewed data from Label Studio...")
            path_new_annotation_file = pull_data_from_label_studio(
                url=self.config.LABEL_STUDIO_URL, 
                api_key=self.config.LABEL_STUDIO_API_KEY, 
                start=self.config.date.get('start'), 
                end=self.config.date.get('end'), 
                project_id=self.config.label_studio_configs['project_id']
            )
            
            logger.info("Step 2: Pulling historical data from Label Studio...")
            path_old_annotation_file = pull_data_from_label_studio(
                url=self.config.LABEL_STUDIO_URL, 
                api_key=self.config.LABEL_STUDIO_API_KEY, 
                start=self.config.date.get('start'), 
                end=self.config.date.get('end'), 
                project_id=self.config.label_studio_configs['project_id'], 
                is_pull_old_dataset=True
            )
            
            # Step 3: Merge datasets
            logger.info("Step 3: Merging old and new datasets...")
            dataset_path = self.data_manager.merge_old_new_data(
                path_old_annotation_file, 
                path_new_annotation_file
            )
            logger.info(f"Merged dataset saved to {dataset_path}")
            
            # Step 4: Train models with Optuna hyperparameter optimization
            logger.info("Step 4: Training models with Optuna optimization...")
            self.ai_trainer = AITrainer(self.config.ai_trainer_configs)
            self.ai_trainer.train_with_optuna(dataset_path=dataset_path)
            
            # Step 5: Push results to GCS
            logger.info("Step 5: Pushing training results to GCS...")
            status, msg = push_folder_to_gcs(
                Path("tmp/rfdetr_tuning"), 
                self.config.training_pipeline.get('output_gcs_models')
            )
            if not status:
                raise PipelinePhaseError(f"Failed to push results to GCS: {msg}")
            
            logger.info(f"✓ {phase_name} completed successfully")
            return {"status": "success"}
            
        except PipelinePhaseError as e:
            logger.error(f"✗ {phase_name} failed: {str(e)}")
            return {"status": "failed", "error": str(e)}
        except Exception as e:
            logger.error(f"✗ {phase_name} failed with unexpected error: {str(e)}", exc_info=True)
            return {"status": "failed", "error": f"Unexpected error: {str(e)}"}
    
    def run_complete_pipeline(self) -> Dict[str, Any]:
        """Run the complete agentic AI pipeline end-to-end.
        
        Orchestrates both prepare and training pipelines in sequence.
        Stops if any phase fails.
        
        Returns:
            Dict with overall status ('success' or 'failed')
        """
        logger.info("\n" + "=" * 80)
        logger.info("STARTING COMPLETE AGENTIC AI PIPELINE")
        logger.info("=" * 80)
        
        try:
            # Phase 1: Prepare data
            logger.info("\n[1/2] Starting data preparation phase...")
            prep_res = self.run_prepare_data_pipeline()
            if prep_res.get("status") != "success":
                logger.error(f"Data preparation failed: {prep_res.get('error')}")
                return prep_res
            logger.info("✓ Data preparation completed")
            
            # Phase 2: Training
            logger.info("\n[2/2] Starting training phase...")
            train_res = self.run_training_pipeline()
            if train_res.get("status") != "success":
                logger.error(f"Training failed: {train_res.get('error')}")
                return train_res
            logger.info("✓ Training completed")
            
            logger.info("\n" + "=" * 80)
            logger.info("✓ COMPLETE PIPELINE EXECUTED SUCCESSFULLY")
            logger.info("=" * 80)
            return {"status": "success"}
            
        except Exception as e:
            logger.error(f"Complete pipeline failed: {str(e)}", exc_info=True)
            return {"status": "failed", "error": str(e)}