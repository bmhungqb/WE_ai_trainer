"""
Dataset manager for handling mixed datasets.
Manages new vs old data mixing for training.
"""

import logging
import random
from typing import Dict, Any, Tuple, List
from utils.logger import get_logger
from utils.models import ImageRecord, DatasetConfig

logger = get_logger(__name__)


class DatasetManager:
    """Manage dataset composition and versions."""
    
    def __init__(self, config: DatasetConfig = None):
        logger.info("Initializing DatasetManager")
        self.config = config or DatasetConfig()
        logger.info(f"Dataset ratio - New: {self.config.new_data_ratio*100:.1f}%, Old: {self.config.old_data_ratio*100:.1f}%")
    
    def split_new_old_data(self, new_records: List[ImageRecord], old_records: List[ImageRecord]) -> Tuple[List[ImageRecord], List[ImageRecord]]:
        """Split dataset into new and old data based on ratios."""
        logger.info(f"Splitting dataset: {self.config.new_data_ratio*100:.1f}% new, {self.config.old_data_ratio*100:.1f}% old")
        try:
            # We want to select a certain amount based on ratios
            # Assuming we take all new_records up to a certain limit, or scale old_records to match the ratio
            new_data_count = len(new_records)
            desired_old_count = int(new_data_count * (self.config.old_data_ratio / self.config.new_data_ratio))
            
            # Smart sampling: instead of random, we could prioritize hard examples in old_records
            # Here we simulate this by picking random for now, but leaving a hook for smart sampling
            if self.config.balance_classes:
                logger.info("Applying class balancing to old data selection")
                # Group by defect type
                grouped = {}
                for r in old_records:
                    defect = r.annotations[0].defect_type if r.annotations else "normal"
                    if defect not in grouped:
                        grouped[defect] = []
                    grouped[defect].append(r)
                
                # Sample evenly from groups
                selected_old = []
                samples_per_class = max(1, desired_old_count // len(grouped)) if grouped else 0
                for defect_type, records in grouped.items():
                    sampled = random.sample(records, min(len(records), samples_per_class))
                    selected_old.extend(sampled)
            else:
                selected_old = random.sample(old_records, min(len(old_records), desired_old_count))
            
            logger.info(f"Data split: {len(new_records)} new samples, {len(selected_old)} old samples")
            return new_records, selected_old
        except Exception as e:
            logger.error(f"Error splitting dataset: {str(e)}", exc_info=True)
            raise
    
    def create_mixed_dataset(self, new_data: List[ImageRecord], old_data: List[ImageRecord]) -> Dict[str, Any]:
        """Create mixed dataset from new and old data."""
        logger.info("Creating mixed dataset from new and old data")
        try:
            combined = new_data + old_data
            random.shuffle(combined)
            
            mixed_dataset = {
                "records": combined,
                "total_samples": len(combined),
                "composition": f"{self.config.new_data_ratio*100:.1f}% new, {self.config.old_data_ratio*100:.1f}% old"
            }
            logger.info(f"Mixed dataset created: {mixed_dataset['total_samples']} total samples")
            return mixed_dataset
        except Exception as e:
            logger.error(f"Error creating mixed dataset: {str(e)}", exc_info=True)
            raise
    
    def get_train_val_split(self, mixed_dataset: Dict[str, Any]) -> Tuple[List[ImageRecord], List[ImageRecord]]:
        """Split mixed dataset into train and validation sets."""
        logger.info(f"Splitting mixed dataset: {self.config.train_ratio*100:.1f}% train, {(1-self.config.train_ratio)*100:.1f}% val")
        try:
            records = mixed_dataset["records"]
            total_samples = mixed_dataset["total_samples"]
            train_count = int(total_samples * self.config.train_ratio)
            
            train_set = records[:train_count]
            val_set = records[train_count:]
            
            logger.info(f"Train/Val split: {len(train_set)} training, {len(val_set)} validation")
            return train_set, val_set
        except Exception as e:
            logger.error(f"Error splitting train/val: {str(e)}", exc_info=True)
            raise
