"""
Enhanced AI Trainer with Optuna optimization and knowledge distillation.
Trains multiple model versions and manages model evolution.
"""

import logging
import random
import datetime
from typing import List, Dict, Any, Tuple
from utils.logger import get_logger
from utils.models import SampleInfo, TrainedModel, ModelMetrics
import optuna
from rfdetr import RFDETRMedium
from rfdetr.datasets.aug_config import (
    AUG_CONSERVATIVE,
    AUG_AGGRESSIVE,
    AUG_AERIAL,
    AUG_INDUSTRIAL
)
logger = get_logger(__name__)


class AITrainer:
    """Train models with Optuna hyperparameter optimization and knowledge distillation."""
    
    def __init__(self, config: dict):
        logger.info("Initializing AI Trainer")
        self.config = config
    
    def objective(self, trial: optuna.trial.Trial):
        """Objective function for Optuna hyperparameter optimization."""
        
        # output
        output_dir = f"tmp/rfdetr_tunning/trial_{trial.number}"
        os.makedirs(output_dir, exist_ok=True)  

        # Augmentations
        aug_options = {
            "none": {},
            'industrial': AUG_INDUSTRIAL,
            'aggressive': AUG_AGGRESSIVE,
            'custom': {}
        }
        
        # hyperparameter optimization 
        optim_config = {
            "aug_config": trial.suggest_categorical("augmentations", list(aug_options.keys())),
            "lr": trial.suggest_float("lr", 1e-5, 5e-4, log=True),
            "lr_encoder": trial.suggest_float("lr_encoder", 1e-6, 1e-4, log=True),
            "lr_vit_layer_decay": trial.suggest_float("lr_vit_layer_decay", 0.6, 0.95),
            "lr_component_decay": trial.suggest_float("lr_component_decay", 0.5, 0.9),
            "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True),
            "warmup_epochs": trial.suggest_int("warmup_epochs", 0, 5),
            "ema_decay": trial.suggest_float("ema_decay", 0.99, 0.9999, log=True),
            "lr_scheduler": trial.suggest_categorical("lr_scheduler", ["step", "cosine"]),
            "cls_loss_coef": trial.suggest_float("cls_loss_coef", 0.5, 3.0),
            "bbox_loss_coef": trial.suggest_float("bbox_loss_coef", 2.0, 8.0),
            "giou_loss_coef": trial.suggest_float("giou_loss_coef", 1.0, 4.0),
            "pretrained_weights": trial.suggest_categorical("pretrained_weights", [
                self.config['pretrained_weights'],
                "",
            ])
        }
        
        map50, recall, f1_score = self.train_with_config(optim_config)
        
        return map50, recall, f1_score
    
    def train_with_config(self, config: dict):
        """Train model with given configuration."""
        logger.info("Training model with given configuration")
        # Build model
        model = RFDETRMedium(
            **self.config,
            dataset_dir=self.config['dataset_path'],
        )

    def train_with_optuna(self):
        """Train model with Optuna hyperparameter optimization."""
        logger.info("Training model with Optuna hyperparameter optimization")
        try:
            study = optuna.create_study(
                study_name=f"rfdetr-tuning-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                directions=["maximize", "maximize", "maximize"]
            )
            logger.info(f"Starting hyperparameter optimization with {self.config['n_trials']} trials")
            study.optimize(self.objective, n_trials=self.config["n_trials"])
            for t in study.best_trials:
                logger.info(f"Trial {t.number}: MAP50 = {t.values[0]}, Recall = {t.values[1]}, F1 = {t.values[2]} with params {t.params}")
        except Exception as e:
            logger.error(f"Error training with Optuna: {str(e)}", exc_info=True)
            raise
        
        

    