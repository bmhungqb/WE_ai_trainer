"""
Enhanced AI Trainer with Optuna optimization and knowledge distillation.
Trains multiple model versions and manages model evolution.
"""

import logging
import random
import datetime
import os
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Tuple
from utils.logger import get_logger
from utils.schemas import SampleInfo, TrainedModel
import optuna
from rfdetr import RFDETRMedium
from rfdetr.datasets.aug_config import (
    AUG_CONSERVATIVE,
    AUG_AGGRESSIVE,
    AUG_AERIAL,
    AUG_INDUSTRIAL
)
from utils.config import config as app_config

logger = get_logger(__name__)


class AITrainer:
    """Train models with Optuna hyperparameter optimization and knowledge distillation."""
    
    def __init__(self, config: dict = None):
        logger.info("Initializing AI Trainer")
        self.config = config or app_config.ai_trainer_configs
        self.models = []
    
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
            "output_dir": output_dir,
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
        
        map50, recall, f1_score = self.train_with_config(optim_config, task_id=trial.number)
        
        return map50, recall, f1_score
    
    def train_with_config(self, config: dict, task_id: int):
        """Train model with given configuration."""
        logger.info("Training model with given configuration")

        model_ = TrainedModel(
            model_id = task_id,
            config=config,
            model_version="RFDETRMedium",
            output_path=config['output_dir'],
            results={
                "map50": 0,
                "recall": 0,
                "f1_score": 0,
            }
        )

        # Build model
        model = RFDETRMedium(
            **model_kwargs,
            dataset_dir=self.config.get('dataset_path'),
            log_per_class_metrics=True
        )
        
        model.train()

        metrics_path = Path(config['output_dir']) / "metrics.csv"
        if metrics_path.exists():
            metrics_df = pd.read_csv(metrics_path)
            # keep rows that contain validation metrics
            metrics_df = metrics_df[metrics_df["val/mAP_50"].notna()]
            # best epoch
            best_row = metrics_df.loc[metrics_df["val/F1"].idxmax()]

            map50 = float(best_row["val/ema_mAP_50"])
            recall = float(best_row["val/recall"])
            f1_score = float(best_row["val/F1"])
            model_.results = {
                "map50": map50,
                "recall": recall,
                "f1_score": f1_score,
            }
            self.models.append(model_)
            return map50, recall, f1_score
        else:
            self.models.append(model_)
            return 0, 0, 0

    def train_with_optuna(self, dataset_path: str):
        """Train model with Optuna hyperparameter optimization."""
        logger.info("Training model with Optuna hyperparameter optimization")
        try:
            if dataset_path:
                self.config['dataset_path'] = dataset_path
            self.models = []
            study = optuna.create_study(
                study_name=f"rfdetr-tuning-{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
                directions=["maximize", "maximize", "maximize"]
            )
            logger.info(f"Starting hyperparameter optimization with {self.config['n_trials']} trials")
            study.optimize(self.objective, n_trials=self.config["n_trials"])
            for t in study.best_trials:
                logger.info(f"Trial {t.number}: MAP50 = {t.values[0]}, Recall = {t.values[1]}, F1 = {t.values[2]} with params {t.params}")
            return self.models
        except Exception as e:
            logger.error(f"Error training with Optuna: {str(e)}", exc_info=True)
            raise
        
        

    