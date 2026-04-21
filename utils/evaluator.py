"""
Model evaluator for textile defect detection.
Evaluates model performance against defined criteria.
"""

import logging
from typing import Dict, Any
from utils.logger import get_logger
from utils.models import ModelMetrics

logger = get_logger(__name__)


class Evaluator:
    """Evaluate model performance against criteria."""
    
    def __init__(self, criteria: Dict[str, float] = None):
        logger.info("Initializing Evaluator")
        self.criteria = criteria or {
            "accuracy": 0.85,
            "precision": 0.80,
            "recall": 0.80,
            "f1_score": 0.80
        }
        logger.info(f"Evaluation criteria set: {self.criteria}")
    
    def evaluate_model(self, model_predictions: Dict[str, Any], ground_truth: Dict[str, Any]) -> ModelMetrics:
        """Evaluate model performance."""
        logger.info("Starting model evaluation")
        try:
            metrics = ModelMetrics(
                accuracy=0.0,
                precision=0.0,
                recall=0.0,
                f1_score=0.0
            )
            logger.debug(f"Calculated metrics: {metrics}")
            return metrics
        except Exception as e:
            logger.error(f"Error evaluating model: {str(e)}", exc_info=True)
            raise
    
    def check_criteria(self, metrics: ModelMetrics) -> bool:
        """Check if metrics meet criteria."""
        logger.info("Checking if metrics meet evaluation criteria")
        try:
            meets_criteria = True
            
            # Using getattr to get metric fields dynamically based on criteria
            for metric_name, threshold in self.criteria.items():
                if hasattr(metrics, metric_name):
                    metric_value = getattr(metrics, metric_name)
                    meets = metric_value >= threshold
                    logger.debug(f"{metric_name}: {metric_value:.4f} >= {threshold} ? {meets}")
                    meets_criteria = meets_criteria and meets
            
            logger.info(f"Criteria check result: {'PASSED' if meets_criteria else 'FAILED'}")
            return meets_criteria
        except Exception as e:
            logger.error(f"Error checking criteria: {str(e)}", exc_info=True)
            raise
    
    def generate_evaluation_report(self, metrics: ModelMetrics, model_name: str) -> str:
        """Generate evaluation report."""
        logger.info(f"Generating evaluation report for model: {model_name}")
        try:
            report = f"\n{'='*60}\nEvaluation Report - {model_name}\n{'='*60}\n"
            for metric, threshold in self.criteria.items():
                if hasattr(metrics, metric):
                    value = getattr(metrics, metric)
                    status = "✓ PASS" if value >= threshold else "✗ FAIL"
                    report += f"{metric:20s}: {value:.4f} (threshold: {threshold}) {status}\n"
            report += f"{'='*60}\n"
            logger.info("Evaluation report generated")
            return report
        except Exception as e:
            logger.error(f"Error generating report: {str(e)}", exc_info=True)
            raise
