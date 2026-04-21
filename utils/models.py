"""
Data models for the agentic AI textile defect detection pipeline.
Provides strong typing and structured data formats across pipeline phases.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime

@dataclass
class Annotation:
    """Represents a single defect annotation."""
    bbox: List[float]  # [x, y, width, height]
    confidence: float
    defect_type: str = None

@dataclass
class ModelPrediction:
    """Represents a single defect annotation."""
    model_version: str = None
    annotations: List[Annotation] = field(default_factory=list)

@dataclass
class SampleInfo:
    """Represents a single image with its associated metadata, annotations, and predictions."""
    id: str
    img_path: str       
    anno_path: str = None
    width: int = None
    height: int = None
    pre_annotations: List[ModelPrediction] = field(default_factory=list)
    final_pre_annotations: List[Annotation] = field(default_factory=list)
    bucket_name: str = None
        
@dataclass
class ImageRecord:
    """Represents an image with its associated metadata, annotations, and predictions."""
    image_id: str
    path: str
    source: str  # e.g., 'shadow' or 'jetson'
    annotations: List[Annotation] = field(default_factory=list)
    predictions: List[Annotation] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    status: str = "raw"  # raw, preprocessed, verified, reviewed

@dataclass
class VerifyResult:
    """Result of the AI Verification phase."""
    auto_accepted: List[ImageRecord]
    needs_review: List[ImageRecord]
    agreement_score: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

@dataclass
class DatasetConfig:
    """Configuration for dataset generation."""
    train_ratio: float = 0.8
    new_data_ratio: float = 0.4
    old_data_ratio: float = 0.6
    balance_classes: bool = True

@dataclass
class ModelMetrics:
    """Evaluation metrics for a trained model."""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    train_loss: float = 0.0
    val_loss: float = 0.0

@dataclass
class TrainedModel:
    """Represents a trained model version."""
    model_id: str
    training_method: str
    params: Dict[str, Any]
    metrics: ModelMetrics
    teacher_model_id: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
