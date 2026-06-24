"""
Data models for the agentic AI textile defect detection pipeline.
Provides strong typing and structured data formats across pipeline phases.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime
from sahi import AutoDetectionModel

@dataclass
class Annotation:
    """Represents a single defect annotation."""
    bbox: List[float]  # [x1, y1, x2, y2]
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
class DatasetConfig:
    """Configuration for dataset generation."""
    train_ratio: float = 0.8
    new_data_ratio: float = 0.4
    old_data_ratio: float = 0.6
    balance_classes: bool = True

@dataclass
class TrainedModel:
    """Represents a trained model version."""
    model_id: str
    model_version: str
    config: dict
    output_path: str
    results: dict

@dataclass
class ModelVerifier:
    """Represents a verified model version."""
    model_id: str
    model_version: str
    model: AutoDetectionModel
    