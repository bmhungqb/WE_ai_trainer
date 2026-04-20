"""
AI verification module for textile defect detection.
Classifies data, makes predictions with multiple models, and performs voting.
"""

import logging
from typing import List, Dict, Any, Tuple
import numpy as np
import torch
from PIL import Image
from utils.logger import get_logger
from utils.models import SampleInfo, Annotation, VerifyResult, ModelPrediction
from utils.gcs_utils import init_connect_gcs_bucket

from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from rfdetr import RFDETRMedium
from rfdetr_plus import RFDETRXLarge

from utils.constants import DEFECT_CLASSES

logger = get_logger(__name__)

class AIVerify:
    """AI verification system for data classification and multi-model predictions."""
    
    def __init__(self, config: dict):
        logger.info("Initializing AIVerify")
        self.models = self._init_models(config['data_pipeline']['verify_configs']['models'])
        self.verify_config = config['data_pipeline']['verify_configs']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def _init_models(self, models_config: list[dict], threshold: float = 0.5) -> list[Any]:
        models = []
        for model_dict in models_config:
            model_type = model_dict['model_type']
            weight_path = model_dict['weight_path']
            if model_type == 'rfdetrMedium':
                model = RFDETRMedium(weight_path=weight_path)
            elif model_type == 'rfdetrXLarge':
                model = RFDETRXLarge(weight_path=weight_path)
            model.optimize_for_inference()
            detection_model_sahi = AutoDetectionModel.from_pretrained(
                model_type="roboflow",
                model=model,
                category_mapping=DEFECT_CLASSES,
                confidence_threshold=threshold,
                device=self.device,
            )
            models.append(detection_model_sahi)
        logger.info(f"Initialized {len(models)} detection models.")
        return models   
    
    def _convert_sahi_to_annotations(self, prediction) -> List[Annotation]:
        pred_coco_format = prediction.to_coco_annotations()
        annotations = []
        for annotation in pred_coco_format['annotations']:
            annotations.append(Annotation(
                bbox=annotation['bbox'],
                confidence=annotation['score'],
                defect_type=annotation['category_id']
            ))
        return annotations
    
    def inference_with_sahi(self, model: Any, image: Image, iou_threshold: float, confidence_threshold: float) -> ModelPrediction:
        result = get_sliced_prediction(
            image,
            model,
            slice_height=512,
            slice_width=512,
            overlap=0.2,
            verbose=False,
        )
        model_version = model.__class__.__name__
        annotations = self._convert_sahi_to_annotations(result)
        return ModelPrediction(
            model_version=model_version,
            annotations=annotations,
        )
        
    def predict_with_models(self, data: List[SampleInfo]) -> List[SampleInfo]:
        """Make predictions with multiple models on the data."""
        logger.info(f"Making predictions with {len(self.models)} models")
        try:
            list_bucket_client = dict()
            records = []
            for record in data:
                img_path = record.img_path
                anno_path = record.anno_path
                bucker_name = record.bucket_name
                if bucker_name not in list_bucket_client:
                    list_bucket_client[bucker_name] = init_connect_gcs_bucket(bucker_name)
                bucket_client = list_bucket_client[bucker_name]
                
                if anno_path:
                    annotation = bucket_client.blob(anno_path).download_as_text()
                    human_annotated = json.loads(annotation)
                    # logic to pre-process human annotations
                    pre_anno = ModelPrediction(
                        model_version= "human",
                        annotations=[Annotation(
                            
                        )]
                    )
                    #
                    #
                    record.pre_annotations.append(pre_anno)
                
                image_bytes = bucket_client.blob(img_path).download_as_bytes()
                image = Image.open(BytesIO(image_bytes)).convert("RGB")
                for model_idx, model in enumerate(self.models):
                    prediction = self.inference_with_sahi(model, image, iou_threshold=0.1, confidence_threshold=0.1)
                    record.pre_annotations.append(prediction)
                record.final_pre_annotations = self.merge_predictions(record.pre_annotations)
                records.append(record)
            return records
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}", exc_info=True)
            raise
    
    def merge_predictions(self, pre_annotations: List[ModelPrediction]) -> List[Annotation]:
        # TODO: implement merging logic using nms for overlapping annotations
        annotations = []
        for pre_annotation in pre_annotations:
            annotations.extend(pre_annotation.annotations)
        return annotations
