"""
AI verification module for textile defect detection.
Classifies data, makes predictions with multiple models, and performs voting.
"""

import logging
from typing import List, Dict, Any, Tuple
import numpy as np
import torch
import json
from io import BytesIO
from PIL import Image
from utils.logger import get_logger
from utils.schemas import SampleInfo, Annotation, VerifyResult, ModelPrediction, VerifiedModel
from utils.gcs_utils import init_connect_gcs_bucket

from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from rfdetr import RFDETRMedium, RFDETRLarge
from rfdetr_plus import RFDETRXLarge

from utils.constants import DEFECT_CLASSES, MAPPING_CLASSES

logger = get_logger(__name__)

class AIVerify:
    """AI verification system for data classification and multi-model predictions."""
    
    def __init__(self, config: dict):
        logger.info("Initializing AIVerify")
        self.models = self._init_models(config['verify_configs']['models'])
        self.verify_config = config['verify_configs']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def _init_models(self, models_config: list[dict], threshold: float = 0.5) -> list[Any]:
        models = []
        for model_dict in models_config:
            model = VerifiedModel(
                model_id=model_dict['model_id'],
                model_version=model_dict['model_version'],
            )
            model_type = model_dict['model_type']
            weight_path = model_dict['weight_path']
            if model_type == 'rfdetrMedium':
                init_model = RFDETRMedium(pretrain_weights=weight_path)
            elif model_type == 'rfdetrXLarge':
                init_model = RFDETRXLarge(pretrain_weights=weight_path)
            elif model_type == 'rfdetrLarge':
                init_model = RFDETRLarge(pretrain_weights=weight_path)
            init_model.optimize_for_inference()
            model.model = AutoDetectionModel.from_pretrained(
                model_type="roboflow",
                model=init_model,
                category_mapping=DEFECT_CLASSES,
                confidence_threshold=threshold,
            )
            models.append(model)
        logger.info(f"Initialized {len(models)} detection models.")
        return models   
    
    def _convert_sahi_to_annotations(self, prediction) -> List[Annotation]:
        pred_coco_format = prediction.to_coco_annotations()
        annotations = []
        for annotation in pred_coco_format:
            x1, y1, x2, y2 = annotation['bbox'] 
            annotations.append(Annotation(
                bbox=[x1, y1, x2, y2],
                confidence=annotation['score'],
                defect_type=DEFECT_CLASSES.get(annotation['category_id'])
            ))
        return annotations
    
    def inference_with_sahi(self, model: Any, image: Image) -> ModelPrediction:
        result = get_sliced_prediction(
            image,
            model.model,
            slice_height=512,
            slice_width=512,
            verbose=False,
        )
        model_version = model.model_version
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
                    # Strip the gs:// URI prefix to get the relative object path
                    if anno_path.startswith(f"gs://{bucker_name}/"):
                        anno_path = anno_path.split(f"gs://{bucker_name}/")[1]
                    elif anno_path.startswith("gs://"):
                        anno_path = "/".join(anno_path.split("/")[3:])
                    annotation = bucket_client.blob(anno_path).download_as_text()
                    human_annotated = json.loads(annotation)

                    # logic to pre-process human annotations
                    preds = human_annotated.get('pos', [])
                    ground_truth = human_annotated.get('gt', 'unknown')
                    annos = []
                    for anno_str in preds:
                        parts = anno_str.split(' ')
                        if len(parts) != 5:
                            continue
                        cx, cy, w, h, score = map(float, parts[1:6])
                        lbl = MAPPING_CLASSES.get(parts[0], 'unknown')  
                        x1 = cx - w/2
                        y1 = cy - h/2
                        x2 = cx + w/2
                        y2 = cy + h/2
                        annos.append(Annotation(
                            bbox=[x1, y1, x2, y2],
                            confidence=score,
                            defect_type=lbl
                        ))
                    pre_anno = ModelPrediction(
                        model_version= "human",
                        annotations=annos
                    )
                    record.pre_annotations.append(pre_anno)
                
                # Strip the gs:// URI prefix for img_path as well
                if img_path.startswith(f"gs://{bucker_name}/"):
                    img_path = img_path.split(f"gs://{bucker_name}/")[1]
                elif img_path.startswith("gs://"):
                    img_path = "/".join(img_path.split("/")[3:])
                image_bytes = bucket_client.blob(img_path).download_as_bytes()
                image = Image.open(BytesIO(image_bytes)).convert("RGB")
                record.width, record.height = image.size
                for model_idx, model in enumerate(self.models):
                    prediction = self.inference_with_sahi(model, image)
                    record.pre_annotations.append(prediction)
                record.final_pre_annotations = self.merge_predictions(record.pre_annotations)
                records.append(record)
            return records
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}", exc_info=True)
            raise
    
    def _calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        
        box1_area = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
        box2_area = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
        
        union_area = box1_area + box2_area - inter_area
        if union_area == 0:
            return 0
        return inter_area / union_area

    def merge_predictions(self, pre_annotations: List[ModelPrediction], iou_threshold: float = 0.5) -> List[Annotation]:
        num_models = len(pre_annotations)
        if num_models == 0:
            return []
            
        all_annos = []
        for model_idx, pre_annotation in enumerate(pre_annotations):
            for anno in pre_annotation.annotations:
                all_annos.append({
                    "anno": anno,
                    "model_idx": model_idx
                })
                
        # Sort by confidence descending so max score is always first
        all_annos.sort(key=lambda x: x["anno"].confidence, reverse=True)
        
        groups = []
        for item in all_annos:
            anno = item["anno"]
            
            # Find a matching group based on IoU overlap
            matched_group = None
            for group in groups:
                rep_bbox = group["items"][0]["anno"].bbox
                if self._calculate_iou(anno.bbox, rep_bbox) >= iou_threshold:
                    matched_group = group
                    break
            
            if matched_group is not None:
                matched_group["items"].append(item)
            else:
                groups.append({"items": [item]})
                
        final_annotations = []
        for group in groups:
            items = group["items"]
            unique_models = set(item["model_idx"] for item in items)
            
            # Condition: (> 0.5) of models must predict this bbox
            if len(unique_models) / num_models >= 0.5:
                # Voting for label (majority vote)
                label_counts = {}
                for item in items:
                    lbl = item["anno"].defect_type
                    label_counts[lbl] = label_counts.get(lbl, 0) + 1
                    
                best_label = None
                max_count = -1
                for item in items:
                    lbl = item["anno"].defect_type
                    if label_counts[lbl] > max_count:
                        max_count = label_counts[lbl]
                        best_label = lbl
                        
                # Max score is the first item since it's sorted
                max_score = items[0]["anno"].confidence
                best_bbox = items[0]["anno"].bbox
                
                final_annotations.append(Annotation(
                    bbox=best_bbox,
                    confidence=max_score,
                    defect_type=best_label
                ))
                
        return final_annotations
