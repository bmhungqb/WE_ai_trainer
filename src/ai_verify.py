"""
AI verification module for textile defect detection.
Classifies data, makes predictions with multiple models, and performs voting.
"""

from typing import List, Dict, Any, Tuple
import numpy as np
import torch
import json
from io import BytesIO
from PIL import Image
from pathlib import Path
from copy import deepcopy

from utils.logger import get_logger
from utils.schemas import Annotation, ModelPrediction, ModelVerifier, SampleInfo
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
        self.config = config
        self.models = self._init_models(config['models'])
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def _init_models(self, models_config: list[dict], threshold: float = 0.5) -> list[Any]:
        models = []
        for model_dict in models_config:
            category_mapping = model_dict.get('category_mapping', DEFECT_CLASSES)
            verifier = ModelVerifier(
                model_id=model_dict['model_id'],
                model_version=f"prediction_{model_dict['model_id']}_{model_dict['model_type']}",
                model=None,
                category_mapping=category_mapping,
            )
            model_type = model_dict['model_type']
            weight_path = model_dict['weight_path']
            if model_type == 'rfdetrMedium':
                init_model = RFDETRMedium(pretrain_weights=weight_path)
            elif model_type == 'rfdetrXLarge':
                init_model = RFDETRXLarge(pretrain_weights=weight_path)
            elif model_type == 'rfdetrLarge':
                init_model = RFDETRLarge(pretrain_weights=weight_path)
            else:
                raise ValueError(f"Unknown model_type: {model_type}")
            init_model.optimize_for_inference()
            verifier.model = AutoDetectionModel.from_pretrained(
                model_type="roboflow",
                model=init_model,
                category_mapping=category_mapping,
                confidence_threshold=threshold,
            )
            models.append(verifier)
        logger.info(f"Initialized {len(models)} detection models.")
        return models

    def _convert_sahi_to_annotations(self, prediction, category_mapping: dict) -> List[Annotation]:
        pred_coco_format = prediction.to_coco_annotations()
        annotations = []
        for annotation in pred_coco_format:
            x1, y1, w, h = annotation['bbox']
            x2 = x1 + w
            y2 = y1 + h
            annotations.append(Annotation(
                bbox=[x1, y1, x2, y2],
                confidence=annotation['score'],
                defect_type=category_mapping.get(annotation['category_id'])
            ))
        return annotations
    
    def _download_gcs_blob(self, bucket_client, blob_path: str, bucket_name: str, is_binary: bool = True) -> Any:
        """Safely download a blob from GCS with validation.
        
        Args:
            bucket_client: GCS bucket client object
            blob_path: Path to the blob (without gs://bucket/ prefix)
            bucket_name: Name of the bucket
            is_binary: If True, download as bytes; if False, download as text
            
        Returns:
            Downloaded content (bytes or str) or None if download fails
            
        Raises:
            None - logs errors and returns None
        """
        try:
            blob = bucket_client.blob(blob_path)
            
            # Check if blob exists
            if not blob.exists():
                logger.error(f"Blob does not exist: gs://{bucket_name}/{blob_path}")
                return None
            
            # Download content
            if is_binary:
                content = blob.download_as_bytes()
                if not content or len(content) == 0:
                    logger.error(f"Downloaded empty binary data from gs://{bucket_name}/{blob_path}")
                    return None
            else:
                content = blob.download_as_text()
                if not content:
                    logger.warning(f"Downloaded empty text data from gs://{bucket_name}/{blob_path}")
                    return None
            
            logger.debug(f"Successfully downloaded {len(content) if isinstance(content, bytes) else len(content.encode())} bytes from gs://{bucket_name}/{blob_path}")
            return content
            
        except Exception as e:
            logger.error(f"Failed to download gs://{bucket_name}/{blob_path}: {str(e)}")
            return None
    
    def _get_bbox_intersection(self, bbox: List[float], slice_coords: Dict[str, float]) -> Tuple[List[float], bool]:
        """
        Get the intersection of a bbox with a slice region and transform to slice coordinates.
        
        Args:
            bbox: [x1, y1, x2, y2] in original image coordinates
            slice_coords: {x_min, y_min, x_max, y_max} - slice boundaries in original image
            
        Returns:
            Tuple of (transformed_bbox in slice coords, has_intersection)
        """
        x1_orig, y1_orig, x2_orig, y2_orig = bbox
        x_min, y_min, x_max, y_max = slice_coords['x_min'], slice_coords['y_min'], slice_coords['x_max'], slice_coords['y_max']
        
        # Check if bbox intersects with slice
        intersects = not (x2_orig < x_min or x1_orig > x_max or y2_orig < y_min or y1_orig > y_max)
        if not intersects:
            return None, False
        
        # Calculate intersection
        x1_inter = max(x1_orig, x_min)
        y1_inter = max(y1_orig, y_min)
        x2_inter = min(x2_orig, x_max)
        y2_inter = min(y2_orig, y_max)
        
        # Transform to slice coordinates
        x1_slice = x1_inter - x_min
        y1_slice = y1_inter - y_min
        x2_slice = x2_inter - x_min
        y2_slice = y2_inter - y_min
        
        return [x1_slice, y1_slice, x2_slice, y2_slice], True
    
    def inference_with_sahi(self, model: Any, image: Image) -> ModelPrediction:
        result = get_sliced_prediction(
            image,
            model.model,
            slice_height=512,
            slice_width=512,
            verbose=False,
        )
        model_version = model.model_version
        annotations = self._convert_sahi_to_annotations(result, model.category_mapping or DEFECT_CLASSES)
        return ModelPrediction(
            model_version=model_version,
            annotations=annotations,
        )
        
    def predict_with_models(self, data: List[SampleInfo], output_local_path: Path, gcs_path: str) -> List[SampleInfo]:
        """Make predictions with multiple models on the data."""
        logger.info(f"Making predictions with {len(self.models)} models")
        try:
            list_bucket_client = dict()
            records = []
            for record in data:
                img_path = record.img_path
                anno_path = record.anno_path
                bucket_name = record.bucket_name
                
                is_slice = False
                
                if bucket_name not in list_bucket_client:
                    list_bucket_client[bucket_name] = init_connect_gcs_bucket(bucket_name)
                bucket_client = list_bucket_client[bucket_name]
                
                # handle image: Strip the gs:// URI prefix for img_path
                if img_path.startswith(f"gs://{bucket_name}/"):
                    img_path = img_path.split(f"gs://{bucket_name}/")[1]
                elif img_path.startswith("gs://"):
                    img_path = "/".join(img_path.split("/")[3:])
                
                logger.info(f"Downloading image from GCS path: {img_path} in bucket: {bucket_name}")
                
                try:
                    # Validate blob exists before download
                    blob = bucket_client.blob(img_path)
                    if not blob.exists():
                        logger.error(f"Blob does not exist: gs://{bucket_name}/{img_path}")
                        continue
                    
                    # Download image bytes
                    image_bytes = blob.download_as_bytes()
                    
                    # Validate image data
                    if not image_bytes or len(image_bytes) == 0:
                        logger.error(f"Downloaded empty image data from gs://{bucket_name}/{img_path}")
                        continue
                    
                    # Try to open and validate image
                    try:
                        image = Image.open(BytesIO(image_bytes)).convert("RGB")
                        record.width, record.height = image.size
                        logger.info(f"Successfully loaded image {record.id}: {record.width}x{record.height}")
                    except Exception as img_err:
                        logger.error(f"Failed to open image {record.id}: {str(img_err)}")
                        logger.error(f"Image data size: {len(image_bytes)} bytes")
                        continue
                        
                except Exception as download_err:
                    logger.error(f"Failed to download image from {img_path}: {str(download_err)}")
                    continue
                
                if record.width != self.config['image_size'][0] or record.height != self.config['image_size'][1]:
                    is_slice = True
                # Load annotation if exists
                human_annotations = []
                if anno_path:
                    anno_path_normalized = anno_path
                    # Strip the gs:// URI prefix to get the relative object path
                    if anno_path_normalized.startswith(f"gs://{bucket_name}/"):
                        anno_path_normalized = anno_path_normalized.split(f"gs://{bucket_name}/")[1]
                    elif anno_path_normalized.startswith("gs://"):
                        anno_path_normalized = "/".join(anno_path_normalized.split("/")[3:])
                    
                    try:
                        anno_blob = bucket_client.blob(anno_path_normalized)
                        if not anno_blob.exists():
                            logger.warning(f"Annotation file does not exist: gs://{bucket_name}/{anno_path_normalized}")
                        else:
                            annotation = anno_blob.download_as_text()
                            if annotation:
                                try:
                                    human_annotated = json.loads(annotation)
                                    
                                    # Parse human annotations
                                    preds = human_annotated.get('pos', [])
                                    for anno_str in preds:
                                        parts = anno_str.split(' ')
                                        if len(parts) < 6:
                                            continue
                                        cx, cy, w, h, score = map(float, parts[1:])
                                        lbl = MAPPING_CLASSES.get(parts[0], 'unknown')  
                                        x1 = (cx - w/2) * record.width
                                        y1 = (cy - h/2) * record.height
                                        x2 = (cx + w/2) * record.width
                                        y2 = (cy + h/2) * record.height
                                        human_annotations.append(Annotation(
                                            bbox=[x1, y1, x2, y2],
                                            confidence=score,
                                            defect_type=lbl
                                        ))
                                    logger.info(f"Loaded {len(human_annotations)} human annotations for {record.id}")
                                except json.JSONDecodeError as json_err:
                                    logger.warning(f"Failed to parse annotation JSON for {record.id}: {str(json_err)}")
                            else:
                                logger.warning(f"Annotation file is empty: gs://{bucket_name}/{anno_path_normalized}")
                    except Exception as anno_err:
                        logger.warning(f"Could not load annotation for {record.id}: {str(anno_err)}")
                
                # Slice and create individual records
                if is_slice:
                    logger.info(f"Processing sliced image: {record.id}")
                    output_local_path.mkdir(parents=True, exist_ok=True)
                    
                    # Slice image into 576x576 tiles using SAHI
                    from sahi.slicing import slice_image
                    sliced_image_result = slice_image(
                        image=image,
                        slice_height=576,
                        slice_width=576,
                        overlap_height_ratio=0,
                        overlap_width_ratio=0
                    )
                    
                    # Process each slice as a separate record
                    slice_count = 0
                    for sliced_item in sliced_image_result:
                        # Extract image and metadata from sliced result
                        if isinstance(sliced_item, dict):
                            sliced_image = sliced_item.get('image')
                            starting_pixel = sliced_item.get('starting_pixel')
                        else:
                            sliced_image = sliced_item.image if hasattr(sliced_item, 'image') else sliced_item
                            starting_pixel = sliced_item.starting_pixel if hasattr(sliced_item, 'starting_pixel') else [0, 0]
                        
                        if sliced_image is None:
                            continue
                        
                        # Get slice coordinates in original image space
                        slice_x_min, slice_y_min = starting_pixel
                        slice_height_actual = sliced_image.height if hasattr(sliced_image, 'height') else sliced_image.shape[0]
                        slice_width_actual = sliced_image.width if hasattr(sliced_image, 'width') else sliced_image.shape[1]
                        
                        # Handle numpy array conversion if needed
                        if isinstance(sliced_image, np.ndarray):
                            slice_height_actual, slice_width_actual = sliced_image.shape[:2]
                            sliced_image = Image.fromarray(sliced_image.astype('uint8'))
                        
                        slice_x_max = slice_x_min + slice_width_actual
                        slice_y_max = slice_y_min + slice_height_actual
                        
                        slice_coords = {
                            'x_min': slice_x_min,
                            'y_min': slice_y_min,
                            'x_max': slice_x_max,
                            'y_max': slice_y_max
                        }
                        
                        # Create a new record for this slice
                        slice_record = deepcopy(record)
                        slice_record.id = f"{record.id}_slice_{slice_count:04d}"
                        slice_record.width = slice_width_actual
                        slice_record.height = slice_height_actual
                        slice_record.pre_annotations = []
                        
                        # Map human annotations to this slice
                        slice_human_annos = []
                        for human_anno in human_annotations:
                            transformed_bbox, has_intersection = self._get_bbox_intersection(
                                human_anno.bbox, slice_coords
                            )
                            if has_intersection:
                                slice_human_annos.append(Annotation(
                                    bbox=transformed_bbox,
                                    confidence=human_anno.confidence,
                                    defect_type=human_anno.defect_type
                                ))
                        
                        # Add human annotations for this slice
                        if slice_human_annos:
                            pre_anno = ModelPrediction(
                                model_version="human",
                                annotations=slice_human_annos
                            )
                            slice_record.pre_annotations.append(pre_anno)
                        
                        # Run inference on this slice
                        for model in self.models:
                            prediction = self.inference_with_sahi(model, sliced_image)
                            # Transform predictions back to original image coordinates
                            transformed_annotations = []
                            for anno in prediction.annotations:
                                x1_orig = anno.bbox[0] + slice_x_min
                                y1_orig = anno.bbox[1] + slice_y_min
                                x2_orig = anno.bbox[2] + slice_x_min
                                y2_orig = anno.bbox[3] + slice_y_min
                                transformed_annotations.append(Annotation(
                                    bbox=[x1_orig, y1_orig, x2_orig, y2_orig],
                                    confidence=anno.confidence,
                                    defect_type=anno.defect_type
                                ))
                            
                            if transformed_annotations:
                                slice_record.pre_annotations.append(ModelPrediction(
                                    model_version=prediction.model_version,
                                    annotations=transformed_annotations
                                ))
                        
                        # Merge predictions for this slice
                        slice_record.final_pre_annotations = self.merge_predictions(slice_record.pre_annotations)
                        
                        # Save sliced image
                        slice_filename = f"{record.id.replace('/', '_')}_{slice_count:04d}.jpg"
                        slice_path = output_local_path / slice_filename
                        slice_record.img_path = f"gs://{gcs_path}/{output_local_path.name}/{slice_filename}"
                        sliced_image.save(str(slice_path))
                        
                        # Append this slice as a separate record
                        records.append(slice_record)
                        slice_count += 1
                    continue
                
                # Standard processing for non-N folder samples
                if human_annotations:
                    pre_anno = ModelPrediction(
                        model_version="human",
                        annotations=human_annotations
                    )
                    record.pre_annotations.append(pre_anno)
                
                for model in self.models:
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
