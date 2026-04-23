"""
Data processing module for textile defect detection.
Handles data from Google Cloud, Shadow, and Jetson sources.
"""

import logging
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
from utils.logger import get_logger
from utils.schemas import SampleInfo, Annotation
from utils.gcs_utils import init_connect_gcs_bucket
from datetime import datetime
logger = get_logger(__name__)


class DataProcessor:
    """Process data from various sources (Shadow, Jetson)."""
    
    def __init__(self):
        logger.info("Initializing DataProcessor")
    
    def download_data_from_gcs(self, buckets_info: List[Dict[str, Any]]) -> List[SampleInfo]:
        """Connect to Google Cloud Storage and download data info."""
        logger.info(f"Connecting to Google Cloud Storage buckets: {buckets_info} in data_processor")
        try:
            download_dir = Path("tmp/data_from_gcs_info.json")
            download_dir.parent.mkdir(parents=True, exist_ok=True)

            results = []
            sample_records = []

            for bucket_info in buckets_info:
                bucket_name = bucket_info.get("gcs_name")
                folder_paths = bucket_info.get("folder_paths", [])
                start_str = bucket_info.get("start")
                end_str = bucket_info.get("end")
                is_require_anno = bucket_info.get("is_require_anno_file", False)

                if not bucket_name:
                    continue

                bucket = init_connect_gcs_bucket(bucket_name)
                
                start_date = datetime.strptime(start_str, "%Y-%m-%d").date() if start_str else None
                end_date = datetime.strptime(end_str, "%Y-%m-%d").date() if end_str else None

                # List blobs from specified folder paths
                blobs = []
                if folder_paths:
                    for folder_path in folder_paths:
                        blobs.extend(list(bucket.list_blobs(prefix=folder_path)))
                else:
                    blobs = list(bucket.list_blobs())
                
                # Group blobs by their stem (filename without extension)
                files_by_stem = {}
                for blob in blobs:
                    if start_date and end_date and blob.time_created:
                        blob_date = blob.time_created.date()
                        if not (start_date <= blob_date <= end_date):
                            continue

                    path = Path(blob.name)
                    stem = str(path.parent / path.stem)
                    ext = path.suffix.lower()

                    if stem not in files_by_stem:
                        files_by_stem[stem] = {}
                    files_by_stem[stem][ext] = f"gs://{bucket_name}/{blob.name}"

                for stem, exts in files_by_stem.items():
                    img_path = exts.get('.jpg')
                    anno_path = exts.get('.json')

                    if not img_path:
                        continue
                        
                    if is_require_anno and not anno_path:
                        continue

                    item = {
                        "id": stem,
                        "img_path": img_path
                    }
                    
                    if anno_path:
                        item["anno_path"] = anno_path

                    results.append(item)
                    
                    record = SampleInfo(
                        id=stem,
                        img_path=img_path,
                        anno_path=anno_path,
                        bucket_name=bucket_name
                    )
                    sample_records.append(record)

            with open(download_dir, 'w') as f:
                json.dump(results, f, indent=4)
                
            logger.info(f"Saved {len(results)} items to {download_dir}")
            return sample_records
                
        except Exception as e:
            logger.error(f"Error downloading data from Google Cloud Storage: {str(e)}", exc_info=True)
            raise

    def get_label_studio_format_json(self, verified_records: List[SampleInfo]) -> str:
        """Get label studio format json file."""
        #TODO: improve the logic, instead of pushing all as pre-annotation, consider samples with high confidence score as annotation (without human review).
        records = []
        for record in verified_records:
            sample = {
                "data": {
                    "image": record.img_path
                }   
            }
            origin_width, origin_height = record.width, record.height
            sample["predictions"] = []
            preds = []
            for pred in record.final_pre_annotations:
                preds.append({
                    "from_name": "label",
                    "to_name": "image",
                    "type": "rectanglelabels",
                    "original_width": origin_width,
                    "original_height": origin_height,
                    "value": {
                        "x": (pred.bbox[0]/origin_width)*100,
                        "y": (pred.bbox[1]/origin_height)*100,
                        "width": (pred.bbox[2]/origin_width)*100,
                        "height": (pred.bbox[3]/origin_height)*100,
                        "rotation": 0,
                        "rectanglelabels": [pred.defect_type]
                    }
                })
            sample["predictions"].append({
                'model_version': 'final_preannotation',
                'result': preds
            })
            records.append(sample)
        
        file_name = f"tasks_label_studio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        file_path = Path("tmp/" + file_name)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(records, f, indent=4)
        logger.info(f"Saved {len(records)} items to {file_path}")
        return file_path
        