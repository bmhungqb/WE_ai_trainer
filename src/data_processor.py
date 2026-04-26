"""
Data processing module for textile defect detection.
Handles data from Google Cloud, Shadow, and Jetson sources.
Provides utilities for downloading images, annotations, and formatting data for Label Studio.
"""

import logging
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from utils.logger import get_logger
from utils.schemas import SampleInfo, Annotation
from utils.gcs_utils import init_connect_gcs_bucket

logger = get_logger(__name__)

# Constants
DEFAULT_DATE_FORMAT = "%Y-%m-%d"
TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"
SUPPORTED_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png'}
ANNOTATION_EXTENSION = '.json'


class DataProcessor:
    """Process data from various sources (GCS, Shadow, Jetson).
    
    Responsibilities:
    - Download image data and annotations from Google Cloud Storage
    - Format predictions for Label Studio ingestion
    - Handle date filtering and file organization
    """
    
    def __init__(self):
        """Initialize the DataProcessor."""
        logger.info("Initializing DataProcessor")
    
    def _parse_date(self, date_str: Optional[str]) -> Optional[Any]:
        """Parse a date string in ISO format (YYYY-MM-DD).
        
        Args:
            date_str: Date string in format YYYY-MM-DD or None
            
        Returns:
            A date object or None if date_str is None
            
        Raises:
            ValueError: If date string is not in expected format
        """
        if not date_str:
            return None
        try:
            return datetime.strptime(date_str, DEFAULT_DATE_FORMAT).date()
        except ValueError as e:
            logger.error(f"Invalid date format '{date_str}'. Expected format: {DEFAULT_DATE_FORMAT}")
            raise

    def download_data_from_gcs(
        self, 
        buckets_info: List[Dict[str, Any]], 
        start_date_str: Optional[str] = None,
        end_date_str: Optional[str] = None
    ) -> List[SampleInfo]:
        """Download data information from Google Cloud Storage buckets.
        
        Args:
            buckets_info: List of bucket configurations, each containing:
                - gcs_name: GCS bucket name
                - folder_paths: List of folder prefixes to search
                - is_require_anno_file: Whether annotation file is required
            start_date_str: Start date for filtering (YYYY-MM-DD format), overrides bucket config
            end_date_str: End date for filtering (YYYY-MM-DD format), overrides bucket config
                
        Returns:
            List of SampleInfo objects with GCS paths
            
        Raises:
            Exception: If GCS connection fails or data loading fails
        """
        logger.info(f"Connecting to Google Cloud Storage buckets: {[b.get('gcs_name') for b in buckets_info]}")
        logger.info(f"Filtering by date range: {start_date_str} to {end_date_str}")
        try:
            download_dir = Path("tmp/data_from_gcs_info.json")
            download_dir.parent.mkdir(parents=True, exist_ok=True)

            results = []
            sample_records = []

            for bucket_info in buckets_info:
                bucket_name = bucket_info.get("gcs_name")
                folder_paths = bucket_info.get("folder_paths", [])
                is_require_anno = bucket_info.get("is_require_anno_file", False)

                if not bucket_name:
                    logger.warning("Bucket name not specified, skipping bucket")
                    continue

                try:
                    bucket = init_connect_gcs_bucket(bucket_name)
                except Exception as e:
                    logger.error(f"Failed to connect to bucket {bucket_name}: {str(e)}")
                    continue
                
                # Parse dates from function parameters
                start_date = self._parse_date(start_date_str) if start_date_str else None
                end_date = self._parse_date(end_date_str) if end_date_str else None

                # List blobs from specified folder paths
                blobs = []
                if folder_paths:
                    for folder_path in folder_paths:
                        try:
                            blobs.extend(list(bucket.list_blobs(prefix=folder_path)))
                        except Exception as e:
                            logger.warning(f"Failed to list blobs in {folder_path}: {str(e)}")
                else:
                    blobs = list(bucket.list_blobs())
                
                # Group blobs by their stem (filename without extension)
                files_by_stem = self._group_files_by_stem(blobs, bucket_name, start_date, end_date)

                # Extract image and annotation pairs
                for stem, exts in files_by_stem.items():
                    pair = self._extract_file_pair(stem, exts, is_require_anno, bucket_name)
                    if pair:
                        item, record = pair
                        results.append(item)
                        sample_records.append(record)

            # Save metadata to local file
            with open(download_dir, 'w') as f:
                json.dump(results, f, indent=4)
                
            logger.info(f"Downloaded {len(sample_records)} samples from GCS. Saved metadata to {download_dir}")
            return sample_records
                
        except Exception as e:
            logger.error(f"Error downloading data from Google Cloud Storage: {str(e)}", exc_info=True)
            raise

    def _group_files_by_stem(
        self, 
        blobs: List[Any], 
        bucket_name: str,
        start_date: Optional[Any] = None,
        end_date: Optional[Any] = None
    ) -> Dict[str, Dict[str, str]]:
        """Group blob files by their stem (filename without extension).
        
        Args:
            blobs: List of GCS blob objects
            bucket_name: Name of the bucket
            start_date: Optional start date filter
            end_date: Optional end date filter
            
        Returns:
            Dictionary mapping file stem to dict of {extension: gcs_path}
        """
        files_by_stem = {}
        
        for blob in blobs:
            # Apply date filtering if specified
            if start_date and end_date and blob.time_created:
                blob_date = blob.time_created.date()
                if not (start_date <= blob_date <= end_date):
                    continue

            # Extract file stem and extension
            path = Path(blob.name)
            stem = str(path.parent / path.stem)
            ext = path.suffix.lower()

            if stem not in files_by_stem:
                files_by_stem[stem] = {}
            
            files_by_stem[stem][ext] = f"gs://{bucket_name}/{blob.name}"
        
        return files_by_stem

    def _extract_file_pair(
        self, 
        stem: str, 
        exts: Dict[str, str],
        is_require_anno: bool,
        bucket_name: str
    ) -> Optional[Tuple[Dict[str, Any], SampleInfo]]:
        """Extract image and annotation pair from file extensions.
        
        Args:
            stem: File stem (name without extension)
            exts: Dictionary of {extension: gcs_path}
            is_require_anno: Whether annotation file is required
            bucket_name: GCS bucket name
            
        Returns:
            Tuple of (metadata_dict, SampleInfo) or None if invalid
        """
        # Find image file
        img_path = None
        for ext in SUPPORTED_IMAGE_EXTENSIONS:
            if ext in exts:
                img_path = exts[ext]
                break
        
        if not img_path:
            return None
        
        # Find annotation file
        anno_path = exts.get(ANNOTATION_EXTENSION)
        
        if is_require_anno and not anno_path:
            logger.debug(f"Annotation required but not found for {stem}")
            return None

        # Create metadata item
        item = {
            "id": stem,
            "img_path": img_path
        }
        
        if anno_path:
            item["anno_path"] = anno_path

        # Create SampleInfo record
        record = SampleInfo(
            id=stem,
            img_path=img_path,
            anno_path=anno_path,
            bucket_name=bucket_name
        )
        
        return item, record

    def get_label_studio_format_json(self, verified_records: List[SampleInfo], json_path: Path) -> Path:
        """Convert verified predictions to Label Studio format.
        
        Formats pre-annotations for Label Studio ingestion. Note: In future versions,
        high-confidence predictions could be added as final annotations without requiring
        human review.
        
        Args:
            verified_records: List of SampleInfo objects with predictions
            json_path: Path to the output JSON file
        Returns:
            Path to generated JSON file
            
        Raises:
            IOError: If file cannot be written
        """
        records = []
        invalid_count = 0
        
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
                # Validate bbox before adding
                if not self._is_valid_bbox(pred.bbox):
                    logger.warning(
                        f"Skipping invalid prediction for sample {record.id}: "
                        f"bbox={pred.bbox}, confidence={pred.confidence}"
                    )
                    invalid_count += 1
                    continue
                
                preds.append({
                    "from_name": "label",
                    "to_name": "image",
                    "type": "rectanglelabels",
                    "original_width": origin_width,
                    "original_height": origin_height,
                    "value": {
                        "x": (pred.bbox[0] / origin_width) * 100,
                        "y": (pred.bbox[1] / origin_height) * 100,
                        "width": ((pred.bbox[2] - pred.bbox[0]) / origin_width) * 100,
                        "height": ((pred.bbox[3] - pred.bbox[1]) / origin_height) * 100,
                        "rotation": 0,
                        "rectanglelabels": [pred.defect_type]
                    },
                    "score": pred.confidence
                })
            
            sample["predictions"].append({
                'model_version': 'final_preannotation',
                'result': preds
            })
            records.append(sample)
        
        # Generate output path with timestamp
        json_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(json_path, 'w') as f:
            json.dump(records, f, indent=4)
        
        logger.info(
            f"Saved {len(records)} samples to {json_path}. "
            f"Skipped {invalid_count} invalid predictions."
        )
        return json_path

    @staticmethod
    def _is_valid_bbox(bbox: List[float]) -> bool:
        """Validate a bounding box.
        
        Args:
            bbox: Bounding box as [x1, y1, x2, y2]
            
        Returns:
            True if bbox is valid (all values are positive numbers), False otherwise
        """
        if not bbox or len(bbox) != 4:
            return False
        
        return all(v is not None and v == v and v >= 0 for v in bbox)  # v == v checks for NaN