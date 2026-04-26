"""
Dataset manager for handling mixed datasets.
Manages merging of new and old data for training with stratified splitting.

This module provides tools for:
- Stratified splitting of COCO-format datasets
- Merging datasets with configurable mixing ratios
- ID re-indexing to maintain COCO format consistency
"""

import json
import os
import copy
import random
import datetime
from collections import Counter, defaultdict
from typing import Dict, Any, Tuple, List, Optional, Set
from utils.logger import get_logger
from utils.schemas import SampleInfo
from utils.config import config as app_config

logger = get_logger(__name__)

# Constants
_BACKGROUND_CLASS = "background"
DEFAULT_SPLIT_INFO_PATH = "tmp/split_info.json"


class DatasetManager:
    """Manage dataset composition, versioning, and merging.
    
    Handles:
    - Stratified train/val/test splitting
    - COCO format dataset manipulation
    - Merging old and new datasets with configurable ratios
    - ID reindexing to maintain COCO format consistency
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize DatasetManager with configuration.
        
        Args:
            config: Configuration dictionary with:
                - new_split_ratio: [train_ratio, val_ratio, test_ratio]
                - split_info_file: Path to split info file (optional)
                - mixing_ratio: {new_data_ratio, old_data_ratio}
        """
        logger.info("Initializing DatasetManager")
        self.config = config or app_config.data_management
        self.split_ratio = self.config.get('new_split_ratio', [0.7, 0.2, 0.1])
        self._validate_split_ratio()

    def _validate_split_ratio(self) -> None:
        """Validate that split ratios sum to 1.0.
        
        Raises:
            AssertionError: If ratios don't sum to 1.0
        """
        train_r, val_r, test_r = self.split_ratio
        assert abs(train_r + val_r + test_r - 1.0) < 1e-6, (
            f"Split ratios must sum to 1.0, got {train_r + val_r + test_r}"
        )

    @staticmethod
    def _load_coco(path: str) -> Dict[str, Any]:
        """Load a COCO-format JSON annotation file.
        
        Args:
            path: Path to COCO JSON file
            
        Returns:
            Dictionary containing COCO data
            
        Raises:
            FileNotFoundError: If file doesn't exist
            json.JSONDecodeError: If file is not valid JSON
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"COCO file not found: {path}")
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in {path}: {e}")
            raise

    @staticmethod
    def _save_coco(data: Dict[str, Any], path: str) -> str:
        """Save data in COCO format to JSON file.
        
        Creates parent directories as needed.
        
        Args:
            data: COCO format dictionary
            path: Output file path
            
        Returns:
            The path where file was saved
            
        Raises:
            IOError: If file cannot be written
        """
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
            return path
        except IOError as e:
            logger.error(f"Failed to save COCO file to {path}: {e}")
            raise

    @staticmethod
    def _get_primary_class(image_id: int, image_annotations: List[Dict[str, Any]]) -> str:
        """Determine the primary (most frequent) class for an image.
        
        Used for stratified splitting - assigns images to splits based on their
        most common defect type. Background images (no annotations) are marked as background.
        
        Args:
            image_id: Image ID to find primary class for
            image_annotations: List of all annotations in dataset
            
        Returns:
            Category ID as string, or "background" if no annotations
        """
        img_annos = [a for a in image_annotations if a['image_id'] == image_id]
        if not img_annos:
            return _BACKGROUND_CLASS
        cat_counts = Counter(a['category_id'] for a in img_annos)
        # Return the category with the highest count
        primary_cat = cat_counts.most_common(1)[0][0]
        return str(primary_cat)

    @staticmethod
    def _build_coco_subset(
        image_entries: List[Dict[str, Any]],
        all_annotations: List[Dict[str, Any]],
        categories: List[Dict[str, Any]],
        info: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Build a COCO dataset from a subset of images with reindexed IDs.
        
        Takes a subset of images and reindexes image IDs (1..N) and annotation IDs
        to maintain COCO format consistency.
        
        Args:
            image_entries: List of image dictionaries to include
            all_annotations: All annotations (will be filtered to selected images)
            categories: Categories list from original COCO file
            info: Optional info dict for COCO metadata
            
        Returns:
            New COCO dictionary with reindexed IDs
        """
        # Collect the original image IDs that belong to this subset
        selected_img_ids: Set[int] = {img['id'] for img in image_entries}

        # Re-index images 1..N
        new_images = []
        old_to_new_img_id = {}
        for idx, img in enumerate(image_entries, start=1):
            old_to_new_img_id[img['id']] = idx
            new_img = copy.deepcopy(img)
            new_img['id'] = idx
            new_images.append(new_img)

        # Filter and re-index annotations
        new_annotations = []
        anno_idx = 1
        for anno in all_annotations:
            if anno['image_id'] in selected_img_ids:
                new_anno = copy.deepcopy(anno)
                new_anno['id'] = anno_idx
                new_anno['image_id'] = old_to_new_img_id[anno['image_id']]
                new_annotations.append(new_anno)
                anno_idx += 1

        return {
            "info": info or {},
            "licenses": [],
            "images": new_images,
            "annotations": new_annotations,
            "categories": copy.deepcopy(categories),
        }

    def _split_data_train_val_test(
        self, annotation_file_path: str
    ) -> Tuple[str, str, str]:
        """
        Split a COCO annotation file into train / val / test sets.
        """
        try:
            train_r, val_r, test_r = self.split_ratio
            assert abs(train_r + val_r + test_r - 1.0) < 1e-6, (
                f"Ratios must sum to 1.0, got {train_r + val_r + test_r}"
            )

            coco = self._load_coco(annotation_file_path)
            images = coco['images']
            annotations = coco['annotations']
            categories = coco.get('categories', [])

            # 1. Build a mapping: image_id → image entry for fast lookup
            img_by_id = {img['id']: img for img in images}

            # 2. Build annotation index: image_id → [annotations]
            annos_by_img = defaultdict(list)
            for anno in annotations:
                annos_by_img[anno['image_id']].append(anno)

            # 3. Group images by primary class
            class_to_images: Dict[str, List[Dict]] = defaultdict(list)
            for img in images:
                primary_cls = self._get_primary_class(img['id'], annos_by_img[img['id']])
                class_to_images[primary_cls].append(img)

            # 4. Stratified split per class
            train_images, val_images, test_images = [], [], []

            for cls, cls_images in class_to_images.items():
                random.shuffle(cls_images)
                n = len(cls_images)
                n_train = max(1, round(n * train_r))  # at least 1 in train
                n_val = max(0, round(n * val_r))
                # Ensure we don't exceed total
                n_test = n - n_train - n_val

                # Edge case: if very few samples, prioritize train > val > test
                if n_test < 0:
                    n_val = n - n_train
                    n_test = 0
                if n_val < 0:
                    n_val = 0
                    n_test = 0
                    n_train = n

                train_images.extend(cls_images[:n_train])
                val_images.extend(cls_images[n_train:n_train + n_val])
                test_images.extend(cls_images[n_train + n_val:])

            random.shuffle(train_images)
            random.shuffle(val_images)
            random.shuffle(test_images)

            info_base = coco.get('info', {})
            train_coco = self._build_coco_subset(train_images, annotations, categories, info_base)
            val_coco = self._build_coco_subset(val_images, annotations, categories, info_base)
            test_coco = self._build_coco_subset(test_images, annotations, categories, info_base)

            ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            base_dir = os.path.dirname(annotation_file_path) or "tmp"
            train_path = self._save_coco(train_coco, os.path.join(base_dir, f"train_{ts}.json"))
            val_path = self._save_coco(val_coco, os.path.join(base_dir, f"val_{ts}.json"))
            test_path = self._save_coco(test_coco, os.path.join(base_dir, f"test_{ts}.json"))

            logger.info(
                f"Stratified split complete — "
                f"train: {len(train_images)} imgs, "
                f"val: {len(val_images)} imgs, "
                f"test: {len(test_images)} imgs"
            )
            return train_path, val_path, test_path

        except Exception as e:
            logger.error(f"Error splitting data: {str(e)}", exc_info=True)
            raise

    def _combine_coco_dicts(self, coco1: Dict, coco2: Dict) -> Dict:
        """Helper to combine two COCO dicts and reindex IDs."""
        categories = coco2.get('categories', coco1.get('categories', []))
        all_images = []
        old_to_new_img = {}
        idx = 1
        
        for img in coco1.get('images', []):
            old_to_new_img[('coco1', img['id'])] = idx
            new_img = copy.deepcopy(img)
            new_img['id'] = idx
            all_images.append(new_img)
            idx += 1
            
        for img in coco2.get('images', []):
            old_to_new_img[('coco2', img['id'])] = idx
            new_img = copy.deepcopy(img)
            new_img['id'] = idx
            all_images.append(new_img)
            idx += 1
            
        all_annotations = []
        anno_idx = 1
        
        for anno in coco1.get('annotations', []):
            new_anno = copy.deepcopy(anno)
            new_anno['id'] = anno_idx
            new_anno['image_id'] = old_to_new_img[('coco1', anno['image_id'])]
            all_annotations.append(new_anno)
            anno_idx += 1
            
        for anno in coco2.get('annotations', []):
            new_anno = copy.deepcopy(anno)
            new_anno['id'] = anno_idx
            new_anno['image_id'] = old_to_new_img[('coco2', anno['image_id'])]
            all_annotations.append(new_anno)
            anno_idx += 1
            
        return {
            "info": coco2.get('info', {}),
            "licenses": [],
            "images": all_images,
            "annotations": all_annotations,
            "categories": categories,
        }

    def merge_old_new_data(self, old_data_pth: str, new_data_pth: str) -> str:
        """
        Merge old and new COCO annotation files into a single dataset.
        """
        try:
            old_coco = self._load_coco(old_data_pth)
            new_coco = self._load_coco(new_data_pth)

            # split new_coco following new_split_ratio
            new_train_pth, new_val_pth, new_test_pth = self._split_data_train_val_test(new_data_pth)
            new_train_coco = self._load_coco(new_train_pth)
            new_val_coco = self._load_coco(new_val_pth)
            new_test_coco = self._load_coco(new_test_pth)

            # get data from old_coco
            # get full train, valid, test from old_coc follow file
            split_info_file = self.config.get('split_info_file', 'tmp/split_info.json')
            if not split_info_file:
                split_info_file = 'tmp/split_info.json'
                
            with open(split_info_file, 'r') as f:
                split_info = json.load(f)

            # The split info contains file names
            train_file_names = set(split_info.get('train', []))
            val_file_names = set(split_info.get('val', []))
            test_file_names = set(split_info.get('test', []))

            old_images = old_coco.get('images', [])
            old_annotations = old_coco.get('annotations', [])
            categories = old_coco.get('categories', [])
            info = old_coco.get('info', {})

            old_train_images = [img for img in old_images if img.get('file_name') in train_file_names]
            old_val_images = [img for img in old_images if img.get('file_name') in val_file_names]
            old_test_images = [img for img in old_images if img.get('file_name') in test_file_names]

            # get mixing_ratio and random get exact number sample from old_coco (note get full valid, test, only consider ratio for train split)
            mixing_ratio = self.config.get('mixing_ratio', {})
            new_data_ratio = mixing_ratio.get('new_data_ratio', 0.4)
            old_data_ratio = mixing_ratio.get('old_data_ratio', 0.6)

            num_new_train = len(new_train_coco.get('images', []))
            
            if new_data_ratio > 0:
                target_old_train = int(num_new_train * (old_data_ratio / new_data_ratio))
            else:
                target_old_train = len(old_train_images)

            # Sample from old_train_images if we have more than needed
            if len(old_train_images) > target_old_train and target_old_train > 0:
                old_train_images = random.sample(old_train_images, target_old_train)
            elif target_old_train == 0:
                old_train_images = []

            # Build old coco subsets
            old_train_coco = self._build_coco_subset(old_train_images, old_annotations, categories, info)
            old_val_coco = self._build_coco_subset(old_val_images, old_annotations, categories, info)
            old_test_coco = self._build_coco_subset(old_test_images, old_annotations, categories, info)

            # Merge old and new datasets
            merged_train_coco = self._combine_coco_dicts(old_train_coco, new_train_coco)
            merged_val_coco = self._combine_coco_dicts(old_val_coco, new_val_coco)
            merged_test_coco = self._combine_coco_dicts(old_test_coco, new_test_coco)

            # Save the merged datasets
            ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            base_dir = f"tmp/merged_dataset_{ts}"
            os.makedirs(base_dir, exist_ok=True)
            
            train_path = self._save_coco(merged_train_coco, os.path.join(base_dir, "train.json"))
            val_path = self._save_coco(merged_val_coco, os.path.join(base_dir, "val.json"))
            test_path = self._save_coco(merged_test_coco, os.path.join(base_dir, "test.json"))

            # Update the global app_config dataset_path for the AI trainer
            app_config.ai_trainer_configs['dataset_path'] = base_dir

            logger.info(
                f"Merged datasets saved to {base_dir}. "
                f"Train: {len(merged_train_coco['images'])} imgs, "
                f"Val: {len(merged_val_coco['images'])} imgs, "
                f"Test: {len(merged_test_coco['images'])} imgs."
            )
            
            return base_dir

        except Exception as e:
            logger.error(f"Error merging datasets: {str(e)}", exc_info=True)
            raise