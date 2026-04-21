"""
Dataset manager for handling mixed datasets.
Manages new vs old data mixing for training.
"""

import json
import os
import copy
import random
import datetime
from collections import Counter, defaultdict
from typing import Dict, Any, Tuple, List
from utils.logger import get_logger
from utils.models import SampleInfo

logger = get_logger(__name__)

class DatasetManager:
    """Manage dataset composition and versions."""
    
    def __init__(self, config: dict = None):
        logger.info("Initializing DatasetManager")
        self.config = config 
        self.split_ratio = config['new_split_ratio']

    @staticmethod
    def _load_coco(path: str) -> Dict[str, Any]:
        """Load a COCO-format JSON file."""
        with open(path, 'r') as f:
            return json.load(f)

    @staticmethod
    def _save_coco(data: Dict[str, Any], path: str) -> str:
        """Save a COCO-format JSON file, creating dirs as needed."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        return path

    @staticmethod
    def _get_primary_class(image_id: int, annotations: List[Dict]) -> str:
        """Determine the primary class for an image based on its annotations.
        
        Primary class = the most frequent category_id among the image's annotations.
        If the image has no annotations it is treated as a background/negative sample.
        """
        img_annos = [a for a in annotations if a['image_id'] == image_id]
        if not img_annos:
            return _BACKGROUND_CLASS
        cat_counts = Counter(a['category_id'] for a in img_annos)
        # Return the category with the highest count (tie-broken arbitrarily)
        return str(cat_counts.most_common(1)[0][0])

    @staticmethod
    def _build_coco_subset(
        image_entries: List[Dict],
        all_annotations: List[Dict],
        categories: List[Dict],
        info: Dict = None,
    ) -> Dict[str, Any]:
        """Build a new COCO dict for a subset of images, re-indexing IDs."""
        # Collect the original image IDs that belong to this subset
        selected_img_ids = {img['id'] for img in image_entries}

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

    def merge_old_new_data(self, old_data_pth: str, new_data_pth: str) -> str:
        """Merge old and new COCO annotation files into a single dataset.

        Image and annotation IDs are re-indexed to avoid conflicts.

        Args:
            old_data_pth: Path to the old COCO JSON.
            new_data_pth: Path to the new COCO JSON.

        Returns:
            Path to the merged COCO JSON file.
        """
        try:
            old_coco = self._load_coco(old_data_pth)
            new_coco = self._load_coco(new_data_pth)

            categories = new_coco.get('categories', old_coco.get('categories', []))

            # Combine images, re-index
            all_images = []
            old_to_new_img = {}
            idx = 1
            for img in old_coco['images']:
                old_to_new_img[('old', img['id'])] = idx
                new_img = copy.deepcopy(img)
                new_img['id'] = idx
                all_images.append(new_img)
                idx += 1
            for img in new_coco['images']:
                old_to_new_img[('new', img['id'])] = idx
                new_img = copy.deepcopy(img)
                new_img['id'] = idx
                all_images.append(new_img)
                idx += 1

            # Combine annotations, re-index
            all_annotations = []
            anno_idx = 1
            for anno in old_coco['annotations']:
                new_anno = copy.deepcopy(anno)
                new_anno['id'] = anno_idx
                new_anno['image_id'] = old_to_new_img[('old', anno['image_id'])]
                all_annotations.append(new_anno)
                anno_idx += 1
            for anno in new_coco['annotations']:
                new_anno = copy.deepcopy(anno)
                new_anno['id'] = anno_idx
                new_anno['image_id'] = old_to_new_img[('new', anno['image_id'])]
                all_annotations.append(new_anno)
                anno_idx += 1

            merged = {
                "info": new_coco.get('info', {}),
                "licenses": [],
                "images": all_images,
                "annotations": all_annotations,
                "categories": categories,
            }

            ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            merged_path = f"tmp/merged_{ts}.json"
            self._save_coco(merged, merged_path)

            logger.info(
                f"Merged dataset: {len(old_coco['images'])} old + "
                f"{len(new_coco['images'])} new = {len(all_images)} total images"
            )
            return merged_path

        except Exception as e:
            logger.error(f"Error merging datasets: {str(e)}", exc_info=True)
            raise