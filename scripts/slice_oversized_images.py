"""
Slice oversized images (e.g. 1280x1280) in a COCO train/valid split down to
576x576 tiles, and rewrite the annotations to match - fixes any downstream
code that assumes a fixed 576x576 canvas (see scripts/stats_image_sizes.py,
which found 12 train + 4 valid images at 1280x1280 while everything else is
576x576).

Uses the same SAHI slicing convention as src/ai_verify.py::predict_with_models
(sahi.slicing.slice_image, slice_height=slice_width=576, no overlap) and the
same box-clipping convention as
src/ai_verify.py::AIVerify._get_bbox_intersection: an annotation box is kept
in EVERY tile it intersects, clipped to that tile's local bounds, so a
defect spanning a tile boundary gets a (partial) annotation in each tile it
touches rather than being dropped or arbitrarily assigned to one tile.

Only images at or above --min-size in either dimension are sliced; images
already <= 576x576 are left untouched. Each oversized image is REPLACED by
its tiles (the original oversized file is deleted after slicing) - not
appended alongside it, so the split ends up fully 576x576-consistent.

Reads:
    <dataset>/train/_annotations.coco.json + images
    <dataset>/valid/_annotations.coco.json + images

Writes (in place):
    <dataset>/train/_annotations.coco.json   (rewritten; backed up as *.bak)
    <dataset>/train/<name>_slice_0000.jpg, _slice_0001.jpg, ...  (new tiles)
    (original oversized image file removed after slicing)

Usage:
    python scripts/slice_oversized_images.py --dataset dataset --slice-size 576
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.logger import setup_logger, get_logger


def get_bbox_intersection(bbox: list, slice_coords: dict):
    """bbox: [x1, y1, x2, y2] in original image coords.
    slice_coords: {x_min, y_min, x_max, y_max} in original image coords.
    Returns (clipped_bbox_in_slice_coords, has_intersection)."""
    x1_orig, y1_orig, x2_orig, y2_orig = bbox
    x_min, y_min, x_max, y_max = (
        slice_coords["x_min"], slice_coords["y_min"], slice_coords["x_max"], slice_coords["y_max"],
    )

    intersects = not (x2_orig < x_min or x1_orig > x_max or y2_orig < y_min or y1_orig > y_max)
    if not intersects:
        return None, False

    x1_inter = max(x1_orig, x_min)
    y1_inter = max(y1_orig, y_min)
    x2_inter = min(x2_orig, x_max)
    y2_inter = min(y2_orig, y_max)

    return [x1_inter - x_min, y1_inter - y_min, x2_inter - x_min, y2_inter - y_min], True


def slice_split(split_dir: Path, slice_size: int, min_size: int) -> dict:
    from PIL import Image
    from sahi.slicing import slice_image

    logger = get_logger(__name__)
    coco_path = split_dir / "_annotations.coco.json"
    if not coco_path.exists():
        logger.warning(f"{coco_path} not found, skipping split")
        return {"oversized_images": 0, "tiles_written": 0, "annotations_written": 0}

    with open(coco_path) as f:
        coco = json.load(f)
    shutil.copy2(coco_path, coco_path.with_name(coco_path.name + ".bak"))

    annos_by_image = {}
    for a in coco["annotations"]:
        annos_by_image.setdefault(a["image_id"], []).append(a)

    next_image_id = max((img["id"] for img in coco["images"]), default=0) + 1
    next_anno_id = max((a["id"] for a in coco["annotations"]), default=0) + 1

    kept_images = []
    kept_annotations = []
    oversized_count, tiles_written = 0, 0
    removed_original_ids = set()

    for img_meta in coco["images"]:
        image_path = split_dir / img_meta["file_name"]
        if not image_path.exists():
            logger.warning(f"Missing image {image_path}, dropping from output")
            continue

        try:
            with Image.open(image_path) as im:
                w, h = im.size
        except Exception as e:
            logger.warning(f"Failed to open {image_path}: {e}")
            continue

        if w < min_size and h < min_size:
            # Not oversized - keep as-is.
            kept_images.append(img_meta)
            kept_annotations.extend(annos_by_image.get(img_meta["id"], []))
            continue

        oversized_count += 1
        removed_original_ids.add(img_meta["id"])
        original_annos = annos_by_image.get(img_meta["id"], [])
        original_boxes = [
            [a["bbox"][0], a["bbox"][1], a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]]
            for a in original_annos
        ]

        image = Image.open(image_path).convert("RGB")
        sliced_result = slice_image(
            image=image,
            slice_height=slice_size,
            slice_width=slice_size,
            overlap_height_ratio=0,
            overlap_width_ratio=0,
        )

        stem = Path(img_meta["file_name"]).stem
        for slice_idx, sliced_item in enumerate(sliced_result):
            if isinstance(sliced_item, dict):
                sliced_image = sliced_item.get("image")
                starting_pixel = sliced_item.get("starting_pixel")
            else:
                sliced_image = sliced_item.image
                starting_pixel = sliced_item.starting_pixel

            import numpy as np
            if isinstance(sliced_image, np.ndarray):
                slice_h, slice_w = sliced_image.shape[:2]
                sliced_image = Image.fromarray(sliced_image.astype("uint8"))
            else:
                slice_w, slice_h = sliced_image.size

            x_min, y_min = starting_pixel
            slice_coords = {"x_min": x_min, "y_min": y_min, "x_max": x_min + slice_w, "y_max": y_min + slice_h}

            tile_boxes = []
            for orig_box, orig_anno in zip(original_boxes, original_annos):
                clipped, has_intersection = get_bbox_intersection(orig_box, slice_coords)
                if not has_intersection:
                    continue
                cx1, cy1, cx2, cy2 = clipped
                cw, ch = cx2 - cx1, cy2 - cy1
                if cw <= 0 or ch <= 0:
                    continue
                tile_boxes.append((orig_anno, [cx1, cy1, cw, ch]))

            tile_name = f"{stem}_slice_{slice_idx:04d}.jpg"
            tile_path = split_dir / tile_name
            sliced_image.save(tile_path, quality=95)
            tiles_written += 1

            new_image_id = next_image_id
            next_image_id += 1
            kept_images.append({
                "id": new_image_id,
                "task_id": img_meta.get("task_id"),
                "file_name": tile_name,
                "width": slice_w,
                "height": slice_h,
            })
            for orig_anno, bbox in tile_boxes:
                kept_annotations.append({
                    "id": next_anno_id,
                    "image_id": new_image_id,
                    "category_id": orig_anno["category_id"],
                    "bbox": bbox,
                    "area": bbox[2] * bbox[3],
                    "iscrowd": orig_anno.get("iscrowd", 0),
                })
                next_anno_id += 1

        image_path.unlink()
        logger.info(f"Sliced {img_meta['file_name']} ({w}x{h}) into tiles, removed original")

    coco["images"] = kept_images
    coco["annotations"] = kept_annotations

    with open(coco_path, "w") as f:
        json.dump(coco, f, indent=2)

    logger.info(
        f"{split_dir}: {oversized_count} oversized image(s) sliced into {tiles_written} tile(s), "
        f"{len(kept_images)} total images / {len(kept_annotations)} total annotations remain. "
        f"Backup written to {coco_path.name}.bak"
    )
    return {"oversized_images": oversized_count, "tiles_written": tiles_written, "annotations_written": len(kept_annotations)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="dataset", help="Dataset directory (contains train/, valid/)")
    parser.add_argument("--slice-size", type=int, default=576, help="Target tile size (square)")
    parser.add_argument("--min-size", type=int, default=577, help="Images with width or height >= this are sliced")
    parser.add_argument("--log-dir", default="./logs", help="Directory for log files")
    return parser


def main():
    args = build_parser().parse_args()
    setup_logger(log_dir=args.log_dir)

    dataset_path = Path(args.dataset)
    for split in ("train", "valid"):
        slice_split(dataset_path / split, args.slice_size, args.min_size)
    return 0


if __name__ == "__main__":
    sys.exit(main())
