"""
Task 4 - Merge all annotation sources into a unified per-image JSON.

Reads:
    dataset/<folder>/<name>.json   - raw device JSON: {"pos": <entry or list of
        entries>, "gt": <worker-corrected true label>}. Each "pos" entry is a
        whitespace-separated "<class> <cx> <cy> <w> <h> <conf>" string (6
        tokens, normalized 0-1 box); entries with fewer tokens have no
        confirmed detection and are skipped. "pos" is the on-device
        production prediction (bbox + predicted class + confidence);
        "gt" is the worker-corrected class for that same bbox.
    reports/predictions_rfdetr_v1.json
    reports/predictions_rfdetr_v2.json
    reports/predictions_new_model.json

Writes:
    results/<folder>/<name>.jpg (copied)
    results/<folder>/<name>.json

    {
        "image": "TPWL/image001.jpg",
        "annotations": {
            "production": [{"bbox": [...], "confidence": 0.9, "class": "pleat"}, ...],
            "ground_truth": [...],
            "rfdetr_v1": [...],
            "rfdetr_v2": [...],
            "new_model": [...]
        }
    }

    Any source whose predictions file is missing is simply omitted/empty -
    this script works whether you ran inference.py (v1/v2 pair),
    inference_new_model.py (single new checkpoint), or both.

Usage:
    python scripts/merge_annotations.py --dataset dataset --predictions-dir reports --output results
"""

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.logger import setup_logger, get_logger
from utils.constants import DEFECT_CLASSES, CANONICAL_LABELS

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
PREDICTION_SOURCES = ("rfdetr_v1", "rfdetr_v2", "new_model")


def _label_for(raw_label) -> str:
    """Normalize a raw "gt" or "pos" class label (int index, or any of the
    old VN-space / VN-underscore / new-English string vocabularies - see
    utils/constants.py::CANONICAL_LABELS) onto one canonical class name."""
    if raw_label is None:
        return "unknown"
    if isinstance(raw_label, str) and raw_label.lstrip("-").isdigit():
        raw_label = int(raw_label)
    if isinstance(raw_label, int):
        raw_label = DEFECT_CLASSES.get(raw_label, str(raw_label))
    return CANONICAL_LABELS.get(raw_label, str(raw_label))


def _parse_pos_entries(pos) -> list:
    """Parse the 'pos' field (a single entry, or a list of entries) into
    (pred_class_raw, cx, cy, w, h, conf) tuples. Each entry is a
    whitespace-separated "<class> <cx> <cy> <w> <h> <conf>" string (6
    tokens); entries with fewer tokens are not a confirmed detection and
    are skipped."""
    entries = [pos] if isinstance(pos, str) else (pos or [])

    parsed = []
    for entry in entries:
        if not isinstance(entry, str):
            continue
        parts = entry.strip().split(" ")
        if len(parts) < 6:
            continue
        try:
            cx, cy, w, h = map(float, parts[1:5])
            conf = float(parts[5])
        except ValueError:
            continue
        if any(math.isnan(v) for v in (cx, cy, w, h, conf)):
            continue
        parsed.append((parts[0], cx, cy, w, h, conf))
    return parsed


def extract_boxes(sample_json: dict, width: int, height: int, ground_truth: bool = False) -> list:
    """Extract boxes from 'pos'. Production uses the predicted class embedded
    in each entry (parts[0]); ground truth keeps the same bbox but uses the
    sample-level worker-corrected "gt" label instead."""
    gt_label = _label_for(sample_json.get("gt")) if ground_truth else None

    boxes = []
    for pred_class_raw, cx, cy, w, h, conf in _parse_pos_entries(sample_json.get("pos")):
        boxes.append({
            "bbox": [
                (cx - w / 2) * width,
                (cy - h / 2) * height,
                (cx + w / 2) * width,
                (cy + h / 2) * height,
            ],
            "confidence": 1.0 if ground_truth else conf,
            "class": gt_label if ground_truth else _label_for(pred_class_raw),
        })
    return boxes


def merge_dataset(dataset_dir: str, predictions_dir: str, output_dir: str):
    from PIL import Image

    logger = get_logger(__name__)
    dataset_path = Path(dataset_dir)
    output_path = Path(output_dir)

    source_predictions = {}
    for source in PREDICTION_SOURCES:
        pred_path = Path(predictions_dir) / f"predictions_{source}.json"
        if pred_path.exists():
            with open(pred_path, "r") as f:
                source_predictions[source] = json.load(f)
        else:
            logger.warning(f"Missing {pred_path}; {source} annotations will be empty")
            source_predictions[source] = {}

    merged, skipped, missing_source = 0, 0, 0
    for json_path in sorted(dataset_path.rglob("*.json")):
        image_path = next(
            (json_path.with_suffix(ext) for ext in IMAGE_EXTS if json_path.with_suffix(ext).exists()),
            None,
        )
        if image_path is None:
            logger.warning(f"No image found for {json_path}, skipping")
            skipped += 1
            continue

        try:
            with open(json_path, "r") as f:
                sample = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Broken JSON {json_path}, skipping: {e}")
            skipped += 1
            continue

        try:
            with Image.open(image_path) as img:
                width, height = img.size
        except Exception as e:
            logger.warning(f"Broken image {image_path}, skipping: {e}")
            skipped += 1
            continue

        folder = image_path.parent.name
        rel_image = f"{folder}/{image_path.name}"

        source_annotations = {}
        any_missing = False
        for source in PREDICTION_SOURCES:
            # Only flag as "missing" if that source's predictions file was
            # actually loaded (non-empty) but doesn't cover this image -
            # a source with no predictions file at all (e.g. rfdetr_v1/v2
            # when only the new model was run) is expected, not an error.
            if source_predictions[source] and rel_image not in source_predictions[source]:
                any_missing = True
            source_annotations[source] = source_predictions[source].get(rel_image) or []
        if any_missing:
            missing_source += 1

        merged_record = {
            "image": rel_image,
            "captured_at": sample.get("_captured_at"),
            "annotations": {
                "production": extract_boxes(sample, width, height, ground_truth=False),
                "ground_truth": extract_boxes(sample, width, height, ground_truth=True),
                **source_annotations,
            },
        }

        out_folder = output_path / folder
        out_folder.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image_path, out_folder / image_path.name)
        with open(out_folder / f"{image_path.stem}.json", "w") as f:
            json.dump(merged_record, f, indent=2)

        merged += 1

    logger.info(
        f"Merged {merged} samples into {output_path} "
        f"({skipped} skipped due to broken files, {missing_source} missing one or more prediction sources)"
    )
    return merged


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge ground truth, production, and RFDETR annotations")
    parser.add_argument("--dataset", default="dataset", help="Dataset directory")
    parser.add_argument("--predictions-dir", default="reports", help="Directory with predictions_rfdetr_v*.json")
    parser.add_argument("--output", default="results", help="Output directory")
    parser.add_argument("--log-dir", default="./logs", help="Directory for log files")
    return parser


def main():
    args = build_parser().parse_args()
    setup_logger(log_dir=args.log_dir)
    merge_dataset(args.dataset, args.predictions_dir, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
