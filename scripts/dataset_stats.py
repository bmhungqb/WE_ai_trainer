"""
Report statistics for the dataset used to train rfdetr.

Supports two dataset layouts:
    1. COCO splits (output of build_train_valid_dataset.py):
        dataset/train/_annotations.coco.json
        dataset/valid/_annotations.coco.json
    2. Flat sidecar-JSON folders (output of download_dataset.py):
        dataset/TPWL/*.jpg + *.json
        dataset/TPRL/*.jpg + *.json

For each split/folder found, reports image count, annotation count, and
per-class distribution. Writes the combined report to reports/dataset_stats.json.

Usage:
    python scripts/dataset_stats.py --dataset dataset --report reports/dataset_stats.json
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.logger import setup_logger, get_logger

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def stats_from_coco(coco_path: Path) -> dict:
    with open(coco_path) as f:
        coco = json.load(f)

    cat_names = {c["id"]: c["name"] for c in coco.get("categories", [])}
    class_counts = Counter(cat_names.get(a["category_id"], a["category_id"]) for a in coco["annotations"])

    images_with_annos = {a["image_id"] for a in coco["annotations"]}

    return {
        "format": "coco",
        "num_images": len(coco["images"]),
        "num_annotations": len(coco["annotations"]),
        "images_without_annotations": len(coco["images"]) - len(images_with_annos),
        "class_counts": dict(class_counts.most_common()),
    }


def stats_from_sidecar_folder(folder: Path) -> dict:
    num_images = 0
    num_annotations = 0
    class_counts = Counter()

    for image_path in sorted(folder.iterdir()):
        if image_path.suffix.lower() not in IMAGE_EXTS:
            continue
        num_images += 1
        json_path = image_path.with_suffix(".json")
        if not json_path.exists():
            continue
        try:
            with open(json_path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        gt = data.get("gt")
        pos = data.get("pos")
        labels = gt if isinstance(gt, list) else [gt] if gt is not None else []
        boxes = pos if isinstance(pos, list) else [pos] if pos is not None else []
        n = max(len(labels), len(boxes), 1 if (labels or boxes) else 0)
        num_annotations += n
        for label in labels:
            class_counts[label] += 1

    return {
        "format": "sidecar",
        "num_images": num_images,
        "num_annotations": num_annotations,
        "class_counts": dict(class_counts.most_common()),
    }


def collect_stats(dataset_dir: str) -> dict:
    logger = get_logger(__name__)
    dataset_path = Path(dataset_dir)

    report = {}

    if not dataset_path.is_dir():
        logger.warning(f"Dataset directory not found: {dataset_path}")
        return report

    for coco_json in sorted(dataset_path.glob("*/_annotations.coco.json")):
        split_name = coco_json.parent.name
        logger.info(f"Reading COCO split: {split_name}")
        report[split_name] = stats_from_coco(coco_json)

    for sub in sorted(dataset_path.iterdir()):
        if not sub.is_dir() or sub.name in report:
            continue
        if (sub / "_annotations.coco.json").exists():
            continue
        has_images = any(p.suffix.lower() in IMAGE_EXTS for p in sub.iterdir())
        if has_images:
            logger.info(f"Reading sidecar folder: {sub.name}")
            report[sub.name] = stats_from_sidecar_folder(sub)

    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="dataset", help="Path to dataset directory")
    parser.add_argument("--report", default="reports/dataset_stats.json", help="Output report path")
    args = parser.parse_args()

    setup_logger()
    logger = get_logger(__name__)

    report = collect_stats(args.dataset)

    if not report:
        logger.warning(f"No recognizable splits/folders found under {args.dataset}")

    for name, s in report.items():
        print(f"\n{name} ({s['format']}):")
        print(f"  images:      {s['num_images']}")
        print(f"  annotations: {s['num_annotations']}")
        print(f"  classes:")
        for cls, count in s["class_counts"].items():
            print(f"    {cls}: {count}")

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nReport written to {report_path}")


if __name__ == "__main__":
    main()
