"""
Report the actual on-disk pixel dimensions of every image in a dataset
directory (COCO train/valid splits, and/or flat sidecar folders like
dataset_june/TPWL, TPRL), plus a size-frequency histogram.

For COCO splits, also cross-checks each image's ACTUAL on-disk size against
the width/height recorded in _annotations.coco.json - a mismatch there
means any bbox math done against the recorded width/height (e.g. Label
Studio percent-coordinate conversions) will be scaled wrong for that image.

Reads:
    <dataset>/train/_annotations.coco.json + images
    <dataset>/valid/_annotations.coco.json + images
    <dataset>/<folder>/*.jpg|.png   (flat sidecar layout, no COCO json)

Writes:
    reports/image_size_stats.json

Usage:
    python scripts/stats_image_sizes.py --dataset dataset --report reports/image_size_stats.json
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.logger import setup_logger, get_logger

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def stats_for_coco_split(coco_path: Path, split_dir: Path) -> dict:
    from PIL import Image

    logger = get_logger(__name__)
    with open(coco_path) as f:
        coco = json.load(f)

    size_counts = Counter()
    mismatches = []
    checked, missing = 0, 0

    for img_meta in coco["images"]:
        image_path = split_dir / img_meta["file_name"]
        if not image_path.exists():
            missing += 1
            continue
        try:
            with Image.open(image_path) as im:
                actual_w, actual_h = im.size
        except Exception as e:
            logger.warning(f"Failed to open {image_path}: {e}")
            continue

        size_counts[f"{actual_w}x{actual_h}"] += 1
        checked += 1

        recorded_w, recorded_h = img_meta.get("width"), img_meta.get("height")
        if recorded_w is not None and recorded_h is not None:
            if (recorded_w, recorded_h) != (actual_w, actual_h):
                mismatches.append({
                    "file_name": img_meta["file_name"],
                    "recorded": [recorded_w, recorded_h],
                    "actual": [actual_w, actual_h],
                })

    return {
        "format": "coco",
        "checked": checked,
        "missing_on_disk": missing,
        "size_distribution": dict(size_counts.most_common()),
        "num_size_mismatches": len(mismatches),
        "mismatches_sample": mismatches[:20],
    }


def stats_for_sidecar_folder(folder: Path) -> dict:
    from PIL import Image

    logger = get_logger(__name__)
    size_counts = Counter()
    checked = 0

    for image_path in sorted(folder.iterdir()):
        if image_path.suffix.lower() not in IMAGE_EXTS:
            continue
        try:
            with Image.open(image_path) as im:
                w, h = im.size
        except Exception as e:
            logger.warning(f"Failed to open {image_path}: {e}")
            continue
        size_counts[f"{w}x{h}"] += 1
        checked += 1

    return {
        "format": "sidecar",
        "checked": checked,
        "size_distribution": dict(size_counts.most_common()),
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
        logger.info(f"Scanning COCO split: {split_name}")
        report[split_name] = stats_for_coco_split(coco_json, coco_json.parent)

    for sub in sorted(dataset_path.iterdir()):
        if not sub.is_dir() or sub.name in report:
            continue
        if (sub / "_annotations.coco.json").exists():
            continue
        has_images = any(p.suffix.lower() in IMAGE_EXTS for p in sub.iterdir())
        if has_images:
            logger.info(f"Scanning sidecar folder: {sub.name}")
            report[sub.name] = stats_for_sidecar_folder(sub)

    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="dataset", help="Path to dataset directory")
    parser.add_argument("--report", default="reports/image_size_stats.json", help="Output report path")
    parser.add_argument("--log-dir", default="./logs", help="Directory for log files")
    return parser


def main():
    args = build_parser().parse_args()
    setup_logger(log_dir=args.log_dir)
    logger = get_logger(__name__)

    report = collect_stats(args.dataset)
    if not report:
        logger.warning(f"No recognizable splits/folders found under {args.dataset}")

    for name, s in report.items():
        print(f"\n{name} ({s['format']}):")
        print(f"  images checked: {s['checked']}")
        if "missing_on_disk" in s:
            print(f"  missing on disk: {s['missing_on_disk']}")
        print(f"  size distribution:")
        for size, count in s["size_distribution"].items():
            print(f"    {size}: {count}")
        if "num_size_mismatches" in s:
            print(f"  recorded-vs-actual size mismatches: {s['num_size_mismatches']}")
            for m in s["mismatches_sample"]:
                print(f"    {m['file_name']}: recorded={m['recorded']} actual={m['actual']}")

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nReport written to {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
