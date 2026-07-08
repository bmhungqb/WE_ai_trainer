"""
Task 2 - Validate the downloaded dataset.

For every image found under the dataset directory, checks:
    - a paired JSON sidecar exists
    - the JSON can be parsed
    - the image can be opened

Writes a summary report to reports/report_dataset.json:
    {
        "total_images": 0,
        "missing_json": 0,
        "broken_json": 0,
        "broken_image": 0
    }

Usage:
    python scripts/validate_dataset.py --dataset dataset --report reports/report_dataset.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.logger import setup_logger, get_logger

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def validate_dataset(dataset_dir: str) -> dict:
    logger = get_logger(__name__)
    dataset_path = Path(dataset_dir)

    report = {
        "total_images": 0,
        "missing_json": 0,
        "broken_json": 0,
        "broken_image": 0,
        "issues": [],
    }

    from PIL import Image

    for image_path in sorted(dataset_path.rglob("*")):
        if image_path.suffix.lower() not in IMAGE_EXTS:
            continue

        report["total_images"] += 1
        json_path = image_path.with_suffix(".json")

        if not json_path.exists():
            report["missing_json"] += 1
            report["issues"].append({"image": str(image_path), "issue": "missing_json"})
            logger.warning(f"Missing JSON for {image_path}")
            continue

        try:
            with open(json_path, "r") as f:
                json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            report["broken_json"] += 1
            report["issues"].append({"image": str(image_path), "issue": "broken_json", "detail": str(e)})
            logger.warning(f"Broken JSON for {image_path}: {e}")

        try:
            with Image.open(image_path) as img:
                img.verify()
        except Exception as e:
            report["broken_image"] += 1
            report["issues"].append({"image": str(image_path), "issue": "broken_image", "detail": str(e)})
            logger.warning(f"Broken image {image_path}: {e}")

    logger.info(
        f"Validated {report['total_images']} images: "
        f"missing_json={report['missing_json']}, broken_json={report['broken_json']}, "
        f"broken_image={report['broken_image']}"
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate the downloaded dataset")
    parser.add_argument("--dataset", default="dataset", help="Dataset directory")
    parser.add_argument("--report", default="reports/report_dataset.json", help="Output report path")
    parser.add_argument("--log-dir", default="./logs", help="Directory for log files")
    return parser


def main():
    args = build_parser().parse_args()
    setup_logger(log_dir=args.log_dir)

    report = validate_dataset(args.dataset)

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Report written to {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
