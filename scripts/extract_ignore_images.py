"""
Copy every image in a COCO train split that contains at least one "ignore"
annotation into its own folder, along with a COCO annos.json scoped to just
those images (all of that image's annotations are kept, not only the
ignore boxes, so real-defect context on the same image isn't lost).

Reads:
    <dataset>/train/_annotations.coco.json + images

Writes:
    <output>/*.jpg|.png          - copies of every image containing an ignore box
    <output>/annos.json           - COCO json scoped to just those images/annotations

Usage:
    python scripts/extract_ignore_images.py --dataset dataset --output ignore_images
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.logger import setup_logger, get_logger

IGNORE_CLASS_NAME = "ignore"


def extract(dataset_dir: str, split: str, output_dir: str) -> None:
    logger = get_logger(__name__)

    split_dir = Path(dataset_dir) / split
    coco_path = split_dir / "_annotations.coco.json"
    with open(coco_path) as f:
        coco = json.load(f)

    ignore_cat_id = next((c["id"] for c in coco["categories"] if c["name"] == IGNORE_CLASS_NAME), None)
    if ignore_cat_id is None:
        raise SystemExit(f"No '{IGNORE_CLASS_NAME}' category in {coco_path}")

    ignore_image_ids = {a["image_id"] for a in coco["annotations"] if a["category_id"] == ignore_cat_id}

    images = [img for img in coco["images"] if img["id"] in ignore_image_ids]
    annotations = [a for a in coco["annotations"] if a["image_id"] in ignore_image_ids]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    copied, missing = 0, 0
    for img in images:
        src = split_dir / img["file_name"]
        if not src.exists():
            logger.warning(f"Missing image on disk, skipping copy: {src}")
            missing += 1
            continue
        shutil.copy2(src, output_path / img["file_name"])
        copied += 1

    out_coco = {
        "info": coco.get("info", {}),
        "licenses": coco.get("licenses", []),
        "images": images,
        "annotations": annotations,
        "categories": coco["categories"],
    }
    with open(output_path / "annos.json", "w") as f:
        json.dump(out_coco, f, indent=2)

    logger.info(
        f"{len(images)} image(s) contain '{IGNORE_CLASS_NAME}' ({copied} copied, {missing} missing on disk), "
        f"{len(annotations)} total annotation(s) written to {output_path / 'annos.json'}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="dataset", help="Dataset directory (contains train/, valid/)")
    parser.add_argument("--split", default="train", choices=["train", "valid"], help="Which split to scan")
    parser.add_argument("--output", default="ignore_images", help="Output folder for copied images + annos.json")
    parser.add_argument("--log-dir", default="./logs", help="Directory for log files")
    return parser


def main():
    args = build_parser().parse_args()
    setup_logger(log_dir=args.log_dir)
    extract(args.dataset, args.split, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
