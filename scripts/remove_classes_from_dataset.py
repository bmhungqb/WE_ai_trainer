"""Remove one or more classes from a built train/valid COCO dataset in-place.

Drops all annotations whose category name is in --classes, and drops those
categories from the categories list. Images are kept even if they end up with
zero annotations (treated as negative/background samples).

Usage:
    python scripts/remove_classes_from_dataset.py --dataset dataset --classes hard_pleat pleat
"""

import argparse
import json
from pathlib import Path


def remove_classes(annotations_path: Path, classes_to_remove: set):
    with open(annotations_path) as f:
        coco = json.load(f)

    removed_category_ids = {
        cat["id"] for cat in coco["categories"] if cat["name"] in classes_to_remove
    }
    coco["categories"] = [cat for cat in coco["categories"] if cat["id"] not in removed_category_ids]

    before = len(coco["annotations"])
    coco["annotations"] = [
        anno for anno in coco["annotations"] if anno["category_id"] not in removed_category_ids
    ]
    removed = before - len(coco["annotations"])

    with open(annotations_path, "w") as f:
        json.dump(coco, f, indent=2)

    print(
        f"{annotations_path}: removed {removed} annotation(s) for classes {sorted(classes_to_remove)}, "
        f"{len(coco['images'])} images kept, {len(coco['annotations'])} annotations remain"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="dataset", help="Dataset directory (contains train/ and valid/)")
    parser.add_argument("--classes", nargs="+", required=True, help="Class names to remove")
    args = parser.parse_args()

    classes_to_remove = set(args.classes)
    dataset_path = Path(args.dataset)

    for split in ("train", "valid"):
        annotations_path = dataset_path / split / "_annotations.coco.json"
        if not annotations_path.exists():
            print(f"Skipping {split}: {annotations_path} not found")
            continue
        remove_classes(annotations_path, classes_to_remove)


if __name__ == "__main__":
    main()
