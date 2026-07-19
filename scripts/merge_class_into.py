"""Merge one class into another in a built train/valid COCO dataset in-place.

Every annotation with category_id = <from-class> has its category_id
rewritten to <into-class>'s id (bboxes/images untouched). The <from-class>
category entry is then dropped from categories[]. Net effect: the merged
class's annotation count absorbs the source class's, and the dataset ends
up with one fewer category.

Usage:
    python scripts/merge_class_into.py --dataset dataset --from-class hard_pleat --into-class pleat
"""

import argparse
import json
import shutil
from pathlib import Path


def merge_class(annotations_path: Path, from_class: str, into_class: str):
    with open(annotations_path) as f:
        coco = json.load(f)

    name_to_id = {cat["name"]: cat["id"] for cat in coco["categories"]}
    if from_class not in name_to_id:
        print(f"{annotations_path}: no '{from_class}' category found, skipping")
        return
    if into_class not in name_to_id:
        raise SystemExit(f"{annotations_path}: target category '{into_class}' not found")

    from_id = name_to_id[from_class]
    into_id = name_to_id[into_class]

    shutil.copy2(annotations_path, annotations_path.with_name(annotations_path.name + ".bak"))

    merged = 0
    for anno in coco["annotations"]:
        if anno["category_id"] == from_id:
            anno["category_id"] = into_id
            merged += 1

    coco["categories"] = [cat for cat in coco["categories"] if cat["id"] != from_id]

    with open(annotations_path, "w") as f:
        json.dump(coco, f, indent=2)

    into_total = sum(1 for a in coco["annotations"] if a["category_id"] == into_id)
    print(
        f"{annotations_path}: merged {merged} '{from_class}' annotation(s) into '{into_class}' "
        f"({into_total} total '{into_class}' annotations now), '{from_class}' category dropped. "
        f"Backup written to {annotations_path.name}.bak"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="dataset", help="Dataset directory (contains train/ and valid/)")
    parser.add_argument("--from-class", required=True, help="Class to merge (annotations relabeled, category dropped)")
    parser.add_argument("--into-class", required=True, help="Class to merge into (kept, absorbs the annotations)")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)

    for split in ("train", "valid"):
        annotations_path = dataset_path / split / "_annotations.coco.json"
        if not annotations_path.exists():
            print(f"Skipping {split}: {annotations_path} not found")
            continue
        merge_class(annotations_path, args.from_class, args.into_class)


if __name__ == "__main__":
    main()
