"""Randomly downsample one class ("target class", e.g. stain) in a built
train/valid COCO dataset so its annotation count is roughly in line with
another class ("reference class", e.g. weaving).

Only images whose annotations are ALL --target-class are dropped (image file
+ json entries). Images that mix --target-class with other classes are left
alone, since dropping them would also remove the other classes' annotations.

Usage:
    python scripts/balance_dataset.py \\
        --dataset dataset \\
        --split train \\
        --target-class stain \\
        --reference-class weaving \\
        --ratio 1.0
"""

import argparse
import json
import random
from pathlib import Path


def downsample_class(coco: dict, split_dir: Path, target_class: str, reference_class: str, ratio: float, seed: int):
    name_to_id = {cat["name"]: cat["id"] for cat in coco["categories"]}
    if target_class not in name_to_id:
        raise SystemExit(f"Target class '{target_class}' not in dataset categories")
    if reference_class not in name_to_id:
        raise SystemExit(f"Reference class '{reference_class}' not in dataset categories")

    target_id = name_to_id[target_class]
    reference_id = name_to_id[reference_class]

    annos_by_image = {}
    for anno in coco["annotations"]:
        annos_by_image.setdefault(anno["image_id"], []).append(anno)

    reference_count = sum(1 for a in coco["annotations"] if a["category_id"] == reference_id)
    target_count = sum(1 for a in coco["annotations"] if a["category_id"] == target_id)
    goal = round(reference_count * ratio)

    print(f"Before balancing: {target_class}={target_count}, {reference_class}={reference_count}, goal<={goal}")

    if target_count <= goal:
        print("Already within ratio, nothing to drop.")
        return

    # Only consider images whose annotations are ALL target_class (safe to drop entirely).
    pure_target_image_ids = [
        image_id
        for image_id, annos in annos_by_image.items()
        if annos and all(a["category_id"] == target_id for a in annos)
    ]

    rng = random.Random(seed)
    rng.shuffle(pure_target_image_ids)

    to_drop_image_ids = set()
    dropped_annos = 0
    for image_id in pure_target_image_ids:
        if target_count - dropped_annos <= goal:
            break
        to_drop_image_ids.add(image_id)
        dropped_annos += len(annos_by_image[image_id])

    if not to_drop_image_ids:
        print("No pure-target images available to drop; target class remains mixed with others.")
        return

    id_to_filename = {img["id"]: img["file_name"] for img in coco["images"]}

    coco["images"] = [img for img in coco["images"] if img["id"] not in to_drop_image_ids]
    coco["annotations"] = [a for a in coco["annotations"] if a["image_id"] not in to_drop_image_ids]

    for image_id in to_drop_image_ids:
        file_path = split_dir / id_to_filename[image_id]
        file_path.unlink(missing_ok=True)

    new_target_count = sum(1 for a in coco["annotations"] if a["category_id"] == target_id)
    print(
        f"Dropped {len(to_drop_image_ids)} pure-{target_class} image(s) / {dropped_annos} annotation(s). "
        f"After balancing: {target_class}={new_target_count}, {reference_class}={reference_count}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="dataset", help="Dataset directory (contains train/, valid/)")
    parser.add_argument("--split", default="train", choices=["train", "valid"], help="Which split to modify")
    parser.add_argument("--target-class", required=True, help="Class to randomly downsample, e.g. stain")
    parser.add_argument("--reference-class", required=True, help="Class to balance against, e.g. weaving")
    parser.add_argument("--ratio", type=float, default=1.0, help="target_count <= ratio * reference_count")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    split_dir = Path(args.dataset) / args.split
    annotations_path = split_dir / "_annotations.coco.json"
    with open(annotations_path) as f:
        coco = json.load(f)

    downsample_class(coco, split_dir, args.target_class, args.reference_class, args.ratio, args.seed)

    with open(annotations_path, "w") as f:
        json.dump(coco, f, indent=2)

    print(f"Saved {annotations_path}")


if __name__ == "__main__":
    main()
