"""Randomly downsample negative images (images with zero annotations) in a
built train/valid COCO dataset so negatives make up roughly --target-ratio
of the split's total image count (default 10%).

Usage:
    python scripts/reduce_negative_samples.py --dataset dataset_gold --target-ratio 0.10
"""

import argparse
import json
import random
from pathlib import Path


def reduce_negatives(coco: dict, split_dir: Path, target_ratio: float, seed: int):
    images_with_annos = {a["image_id"] for a in coco["annotations"]}
    negative_images = [img for img in coco["images"] if img["id"] not in images_with_annos]
    positive_count = len(coco["images"]) - len(negative_images)

    # Solve for the negative count N such that N / (positive_count + N) = target_ratio.
    if target_ratio >= 1:
        raise SystemExit("--target-ratio must be < 1")
    goal = round(target_ratio * positive_count / (1 - target_ratio))

    print(f"Before: positive={positive_count}, negative={len(negative_images)}, goal_negative<={goal}")

    if len(negative_images) <= goal:
        print("Already within target ratio, nothing to drop.")
        return

    rng = random.Random(seed)
    rng.shuffle(negative_images)

    to_drop = negative_images[: len(negative_images) - goal]
    to_drop_ids = {img["id"] for img in to_drop}

    id_to_filename = {img["id"]: img["file_name"] for img in coco["images"]}

    coco["images"] = [img for img in coco["images"] if img["id"] not in to_drop_ids]

    for image_id in to_drop_ids:
        file_path = split_dir / id_to_filename[image_id]
        file_path.unlink(missing_ok=True)

    new_negative_count = len(negative_images) - len(to_drop)
    new_total = positive_count + new_negative_count
    print(
        f"Dropped {len(to_drop_ids)} negative image(s). "
        f"After: positive={positive_count}, negative={new_negative_count}, "
        f"negative_ratio={new_negative_count / new_total:.3f}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="dataset", help="Dataset directory (contains train/, valid/)")
    parser.add_argument(
        "--target-ratio", type=float, default=0.10,
        help="Target fraction of negative (zero-annotation) images in each split's total image count",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dataset_path = Path(args.dataset)

    for split in ("train", "valid"):
        split_dir = dataset_path / split
        annotations_path = split_dir / "_annotations.coco.json"
        if not annotations_path.exists():
            print(f"Skipping {split}: {annotations_path} not found")
            continue

        with open(annotations_path) as f:
            coco = json.load(f)

        print(f"-- {split} --")
        reduce_negatives(coco, split_dir, args.target_ratio, args.seed)

        with open(annotations_path, "w") as f:
            json.dump(coco, f, indent=2)
        print(f"Saved {annotations_path}")


if __name__ == "__main__":
    main()
