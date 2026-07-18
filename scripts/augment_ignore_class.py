"""
Boost the "ignore" class's image count in a COCO train split (see
scripts/dataset_stats.py - train currently has only 336 images with an
ignore box, vs. 3003 for weaving) by copy-pasting real ignore-labeled crops
onto OTHER images in the dataset, each paste becoming a NEW synthetic image
+ a new "ignore" annotation at the paste location. This never touches or
overwrites existing images/annotations - every host image is written out
as a brand-new file with a new image id.

Host image selection (targets --target-count new images total):
    - ALL current negative images (no annotations at all) are used once each.
    - The remaining budget is split: --hard-pleat-pleat-share (default 0.5)
      of it drawn from images containing hard_pleat/pleat boxes, and
      --weaving-stain-share (default 0.05) from images containing
      weaving/stain boxes. Any leftover budget beyond that (since the
      shares don't have to sum to 1) is simply not used.
    - Host images can be drawn more than once if the target exceeds the
      pool size; each draw still produces a distinct output file.

Placement: for each host image, 1+ ignore crops (randomly chosen from the
pool of existing ignore-labeled boxes) are pasted at random positions that
do NOT overlap (IoU-free, checked by simple rectangle intersection) any of
the host's existing annotation boxes, so real defect labels are never
occluded/contradicted. Crops are blended in with a feathered (Gaussian
alpha) edge instead of a hard paste, to avoid an obvious seam.

Reads:
    <dataset>/train/_annotations.coco.json + images

Writes:
    <dataset>/train/_annotations.coco.json   (extended in place; a backup
        of the previous file is written alongside as *.bak)
    <dataset>/train/aug_ignore_<n>.jpg       (new synthetic images)

Usage:
    python scripts/augment_ignore_class.py --dataset dataset --target-count 3003
"""

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.logger import setup_logger, get_logger

IGNORE_CLASS_NAME = "ignore"
HARD_PLEAT_PLEAT = ("hard_pleat", "pleat")
WEAVING_STAIN = ("weaving", "stain")


def boxes_overlap(a: list, b: list) -> bool:
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    return not (ax2 <= bx1 or bx2 <= ax1 or ay2 <= by1 or by2 <= ay1)


def find_free_position(host_w: int, host_h: int, crop_w: int, crop_h: int,
                        existing_boxes: list, max_attempts: int = 40, rng: random.Random = None):
    rng = rng or random
    if crop_w >= host_w or crop_h >= host_h:
        return None
    for _ in range(max_attempts):
        x = rng.randint(0, host_w - crop_w)
        y = rng.randint(0, host_h - crop_h)
        candidate = [x, y, crop_w, crop_h]
        if not any(boxes_overlap(candidate, b) for b in existing_boxes):
            return x, y
    return None


def feathered_paste(host_img, crop_img, x: int, y: int, feather: int = 8):
    """Alpha-blend crop_img onto host_img at (x, y) with a soft (Gaussian-like
    linear ramp) edge instead of a hard rectangular seam."""
    from PIL import Image, ImageFilter
    import numpy as np

    w, h = crop_img.size
    mask = Image.new("L", (w, h), 255)
    if feather > 0 and w > feather * 2 and h > feather * 2:
        mask_arr = np.full((h, w), 255, dtype=np.uint8)
        ramp = np.linspace(0, 255, feather).astype(np.uint8)
        mask_arr[:feather, :] = np.minimum(mask_arr[:feather, :], ramp[:, None])
        mask_arr[-feather:, :] = np.minimum(mask_arr[-feather:, :], ramp[::-1][:, None])
        mask_arr[:, :feather] = np.minimum(mask_arr[:, :feather], ramp[None, :])
        mask_arr[:, -feather:] = np.minimum(mask_arr[:, -feather:], ramp[::-1][None, :])
        mask = Image.fromarray(mask_arr, mode="L").filter(ImageFilter.GaussianBlur(radius=2))

    host_img.paste(crop_img, (x, y), mask)


def load_coco(coco_path: Path) -> dict:
    with open(coco_path) as f:
        return json.load(f)


def build_ignore_crop_pool(coco: dict, images_dir: Path, ignore_cat_id: int) -> list:
    """List of (PIL.Image crop, width, height) for every ignore-labeled box."""
    from PIL import Image

    logger = get_logger(__name__)
    images_by_id = {img["id"]: img for img in coco["images"]}
    crops = []
    for anno in coco["annotations"]:
        if anno["category_id"] != ignore_cat_id:
            continue
        img_meta = images_by_id.get(anno["image_id"])
        if img_meta is None:
            continue
        image_path = images_dir / img_meta["file_name"]
        if not image_path.exists():
            continue
        x, y, w, h = [int(round(v)) for v in anno["bbox"]]
        if w <= 0 or h <= 0:
            continue
        try:
            with Image.open(image_path) as full_img:
                crop = full_img.convert("RGB").crop((x, y, x + w, y + h))
                crop.load()
        except Exception as e:
            logger.warning(f"Failed to crop ignore box from {image_path}: {e}")
            continue
        crops.append(crop)

    logger.info(f"Built ignore crop pool: {len(crops)} crops")
    return crops


def select_host_pool(coco: dict, class_names: tuple, cat_name_by_id: dict) -> list:
    """image_ids containing at least one box of any class in class_names."""
    target_ids = {cid for cid, name in cat_name_by_id.items() if name in class_names}
    image_ids = {a["image_id"] for a in coco["annotations"] if a["category_id"] in target_ids}
    return sorted(image_ids)


def augment(dataset_dir: str, target_count: int, hard_pleat_pleat_share: float,
            weaving_stain_share: float, min_crops_per_image: int, max_crops_per_image: int,
            feather: int, seed: int) -> None:
    from PIL import Image

    logger = get_logger(__name__)
    rng = random.Random(seed)

    split_dir = Path(dataset_dir) / "train"
    coco_path = split_dir / "_annotations.coco.json"
    coco = load_coco(coco_path)
    shutil.copy2(coco_path, coco_path.with_name(coco_path.name + ".bak"))

    cat_name_by_id = {c["id"]: c["name"] for c in coco["categories"]}
    name_to_cat_id = {v: k for k, v in cat_name_by_id.items()}
    ignore_cat_id = name_to_cat_id.get(IGNORE_CLASS_NAME)
    if ignore_cat_id is None:
        raise SystemExit(f"No '{IGNORE_CLASS_NAME}' category in {coco_path}")

    ignore_crops = build_ignore_crop_pool(coco, split_dir, ignore_cat_id)
    if not ignore_crops:
        raise SystemExit("No ignore-labeled boxes found to build a crop pool from")

    annos_by_image = {}
    for a in coco["annotations"]:
        annos_by_image.setdefault(a["image_id"], []).append(a)

    negative_ids = [img["id"] for img in coco["images"] if img["id"] not in annos_by_image]
    hard_pleat_pleat_ids = select_host_pool(coco, HARD_PLEAT_PLEAT, cat_name_by_id)
    weaving_stain_ids = select_host_pool(coco, WEAVING_STAIN, cat_name_by_id)

    remaining_budget = max(0, target_count - len(negative_ids))
    n_hard_pleat_pleat = round(remaining_budget * hard_pleat_pleat_share)
    n_weaving_stain = round(remaining_budget * weaving_stain_share)

    logger.info(
        f"Host pools: negatives={len(negative_ids)}, hard_pleat/pleat={len(hard_pleat_pleat_ids)}, "
        f"weaving/stain={len(weaving_stain_ids)}"
    )
    logger.info(
        f"Plan: {len(negative_ids)} from negatives (all) + {n_hard_pleat_pleat} from hard_pleat/pleat "
        f"+ {n_weaving_stain} from weaving/stain = {len(negative_ids) + n_hard_pleat_pleat + n_weaving_stain} target"
    )

    host_plan = list(negative_ids)
    if hard_pleat_pleat_ids:
        host_plan += [rng.choice(hard_pleat_pleat_ids) for _ in range(n_hard_pleat_pleat)]
    if weaving_stain_ids:
        host_plan += [rng.choice(weaving_stain_ids) for _ in range(n_weaving_stain)]

    images_by_id = {img["id"]: img for img in coco["images"]}
    next_image_id = max(img["id"] for img in coco["images"]) + 1
    next_anno_id = max((a["id"] for a in coco["annotations"]), default=0) + 1

    new_images = []
    new_annotations = []
    written, skipped_no_space = 0, 0

    for i, host_id in enumerate(host_plan, 1):
        host_meta = images_by_id[host_id]
        host_path = split_dir / host_meta["file_name"]
        if not host_path.exists():
            continue

        try:
            host_img = Image.open(host_path).convert("RGB")
        except Exception as e:
            logger.warning(f"Failed to open host {host_path}: {e}")
            continue

        existing_boxes = [a["bbox"] for a in annos_by_image.get(host_id, [])]
        n_crops = rng.randint(min_crops_per_image, max_crops_per_image)

        placed_boxes = []
        for _ in range(n_crops):
            crop = rng.choice(ignore_crops)
            cw, ch = crop.size
            pos = find_free_position(
                host_img.width, host_img.height, cw, ch,
                existing_boxes + placed_boxes, rng=rng,
            )
            if pos is None:
                continue
            x, y = pos
            feathered_paste(host_img, crop, x, y, feather=feather)
            placed_boxes.append([x, y, cw, ch])

        if not placed_boxes:
            skipped_no_space += 1
            continue

        out_name = f"aug_ignore_{i:05d}.jpg"
        out_path = split_dir / out_name
        host_img.save(out_path, quality=95)

        new_image_id = next_image_id
        next_image_id += 1
        new_images.append({
            "id": new_image_id,
            "task_id": None,
            "file_name": out_name,
            "width": host_img.width,
            "height": host_img.height,
        })
        for box in placed_boxes:
            new_annotations.append({
                "id": next_anno_id,
                "image_id": new_image_id,
                "category_id": ignore_cat_id,
                "bbox": box,
                "area": box[2] * box[3],
                "iscrowd": 0,
            })
            next_anno_id += 1
        written += 1

        if i % 200 == 0:
            logger.info(f"...{i}/{len(host_plan)} hosts processed, {written} written")

    coco["images"].extend(new_images)
    coco["annotations"].extend(new_annotations)

    with open(coco_path, "w") as f:
        json.dump(coco, f, indent=2)

    logger.info(
        f"Done. Wrote {written} new augmented image(s) "
        f"({skipped_no_space} host(s) skipped - no free space for any crop), "
        f"{len(new_annotations)} new 'ignore' annotations. "
        f"Original file backed up as {coco_path.name}.bak"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="dataset", help="Dataset directory (must have train/_annotations.coco.json)")
    parser.add_argument("--target-count", type=int, default=3003, help="Total new augmented images to generate")
    parser.add_argument("--hard-pleat-pleat-share", type=float, default=0.5,
                         help="Share of the post-negatives budget drawn from hard_pleat/pleat host images")
    parser.add_argument("--weaving-stain-share", type=float, default=0.05,
                         help="Share of the post-negatives budget drawn from weaving/stain host images")
    parser.add_argument("--min-crops-per-image", type=int, default=1, help="Min ignore crops pasted per host image")
    parser.add_argument("--max-crops-per-image", type=int, default=2, help="Max ignore crops pasted per host image")
    parser.add_argument("--feather", type=int, default=8, help="Feather blend width in pixels (0 = hard paste)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-dir", default="./logs", help="Directory for log files")
    return parser


def main():
    args = build_parser().parse_args()
    setup_logger(log_dir=args.log_dir)
    augment(
        args.dataset, args.target_count, args.hard_pleat_pleat_share, args.weaving_stain_share,
        args.min_crops_per_image, args.max_crops_per_image, args.feather, args.seed,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
