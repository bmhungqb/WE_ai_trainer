"""
Read a local COCO dataset (e.g. dataset_gold/, output of
build_train_valid_dataset.py), run an RFDETR model built with a near-zero
confidence threshold (--min-confidence, default 0.01 - effectively no
filtering), and compute a per-class precision-recall curve (swept over
confidence threshold) plus the threshold that maximizes F1 for each class -
answers "what --class-confidence-threshold should I actually use per
class" before running scripts/move_review_tasks.py.

Same PR-curve math as scripts/compute_pr_curves.py::match_class_predictions
/ build_pr_curve (greedy highest-confidence-first IoU matching per image,
each GT box claimed at most once). Unlike
scripts/evaluate_class_thresholds.py (which pulls tasks + images from a
Label Studio project over the network), this reads images and ground truth
directly off disk - no LABEL_STUDIO_URL / GCS access needed, just the
dataset directory.

Requires a GPU environment (torch, rfdetr, rfdetr_plus, sahi) to run
inference - not executed in this repo's dev sandbox.

Reads:
    <dataset>/train/_annotations.coco.json + images
    <dataset>/valid/_annotations.coco.json + images

Writes:
    reports/class_thresholds_local.json

    {
      "stain": {
        "curve": [{"threshold": 0.95, "precision": 1.0, "recall": 0.01, "f1": ...}, ...],
        "best_f1": {"threshold": 0.4, "precision": ..., "recall": ..., "f1": ...},
        "num_gt": 706
      },
      "weaving": {...},
      ...
    }

Usage:
    python scripts/evaluate_class_thresholds_local.py \
      --dataset dataset_gold \
      --model 1:rfdetrMedium:<weight_path> \
      --model-class-names 0:pleat,1:stain,2:weaving,4:ignore \
      --iou-threshold 0.5

    # Only evaluate one split instead of both
    python scripts/evaluate_class_thresholds_local.py \
      --dataset dataset_gold --split valid \
      --model 1:rfdetrMedium:<weight_path> \
      --model-class-names 0:pleat,1:stain,2:weaving,4:ignore
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.logger import setup_logger, get_logger
from utils.constants import CANONICAL_LABELS
from src.ai_verify import AIVerify


def _canonical(label: str) -> str:
    return CANONICAL_LABELS.get(label, label)


def iou(box1: list, box2: list) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union else 0.0


def match_class_predictions(class_preds: list, iou_threshold: float) -> list:
    """class_preds: list of (confidence, bbox, gt_boxes_in_same_image) tuples,
    ALL for one class, gathered across the whole dataset. Returns a list of
    (confidence, is_tp) sorted by confidence descending - greedy per-image
    matching against that image's same-class GT boxes only (a GT box can
    only be claimed once WITHIN its own image, standard per-class AP)."""
    by_image = defaultdict(list)
    for conf, bbox, gt_boxes in class_preds:
        by_image[id(gt_boxes)].append((conf, bbox, gt_boxes))

    results = []
    for group in by_image.values():
        group.sort(key=lambda p: p[0], reverse=True)
        gt_boxes = group[0][2]
        claimed = set()
        for conf, bbox, _ in group:
            best_idx, best_iou = None, 0.0
            for i, gt_bbox in enumerate(gt_boxes):
                if i in claimed:
                    continue
                score = iou(bbox, gt_bbox)
                if score > best_iou:
                    best_idx, best_iou = i, score
            if best_idx is not None and best_iou >= iou_threshold:
                claimed.add(best_idx)
                results.append((conf, True))
            else:
                results.append((conf, False))

    results.sort(key=lambda r: r[0], reverse=True)
    return results


def build_pr_curve(matches: list, num_gt: int) -> dict:
    """matches: (confidence, is_tp) sorted by confidence descending.
    Returns the full curve (one point per distinct confidence threshold)
    plus the max-F1 point."""
    curve = []
    tp = 0
    fp = 0
    best = {"threshold": 1.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    i = 0
    n = len(matches)
    while i < n:
        threshold = matches[i][0]
        while i < n and matches[i][0] == threshold:
            if matches[i][1]:
                tp += 1
            else:
                fp += 1
            i += 1

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / num_gt if num_gt else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        point = {"threshold": threshold, "precision": precision, "recall": recall, "f1": f1}
        curve.append(point)
        if f1 > best["f1"]:
            best = point

    return {"curve": curve, "best_f1": best, "num_gt": num_gt}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, help="Dataset directory (contains train/, valid/, each with _annotations.coco.json)")
    parser.add_argument("--split", choices=["train", "valid", "both"], default="both", help="Which split(s) to evaluate")
    parser.add_argument(
        "--model",
        nargs="+",
        metavar="ID:TYPE:WEIGHT_PATH",
        required=True,
        help="Detection model(s) to evaluate, as id:type:weight_path (repeatable)",
    )
    parser.add_argument(
        "--model-class-names",
        default=None,
        help="Comma-separated id:name pairs mapping the model's output category ids to class "
             "names, e.g. '0:pleat,1:stain,2:weaving,4:ignore'. Defaults to the full "
             "DEFECT_CLASSES mapping if omitted.",
    )
    parser.add_argument("--image-size", type=int, nargs=2, metavar=("W", "H"), default=[576, 576], help="Expected raw image size")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument(
        "--min-confidence", type=float, default=0.01,
        help="Confidence threshold to build the model(s) with. Must stay just above 0.0 - at "
             "exactly 0.0, SAHI's postprocessing lets through degenerate/negative-coordinate "
             "boxes from the raw model output that its own BoundingBox validation then rejects, "
             "crashing get_sliced_prediction. A small positive value still captures effectively "
             "the full score range for the PR curve while avoiding that crash.",
    )
    parser.add_argument("--report", default="reports/class_thresholds_local.json", help="Output report path")
    parser.add_argument("--log-dir", default="./logs", help="Directory for log files")
    return parser


def parse_models(model_specs: list[str], class_names: str | None) -> list[dict]:
    category_mapping = None
    if class_names:
        category_mapping = {}
        for entry in class_names.split(","):
            id_str, name = entry.split(":", 1)
            category_mapping[int(id_str)] = name.strip()
    models = []
    for spec in model_specs:
        parts = spec.split(":")
        if len(parts) != 3:
            raise ValueError(f"Invalid --model format '{spec}', expected id:type:weight_path")
        model = {"model_id": parts[0], "model_type": parts[1], "weight_path": parts[2]}
        if category_mapping is not None:
            model["category_mapping"] = category_mapping
        models.append(model)
    return models


def load_coco_split(split_dir: Path):
    annotations_path = split_dir / "_annotations.coco.json"
    with open(annotations_path) as f:
        coco = json.load(f)

    cat_names = {c["id"]: c["name"] for c in coco["categories"]}

    gt_by_image = defaultdict(list)
    for a in coco["annotations"]:
        x, y, w, h = a["bbox"]
        cls = _canonical(cat_names.get(a["category_id"], str(a["category_id"])))
        gt_by_image[a["image_id"]].append({"bbox": [x, y, x + w, y + h], "label": cls})

    images = [
        {"id": img["id"], "path": split_dir / img["file_name"], "annos": gt_by_image.get(img["id"], [])}
        for img in coco["images"]
    ]
    return images


def main():
    args = build_parser().parse_args()
    setup_logger(log_dir=args.log_dir)
    logger = get_logger(__name__)

    from PIL import Image

    dataset_path = Path(args.dataset)
    splits = ["train", "valid"] if args.split == "both" else [args.split]

    images = []
    for split in splits:
        split_dir = dataset_path / split
        annotations_path = split_dir / "_annotations.coco.json"
        if not annotations_path.exists():
            logger.warning(f"Skipping {split}: {annotations_path} not found")
            continue
        split_images = load_coco_split(split_dir)
        logger.info(f"{split}: {len(split_images)} image(s)")
        images.extend(split_images)

    if not images:
        logger.error("No images found, aborting")
        return 1

    logger.info(f"Step 1: initializing model(s) (confidence_threshold={args.min_confidence} - near-zero, we need the full score range)")
    verify_configs = {
        "image_size": args.image_size,
        "models": parse_models(args.model, args.model_class_names),
        "confidence_threshold": args.min_confidence,
    }
    ai_verify = AIVerify(verify_configs)

    logger.info("Step 2: running inference per image and collecting per-class predictions vs. GT")

    per_class_preds = defaultdict(list)
    gt_counts = defaultdict(int)

    n_evaluated, n_failed = 0, 0
    for i, img in enumerate(images, 1):
        if not img["path"].exists():
            logger.warning(f"[{i}/{len(images)}] missing image file {img['path']}, skipping")
            n_failed += 1
            continue

        try:
            image = Image.open(img["path"]).convert("RGB")
        except Exception as e:
            logger.error(f"[{i}/{len(images)}] failed to open {img['path']}: {e}")
            n_failed += 1
            continue

        pre_annotations = []
        for model in ai_verify.models:
            pre_annotations.append(ai_verify.inference_with_sahi(model, image))
        final_annotations = ai_verify.merge_predictions(pre_annotations)

        gt_by_class = defaultdict(list)
        for h in img["annos"]:
            gt_by_class[h["label"]].append(h["bbox"])
            gt_counts[h["label"]] += 1

        for pred in final_annotations:
            cls = _canonical(pred.defect_type)
            gt_boxes = gt_by_class.get(cls, [])
            per_class_preds[cls].append((pred.confidence, pred.bbox, gt_boxes))

        n_evaluated += 1
        if n_evaluated % 50 == 0:
            logger.info(f"[{i}/{len(images)}] evaluated {n_evaluated} image(s) so far")

    logger.info(f"Done evaluating: {n_evaluated} image(s), failed={n_failed}")

    classes = sorted(gt_counts.keys())
    report = {"dataset": args.dataset, "splits": splits, "samples": n_evaluated, "iou_threshold": args.iou_threshold, "classes": {}}

    for cls in classes:
        matches = match_class_predictions(per_class_preds.get(cls, []), args.iou_threshold)
        report["classes"][cls] = build_pr_curve(matches, gt_counts[cls])
        best = report["classes"][cls]["best_f1"]
        logger.info(
            f"  {cls}: best F1={best['f1']:.3f} at threshold={best['threshold']:.3f} "
            f"(P={best['precision']:.3f} R={best['recall']:.3f}, {gt_counts[cls]} GT boxes)"
        )

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Report written to {report_path}")

    print("\nSuggested --class-confidence-threshold value:")
    print(",".join(f"{cls}:{report['classes'][cls]['best_f1']['threshold']:.2f}" for cls in classes))

    return 0


if __name__ == "__main__":
    sys.exit(main())
