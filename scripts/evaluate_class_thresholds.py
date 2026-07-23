"""
Pull annotated tasks from a Label Studio project, run an RFDETR model with
NO confidence filtering, and compute a per-class precision-recall curve
(swept over confidence threshold) plus the threshold that maximizes F1 for
each class - answers "what --class-confidence-threshold should I actually
use per class" before running scripts/move_review_tasks.py.

Same PR-curve math as scripts/compute_pr_curves.py::match_class_predictions
/ build_pr_curve (greedy highest-confidence-first IoU matching per image,
each GT box claimed at most once), but sources predictions/ground truth
live from a Label Studio project instead of results/**/*.json. Labels are
canonicalized via utils/constants.py::CANONICAL_LABELS so old/new label
vocabularies compare consistently.

Requires a GPU environment (torch, rfdetr, rfdetr_plus, sahi) to run
inference - not executed in this repo's dev sandbox.

Reads:
    Label Studio project (LABEL_STUDIO_URL / LABEL_STUDIO_API_KEY from .env)

Writes:
    reports/class_thresholds.json

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
    python scripts/evaluate_class_thresholds.py \
      --project-id 25 \
      --model 1:rfdetrMedium:<weight_path> \
      --model-class-names 0:pleat,1:stain,2:weaving,3:ignore \
      --iou-threshold 0.5
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from io import BytesIO
from pathlib import Path

from dotenv import load_dotenv
from label_studio_sdk import Client
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.logger import setup_logger, get_logger
from utils.constants import CANONICAL_LABELS
from utils.label_studio_utils import process_task
from src.ai_verify import AIVerify

load_dotenv()


def _canonical(label: str) -> str:
    return CANONICAL_LABELS.get(label, label)


def to_xyxy(human_annos: list) -> list:
    """utils/label_studio_utils.py::process_task returns each anno's "bbox"
    as [x1, y1, width, height] - convert to [x1, y1, x2, y2] to match the
    model's Annotation.bbox convention."""
    converted = []
    for h in human_annos:
        x1, y1, w, h_ = h["bbox"]
        converted.append({**h, "bbox": [x1, y1, x1 + w, y1 + h_]})
    return converted


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
    ALL for one class, gathered across the whole project. Returns a list of
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
    parser.add_argument("--project-id", type=int, required=True, help="Label Studio project ID to evaluate against")
    parser.add_argument("--image-size", type=int, nargs=2, metavar=("W", "H"), default=[576, 576], help="Expected raw image size")
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
             "names, e.g. '0:pleat,1:stain,2:weaving,3:ignore'. Defaults to the full "
             "DEFECT_CLASSES mapping if omitted.",
    )
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--min-task-id", type=int, default=None, help="Only evaluate tasks with id >= this value")
    parser.add_argument("--page-size", type=int, default=50, help="Label Studio task pagination page size")
    parser.add_argument("--report", default="reports/class_thresholds.json", help="Output report path")
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


def resolve_image_url(raw_image: str) -> str:
    """Handle Label Studio local-storage proxy URLs (data/local-files/?d=...&fileuri=...)."""
    import base64
    if "fileuri=" in raw_image:
        b64_str = raw_image.split("fileuri=")[-1].split("&")[0]
        return base64.b64decode(b64_str).decode("utf-8")
    return raw_image


def fetch_tasks(project, page_size: int) -> list:
    logger = get_logger(__name__)
    tasks = []
    page = 1
    while True:
        try:
            resp = project.get_paginated_tasks(page=page, page_size=page_size)
            page_tasks = resp.get("tasks", [])
        except AttributeError:
            page_tasks = project.get_tasks() if page == 1 else []

        if not page_tasks:
            break
        tasks.extend(page_tasks)
        if len(page_tasks) < page_size:
            break
        page += 1

    logger.info(f"Fetched {len(tasks)} task(s) from project")
    return tasks


def download_image_from_gcs(bucket_client_cache: dict, gcs_image_url: str) -> Image.Image:
    from utils.gcs_utils import init_connect_gcs_bucket

    without_scheme = gcs_image_url[len("gs://"):]
    bucket_name, blob_path = without_scheme.split("/", 1)

    if bucket_name not in bucket_client_cache:
        bucket_client_cache[bucket_name] = init_connect_gcs_bucket(bucket_name)
    bucket = bucket_client_cache[bucket_name]

    blob = bucket.blob(blob_path)
    image_bytes = blob.download_as_bytes()
    return Image.open(BytesIO(image_bytes)).convert("RGB")


def main():
    args = build_parser().parse_args()
    setup_logger(log_dir=args.log_dir)
    logger = get_logger(__name__)

    url = os.getenv("LABEL_STUDIO_URL")
    api_key = os.getenv("LABEL_STUDIO_API_KEY")
    if not url or not api_key:
        raise SystemExit("LABEL_STUDIO_URL / LABEL_STUDIO_API_KEY not set (check .env)")

    logger.info(f"Step 1: pulling annotated tasks from Label Studio project {args.project_id}")
    legacy_client = Client(url, api_key)
    project = legacy_client.get_project(args.project_id)
    tasks = fetch_tasks(project, args.page_size)
    if not tasks:
        logger.error("No tasks found, aborting")
        return 1
    if args.min_task_id is not None:
        tasks = [t for t in tasks if t.get("id", 0) >= args.min_task_id]
        logger.info(f"Filtered to {len(tasks)} task(s) with id >= {args.min_task_id}")

    logger.info("Step 2: initializing model(s) (confidence_threshold=0.0 - no filtering, we need the full score range)")
    verify_configs = {
        "image_size": args.image_size,
        "models": parse_models(args.model, args.model_class_names),
        "confidence_threshold": 0.0,
    }
    ai_verify = AIVerify(verify_configs)

    logger.info("Step 3: running inference per task and collecting per-class predictions vs. GT")
    bucket_client_cache = {}

    # class -> list of (confidence, bbox, gt_boxes_for_this_image)
    per_class_preds = defaultdict(list)
    gt_counts = defaultdict(int)

    n_evaluated, n_skipped, n_failed = 0, 0, 0
    for i, task in enumerate(tasks, 1):
        task_id = task["id"]

        sample = process_task(task)
        if not sample:
            n_skipped += 1
            continue
        human_annos = to_xyxy(sample["annos"])

        raw_image = task.get("data", {}).get("image", "")
        gcs_image_url = resolve_image_url(raw_image)
        if not gcs_image_url.startswith("gs://"):
            n_skipped += 1
            continue

        try:
            image = download_image_from_gcs(bucket_client_cache, gcs_image_url)
        except Exception as e:
            logger.error(f"[{i}/{len(tasks)}] task {task_id}: failed to download image: {e}")
            n_failed += 1
            continue

        pre_annotations = []
        for model in ai_verify.models:
            pre_annotations.append(ai_verify.inference_with_sahi(model, image))
        final_annotations = ai_verify.merge_predictions(pre_annotations)

        gt_by_class = defaultdict(list)
        for h in human_annos:
            cls = _canonical(h["label"])
            gt_by_class[cls].append(h["bbox"])
            gt_counts[cls] += 1

        for pred in final_annotations:
            cls = _canonical(pred.defect_type)
            gt_boxes = gt_by_class.get(cls, [])
            per_class_preds[cls].append((pred.confidence, pred.bbox, gt_boxes))

        n_evaluated += 1
        if n_evaluated % 50 == 0:
            logger.info(f"[{i}/{len(tasks)}] evaluated {n_evaluated} task(s) so far")

    logger.info(f"Done evaluating: {n_evaluated} task(s), skipped={n_skipped}, failed={n_failed}")

    classes = sorted(gt_counts.keys())
    report = {"project_id": args.project_id, "samples": n_evaluated, "iou_threshold": args.iou_threshold, "classes": {}}

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
