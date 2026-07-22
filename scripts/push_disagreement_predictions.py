"""
Re-infer every annotated Label Studio task with a new model, IoU-match its
predictions against the task's existing human annotation, and push the new
model's boxes as a NEW prediction version ONLY on tasks where it disagrees
with the human annotation - surfacing exactly the cases worth a reviewer's
attention instead of flooding every task with a redundant prediction.

Disagreement (same IoU-matching convention as
scripts/compute_comparison_metrics.py::greedy_match, canonicalized via
utils/constants.py::CANONICAL_LABELS so old/new label vocabularies don't
cause false "disagreements"): a task is flagged if, after greedy
highest-confidence-first IoU matching (>= --iou-threshold) between the new
model's predictions and the human annotation's boxes,
  - any predicted box has no matching human box (a possible false
    positive / extra detection), OR
  - any predicted box matches a human box of a DIFFERENT class
    (a possible mislabel), OR
  - any human box is left unmatched (a possible missed detection).
Tasks where every predicted box matches a same-class human box, and every
human box is matched, are left alone - no prediction pushed.

Like scripts/update_ignore_predictions.py, this only ADDS a new prediction
(via the Label Studio predictions API, tagged with a distinct
model_version) - it never modifies or deletes the existing human
annotation, which remains the source of truth.

Usage:
    python scripts/push_disagreement_predictions.py \
      --project-id 23 \
      --model 1:rfdetrMedium:weights/weight_rfdetr_m_june_v2.pth \
      --confidence-threshold 0.5 --iou-threshold 0.5

    # Only re-check tasks cloned for the latest training run (task_id >= 198255),
    # and only compare on the classes that model was actually trained on -
    # human boxes for any other class (e.g. pleat/hard_pleat) are dropped
    # before matching, so they can't show up as false "missed detections":
    python scripts/push_disagreement_predictions.py \
      --project-id 23 \
      --model 1:rfdetrMedium:weights/weight_rfdetr_m_stain_weaving_ignore.pth \
      --min-task-id 198255 \
      --classes stain weaving ignore \
      --confidence-threshold 0.5 --iou-threshold 0.5
"""

import argparse
import datetime
import os
import sys
from io import BytesIO
from pathlib import Path

from dotenv import load_dotenv
from label_studio_sdk import Client, LabelStudio
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
    as [x1, y1, width, height] (pixel origin + pixel size), NOT [x1, y1, x2,
    y2] - convert here so iou()/greedy_match() (which expect [x1,y1,x2,y2],
    same as the model's Annotation.bbox) compare like with like. Without
    this, width/height get treated as x2/y2 and every human box's "end"
    corner lands far short of where it should, producing IoU ~0 against
    even a pixel-perfect matching prediction."""
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


def greedy_match(predicted: list, human: list, iou_threshold: float):
    """predicted: list of Annotation (bbox, confidence, defect_type).
    human: list of {"label", "bbox", ...} (from process_task).
    Returns (pairs, matched_human_idx_set) - pairs is
    [(pred, matched_human_or_None), ...], highest-confidence prediction first."""
    matched = set()
    pairs = []
    for pred in sorted(predicted, key=lambda p: p.confidence, reverse=True):
        best_idx, best_iou = None, 0.0
        for i, h in enumerate(human):
            if i in matched:
                continue
            score = iou(pred.bbox, h["bbox"])
            if score > best_iou:
                best_idx, best_iou = i, score
        if best_idx is not None and best_iou >= iou_threshold:
            matched.add(best_idx)
            pairs.append((pred, human[best_idx]))
        else:
            pairs.append((pred, None))
    return pairs, matched


def has_disagreement(predicted: list, human: list, iou_threshold: float) -> bool:
    pairs, matched = greedy_match(predicted, human, iou_threshold)
    for pred, h in pairs:
        if h is None:
            return True  # unmatched prediction - possible false positive
        if _canonical(pred.defect_type) != _canonical(h["label"]):
            return True  # matched but different class - possible mislabel
    if len(matched) < len(human):
        return True  # a human box was never matched - possible missed detection
    return False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Push new-model predictions only for tasks that disagree with the existing human annotation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--project-id", type=int, default=23, help="Label Studio project ID")
    parser.add_argument("--image-size", type=int, nargs=2, metavar=("W", "H"), default=[576, 576], help="Expected raw image size")
    parser.add_argument(
        "--model",
        nargs="+",
        metavar="ID:TYPE:WEIGHT_PATH",
        required=True,
        help="Detection model(s) to compare against human annotations, as id:type:weight_path (repeatable)",
    )
    parser.add_argument(
        "--model-class-names",
        default=None,
        help="Comma-separated class names in the model's output category-id order, e.g. "
             "'stain,weaving' for a 2-class checkpoint. Applies to all --model entries. "
             "Defaults to the full DEFECT_CLASSES mapping if omitted - required whenever the "
             "model wasn't trained on the full 5-class vocabulary, otherwise predicted "
             "category ids get mapped to the wrong class names.",
    )
    parser.add_argument("--confidence-threshold", type=float, default=0.5)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument(
        "--min-task-id", type=int, default=None, help="Only re-check tasks with id >= this value"
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=None,
        metavar="CLASS",
        help="Restrict comparison to these class names (canonicalized via CANONICAL_LABELS): "
             "human boxes and model predictions for any other class are dropped before "
             "IoU-matching. Defaults to all classes if omitted.",
    )
    parser.add_argument("--page-size", type=int, default=50, help="Label Studio task pagination page size")
    parser.add_argument("--dry-run", action="store_true", help="Run inference and report disagreements, but don't push predictions")
    parser.add_argument("--log-dir", default="./logs", help="Directory for log files")
    return parser


def parse_models(model_specs: list[str], class_names: str | None) -> list[dict]:
    category_mapping = (
        {i: name.strip() for i, name in enumerate(class_names.split(","))} if class_names else None
    )
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


def build_prediction_result(annotations, origin_width: int, origin_height: int) -> list:
    result = []
    for anno in annotations:
        bbox = anno.bbox
        if not bbox or len(bbox) != 4 or any(v is None or v != v or v < 0 for v in bbox):
            continue
        result.append({
            "from_name": "label",
            "to_name": "image",
            "type": "rectanglelabels",
            "original_width": origin_width,
            "original_height": origin_height,
            "value": {
                "x": (bbox[0] / origin_width) * 100,
                "y": (bbox[1] / origin_height) * 100,
                "width": ((bbox[2] - bbox[0]) / origin_width) * 100,
                "height": ((bbox[3] - bbox[1]) / origin_height) * 100,
                "rotation": 0,
                "rectanglelabels": [anno.defect_type],
            },
            "score": anno.confidence,
        })
    return result


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

    allowed_classes = {_canonical(c) for c in args.classes} if args.classes else None
    if allowed_classes:
        logger.info(f"Restricting comparison to classes: {sorted(allowed_classes)}")

    logger.info("Step 2: initializing model(s)")
    verify_configs = {"image_size": args.image_size, "models": parse_models(args.model, args.model_class_names)}
    ai_verify = AIVerify(verify_configs)

    logger.info("Step 3: comparing new model predictions vs. human annotations per task")
    ls_client = LabelStudio(base_url=url, api_key=api_key)
    bucket_client_cache = {}

    disagreements, agreements, skipped, failed = 0, 0, 0, 0
    for i, task in enumerate(tasks, 1):
        task_id = task["id"]

        sample = process_task(task)
        if not sample:
            logger.info(f"[{i}/{len(tasks)}] task {task_id}: no human annotation, skipping")
            skipped += 1
            continue
        human_annos = to_xyxy(sample["annos"])
        if allowed_classes is not None:
            human_annos = [h for h in human_annos if _canonical(h["label"]) in allowed_classes]

        raw_image = task.get("data", {}).get("image", "")
        gcs_image_url = resolve_image_url(raw_image)
        if not gcs_image_url.startswith("gs://"):
            logger.warning(f"[{i}/{len(tasks)}] task {task_id}: not a GCS image URL ({gcs_image_url}), skipping")
            skipped += 1
            continue

        try:
            image = download_image_from_gcs(bucket_client_cache, gcs_image_url)
        except Exception as e:
            logger.error(f"[{i}/{len(tasks)}] task {task_id}: failed to download image: {e}")
            failed += 1
            continue

        width, height = image.size
        pre_annotations = []
        for model in ai_verify.models:
            pre_annotations.append(ai_verify.inference_with_sahi(model, image))
        final_annotations = ai_verify.merge_predictions(pre_annotations)
        final_annotations = [a for a in final_annotations if a.confidence >= args.confidence_threshold]
        if allowed_classes is not None:
            final_annotations = [a for a in final_annotations if _canonical(a.defect_type) in allowed_classes]

        if not has_disagreement(final_annotations, human_annos, args.iou_threshold):
            logger.info(f"[{i}/{len(tasks)}] task {task_id}: agrees with human annotation, skipping")
            agreements += 1
            continue

        disagreements += 1
        result = build_prediction_result(final_annotations, width, height)
        model_version = f"disagreement_{args.model[0].split(':')[0]}_{datetime.date.today().isoformat()}"

        logger.info(
            f"[{i}/{len(tasks)}] task {task_id}: DISAGREES with human annotation, "
            f"{'(dry-run) would push' if args.dry_run else 'pushing'} prediction "
            f"({len(result)} boxes, model_version={model_version})"
        )

        if args.dry_run:
            continue

        try:
            ls_client.predictions.create(task=task_id, result=result, model_version=model_version)
        except Exception as e:
            logger.error(f"[{i}/{len(tasks)}] task {task_id}: failed to push prediction: {e}")
            failed += 1
            continue

    if args.dry_run:
        logger.info(
            f"Done (dry-run). disagreements={disagreements} agreements={agreements} "
            f"skipped={skipped} failed={failed} total={len(tasks)} -- nothing was pushed"
        )
    else:
        logger.info(
            f"Done. disagreements_pushed={disagreements} agreements={agreements} "
            f"skipped={skipped} failed={failed} total={len(tasks)}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
