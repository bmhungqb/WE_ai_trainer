"""
Clone annotated tasks from a Label Studio project, re-infer each with an
RFDETR model, and move any task whose human annotation EXACTLY matches the
model's predictions (same boxes, same classes, nothing extra/missing) to a
separate "gold dataset" project.

"Exact match" here means: after greedy highest-confidence-first IoU matching
(>= --iou-threshold, same convention as scripts/push_disagreement_predictions.py
::greedy_match / has_disagreement, canonicalized via
utils/constants.py::CANONICAL_LABELS) between the model's predictions and the
human annotation's boxes,
  - every predicted box matches a human box of the SAME class, AND
  - every human box is matched, AND
  - the counts are equal (so there's no extra prediction/human box left over).
This is the exact inverse of push_disagreement_predictions.py::has_disagreement
- a task is "gold" iff it does NOT disagree.

On top of the exact-match check, a task is only eligible for gold if:
  - its human annotation contains at least one stain, weaving, and/or
    ignore box (tasks made up solely of other classes, e.g. only
    pleat/hard_pleat, are excluded), AND
  - every weaving-class human box is a tall, thin streak: height > 3x width.

Moving a task = importing it (with its existing annotation as the task's
data + a completed annotation) into the target project via the Label Studio
import API, then deleting it from the source project. The move only happens
after a successful import, so a failed import never loses the source task.

Requires a GPU environment (torch, rfdetr, rfdetr_plus, sahi) to run
inference - not executed in this repo's dev sandbox.

Usage:
    python scripts/move_gold_tasks.py \
      --source-project-id 23 --target-project-id 45 \
      --model 1:rfdetrMedium:weights/weight_rfdetr_m_june_v2.pth \
      --confidence-threshold 0.5 --iou-threshold 0.5

    # Dry-run: report which tasks would move, without importing/deleting anything
    python scripts/move_gold_tasks.py \
      --source-project-id 23 --target-project-id 45 \
      --model 1:rfdetrMedium:weights/weight_rfdetr_m_june_v2.pth \
      --dry-run
"""

import argparse
import os
import sys
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
    as [x1, y1, width, height] (pixel origin + pixel size), NOT [x1, y1, x2,
    y2] - convert here so iou()/greedy_match() (which expect [x1,y1,x2,y2],
    same as the model's Annotation.bbox) compare like with like."""
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


def is_exact_match(predicted: list, human: list, iou_threshold: float) -> bool:
    """Inverse of push_disagreement_predictions.py::has_disagreement - a task
    is "gold" iff every prediction matches a same-class human box, every
    human box gets matched, and there's no count mismatch left over."""
    if len(predicted) != len(human):
        return False
    pairs, matched = greedy_match(predicted, human, iou_threshold)
    for pred, h in pairs:
        if h is None:
            return False  # unmatched prediction
        if _canonical(pred.defect_type) != _canonical(h["label"]):
            return False  # matched but different class
    if len(matched) < len(human):
        return False  # a human box was never matched
    return True


GOLD_REQUIRED_CLASSES = {"stain", "weaving", "ignore"}
WEAVING_LABEL = "weaving"
WEAVING_MIN_HEIGHT_TO_WIDTH_RATIO = 3.0


def has_required_class(human: list) -> bool:
    """Only consider a task gold if its human annotation contains at least
    one box of stain, weaving, and/or ignore (any combination) - tasks made
    up solely of other classes (e.g. only pleat/hard_pleat) are excluded."""
    return any(_canonical(h["label"]) in GOLD_REQUIRED_CLASSES for h in human)


def weaving_boxes_pass_shape(human: list) -> bool:
    """Every weaving-class human box must be a tall, thin streak: height
    > 3x width. If any weaving box fails this, the task isn't gold."""
    for h in human:
        if _canonical(h["label"]) != WEAVING_LABEL:
            continue
        x1, y1, x2, y2 = h["bbox"]
        width, height = x2 - x1, y2 - y1
        if width <= 0 or height <= WEAVING_MIN_HEIGHT_TO_WIDTH_RATIO * width:
            return False
    return True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Move tasks whose human annotation exactly matches RFDETR predictions "
                    "into a separate gold-dataset project",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source-project-id", type=int, required=True, help="Label Studio project ID to pull tasks from")
    parser.add_argument("--target-project-id", type=int, required=True, help="Label Studio project ID to move gold tasks into")
    parser.add_argument("--image-size", type=int, nargs=2, metavar=("W", "H"), default=[576, 576], help="Expected raw image size")
    parser.add_argument(
        "--model",
        nargs="+",
        metavar="ID:TYPE:WEIGHT_PATH",
        required=True,
        help="Detection model(s) to compare against human annotations, as id:type:weight_path (repeatable)",
    )
    parser.add_argument("--confidence-threshold", type=float, default=0.5)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--min-task-id", type=int, default=None, help="Only check tasks with id >= this value")
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
    parser.add_argument(
        "--delete-source",
        action="store_true",
        help="Delete each task from the source project after it's successfully imported into "
             "the target project. Without this flag, tasks are copied but left in place "
             "(safer default - re-run with --delete-source once you've spot-checked the target project).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Report which tasks would move, but don't import or delete anything")
    parser.add_argument("--log-dir", default="./logs", help="Directory for log files")
    return parser


def parse_models(model_specs: list[str]) -> list[dict]:
    models = []
    for spec in model_specs:
        parts = spec.split(":")
        if len(parts) != 3:
            raise ValueError(f"Invalid --model format '{spec}', expected id:type:weight_path")
        models.append({"model_id": parts[0], "model_type": parts[1], "weight_path": parts[2]})
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


def import_task_to_target(target_project, target_project_id: int, task: dict) -> bool:
    """Import one task (its original "data" + its final annotation, verbatim)
    into the target project via Project.import_tasks, so the task's human
    annotation carries over as a completed annotation rather than just raw
    data."""
    logger = get_logger(__name__)
    annotations = task.get("annotations", [])
    payload = [{
        "data": task["data"],
        "annotations": [
            {"result": a["result"], "was_cancelled": a.get("was_cancelled", False)}
            for a in annotations
        ],
    }]

    try:
        target_project.import_tasks(payload)
    except Exception as e:
        logger.error(f"Failed to import task {task['id']} into project {target_project_id}: {e}")
        return False
    return True


def main():
    args = build_parser().parse_args()
    setup_logger(log_dir=args.log_dir)
    logger = get_logger(__name__)

    url = os.getenv("LABEL_STUDIO_URL")
    api_key = os.getenv("LABEL_STUDIO_API_KEY")
    if not url or not api_key:
        raise SystemExit("LABEL_STUDIO_URL / LABEL_STUDIO_API_KEY not set (check .env)")

    logger.info(f"Step 1: pulling annotated tasks from Label Studio project {args.source_project_id}")
    legacy_client = Client(url, api_key)
    source_project = legacy_client.get_project(args.source_project_id)
    target_project = legacy_client.get_project(args.target_project_id)
    tasks = fetch_tasks(source_project, args.page_size)
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
    verify_configs = {"image_size": args.image_size, "models": parse_models(args.model)}
    ai_verify = AIVerify(verify_configs)

    logger.info("Step 3: comparing predictions vs. human annotations per task")
    bucket_client_cache = {}

    moved, kept, skipped, failed = 0, 0, 0, 0
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

        if not has_required_class(human_annos):
            logger.info(f"[{i}/{len(tasks)}] task {task_id}: no stain/weaving/ignore box, skipping")
            kept += 1
            continue
        if not weaving_boxes_pass_shape(human_annos):
            logger.info(f"[{i}/{len(tasks)}] task {task_id}: weaving box fails height>3x width shape check, skipping")
            kept += 1
            continue

        try:
            image = download_image_from_gcs(bucket_client_cache, gcs_image_url)
        except Exception as e:
            logger.error(f"[{i}/{len(tasks)}] task {task_id}: failed to download image: {e}")
            failed += 1
            continue

        pre_annotations = []
        for model in ai_verify.models:
            pre_annotations.append(ai_verify.inference_with_sahi(model, image))
        final_annotations = ai_verify.merge_predictions(pre_annotations)
        final_annotations = [a for a in final_annotations if a.confidence >= args.confidence_threshold]
        if allowed_classes is not None:
            final_annotations = [a for a in final_annotations if _canonical(a.defect_type) in allowed_classes]

        if not is_exact_match(final_annotations, human_annos, args.iou_threshold):
            logger.info(f"[{i}/{len(tasks)}] task {task_id}: does not exactly match model predictions, keeping in source")
            kept += 1
            continue

        logger.info(
            f"[{i}/{len(tasks)}] task {task_id}: GOLD (exact match, {len(human_annos)} box(es)), "
            f"{'(dry-run) would move' if args.dry_run else 'moving'} to project {args.target_project_id}"
        )

        if args.dry_run:
            moved += 1
            continue

        if not import_task_to_target(target_project, args.target_project_id, task):
            failed += 1
            continue

        if args.delete_source:
            try:
                source_project.delete_task(task_id)
            except Exception as e:
                logger.error(f"[{i}/{len(tasks)}] task {task_id}: imported to target but failed to delete from source: {e}")
                failed += 1
                continue

        moved += 1

    if args.dry_run:
        logger.info(
            f"Done (dry-run). would_move={moved} kept={kept} skipped={skipped} failed={failed} "
            f"total={len(tasks)} -- nothing was imported or deleted"
        )
    else:
        logger.info(
            f"Done. moved={moved} kept={kept} skipped={skipped} failed={failed} total={len(tasks)} "
            f"(source_deleted={'yes' if args.delete_source else 'no, --delete-source not passed'})"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
