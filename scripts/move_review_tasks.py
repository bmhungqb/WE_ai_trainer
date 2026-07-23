"""
Re-infer every non-cancelled task (annotated or not) in a Label Studio
project with an RFDETR model, and move any task whose predictions contain
a qualifying weaving box (height > 2x width) or ANY ignore box into a
separate "needs review" project.

This is a pure model-detection filter - it does NOT compare against the
task's existing human annotation (if any). The detected box(es) that
triggered the move are imported into the target project as the task's
completed annotation.

A task is skipped if its most recent annotation was cancelled
(was_cancelled=True) - same convention as
utils/label_studio_utils.py::process_task. Tasks with no annotation at all
are NOT skipped (they're exactly the "not annotated" tasks this script is
meant to also cover).

Tasks whose image URL already exists in the target project are skipped
before inference even runs, so re-running this script is safe / idempotent
- it never double-imports a task that's already been moved.

Moving a task = importing it into the target project via
Project.import_tasks, then deleting it from the source project. The move
only happens after a successful import, so a failed import never loses the
source task.

Requires a GPU environment (torch, rfdetr, rfdetr_plus, sahi) to run
inference - not executed in this repo's dev sandbox.

Usage:
    python scripts/move_review_tasks.py \
      --source-project-id 23 --target-project-id 25 \
      --model 1:rfdetrMedium:<weight_path> \
      --model-class-names 0:pleat,1:stain,2:weaving,3:ignore \
      --confidence-threshold 0.5 \
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
from src.ai_verify import AIVerify

load_dotenv()

WEAVING_LABEL = "weaving"
IGNORE_LABEL = "ignore"
WEAVING_MIN_HEIGHT_TO_WIDTH_RATIO = 2.0


def _canonical(label: str) -> str:
    return CANONICAL_LABELS.get(label, label)


def is_cancelled(task: dict) -> bool:
    """Same convention as utils/label_studio_utils.py::process_task - only
    the most recent annotation's was_cancelled flag matters. Tasks with no
    annotations at all are NOT cancelled."""
    annotations = task.get("annotations", [])
    if not annotations:
        return False
    return annotations[-1].get("was_cancelled", False)


def box_passes_shape(bbox: list, min_ratio: float) -> bool:
    x1, y1, x2, y2 = bbox
    width, height = x2 - x1, y2 - y1
    return width > 0 and height > min_ratio * width


def find_trigger_boxes(predicted: list) -> list:
    """Return predicted boxes that should trigger a move: any ignore box,
    or a weaving box whose height > WEAVING_MIN_HEIGHT_TO_WIDTH_RATIO x width."""
    triggers = []
    for pred in predicted:
        label = _canonical(pred.defect_type)
        if label == IGNORE_LABEL:
            triggers.append(pred)
        elif label == WEAVING_LABEL and box_passes_shape(pred.bbox, WEAVING_MIN_HEIGHT_TO_WIDTH_RATIO):
            triggers.append(pred)
    return triggers


def build_annotation_result(annotations, origin_width: int, origin_height: int) -> list:
    """Convert Annotation objects (pixel [x1,y1,x2,y2] bbox) into a Label
    Studio rectanglelabels result list, same format as
    push_disagreement_predictions.py::build_prediction_result."""
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
        })
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Move tasks whose predictions contain a qualifying weaving or any ignore box "
                    "into a separate review project",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source-project-id", type=int, required=True, help="Label Studio project ID to pull tasks from")
    parser.add_argument("--target-project-id", type=int, required=True, help="Label Studio project ID to move flagged tasks into")
    parser.add_argument("--image-size", type=int, nargs=2, metavar=("W", "H"), default=[576, 576], help="Expected raw image size")
    parser.add_argument(
        "--model",
        nargs="+",
        metavar="ID:TYPE:WEIGHT_PATH",
        required=True,
        help="Detection model(s) to run inference with, as id:type:weight_path (repeatable)",
    )
    parser.add_argument(
        "--model-class-names",
        default=None,
        help="Comma-separated id:name pairs mapping the model's output category ids to class "
             "names, e.g. '0:pleat,1:stain,2:weaving,3:ignore'. Defaults to the full "
             "DEFECT_CLASSES mapping if omitted - required whenever the model's category ids "
             "don't match DEFECT_CLASSES exactly.",
    )
    parser.add_argument("--confidence-threshold", type=float, default=0.5)
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


def import_task_to_target(target_project, target_project_id: int, task: dict, result: list) -> bool:
    """Import one task into the target project via Project.import_tasks, with
    the model's trigger box(es) as a single completed annotation."""
    logger = get_logger(__name__)
    payload = [{
        "data": task["data"],
        "annotations": [{"result": result, "was_cancelled": False}],
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

    logger.info(f"Step 1: pulling tasks from Label Studio project {args.source_project_id}")
    legacy_client = Client(url, api_key)
    source_project = legacy_client.get_project(args.source_project_id)
    target_project = legacy_client.get_project(args.target_project_id)
    tasks = fetch_tasks(source_project, args.page_size)
    if not tasks:
        logger.error("No tasks found, aborting")
        return 1

    logger.info(f"Fetching existing tasks in target project {args.target_project_id} to skip already-moved ones")
    target_tasks = fetch_tasks(target_project, args.page_size)
    already_moved_image_urls = {
        resolve_image_url(t.get("data", {}).get("image", "")) for t in target_tasks
    }
    already_moved_image_urls.discard("")
    logger.info(f"{len(already_moved_image_urls)} image(s) already present in target project")

    logger.info("Step 2: initializing model(s)")
    verify_configs = {"image_size": args.image_size, "models": parse_models(args.model, args.model_class_names)}
    ai_verify = AIVerify(verify_configs)

    logger.info("Step 3: running inference per task")
    bucket_client_cache = {}

    moved, kept, skipped, failed = 0, 0, 0, 0
    for i, task in enumerate(tasks, 1):
        task_id = task["id"]

        if is_cancelled(task):
            logger.info(f"[{i}/{len(tasks)}] task {task_id}: cancelled annotation, skipping")
            skipped += 1
            continue

        raw_image = task.get("data", {}).get("image", "")
        gcs_image_url = resolve_image_url(raw_image)
        if not gcs_image_url.startswith("gs://"):
            logger.warning(f"[{i}/{len(tasks)}] task {task_id}: not a GCS image URL ({gcs_image_url}), skipping")
            skipped += 1
            continue

        if gcs_image_url in already_moved_image_urls:
            logger.info(f"[{i}/{len(tasks)}] task {task_id}: already present in target project, skipping")
            skipped += 1
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

        triggers = find_trigger_boxes(final_annotations)
        if not triggers:
            kept += 1
            continue

        width, height = image.size
        result = build_annotation_result(triggers, width, height)
        trigger_classes = sorted({_canonical(t.defect_type) for t in triggers})

        logger.info(
            f"[{i}/{len(tasks)}] task {task_id}: FLAGGED ({', '.join(trigger_classes)}, {len(triggers)} box(es)), "
            f"{'(dry-run) would move' if args.dry_run else 'moving'} to project {args.target_project_id}"
        )

        if args.dry_run:
            moved += 1
            continue

        if not import_task_to_target(target_project, args.target_project_id, task, result):
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
