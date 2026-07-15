"""
Pull existing tasks from a Label Studio project (already-imported June samples),
re-run inference with a new rfdetr model that has the "ignore" class, and push
a new prediction (new model_version) onto each task where "ignore" was detected.

Unlike generate_ignore_tasks.py (which re-downloads raw samples from GCS and
writes a brand-new local task JSON), this script works against tasks that are
ALREADY in Label Studio: it reads task["data"]["image"] (a GCS URI) to fetch
the image, runs the new model, and calls the Label Studio predictions API to
attach a fresh pre-annotation to that same task_id -- no re-import, no
duplicate tasks.

Usage:
    python scripts/update_ignore_predictions.py \
        --project-id 23 \
        --start-date 2026-06-01 --end-date 2026-06-30 \
        --model 1:rfdetrMedium:weights/weight_rfdetr_m_ignore_v1.pth
"""

import argparse
import datetime
import os
import sys
from io import BytesIO
from pathlib import Path

import requests
from dotenv import load_dotenv
from label_studio_sdk import Client, LabelStudio
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.logger import setup_logger, get_logger
from src.ai_verify import AIVerify

load_dotenv()

IGNORE_CLASS = "ignore"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Re-infer existing Label Studio tasks with the new ignore-class model "
                    "and push updated pre-annotations for tasks where 'ignore' is predicted",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--project-id", type=int, default=23, help="Label Studio project ID")
    parser.add_argument("--start-date", default="2026-06-01", help="Start date (YYYY-MM-DD), inclusive, filtered on task created_at")
    parser.add_argument("--end-date", default="2026-06-30", help="End date (YYYY-MM-DD), inclusive, filtered on task created_at")
    parser.add_argument("--image-size", type=int, nargs=2, metavar=("W", "H"), default=[576, 576], help="Expected raw image size")
    parser.add_argument(
        "--model",
        nargs="+",
        metavar="ID:TYPE:WEIGHT_PATH",
        required=True,
        help="Detection model(s) with the new 'ignore' class, as id:type:weight_path (repeatable)",
    )
    parser.add_argument("--page-size", type=int, default=50, help="Label Studio task pagination page size")
    parser.add_argument("--dry-run", action="store_true", help="Run inference and report matches, but don't push predictions")
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


def fetch_tasks_in_range(project, start_date, end_date, page_size: int) -> list:
    logger = get_logger(__name__)
    matched = []
    page = 1
    scanned = 0
    while True:
        try:
            resp = project.get_paginated_tasks(page=page, page_size=page_size)
            tasks = resp.get("tasks", [])
        except AttributeError:
            tasks = project.get_tasks() if page == 1 else []

        if not tasks:
            break
        scanned += len(tasks)

        for task in tasks:
            created_at_str = task["created_at"]
            created_date = datetime.datetime.fromisoformat(created_at_str.replace("Z", "+00:00")).date()
            if start_date <= created_date <= end_date:
                matched.append(task)

        if len(tasks) < page_size:
            break
        page += 1

    logger.info(f"Scanned {scanned} task(s), {len(matched)} in range [{start_date}, {end_date}]")
    return matched


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

    start_date = datetime.datetime.strptime(args.start_date, "%Y-%m-%d").date()
    end_date = datetime.datetime.strptime(args.end_date, "%Y-%m-%d").date()

    logger.info(f"Step 1: pulling tasks from Label Studio project {args.project_id} "
                f"({args.start_date} .. {args.end_date})")
    legacy_client = Client(url, api_key)
    project = legacy_client.get_project(args.project_id)
    tasks = fetch_tasks_in_range(project, start_date, end_date, args.page_size)
    if not tasks:
        logger.error("No tasks found in the given date range, aborting")
        return 1

    logger.info(f"Step 2: initializing model(s) with '{IGNORE_CLASS}' class support")
    verify_configs = {"image_size": args.image_size, "models": parse_models(args.model)}
    ai_verify = AIVerify(verify_configs)

    logger.info("Step 3: running inference per task and checking for 'ignore' predictions")
    ls_client = LabelStudio(base_url=url, api_key=api_key)
    bucket_client_cache = {}

    updated, skipped, failed = 0, 0, 0
    for i, task in enumerate(tasks, 1):
        task_id = task["id"]
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

        has_ignore = any(anno.defect_type == IGNORE_CLASS for anno in final_annotations)
        if not has_ignore:
            logger.info(f"[{i}/{len(tasks)}] task {task_id}: no '{IGNORE_CLASS}' prediction, skipping")
            skipped += 1
            continue

        result = build_prediction_result(final_annotations, width, height)
        model_version = f"ignore_reinfer_{args.model[0].split(':')[0]}_{datetime.date.today().isoformat()}"

        logger.info(f"[{i}/{len(tasks)}] task {task_id}: '{IGNORE_CLASS}' detected, "
                    f"{'(dry-run) would push' if args.dry_run else 'pushing'} prediction "
                    f"({len(result)} boxes, model_version={model_version})")

        if not args.dry_run:
            try:
                ls_client.predictions.create(task=task_id, result=result, model_version=model_version)
            except Exception as e:
                logger.error(f"[{i}/{len(tasks)}] task {task_id}: failed to push prediction: {e}")
                failed += 1
                continue

        updated += 1

    logger.info(f"Done. updated={updated} skipped={skipped} failed={failed} total={len(tasks)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
