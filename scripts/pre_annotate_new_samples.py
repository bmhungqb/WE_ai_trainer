"""
Pull unannotated tasks created after a given date from a Label Studio project,
run inference with a trained model, and push the predictions as
pre-annotations on those same tasks (no re-import, no duplicate tasks).

Usage:
    python scripts/pre_annotate_new_samples.py \
        --project-id 23 \
        --created-after 2026-06-01 \
        --model 1:rfdetrMedium:weights/weight_rfdetr_m_v2.pth
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
from src.ai_verify import AIVerify

load_dotenv()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pre-annotate unannotated Label Studio tasks created after a given date with a trained model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--project-id", type=int, default=23, help="Label Studio project ID")
    parser.add_argument("--created-after", required=True, help="Only tasks created on/after this date (YYYY-MM-DD)")
    parser.add_argument("--created-before", default=None, help="Only tasks created on/before this date (YYYY-MM-DD)")
    parser.add_argument("--image-size", type=int, nargs=2, metavar=("W", "H"), default=[576, 576], help="Expected raw image size")
    parser.add_argument(
        "--model",
        nargs="+",
        metavar="ID:TYPE:WEIGHT_PATH",
        required=True,
        help="Detection model(s), as id:type:weight_path (repeatable)",
    )
    parser.add_argument("--page-size", type=int, default=50, help="Label Studio task pagination page size")
    parser.add_argument("--dry-run", action="store_true", help="Run inference and report, but don't push predictions")
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


def fetch_unannotated_tasks(project, created_after, created_before, page_size: int) -> list:
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
            if task.get("annotations"):
                continue  # already annotated, skip

            created_at_str = task["created_at"]
            created_date = datetime.datetime.fromisoformat(created_at_str.replace("Z", "+00:00")).date()
            if created_date < created_after:
                continue
            if created_before is not None and created_date > created_before:
                continue
            matched.append(task)

        if len(tasks) < page_size:
            break
        page += 1

    logger.info(f"Scanned {scanned} task(s), {len(matched)} unannotated and in range")
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

    created_after = datetime.datetime.strptime(args.created_after, "%Y-%m-%d").date()
    created_before = (
        datetime.datetime.strptime(args.created_before, "%Y-%m-%d").date() if args.created_before else None
    )

    logger.info(f"Step 1: pulling unannotated tasks from Label Studio project {args.project_id} "
                f"(created_after={created_after}, created_before={created_before})")
    legacy_client = Client(url, api_key)
    project = legacy_client.get_project(args.project_id)
    tasks = fetch_unannotated_tasks(project, created_after, created_before, args.page_size)
    if not tasks:
        logger.error("No matching unannotated tasks found, aborting")
        return 1

    logger.info("Step 2: initializing model(s)")
    verify_configs = {"image_size": args.image_size, "models": parse_models(args.model)}
    ai_verify = AIVerify(verify_configs)

    logger.info("Step 3: running inference per task and pushing pre-annotations")
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

        result = build_prediction_result(final_annotations, width, height)
        model_version = f"preannotate_{args.model[0].split(':')[0]}_{datetime.date.today().isoformat()}"

        logger.info(f"[{i}/{len(tasks)}] task {task_id}: {len(result)} box(es) predicted, "
                    f"{'(dry-run) would push' if args.dry_run else 'pushing'} (model_version={model_version})")

        if args.dry_run:
            continue

        try:
            ls_client.predictions.create(task=task_id, result=result, model_version=model_version)
        except Exception as e:
            logger.error(f"[{i}/{len(tasks)}] task {task_id}: failed to push prediction: {e}")
            failed += 1
            continue

        updated += 1

    if args.dry_run:
        logger.info(f"Done (dry-run). total={len(tasks)} -- nothing was pushed to Label Studio")
    else:
        logger.info(f"Done. updated={updated} skipped={skipped} failed={failed} total={len(tasks)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
