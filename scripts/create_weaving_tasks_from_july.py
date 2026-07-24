"""
Run the new model over the already-downloaded local dataset_july/ folder
(same layout as scripts/download_dataset.py's output: <folder>/<name>.jpg +
<name>.json), and for every image where EITHER

    - the model detects at least one "weaving" box, OR
    - the sample's own worker-corrected ground truth ("gt" field in the
      JSON sidecar) is Loi_soi / weaving (same label, old VN-underscore vs.
      new English name - see utils/constants.py::CANONICAL_LABELS)

create a brand-new Label Studio task in a target project. Only the model's
detected boxes are attached as a prediction (pre-annotation) for a human
reviewer to confirm - ground truth alone, with no model detection, still
qualifies the image but contributes no prediction boxes (nothing to
attach), so the reviewer draws the box themselves.

Only new images are created: any image already present in the target
project (matched by its GCS gs://<bucket>/<folder>/<filename> path,
independent of the signed-URL query string) is skipped entirely.

Does NOT pull anything from GCS - if dataset_july/ doesn't exist locally,
this aborts with an error telling you to run scripts/download_dataset.py
first (or point --dataset at wherever it actually lives).

Each new task's "data.image" is the raw gs://<bucket>/<folder>/<filename>
URI. This project already has a GCS Import Storage configured for this
exact bucket with use_blob_urls=true (verified via GET
/api/storages?project=<id>) - Label Studio matches a raw gs:// data.image
value against that storage automatically and serves it through its own
/tasks/<id>/resolve/ proxy, the same mechanism every existing task in this
project already uses. No signed URL / CORS setup needed.

Requires a GPU environment (torch, rfdetr, rfdetr_plus, sahi) to run
inference - not executed in this repo's dev sandbox.

Usage:
    python scripts/create_weaving_tasks_from_july.py \
      --dataset dataset_july \
      --target-project-id 25 \
      --model 1:rfdetrMedium:<weight_path> \
      --model-class-names 0:pleat,1:stain,2:weaving,3:hard_pleat,4:ignore \
      --confidence-threshold 0.5

    # Dry-run: report which images would get a new task, without creating anything
    python scripts/create_weaving_tasks_from_july.py \
      --dataset dataset_july \
      --target-project-id 25 \
      --model 1:rfdetrMedium:<weight_path> \
      --model-class-names 0:pleat,1:stain,2:weaving,3:hard_pleat,4:ignore \
      --dry-run
"""

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from label_studio_sdk import Client

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.logger import setup_logger, get_logger
from utils.constants import CANONICAL_LABELS, DEFECT_CLASSES
from src.ai_verify import AIVerify

load_dotenv()

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
WEAVING_LABEL = "weaving"


def _canonical(label: str) -> str:
    return CANONICAL_LABELS.get(label, label)


def _gt_label_for(raw_label) -> str:
    """Same normalization as merge_annotations.py::_label_for - a raw "gt"
    value can be an int class index or any of the old VN-underscore / new
    English label vocabularies; canonicalize onto one class name."""
    if raw_label is None:
        return "unknown"
    if isinstance(raw_label, str) and raw_label.lstrip("-").isdigit():
        raw_label = int(raw_label)
    if isinstance(raw_label, int):
        raw_label = DEFECT_CLASSES.get(raw_label, str(raw_label))
    return CANONICAL_LABELS.get(raw_label, str(raw_label))


def ground_truth_is_weaving(sample_json: dict) -> bool:
    return _gt_label_for(sample_json.get("gt")) == WEAVING_LABEL


def resolve_image_url(raw_image: str) -> str:
    """Handle Label Studio local-storage proxy URLs (data/local-files/?d=...&fileuri=...)."""
    import base64
    if "fileuri=" in raw_image:
        b64_str = raw_image.split("fileuri=")[-1].split("&")[0]
        return base64.b64decode(b64_str).decode("utf-8")
    return raw_image


def gcs_path_from_url(url: str) -> str:
    """Normalize any of: a raw gs://bucket/path URI, a signed
    https://storage.googleapis.com/bucket/path?... URL, or a Label Studio
    local-files proxy URL, down to just "bucket/path" - so existing tasks
    (which may carry any of these forms depending on how they were
    created) and freshly-signed URLs can be deduplicated against each
    other regardless of query string / scheme differences."""
    resolved = resolve_image_url(url)
    if resolved.startswith("gs://"):
        return resolved[len("gs://"):]
    if "storage.googleapis.com/" in resolved:
        return resolved.split("storage.googleapis.com/", 1)[1].split("?")[0]
    return resolved.split("?")[0]


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


def build_prediction_result(annotations, origin_width: int, origin_height: int) -> list:
    """Same percent-based rectanglelabels format as
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
            "score": anno.confidence,
        })
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create new Label Studio tasks for July images with a detected or "
                    "ground-truth weaving defect",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", default="dataset_july", help="Local dataset directory (output of scripts/download_dataset.py)")
    parser.add_argument("--bucket", default="jetson-textile-storage", help="GCS bucket to reconstruct the gs:// image URI from")
    parser.add_argument("--target-project-id", type=int, required=True, help="Label Studio project ID to create new tasks in")
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
             "names, e.g. '0:pleat,1:stain,2:weaving,3:hard_pleat,4:ignore'. Defaults to the "
             "full DEFECT_CLASSES mapping if omitted.",
    )
    parser.add_argument("--image-size", type=int, nargs=2, metavar=("W", "H"), default=[576, 576], help="Expected raw image size")
    parser.add_argument("--confidence-threshold", type=float, default=0.5)
    parser.add_argument("--page-size", type=int, default=50, help="Label Studio task pagination page size")
    parser.add_argument("--dry-run", action="store_true", help="Report which images would get a new task, but don't create anything")
    parser.add_argument("--log-dir", default="./logs", help="Directory for log files")
    return parser


def main():
    args = build_parser().parse_args()
    setup_logger(log_dir=args.log_dir)
    logger = get_logger(__name__)

    dataset_path = Path(args.dataset)
    if not dataset_path.is_dir():
        raise SystemExit(
            f"{dataset_path} does not exist. This script only reads an already-downloaded "
            f"local dataset - run scripts/download_dataset.py --output {dataset_path} first, "
            f"or pass --dataset pointing at wherever the July data actually lives."
        )

    url = os.getenv("LABEL_STUDIO_URL")
    api_key = os.getenv("LABEL_STUDIO_API_KEY")
    if not url or not api_key:
        raise SystemExit("LABEL_STUDIO_URL / LABEL_STUDIO_API_KEY not set (check .env)")

    from PIL import Image

    images = sorted(p for p in dataset_path.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    if not images:
        logger.error(f"No images found under {dataset_path}, aborting")
        return 1
    logger.info(f"Found {len(images)} image(s) under {dataset_path}")

    legacy_client = Client(url, api_key)
    target_project = legacy_client.get_project(args.target_project_id)

    logger.info(f"Fetching existing tasks in target project {args.target_project_id} to skip already-present images")
    existing_tasks = fetch_tasks(target_project, args.page_size)
    already_present_paths = {
        gcs_path_from_url(t.get("data", {}).get("image", "")) for t in existing_tasks
    }
    already_present_paths.discard("")
    logger.info(f"{len(already_present_paths)} image(s) already present in target project")

    logger.info("Initializing model(s)")
    verify_configs = {"image_size": args.image_size, "models": parse_models(args.model, args.model_class_names)}
    ai_verify = AIVerify(verify_configs)

    created, skipped_existing, skipped_no_match, failed = 0, 0, 0, 0
    for i, image_path in enumerate(images, 1):
        rel_path = image_path.relative_to(dataset_path)
        gcs_path = f"{args.bucket}/{rel_path.as_posix()}"

        if gcs_path in already_present_paths:
            skipped_existing += 1
            continue

        json_path = image_path.with_suffix(".json")
        gt_is_weaving = False
        if json_path.exists():
            try:
                with open(json_path) as f:
                    sample_json = json.load(f)
                gt_is_weaving = ground_truth_is_weaving(sample_json)
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"[{i}/{len(images)}] {rel_path}: broken JSON sidecar, treating gt as unknown: {e}")

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            logger.error(f"[{i}/{len(images)}] {rel_path}: failed to open image: {e}")
            failed += 1
            continue

        pre_annotations = []
        for model in ai_verify.models:
            pre_annotations.append(ai_verify.inference_with_sahi(model, image))
        final_annotations = ai_verify.merge_predictions(pre_annotations)
        final_annotations = [a for a in final_annotations if a.confidence >= args.confidence_threshold]

        weaving_boxes = [a for a in final_annotations if _canonical(a.defect_type) == WEAVING_LABEL]
        model_detected_weaving = bool(weaving_boxes)

        if not model_detected_weaving and not gt_is_weaving:
            skipped_no_match += 1
            continue

        width, height = image.size
        result = build_prediction_result(weaving_boxes, width, height)
        reasons = []
        if model_detected_weaving:
            reasons.append(f"model detected {len(weaving_boxes)} weaving box(es)")
        if gt_is_weaving:
            reasons.append("ground truth is Loi_soi/weaving")

        logger.info(
            f"[{i}/{len(images)}] {rel_path}: {' + '.join(reasons)}, "
            f"{'(dry-run) would create' if args.dry_run else 'creating'} task in project {args.target_project_id}"
        )

        if args.dry_run:
            created += 1
            continue

        image_url = f"gs://{gcs_path}"

        task = {"data": {"image": image_url}}
        if result:
            task["predictions"] = [{
                "model_version": f"weaving_from_july_{args.model[0].split(':')[0]}",
                "result": result,
            }]

        try:
            target_project.import_tasks([task])
        except Exception as e:
            logger.error(f"[{i}/{len(images)}] {rel_path}: failed to create task: {e}")
            failed += 1
            continue

        created += 1

    if args.dry_run:
        logger.info(
            f"Done (dry-run). would_create={created} skipped_existing={skipped_existing} "
            f"skipped_no_match={skipped_no_match} failed={failed} total={len(images)} -- nothing was created"
        )
    else:
        logger.info(
            f"Done. created={created} skipped_existing={skipped_existing} "
            f"skipped_no_match={skipped_no_match} failed={failed} total={len(images)}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
