"""
Run the new model over the already-downloaded local dataset_july/ folder
(same layout as scripts/download_dataset.py's output: <folder>/<name>.jpg +
<name>.json), and for every image where the model detects at least one
"ignore" box, create a brand-new Label Studio task in a target project -
the detected box(es) are attached as a prediction (pre-annotation) for a
human reviewer to confirm, not a completed annotation.

Does NOT pull anything from GCS - if dataset_july/ doesn't exist locally,
this aborts with an error telling you to run scripts/download_dataset.py
first (or point --dataset at wherever it actually lives).

Each new task's "data.image" is the raw gs://<bucket>/<folder>/<filename>
URI (mirroring the local <folder>/<filename> layout back onto the GCS
bucket it was downloaded from - see scripts/download_dataset.py). This
works IF AND ONLY IF the target project already has a GCS Import Storage
configured for that exact bucket with use_blob_urls=true (check via GET
/api/storages?project=<id>) - Label Studio then matches the raw gs://
value against that storage automatically and serves it through its own
/tasks/<id>/resolve/ proxy, the same mechanism every task created by that
storage's own sync uses. Without such a storage configured, a raw gs://
URI won't render in the browser at all - use a signed HTTPS URL
(google.cloud.storage Blob.generate_signed_url) instead in that case, and
make sure the bucket's CORS policy allows the Label Studio origin.

Requires a GPU environment (torch, rfdetr, rfdetr_plus, sahi) to run
inference - not executed in this repo's dev sandbox.

Usage:
    python scripts/create_ignore_tasks_from_july.py \
      --dataset dataset_july \
      --target-project-id 25 \
      --model 1:rfdetrMedium:<weight_path> \
      --model-class-names 0:pleat,1:stain,2:weaving,3:hard_pleat,4:ignore \
      --confidence-threshold 0.5

    # Dry-run: report which images would get a new task, without creating anything
    python scripts/create_ignore_tasks_from_july.py \
      --dataset dataset_july \
      --target-project-id 25 \
      --model 1:rfdetrMedium:<weight_path> \
      --model-class-names 0:pleat,1:stain,2:weaving,3:hard_pleat,4:ignore \
      --dry-run
"""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from label_studio_sdk import Client

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.logger import setup_logger, get_logger
from utils.constants import CANONICAL_LABELS
from src.ai_verify import AIVerify

load_dotenv()

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
IGNORE_LABEL = "ignore"


def _canonical(label: str) -> str:
    return CANONICAL_LABELS.get(label, label)


def sign_gcs_url(bucket_cache: dict, bucket_name: str, blob_path: str, days: int) -> str:
    """Only used with --use-signed-url, for target projects without a GCS
    Import Storage configured (use_blob_urls=true) for --bucket. Without
    that storage, a raw gs:// URI can't be rendered by the browser at all -
    generate_signed_url() makes a plain https:// URL that works without
    changing the object's bucket ACL, but requires the bucket's CORS policy
    to allow the Label Studio origin."""
    import datetime

    from utils.gcs_utils import init_connect_gcs_bucket

    if bucket_name not in bucket_cache:
        bucket_cache[bucket_name] = init_connect_gcs_bucket(bucket_name)
    bucket = bucket_cache[bucket_name]
    blob = bucket.blob(blob_path)
    return blob.generate_signed_url(
        version="v4",
        expiration=datetime.timedelta(days=days),
        method="GET",
    )


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
    push_disagreement_predictions.py::build_prediction_result /
    build_tasks_from_data_js.py::build_task."""
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
        description="Create new Label Studio tasks (with an 'ignore' prediction) from local dataset_july/ images",
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
    parser.add_argument(
        "--use-signed-url", action="store_true",
        help="Use a signed HTTPS URL for data.image instead of a raw gs:// URI. Only needed if "
             "the target project does NOT have a GCS Import Storage configured for --bucket with "
             "use_blob_urls=true (check via GET /api/storages?project=<id>) - if it does, the raw "
             "gs:// URI (the default) resolves the same way every existing task in that project "
             "already does, with no CORS setup needed.",
    )
    parser.add_argument(
        "--signed-url-days", type=int, default=6,
        help="Validity period (days) for the signed HTTPS image URL, only used with --use-signed-url. "
             "GCS signed URLs (V4) cap out at 7 days - keep this at or below 7.",
    )
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

    logger.info("Initializing model(s)")
    verify_configs = {"image_size": args.image_size, "models": parse_models(args.model, args.model_class_names)}
    ai_verify = AIVerify(verify_configs)

    legacy_client = Client(url, api_key)
    target_project = legacy_client.get_project(args.target_project_id)
    bucket_cache = {}

    created, skipped, failed = 0, 0, 0
    for i, image_path in enumerate(images, 1):
        rel_path = image_path.relative_to(dataset_path)

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

        ignore_boxes = [a for a in final_annotations if _canonical(a.defect_type) == IGNORE_LABEL]
        if not ignore_boxes:
            skipped += 1
            continue

        width, height = image.size
        result = build_prediction_result(ignore_boxes, width, height)

        logger.info(
            f"[{i}/{len(images)}] {rel_path}: detected {len(ignore_boxes)} 'ignore' box(es), "
            f"{'(dry-run) would create' if args.dry_run else 'creating'} task in project {args.target_project_id}"
        )

        if args.dry_run:
            created += 1
            continue

        if args.use_signed_url:
            try:
                image_url = sign_gcs_url(bucket_cache, args.bucket, rel_path.as_posix(), args.signed_url_days)
            except Exception as e:
                logger.error(f"[{i}/{len(images)}] {rel_path}: failed to sign image URL: {e}")
                failed += 1
                continue
        else:
            image_url = f"gs://{args.bucket}/{rel_path.as_posix()}"

        task = {
            "data": {"image": image_url},
            "predictions": [{
                "model_version": f"ignore_from_july_{args.model[0].split(':')[0]}",
                "result": result,
            }],
        }

        try:
            target_project.import_tasks([task])
        except Exception as e:
            logger.error(f"[{i}/{len(images)}] {rel_path}: failed to create task: {e}")
            failed += 1
            continue

        created += 1

    if args.dry_run:
        logger.info(
            f"Done (dry-run). would_create={created} skipped={skipped} failed={failed} "
            f"total={len(images)} -- nothing was created"
        )
    else:
        logger.info(f"Done. created={created} skipped={skipped} failed={failed} total={len(images)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
