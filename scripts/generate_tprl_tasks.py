"""
Generate a Label Studio task JSON (tmp/tasks_<start>_<end>.json) for samples in the
TPRL folder of the jetson-textile-storage GCS bucket, filtered by capture date.

Reuses the same three steps as AgenticAIPipeline.run_prepare_data_pipeline()
(download -> AI verify -> format for Label Studio) but stops there: it does NOT
push sliced images to GCS, does NOT push the task JSON to GCS, and does NOT push
samples to Label Studio. The output file is written locally for review only.

Usage:
    python scripts/generate_tprl_tasks.py \
        --start-date 2026-05-01 --end-date 2026-06-30
"""

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.logger import setup_logger, get_logger
from src.data_processor import DataProcessor
from src.ai_verify import AIVerify

load_dotenv()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a local Label Studio task JSON for TPRL samples in a date range",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--bucket", default="jetson-textile-storage", help="Source GCS bucket")
    parser.add_argument("--folder", default="TPRL", help="Folder inside the bucket to pull samples from")
    parser.add_argument("--start-date", default="2026-05-01", help="Start date (YYYY-MM-DD), inclusive")
    parser.add_argument("--end-date", default="2026-06-30", help="End date (YYYY-MM-DD), inclusive")
    parser.add_argument("--require-anno", action=argparse.BooleanOptionalAction, default=True, help="Require annotation sidecar file for each image")
    parser.add_argument("--image-size", type=int, nargs=2, metavar=("W", "H"), default=[576, 576], help="Expected raw image size")
    parser.add_argument(
        "--model",
        nargs="+",
        metavar="ID:TYPE:WEIGHT_PATH",
        default=["1:rfdetrMedium:weights/weight_rfdetr_m_slice_dinov3_v3.pth"],
        help="Detection models as id:type:weight_path (repeatable)",
    )
    parser.add_argument("--output-dir", default="tmp", help="Local directory to write the task JSON and slice images into")
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


def main():
    args = build_parser().parse_args()
    setup_logger(log_dir=args.log_dir)
    logger = get_logger(__name__)

    output_dir = Path(args.output_dir)

    # Step 1: download sample metadata from GCS, filtered to TPRL + date range
    logger.info(f"Step 1: downloading samples from gs://{args.bucket}/{args.folder} "
                f"({args.start_date} .. {args.end_date})")
    data_processor = DataProcessor()
    sample_records = data_processor.download_data_from_gcs(
        [{
            "gcs_name": args.bucket,
            "folder_paths": [args.folder],
            "is_require_anno_file": args.require_anno,
        }],
        start_date_str=args.start_date,
        end_date_str=args.end_date,
    )
    logger.info(f"Downloaded {len(sample_records)} samples")
    if not sample_records:
        logger.error("No samples found for the given folder/date range, aborting")
        return 1

    # Step 2: run AI verification (predictions) on those samples
    logger.info("Step 2: running AI verification...")
    verify_configs = {"image_size": args.image_size, "models": parse_models(args.model)}
    ai_verify = AIVerify(verify_configs)
    slice_images_local_path = output_dir / f"slice_images_{args.folder}_{args.start_date}_{args.end_date}"
    verified_records = ai_verify.predict_with_models(
        sample_records,
        slice_images_local_path,
        gcs_path=f"{args.bucket}/{args.folder}",  # only used to build slice img_path strings; nothing is uploaded
    )
    logger.info(f"Verified {len(verified_records)} records")

    # Step 3: format predictions as a Label Studio task JSON, saved locally only
    json_path = output_dir / f"tasks_{args.folder}_{args.start_date}_{args.end_date}.json"
    data_processor.get_label_studio_format_json(verified_records, json_path)
    logger.info(f"✓ Wrote {json_path} (local only, not pushed to GCS or Label Studio)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
