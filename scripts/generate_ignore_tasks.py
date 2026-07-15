"""
Run a newly trained model (with the new "ignore" class) over T5/T6 (May/June 2026)
TPWL + TPRL samples, and write a Label Studio task JSON containing only the samples
where the model predicted "ignore" at least once.

Same download -> AI verify -> format-for-Label-Studio pipeline as
generate_tprl_tasks.py, but adds an "ignore" filter before writing the task JSON,
and defaults to both TPWL and TPRL folders (matching plan.md's T5/T6 dataset scope).

Usage:
    python scripts/generate_ignore_tasks.py \
        --start-date 2026-05-01 --end-date 2026-06-30 \
        --model 1:rfdetrMedium:weights/weight_rfdetr_m_ignore_v1.pth
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

IGNORE_CLASS = "ignore"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a local Label Studio task JSON for T5/T6 samples with 'ignore' predictions",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--bucket", default="jetson-textile-storage", help="Source GCS bucket")
    parser.add_argument("--folders", nargs="+", default=["TPWL", "TPRL"], help="Folders inside the bucket to pull samples from")
    parser.add_argument("--start-date", default="2026-05-01", help="Start date (YYYY-MM-DD), inclusive")
    parser.add_argument("--end-date", default="2026-06-30", help="End date (YYYY-MM-DD), inclusive")
    parser.add_argument("--require-anno", action=argparse.BooleanOptionalAction, default=True, help="Require annotation sidecar file for each image")
    parser.add_argument("--image-size", type=int, nargs=2, metavar=("W", "H"), default=[576, 576], help="Expected raw image size")
    parser.add_argument(
        "--model",
        nargs="+",
        metavar="ID:TYPE:WEIGHT_PATH",
        required=True,
        help="Detection model(s) with the new 'ignore' class, as id:type:weight_path (repeatable)",
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


def filter_ignore_predictions(verified_records: list) -> list:
    return [
        record for record in verified_records
        if any(anno.defect_type == IGNORE_CLASS for anno in record.final_pre_annotations)
    ]


def main():
    args = build_parser().parse_args()
    setup_logger(log_dir=args.log_dir)
    logger = get_logger(__name__)

    output_dir = Path(args.output_dir)

    logger.info(f"Step 1: downloading samples from gs://{args.bucket}/{{{','.join(args.folders)}}} "
                f"({args.start_date} .. {args.end_date})")
    data_processor = DataProcessor()
    sample_records = data_processor.download_data_from_gcs(
        [{
            "gcs_name": args.bucket,
            "folder_paths": args.folders,
            "is_require_anno_file": args.require_anno,
        }],
        start_date_str=args.start_date,
        end_date_str=args.end_date,
    )
    logger.info(f"Downloaded {len(sample_records)} samples")
    if not sample_records:
        logger.error("No samples found for the given folder/date range, aborting")
        return 1

    logger.info("Step 2: running AI verification...")
    verify_configs = {"image_size": args.image_size, "models": parse_models(args.model)}
    ai_verify = AIVerify(verify_configs)
    folders_tag = "_".join(args.folders)
    slice_images_local_path = output_dir / f"slice_images_{folders_tag}_{args.start_date}_{args.end_date}"
    verified_records = ai_verify.predict_with_models(
        sample_records,
        slice_images_local_path,
        gcs_path=f"{args.bucket}/{folders_tag}",  # only used to build slice img_path strings; nothing is uploaded
    )
    logger.info(f"Verified {len(verified_records)} records")

    logger.info(f"Step 3: filtering for '{IGNORE_CLASS}' predictions...")
    ignore_records = filter_ignore_predictions(verified_records)
    logger.info(f"{len(ignore_records)}/{len(verified_records)} records have an '{IGNORE_CLASS}' prediction")
    if not ignore_records:
        logger.warning(f"No records with '{IGNORE_CLASS}' predictions found, nothing to write")
        return 0

    json_path = output_dir / f"tasks_ignore_{args.start_date}_{args.end_date}.json"
    data_processor.get_label_studio_format_json(ignore_records, json_path)
    logger.info(f"Wrote {json_path} (local only, not pushed to GCS or Label Studio)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
