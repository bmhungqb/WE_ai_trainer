"""
Main entry point for Agentic AI Textile Defect Detection system.
All pipeline configuration is provided via CLI arguments.
"""

import sys
import argparse

from utils.logger import setup_logger, get_logger
from utils.config import config as app_config
from src.agentic_pipeline import AgenticAIPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Agentic AI Textile Defect Detection Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--mode",
        choices=["prepare", "train", "all"],
        default="prepare",
        help="Pipeline mode",
    )

    # ── Date range ──
    date = parser.add_argument_group("date range")
    date.add_argument("--start-date", required=True, help="Start date (YYYY-MM-DD)")
    date.add_argument("--end-date", required=True, help="End date (YYYY-MM-DD)")

    # ── GCS source ──
    gcs = parser.add_argument_group("GCS source")
    gcs.add_argument("--gcs-bucket", default="jetson-textile-storage", help="Source GCS bucket name")
    gcs.add_argument("--gcs-folder-paths", nargs="+", default=["N"], help="Folder paths inside source bucket")
    gcs.add_argument("--require-anno", action=argparse.BooleanOptionalAction, default=True, help="Require annotation file for each image")

    # ── Verify / models ──
    verify = parser.add_argument_group("verification models")
    verify.add_argument("--image-size", type=int, nargs=2, metavar=("W", "H"), default=[576, 576], help="Expected image size")
    verify.add_argument(
        "--model",
        nargs="+",
        metavar="ID:TYPE:WEIGHT_PATH",
        default=["1:rfdetrMedium:weights/weight_rfdetr_m_slice_dinov3_v3.pth"],
        help="Detection models as id:type:weight_path (repeatable)",
    )

    # ── Output ──
    out = parser.add_argument_group("output")
    out.add_argument("--output-gcs-bucket", default="textile-datasets", help="Destination GCS bucket for outputs")
    out.add_argument("--output-tasks-path", default="auto_training/tasks", help="GCS path for annotation task JSONs")
    out.add_argument("--output-images-path", default="auto_training/data", help="GCS path for sliced images")
    out.add_argument("--output-gcs-models", default="textile-datasets/auto_training/outputs", help="GCS path for trained model outputs")

    # ── Label Studio ──
    ls = parser.add_argument_group("Label Studio")
    ls.add_argument("--label-studio-project-id", type=int, default=22, help="Label Studio project ID")

    # ── Training ──
    train = parser.add_argument_group("training")
    train.add_argument("--pretrained-weights", default="weights/weight_rfdetr_m_slice_dinov3_v3.pth", help="Pretrained weights for training")
    train.add_argument("--n-trials", type=int, default=10, help="Number of Optuna trials")
    train.add_argument("--dataset-path", default="", help="Path to training dataset (usually set by merge step)")

    # ── Data merge ──
    merge = parser.add_argument_group("data merge")
    merge.add_argument("--new-data-ratio", type=float, default=0.4, help="Ratio of new data in merged training set")
    merge.add_argument("--old-data-ratio", type=float, default=0.6, help="Ratio of old data in merged training set")
    merge.add_argument("--split-ratio", type=float, nargs=3, metavar=("TRAIN", "VAL", "TEST"), default=[0.7, 0.2, 0.1], help="Train/val/test split ratios")
    merge.add_argument("--split-info-file", default="", help="Path to split info JSON (optional; derived automatically if absent)")

    # ── Logging ──
    log = parser.add_argument_group("logging")
    log.add_argument("--log-dir", default="./logs", help="Directory for log files")

    return parser


def parse_models(model_specs: list[str]) -> list[dict]:
    models = []
    for spec in model_specs:
        parts = spec.split(":")
        if len(parts) != 3:
            raise ValueError(f"Invalid --model format '{spec}', expected id:type:weight_path")
        models.append({
            "model_id": parts[0],
            "model_type": parts[1],
            "weight_path": parts[2],
        })
    return models


def build_config(args) -> dict:
    return {
        "date": {
            "start": args.start_date,
            "end": args.end_date,
        },
        "label_studio_configs": {
            "project_id": args.label_studio_project_id,
        },
        "data_pipeline": {
            "gcs_buckets": [
                {
                    "gcs_name": args.gcs_bucket,
                    "folder_paths": args.gcs_folder_paths,
                    "is_require_anno_file": args.require_anno,
                }
            ],
            "verify_configs": {
                "image_size": args.image_size,
                "models": parse_models(args.model),
            },
            "output_configs": {
                "gcs_bucket_name": args.output_gcs_bucket,
                "gcs_tasks_folder_path": args.output_tasks_path,
                "gcs_sliced_images_folder_path": args.output_images_path,
            },
        },
        "training_pipeline": {
            "data_merge_config": {
                "mixing_ratio": {
                    "new_data_ratio": args.new_data_ratio,
                    "old_data_ratio": args.old_data_ratio,
                    "new_split_ratio": args.split_ratio,
                },
                "split_info_file": args.split_info_file,
            },
            "ai_trainer_configs": {
                "dataset_path": args.dataset_path,
                "pretrained_weights": args.pretrained_weights,
                "n_trials": args.n_trials,
            },
            "evaluation_configs": {
                "metrics": ["map50", "recall", "f1_score"],
            },
            "output_gcs_models": args.output_gcs_models,
        },
        "data_management": {
            "mixing_ratio": {
                "new_split_ratio": args.split_ratio,
            },
        },
        "logging": {
            "log_dir": args.log_dir,
        },
    }


def main():
    parser = build_parser()
    args = parser.parse_args()

    setup_logger(log_dir=args.log_dir)
    logger = get_logger(__name__)

    logger.info("=" * 80)
    logger.info("AGENTIC AI TEXTILE DEFECT DETECTION SYSTEM")
    logger.info(f"Mode: {args.mode}")
    logger.info("=" * 80)

    try:
        pipeline_config = build_config(args)
        app_config.configure(pipeline_config)

        pipeline = AgenticAIPipeline()

        if args.mode == "prepare":
            result = pipeline.run_prepare_data_pipeline()
        elif args.mode == "train":
            result = pipeline.run_training_pipeline()
        else:
            result = pipeline.run_complete_pipeline()

        if result["status"] == "success":
            logger.info("PIPELINE EXECUTION COMPLETED SUCCESSFULLY")
            return 0
        else:
            logger.error(f"Pipeline execution failed: {result.get('error', 'Unknown error')}")
            return 1

    except Exception as e:
        logger.error(f"Unexpected error occurred: {str(e)}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
