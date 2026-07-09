"""
Backfill "_captured_at" into dataset JSON sidecars downloaded before that
field existed (download_dataset.py now stamps it automatically).

Only lists GCS blob metadata (cheap) and patches the existing local JSON
files in place - does not re-download any images.

Usage:
    python scripts/backfill_capture_time.py --dataset dataset --folders TPWL TPRL
"""

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.logger import setup_logger, get_logger

load_dotenv()

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def backfill(dataset_dir: str, folders: list, bucket_name: str = "jetson-textile-storage"):
    from google.cloud import storage

    logger = get_logger(__name__)
    dataset_path = Path(dataset_dir)

    client = storage.Client()
    bucket = client.bucket(bucket_name)

    updated, already_set, missing = 0, 0, 0
    for folder in folders:
        blob_times = {
            blob.name: (blob.time_created or blob.updated)
            for blob in client.list_blobs(bucket, prefix=f"{folder}/")
        }

        for json_path in sorted((dataset_path / folder).glob("*.json")):
            image_path = next(
                (json_path.with_suffix(ext) for ext in IMAGE_EXTS if json_path.with_suffix(ext).exists()),
                None,
            )
            if image_path is None:
                continue

            try:
                with open(json_path, "r") as f:
                    sample = json.load(f)
            except (json.JSONDecodeError, OSError):
                missing += 1
                continue

            if sample.get("_captured_at"):
                already_set += 1
                continue

            blob_time = blob_times.get(f"{folder}/{image_path.name}")
            if blob_time is None:
                logger.warning(f"No blob metadata for {folder}/{image_path.name}, skipping")
                missing += 1
                continue

            sample["_captured_at"] = blob_time.date().isoformat()
            with open(json_path, "w") as f:
                json.dump(sample, f)
            updated += 1

    logger.info(f"Backfilled {updated} samples ({already_set} already set, {missing} missing/unreadable)")
    return updated


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Backfill _captured_at into dataset JSON sidecars")
    parser.add_argument("--dataset", default="dataset", help="Dataset directory")
    parser.add_argument("--folders", nargs="+", default=["TPWL", "TPRL"], help="Folder names to include")
    parser.add_argument("--bucket", default="jetson-textile-storage", help="Source GCS bucket")
    parser.add_argument("--log-dir", default="./logs", help="Directory for log files")
    return parser


def main():
    args = build_parser().parse_args()
    setup_logger(log_dir=args.log_dir)
    backfill(args.dataset, args.folders, bucket_name=args.bucket)
    return 0


if __name__ == "__main__":
    sys.exit(main())
