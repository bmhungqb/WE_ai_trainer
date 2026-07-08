"""
Task 1 - Download production dataset directly from GCS (bucket: jetson-textile-storage).

Each sample lives as a flat trio directly under a folder (TPWL/, TPRL/):
    TPWL/<name>.jpg
    TPWL/<name>.json   {"d": <predicted label>, "pos": "cx cy w h [score]" (or list of such
                         strings, normalized 0-1), "gt": <true label>}
    TPWL/<name>.txt     YOLO-format lines (not used by this pipeline)

Both the predicted label ("d") and the true label ("gt") describe the same
box(es) in "pos" - Task 4 (merge_annotations.py) turns this into separate
production/ground_truth annotation lists.

Output layout:
    dataset/
        TPWL/
            image001.jpg
            image001.json
        TPRL/
            image002.png
            image002.json

Usage:
    python scripts/download_dataset.py \
        --folders TPWL TPRL \
        --start-date 2026-05-01 --end-date 2026-06-30 \
        --output dataset
"""

import argparse
import datetime
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.logger import setup_logger, get_logger

load_dotenv()

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def find_samples(bucket, folder: str):
    """Group blobs under folder/ by stem, yielding (stem, image_blob, json_blob) for
    pairs that have both an image and a JSON sidecar."""
    by_stem = {}
    for blob in bucket.client.list_blobs(bucket, prefix=f"{folder}/"):
        suffix = Path(blob.name).suffix.lower()
        stem = blob.name[: -len(suffix)] if suffix else blob.name
        entry = by_stem.setdefault(stem, {})
        if suffix in IMAGE_EXTS:
            entry["image"] = blob
        elif suffix == ".json":
            entry["json"] = blob

    for stem, entry in sorted(by_stem.items()):
        if "image" in entry and "json" in entry:
            yield stem, entry["image"], entry["json"]


def download_dataset(output_dir: str, folders: list, start_date: str, end_date: str,
                      bucket_name: str = "jetson-textile-storage", limit: int = None):
    from google.cloud import storage

    logger = get_logger(__name__)
    output_path = Path(output_dir)

    start = datetime.datetime.strptime(start_date, "%Y-%m-%d").date()
    end = datetime.datetime.strptime(end_date, "%Y-%m-%d").date()

    client = storage.Client()
    bucket = client.bucket(bucket_name)

    downloaded, skipped = 0, 0
    for folder in folders:
        folder_dir = output_path / folder
        for stem, image_blob, json_blob in find_samples(bucket, folder):
            if limit is not None and downloaded >= limit:
                break

            sample_date = (image_blob.time_created or image_blob.updated).date()
            if not (start <= sample_date <= end):
                continue

            folder_dir.mkdir(parents=True, exist_ok=True)
            filename = Path(image_blob.name).name
            local_image = folder_dir / filename
            local_json = folder_dir / f"{Path(filename).stem}.json"

            try:
                image_blob.download_to_filename(str(local_image))
                json_blob.download_to_filename(str(local_json))
                downloaded += 1
                logger.info(f"OK: {folder}/{filename} ({sample_date})")
            except Exception as e:
                skipped += 1
                logger.error(f"FAIL: {folder}/{filename} - {e}")

    logger.info(f"Done. Downloaded {downloaded} samples, skipped {skipped}. Output: {output_path}")
    return downloaded, skipped


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download production dataset from GCS")
    parser.add_argument("--bucket", default="jetson-textile-storage", help="Source GCS bucket")
    parser.add_argument("--folders", nargs="+", default=["TPWL", "TPRL"], help="Folder names to include")
    parser.add_argument("--start-date", default="2026-05-01", help="Start date (YYYY-MM-DD), inclusive")
    parser.add_argument("--end-date", default="2026-06-30", help="End date (YYYY-MM-DD), inclusive")
    parser.add_argument("--output", default="dataset", help="Output directory")
    parser.add_argument("--limit", type=int, default=None, help="Max samples to download (for testing)")
    parser.add_argument("--log-dir", default="./logs", help="Directory for log files")
    return parser


def main():
    args = build_parser().parse_args()
    setup_logger(log_dir=args.log_dir)

    download_dataset(args.output, args.folders, args.start_date, args.end_date,
                      bucket_name=args.bucket, limit=args.limit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
