"""
Download images from a task JSON file.
Images are downloaded from GCS and saved locally, organized by folder name.
"""

import json
import argparse
from pathlib import Path
from dotenv import load_dotenv
from google.cloud import storage

load_dotenv()


def download_images(input_path: str, output_dir: str = "tmp/downloaded_images"):
    with open(input_path, "r") as f:
        tasks = json.load(f)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    client = storage.Client()
    bucket_cache = {}
    total = len(tasks)

    for i, task in enumerate(tasks, 1):
        image_url = task.get("data", {}).get("image", "")
        if not image_url.startswith("gs://"):
            print(f"[{i}/{total}] Skip (not GCS): {image_url}")
            continue

        parts = image_url.replace("gs://", "").split("/", 1)
        bucket_name = parts[0]
        blob_path = parts[1]

        if bucket_name not in bucket_cache:
            bucket_cache[bucket_name] = client.bucket(bucket_name)

        bucket = bucket_cache[bucket_name]
        blob = bucket.blob(blob_path)

        local_file = output_path / blob_path
        local_file.parent.mkdir(parents=True, exist_ok=True)

        try:
            blob.download_to_filename(str(local_file))
            print(f"[{i}/{total}] OK: {blob_path}")
        except Exception as e:
            print(f"[{i}/{total}] FAIL: {blob_path} - {e}")

    print(f"\nDone. Images saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download images from task JSON")
    parser.add_argument("input", help="Path to task JSON file")
    parser.add_argument("-o", "--output", default="tmp/downloaded_images", help="Output directory (default: tmp/downloaded_images)")
    args = parser.parse_args()
    download_images(args.input, args.output)
