"""Turn a COCO annotations JSON (file_name = gs:// URI) into a Roboflow-style
train/valid dataset with the images downloaded locally.

Input:
    A COCO json like the ones produced by scripts/pull_label_studio_samples.py,
    where each image's "file_name" is a gs://bucket/path/to/image.png URI.

Output layout:
    dataset/
        train/
            _annotations.coco.json
            image1.png
            image2.png
            ...
        valid/
            _annotations.coco.json
            image1.png
            image2.png
            ...

Usage:
    python scripts/build_train_valid_dataset.py \
        --input tmp/annotations_project23_july_balanced_20260710_154134.json \
        --output dataset \
        --val-ratio 0.2
"""

import argparse
import json
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
from google.cloud import storage

load_dotenv()


def parse_gs_uri(uri: str):
    if not uri.startswith("gs://"):
        raise ValueError(f"Not a gs:// URI: {uri}")
    bucket_name, blob_path = uri.replace("gs://", "").split("/", 1)
    return bucket_name, blob_path


def download_image(client, bucket_cache, image, dest_dir: Path):
    bucket_name, blob_path = parse_gs_uri(image["file_name"])
    if bucket_name not in bucket_cache:
        bucket_cache[bucket_name] = client.bucket(bucket_name)
    bucket = bucket_cache[bucket_name]

    local_name = Path(blob_path).name
    local_path = dest_dir / local_name
    bucket.blob(blob_path).download_to_filename(str(local_path))
    return image["id"], local_name


def split_images(images: list, val_ratio: float, seed: int):
    rng = random.Random(seed)
    shuffled = images[:]
    rng.shuffle(shuffled)
    n_val = round(len(shuffled) * val_ratio)
    val_images = shuffled[:n_val]
    train_images = shuffled[n_val:]
    return train_images, val_images


def write_split(client, bucket_cache, images, annotations_by_image, categories, split_dir: Path, workers: int):
    split_dir.mkdir(parents=True, exist_ok=True)

    coco = {
        "info": {},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": categories,
    }

    old_to_new_image_id = {}
    downloaded_names = {}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(download_image, client, bucket_cache, img, split_dir): img for img in images}
        for i, future in enumerate(as_completed(futures), 1):
            img = futures[future]
            try:
                old_id, local_name = future.result()
                downloaded_names[old_id] = local_name
            except Exception as e:
                print(f"  FAIL {img['file_name']}: {e}")
            if i % 200 == 0:
                print(f"  ...{i}/{len(images)} downloaded")

    for img in images:
        old_id = img["id"]
        if old_id not in downloaded_names:
            continue  # download failed, drop image and its annotations
        new_id = len(coco["images"]) + 1
        old_to_new_image_id[old_id] = new_id
        coco["images"].append({
            "id": new_id,
            "task_id": img.get("task_id"),
            "file_name": downloaded_names[old_id],
            "width": img["width"],
            "height": img["height"],
        })

    for old_id, new_id in old_to_new_image_id.items():
        for anno in annotations_by_image.get(old_id, []):
            coco["annotations"].append({
                "id": len(coco["annotations"]) + 1,
                "image_id": new_id,
                "category_id": anno["category_id"],
                "bbox": anno["bbox"],
                "area": anno["area"],
                "iscrowd": anno["iscrowd"],
            })

    with open(split_dir / "_annotations.coco.json", "w") as f:
        json.dump(coco, f, indent=2)

    print(f"{split_dir}: {len(coco['images'])} images, {len(coco['annotations'])} annotations")


def build_dataset(input_path: str, output_dir: str, val_ratio: float, seed: int, workers: int):
    with open(input_path) as f:
        data = json.load(f)

    images = data["images"]
    categories = data["categories"]

    annotations_by_image = {}
    for anno in data["annotations"]:
        annotations_by_image.setdefault(anno["image_id"], []).append(anno)

    train_images, val_images = split_images(images, val_ratio, seed)
    print(f"Split: {len(train_images)} train / {len(val_images)} valid (val_ratio={val_ratio})")

    client = storage.Client()
    bucket_cache = {}
    output_path = Path(output_dir)

    print("Downloading train split...")
    write_split(client, bucket_cache, train_images, annotations_by_image, categories, output_path / "train", workers)

    print("Downloading valid split...")
    write_split(client, bucket_cache, val_images, annotations_by_image, categories, output_path / "valid", workers)

    print(f"Done. Dataset saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to input COCO json (file_name = gs:// URIs)")
    parser.add_argument("--output", default="dataset", help="Output dataset directory")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Fraction of images to put in valid/")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for the train/valid split")
    parser.add_argument("--workers", type=int, default=16, help="Parallel GCS download workers")
    args = parser.parse_args()

    build_dataset(args.input, args.output, args.val_ratio, args.seed, args.workers)


if __name__ == "__main__":
    main()
