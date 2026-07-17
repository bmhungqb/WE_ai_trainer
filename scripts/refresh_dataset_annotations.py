"""
Refresh only the "annotations" (and "categories") in an existing local
train/valid COCO dataset (the layout produced by
scripts/build_train_valid_dataset.py) from the current state of Label
Studio - without re-downloading any images.

Use this after fixing labels in Label Studio (e.g. via the
extract_missed_as_ignore.py -> annotate_manifest_with_task_id.py workflow):
the images on disk haven't changed, only the annotations, so there's no
need to re-pull them from GCS.

How it matches: each image entry in <dataset>/train|valid/_annotations.coco.json
carries a "task_id" (written by build_train_valid_dataset.py at build time).
This script fetches every annotated task from the given Label Studio
project, decodes its current box/label annotations the same way
scripts/pull_label_studio_samples.py does, and replaces the "annotations"
array for exactly the images whose task_id matches - by image_id, so
existing image ids/file_names/width/height are left untouched. Local images
with no matching Label Studio task (e.g. it was deleted, or the project ID
is wrong) keep their old annotations and are reported as unmatched -
skipped, not dropped.

Reads:
    <dataset>/train/_annotations.coco.json
    <dataset>/valid/_annotations.coco.json
    Label Studio project (LABEL_STUDIO_URL / LABEL_STUDIO_API_KEY from .env)

Writes (in place):
    <dataset>/train/_annotations.coco.json
    <dataset>/valid/_annotations.coco.json
    (the previous file is backed up alongside as _annotations.coco.json.bak)

Usage:
    python scripts/refresh_dataset_annotations.py --dataset dataset --project-id 23
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.constants import DEFECT_CLASSES
from utils.label_studio_utils import process_task
from utils.logger import setup_logger, get_logger

load_dotenv()

SPLITS = ("train", "valid")


def fetch_all_tasks(project, page_size: int) -> list:
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


def build_annotations_by_task_id(tasks: list, label_to_id: dict) -> dict:
    """task_id -> list of COCO-style annotation dicts (no id/image_id yet -
    those are assigned per split when writing out)."""
    logger = get_logger(__name__)
    by_task_id = {}
    for task in tasks:
        sample = process_task(task)
        if not sample:
            continue  # unannotated / cancelled task
        annos = []
        for anno in sample["annos"]:
            annos.append({
                "category_id": label_to_id.get(anno["label"], -1),
                "bbox": anno["bbox"],
                "area": anno["area"],
                "iscrowd": anno["iscrowd"],
            })
        by_task_id[sample["task_id"]] = annos
    logger.info(f"Decoded current annotations for {len(by_task_id)} task(s)")
    return by_task_id


def refresh_split(split_dir: Path, annotations_by_task_id: dict, categories: list) -> tuple:
    logger = get_logger(__name__)
    coco_path = split_dir / "_annotations.coco.json"
    if not coco_path.exists():
        logger.warning(f"{coco_path} not found, skipping split")
        return 0, 0, 0

    with open(coco_path, "r") as f:
        coco = json.load(f)

    shutil.copy2(coco_path, coco_path.with_name(coco_path.name + ".bak"))

    matched, unmatched = 0, 0
    new_annotations = []
    for img in coco["images"]:
        task_id = img.get("task_id")
        if task_id is None or task_id not in annotations_by_task_id:
            # No matching task - keep this image's existing annotations untouched.
            kept = [a for a in coco["annotations"] if a["image_id"] == img["id"]]
            new_annotations.extend(kept)
            unmatched += 1
            continue

        matched += 1
        for anno in annotations_by_task_id[task_id]:
            new_annotations.append({
                "id": len(new_annotations) + 1,
                "image_id": img["id"],
                "category_id": anno["category_id"],
                "bbox": anno["bbox"],
                "area": anno["area"],
                "iscrowd": anno["iscrowd"],
            })

    # Re-number ids sequentially after the merge above.
    for i, anno in enumerate(new_annotations, 1):
        anno["id"] = i

    coco["annotations"] = new_annotations
    if categories:
        coco["categories"] = categories

    with open(coco_path, "w") as f:
        json.dump(coco, f, indent=2)

    logger.info(
        f"{split_dir}: {matched} image(s) refreshed, {unmatched} unmatched (kept as-is), "
        f"{len(new_annotations)} total annotations written"
    )
    return matched, unmatched, len(new_annotations)


def refresh_dataset(dataset_dir: str, project_id: int, page_size: int) -> None:
    from label_studio_sdk import Client
    import os

    logger = get_logger(__name__)

    url = os.getenv("LABEL_STUDIO_URL")
    api_key = os.getenv("LABEL_STUDIO_API_KEY")
    if not url or not api_key:
        raise SystemExit("LABEL_STUDIO_URL / LABEL_STUDIO_API_KEY not set (check .env)")

    label_to_id = {v: k for k, v in DEFECT_CLASSES.items()}
    categories = [{"id": k, "name": v, "supercategory": "defect"} for k, v in DEFECT_CLASSES.items()]

    ls = Client(url, api_key)
    project = ls.get_project(project_id)
    tasks = fetch_all_tasks(project, page_size)
    annotations_by_task_id = build_annotations_by_task_id(tasks, label_to_id)

    dataset_path = Path(dataset_dir)
    for split in SPLITS:
        refresh_split(dataset_path / split, annotations_by_task_id, categories)

    logger.info("Done. Image files were not touched - only *_annotations.coco.json* files were rewritten.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="dataset", help="Existing train/valid dataset directory to refresh in place")
    parser.add_argument("--project-id", type=int, default=23, help="Label Studio project ID")
    parser.add_argument("--page-size", type=int, default=50, help="Label Studio task pagination page size")
    parser.add_argument("--log-dir", default="./logs", help="Directory for log files")
    return parser


def main():
    args = build_parser().parse_args()
    setup_logger(log_dir=args.log_dir)
    refresh_dataset(args.dataset, args.project_id, args.page_size)
    return 0


if __name__ == "__main__":
    sys.exit(main())
