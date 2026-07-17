"""
Join a manifest.json (from scripts/extract_missed_as_ignore.py or
scripts/extract_false_positives.py) with Label Studio task IDs, so each
flagged sample can be opened directly in Label Studio for a labeling fix.

The dataset was pulled straight from GCS (scripts/download_dataset.py), not
from Label Studio, so there's no task_id anywhere in dataset_june/ or
results/ to begin with. This script bridges that gap by listing every task
in a Label Studio project, resolving each task's image field to its GCS URI
(same decoding as scripts/update_ignore_predictions.py::resolve_image_url),
and matching that against manifest.json's "source_image" ("TPWL/<file>") by
suffix.

Reads:
    <manifest>       manifest.json from extract_missed_as_ignore.py /
                      extract_false_positives.py (needs a "source_image" key)

Writes:
    <manifest_dir>/manifest_with_task_id.json   - same entries, each with an
        added "task_id" (or null if no matching task was found) and
        "label_studio_url" (project's task-annotation deep link)

Usage:
    python scripts/annotate_manifest_with_task_id.py \
      --manifest reports/missed_as_ignore_new_model/manifest.json \
      --project-id 23
"""

import argparse
import base64
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.logger import setup_logger, get_logger

load_dotenv()


def resolve_image_url(raw_image: str) -> str:
    """Handle Label Studio local-storage proxy URLs (data/local-files/?d=...&fileuri=...)."""
    if "fileuri=" in raw_image:
        b64_str = raw_image.split("fileuri=")[-1].split("&")[0]
        return base64.b64decode(b64_str).decode("utf-8")
    return raw_image


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


def build_image_to_task_id(tasks: list) -> dict:
    """Maps a GCS image URI's "<folder>/<filename>" suffix (e.g.
    "TPWL/image001.jpg") to its Label Studio task_id."""
    mapping = {}
    for task in tasks:
        raw_image = task.get("data", {}).get("image", "")
        if not raw_image:
            continue
        gcs_url = resolve_image_url(raw_image)
        # gs://bucket/TPWL/image001.jpg -> "TPWL/image001.jpg"
        key = "/".join(gcs_url.split("/")[-2:])
        mapping[key] = task["id"]
    return mapping


def annotate(manifest_path: str, project_id: int, page_size: int, base_url: str) -> list:
    from label_studio_sdk import Client

    logger = get_logger(__name__)

    url = os.getenv("LABEL_STUDIO_URL")
    api_key = os.getenv("LABEL_STUDIO_API_KEY")
    if not url or not api_key:
        raise SystemExit("LABEL_STUDIO_URL / LABEL_STUDIO_API_KEY not set (check .env)")

    ls = Client(url, api_key)
    project = ls.get_project(project_id)
    tasks = fetch_all_tasks(project, page_size)
    image_to_task_id = build_image_to_task_id(tasks)

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    matched, unmatched = 0, 0
    for entry in manifest:
        task_id = image_to_task_id.get(entry["source_image"])
        entry["task_id"] = task_id
        entry["label_studio_url"] = (
            f"{base_url.rstrip('/')}/projects/{project_id}/data?task={task_id}"
            if task_id is not None else None
        )
        if task_id is not None:
            matched += 1
        else:
            unmatched += 1

    logger.info(f"Matched {matched}/{len(manifest)} manifest entries to a task_id ({unmatched} unmatched)")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True, help="Path to manifest.json to annotate")
    parser.add_argument("--project-id", type=int, default=23, help="Label Studio project ID")
    parser.add_argument("--page-size", type=int, default=50, help="Label Studio task pagination page size")
    parser.add_argument("--output", default=None, help="Output path (default: <manifest_dir>/manifest_with_task_id.json)")
    parser.add_argument("--log-dir", default="./logs", help="Directory for log files")
    return parser


def main():
    args = build_parser().parse_args()
    setup_logger(log_dir=args.log_dir)

    base_url = os.getenv("LABEL_STUDIO_URL", "").rstrip("/")
    manifest = annotate(args.manifest, args.project_id, args.page_size, base_url)

    manifest_path = Path(args.manifest)
    output_path = Path(args.output) if args.output else manifest_path.with_name("manifest_with_task_id.json")
    with open(output_path, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
