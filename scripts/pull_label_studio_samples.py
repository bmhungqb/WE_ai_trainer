"""Download annotated samples from a Label Studio project (skips unreviewed tasks).

By default, negative samples (reviewed tasks with no defect annotations) are excluded.
Pass --include-negatives to keep them in the output.

Usage:
    python scripts/pull_label_studio_samples.py --project-id 23           # pull ALL annotated samples
    python scripts/pull_label_studio_samples.py --project-id 23 --limit 5  # pull only the first 5 annotated samples
    python scripts/pull_label_studio_samples.py --project-id 23 --include-negatives  # also keep negative samples

Reads LABEL_STUDIO_URL / LABEL_STUDIO_API_KEY from the environment (.env).
"""
import argparse
import datetime
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from label_studio_sdk import Client

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.label_studio_utils import process_task
from utils.constants import DEFECT_CLASSES
from utils.logger import get_logger

load_dotenv()
logger = get_logger(__name__)

label_to_id = {v: k for k, v in DEFECT_CLASSES.items()}


def pull_sample(
    url: str,
    api_key: str,
    project_id: int,
    limit: int | None,
    output_path: str,
    page_size: int = 50,
    created_after: datetime.date | None = None,
    created_before: datetime.date | None = None,
    include_negatives: bool = False,
):
    ls = Client(url, api_key)
    project = ls.get_project(project_id)

    coco_output_format = {
        "info": {},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [],
    }

    page = 1
    scanned = 0
    while limit is None or len(coco_output_format["images"]) < limit:
        try:
            resp = project.get_paginated_tasks(page=page, page_size=page_size)
            tasks = resp.get("tasks", [])
        except AttributeError:
            # older/newer SDKs may not expose get_paginated_tasks; fall back to a single full fetch
            tasks = project.get_tasks() if page == 1 else []

        if not tasks:
            break
        scanned += len(tasks)

        for task in tasks:
            if created_after is not None or created_before is not None:
                created_at_str = task.get("created_at")
                if not created_at_str:
                    continue
                created_date = datetime.datetime.fromisoformat(
                    created_at_str.replace("Z", "+00:00")
                ).date()
                if created_after is not None and created_date < created_after:
                    continue
                if created_before is not None and created_date > created_before:
                    continue

            sample = process_task(task)
            if not sample:
                continue  # skip unannotated tasks (no review recorded at all)
            if not sample["annos"] and not include_negatives:
                continue  # skip negative samples (reviewed, no defects)

            image_id = len(coco_output_format["images"]) + 1
            coco_output_format["images"].append({
                "id": image_id,
                "file_name": sample["image_url"],
                "width": sample["width"],
                "height": sample["height"],
            })
            for anno in sample["annos"]:
                coco_output_format["annotations"].append({
                    "id": len(coco_output_format["annotations"]) + 1,
                    "image_id": image_id,
                    "category_id": label_to_id.get(anno["label"], -1),
                    "bbox": anno["bbox"],
                    "area": anno["area"],
                    "iscrowd": anno["iscrowd"],
                })
            if limit is not None and len(coco_output_format["images"]) >= limit:
                break

        if len(tasks) < page_size:
            break  # reached the end of the project
        page += 1

    logger.info(
        f"Scanned {scanned} task(s), kept {len(coco_output_format['images'])} sample(s) "
        f"(negatives {'included' if include_negatives else 'excluded'}) from project {project_id}"
    )

    for category_id, category_name in DEFECT_CLASSES.items():
        coco_output_format["categories"].append({
            "id": category_id,
            "name": category_name,
            "supercategory": "defect",
        })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(coco_output_format, f, indent=2)

    logger.info(
        f"Saved {len(coco_output_format['images'])} image(s) / "
        f"{len(coco_output_format['annotations'])} annotation(s) to {output_path}"
    )
    return output_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-id", type=int, default=23)
    parser.add_argument(
        "--limit", type=int, default=None, help="Max number of annotated samples to pull (default: all)"
    )
    parser.add_argument(
        "--created-after", type=str, default=None, help="Only pull tasks created on/after this date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--created-before", type=str, default=None, help="Only pull tasks created on/before this date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--include-negatives", action="store_true",
        help="Keep negative samples (reviewed tasks with no defect annotations) in the output"
    )
    parser.add_argument(
        "--output",
        default=f"tmp/annotations_project23_sample_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
    )
    args = parser.parse_args()

    url = os.getenv("LABEL_STUDIO_URL")
    api_key = os.getenv("LABEL_STUDIO_API_KEY")
    if not url or not api_key:
        raise SystemExit("LABEL_STUDIO_URL / LABEL_STUDIO_API_KEY not set (check .env)")

    created_after = (
        datetime.datetime.strptime(args.created_after, "%Y-%m-%d").date() if args.created_after else None
    )
    created_before = (
        datetime.datetime.strptime(args.created_before, "%Y-%m-%d").date() if args.created_before else None
    )

    path = pull_sample(
        url, api_key, args.project_id, args.limit, args.output,
        created_after=created_after, created_before=created_before,
        include_negatives=args.include_negatives,
    )

    print(f"Done. Output: {path}")


if __name__ == "__main__":
    main()
