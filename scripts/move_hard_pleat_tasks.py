"""
Move every annotated task in a Label Studio project whose human annotation
contains ONLY "hard_pleat" class boxes (no other class mixed in, at least
one box present) into a separate project - no model inference involved,
this is a pure human-annotation filter.

Tasks whose image URL already exists in the target project are skipped, so
re-running this script is safe / idempotent - it never double-imports a
task that's already been moved.

Moving a task = importing it (with its existing human annotation verbatim)
into the target project via Project.import_tasks, then deleting it from the
source project. The move only happens after a successful import, so a
failed import never loses the source task.

Usage:
    python scripts/move_hard_pleat_tasks.py --source-project-id 23 --target-project-id 25

    # Dry-run: report which tasks would move, without importing/deleting anything
    python scripts/move_hard_pleat_tasks.py --source-project-id 23 --target-project-id 25 --dry-run
"""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from label_studio_sdk import Client

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.logger import setup_logger, get_logger
from utils.constants import CANONICAL_LABELS
from utils.label_studio_utils import process_task

load_dotenv()

HARD_PLEAT_LABEL = "hard_pleat"


def _canonical(label: str) -> str:
    return CANONICAL_LABELS.get(label, label)


def is_only_hard_pleat(human_annos: list) -> bool:
    """True iff every box in the annotation is hard_pleat and there's at
    least one box (an empty/negative-sample annotation doesn't count)."""
    return bool(human_annos) and all(_canonical(a["label"]) == HARD_PLEAT_LABEL for a in human_annos)


def resolve_image_url(raw_image: str) -> str:
    """Handle Label Studio local-storage proxy URLs (data/local-files/?d=...&fileuri=...)."""
    import base64
    if "fileuri=" in raw_image:
        b64_str = raw_image.split("fileuri=")[-1].split("&")[0]
        return base64.b64decode(b64_str).decode("utf-8")
    return raw_image


def fetch_tasks(project, page_size: int) -> list:
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


def import_task_to_target(target_project, target_project_id: int, task: dict) -> bool:
    """Import one task (its original "data" + its human annotation, verbatim)
    into the target project via Project.import_tasks."""
    logger = get_logger(__name__)
    annotations = task.get("annotations", [])
    payload = [{
        "data": task["data"],
        "annotations": [
            {"result": a["result"], "was_cancelled": a.get("was_cancelled", False)}
            for a in annotations
        ],
    }]

    try:
        target_project.import_tasks(payload)
    except Exception as e:
        logger.error(f"Failed to import task {task['id']} into project {target_project_id}: {e}")
        return False
    return True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Move tasks whose annotation contains ONLY 'hard_pleat' class boxes into a separate project",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source-project-id", type=int, required=True, help="Label Studio project ID to pull tasks from")
    parser.add_argument("--target-project-id", type=int, required=True, help="Label Studio project ID to move hard_pleat tasks into")
    parser.add_argument("--page-size", type=int, default=50, help="Label Studio task pagination page size")
    parser.add_argument(
        "--delete-source",
        action="store_true",
        help="Delete each task from the source project after it's successfully imported into "
             "the target project. Without this flag, tasks are copied but left in place "
             "(safer default - re-run with --delete-source once you've spot-checked the target project).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Report which tasks would move, but don't import or delete anything")
    parser.add_argument("--log-dir", default="./logs", help="Directory for log files")
    return parser


def main():
    args = build_parser().parse_args()
    setup_logger(log_dir=args.log_dir)
    logger = get_logger(__name__)

    url = os.getenv("LABEL_STUDIO_URL")
    api_key = os.getenv("LABEL_STUDIO_API_KEY")
    if not url or not api_key:
        raise SystemExit("LABEL_STUDIO_URL / LABEL_STUDIO_API_KEY not set (check .env)")

    logger.info(f"Step 1: pulling annotated tasks from Label Studio project {args.source_project_id}")
    legacy_client = Client(url, api_key)
    source_project = legacy_client.get_project(args.source_project_id)
    target_project = legacy_client.get_project(args.target_project_id)
    tasks = fetch_tasks(source_project, args.page_size)
    if not tasks:
        logger.error("No tasks found, aborting")
        return 1

    logger.info(f"Fetching existing tasks in target project {args.target_project_id} to skip already-moved ones")
    target_tasks = fetch_tasks(target_project, args.page_size)
    already_moved_image_urls = {
        resolve_image_url(t.get("data", {}).get("image", "")) for t in target_tasks
    }
    already_moved_image_urls.discard("")
    logger.info(f"{len(already_moved_image_urls)} image(s) already present in target project")

    moved, kept, skipped, failed = 0, 0, 0, 0
    for i, task in enumerate(tasks, 1):
        task_id = task["id"]

        raw_image = task.get("data", {}).get("image", "")
        gcs_image_url = resolve_image_url(raw_image)
        if gcs_image_url in already_moved_image_urls:
            logger.info(f"[{i}/{len(tasks)}] task {task_id}: already present in target project, skipping")
            skipped += 1
            continue

        sample = process_task(task)
        if not sample:
            logger.info(f"[{i}/{len(tasks)}] task {task_id}: no human annotation, skipping")
            skipped += 1
            continue

        if not is_only_hard_pleat(sample["annos"]):
            kept += 1
            continue

        logger.info(
            f"[{i}/{len(tasks)}] task {task_id}: contains ONLY 'hard_pleat' class ({len(sample['annos'])} box(es)), "
            f"{'(dry-run) would move' if args.dry_run else 'moving'} to project {args.target_project_id}"
        )

        if args.dry_run:
            moved += 1
            continue

        if not import_task_to_target(target_project, args.target_project_id, task):
            failed += 1
            continue

        if args.delete_source:
            try:
                source_project.delete_task(task_id)
            except Exception as e:
                logger.error(f"[{i}/{len(tasks)}] task {task_id}: imported to target but failed to delete from source: {e}")
                failed += 1
                continue

        moved += 1

    if args.dry_run:
        logger.info(
            f"Done (dry-run). would_move={moved} kept={kept} skipped={skipped} failed={failed} "
            f"total={len(tasks)} -- nothing was imported or deleted"
        )
    else:
        logger.info(
            f"Done. moved={moved} kept={kept} skipped={skipped} failed={failed} total={len(tasks)} "
            f"(source_deleted={'yes' if args.delete_source else 'no, --delete-source not passed'})"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
