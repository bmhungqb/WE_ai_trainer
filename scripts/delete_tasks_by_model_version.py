"""
Delete every task in a project that has a prediction whose model_version
starts with a given prefix - a targeted cleanup tool for undoing a batch
of tasks created by one of the create_*_tasks_from_july.py scripts (e.g.
after fixing a bad data.image URL format and needing to recreate them),
without touching any other task in the project.

Usage:
    python scripts/delete_tasks_by_model_version.py \
      --project-id 25 --model-version-prefix weaving_from_july_ --dry-run

    python scripts/delete_tasks_by_model_version.py \
      --project-id 25 --model-version-prefix weaving_from_july_
"""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from label_studio_sdk import Client

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.logger import setup_logger, get_logger

load_dotenv()


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


def has_matching_prediction(task: dict, prefix: str) -> bool:
    return any(
        p.get("model_version", "").startswith(prefix)
        for p in task.get("predictions", [])
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project-id", type=int, required=True, help="Label Studio project ID")
    parser.add_argument(
        "--model-version-prefix", required=True,
        help="Delete any task with a prediction whose model_version starts with this prefix",
    )
    parser.add_argument("--page-size", type=int, default=50, help="Label Studio task pagination page size")
    parser.add_argument("--dry-run", action="store_true", help="Report which tasks would be deleted, but don't delete anything")
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

    legacy_client = Client(url, api_key)
    project = legacy_client.get_project(args.project_id)
    tasks = fetch_tasks(project, args.page_size)

    matching = [t for t in tasks if has_matching_prediction(t, args.model_version_prefix)]
    logger.info(f"{len(matching)} of {len(tasks)} task(s) have a prediction with model_version starting '{args.model_version_prefix}'")

    deleted, failed = 0, 0
    for i, task in enumerate(matching, 1):
        task_id = task["id"]
        logger.info(f"[{i}/{len(matching)}] task {task_id}: {'(dry-run) would delete' if args.dry_run else 'deleting'}")
        if args.dry_run:
            deleted += 1
            continue
        try:
            project.delete_task(task_id)
            deleted += 1
        except Exception as e:
            logger.error(f"[{i}/{len(matching)}] task {task_id}: failed to delete: {e}")
            failed += 1

    if args.dry_run:
        logger.info(f"Done (dry-run). would_delete={deleted} failed={failed} -- nothing was deleted")
    else:
        logger.info(f"Done. deleted={deleted} failed={failed}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
