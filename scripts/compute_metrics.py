"""
(Optional) Compute precision/recall/F1 for each prediction source against
ground truth, using IoU-based greedy matching (reuses the IoU logic from
src/ai_verify.py::AIVerify._calculate_iou).

Reads:
    results/**/*.json   (output of merge_annotations.py)

Writes:
    reports/metrics.json

    {
        "production": {"precision": 0, "recall": 0, "f1": 0},
        "rfdetr_v1": {"precision": 0, "recall": 0, "f1": 0},
        "rfdetr_v2": {"precision": 0, "recall": 0, "f1": 0}
    }

Usage:
    python scripts/compute_metrics.py --results results --report reports/metrics.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.logger import setup_logger, get_logger

SOURCES = ("production", "rfdetr_v1", "rfdetr_v2")


def iou(box1: list, box2: list) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union else 0.0


def match_counts(predicted: list, ground_truth: list, iou_threshold: float = 0.5) -> tuple:
    matched_gt = set()
    tp = 0
    for pred in predicted:
        best_idx, best_iou = None, 0.0
        for i, gt in enumerate(ground_truth):
            if i in matched_gt:
                continue
            score = iou(pred["bbox"], gt["bbox"])
            if score > best_iou:
                best_idx, best_iou = i, score
        if best_idx is not None and best_iou >= iou_threshold:
            matched_gt.add(best_idx)
            tp += 1

    fp = len(predicted) - tp
    fn = len(ground_truth) - len(matched_gt)
    return tp, fp, fn


def compute_metrics(results_dir: str, iou_threshold: float = 0.5) -> dict:
    logger = get_logger(__name__)
    totals = {source: {"tp": 0, "fp": 0, "fn": 0} for source in SOURCES}

    samples = 0
    for json_path in sorted(Path(results_dir).rglob("*.json")):
        with open(json_path, "r") as f:
            record = json.load(f)
        annotations = record.get("annotations", {})
        ground_truth = annotations.get("ground_truth", [])

        for source in SOURCES:
            tp, fp, fn = match_counts(annotations.get(source, []), ground_truth, iou_threshold)
            totals[source]["tp"] += tp
            totals[source]["fp"] += fp
            totals[source]["fn"] += fn
        samples += 1

    metrics = {}
    for source in SOURCES:
        tp, fp, fn = totals[source]["tp"], totals[source]["fp"], totals[source]["fn"]
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        metrics[source] = {"precision": precision, "recall": recall, "f1": f1}

    logger.info(f"Computed metrics over {samples} samples: {metrics}")
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute precision/recall/F1 vs ground truth")
    parser.add_argument("--results", default="results", help="Merged results directory")
    parser.add_argument("--report", default="reports/metrics.json", help="Output report path")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--log-dir", default="./logs", help="Directory for log files")
    return parser


def main():
    args = build_parser().parse_args()
    setup_logger(log_dir=args.log_dir)

    metrics = compute_metrics(args.results, args.iou_threshold)

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Report written to {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
