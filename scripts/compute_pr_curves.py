"""
Per-class precision-recall curve (swept over confidence threshold) for a
prediction source against ground truth, plus the confidence threshold that
maximizes F1 for each class - answers "what threshold should I actually
deploy at" per defect class, rather than compute_comparison_metrics.py's
single fixed-threshold (0.5) snapshot.

Matching convention (standard COCO/VOC-style AP, adapted to this dataset):
for a given class, every prediction of that class across the whole dataset
is IoU-matched (>= iou-threshold) against ground-truth boxes of the SAME
class only, greedily, highest-confidence prediction first, each GT box
claimed at most once. A prediction with no class-matching GT box available
(or below the IoU threshold) is a false positive. GT boxes of that class
never matched by any prediction are the "recall ceiling" - true count used
as the recall denominator regardless of threshold.

Ground truth's no-defect label ("Khong_co_loi") and the new model's "ignore"
prediction label are treated as the same class (see
compute_comparison_metrics.py::_canonical_class) for consistency with the
rest of this pipeline's metrics.

Reads:
    results/**/*.json   (output of merge_annotations.py)

Writes:
    reports/pr_curves.json

    {
      "production": {
        "stain": {
          "curve": [{"threshold": 0.95, "precision": 1.0, "recall": 0.01, "f1": ...}, ...],
          "best_f1": {"threshold": 0.4, "precision": ..., "recall": ..., "f1": ...},
          "num_gt": 2497
        },
        ...
      },
      "new_model": {...}
    }

Usage:
    python scripts/compute_pr_curves.py --results results --report reports/pr_curves.json
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.logger import setup_logger, get_logger

SOURCES = ("production", "new_model")
NO_DEFECT_LABEL = "Khong_co_loi"
IGNORE_LABEL = "ignore"


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


def _canonical_class(label: str) -> str:
    return IGNORE_LABEL if label == NO_DEFECT_LABEL else label


def match_class_predictions(class_preds: list, iou_threshold: float) -> list:
    """class_preds: list of (confidence, bbox, gt_boxes_in_same_image) tuples,
    ALL for one class, gathered across the whole dataset. Returns a list of
    (confidence, is_tp) sorted by confidence descending - greedy per-image
    matching against that image's same-class GT boxes only (a GT box can only
    be claimed once WITHIN its own image, matching standard per-class AP)."""
    # Group by image identity (id() of the shared gt list is enough here
    # since each image contributes its own gt_boxes_in_same_image list).
    by_image = defaultdict(list)
    for conf, bbox, gt_boxes in class_preds:
        by_image[id(gt_boxes)].append((conf, bbox, gt_boxes))

    results = []
    for group in by_image.values():
        group.sort(key=lambda p: p[0], reverse=True)
        gt_boxes = group[0][2]
        claimed = set()
        for conf, bbox, _ in group:
            best_idx, best_iou = None, 0.0
            for i, gt_bbox in enumerate(gt_boxes):
                if i in claimed:
                    continue
                score = iou(bbox, gt_bbox)
                if score > best_iou:
                    best_idx, best_iou = i, score
            if best_idx is not None and best_iou >= iou_threshold:
                claimed.add(best_idx)
                results.append((conf, True))
            else:
                results.append((conf, False))

    results.sort(key=lambda r: r[0], reverse=True)
    return results


def build_pr_curve(matches: list, num_gt: int) -> dict:
    """matches: (confidence, is_tp) sorted by confidence descending.
    Returns the full curve (one point per distinct confidence threshold,
    i.e. after each prediction is included) plus the max-F1 point."""
    curve = []
    tp = 0
    fp = 0
    best = {"threshold": 1.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    i = 0
    n = len(matches)
    while i < n:
        threshold = matches[i][0]
        # Consume all predictions tied at this exact confidence together,
        # so each curve point reflects "include everything >= threshold".
        while i < n and matches[i][0] == threshold:
            if matches[i][1]:
                tp += 1
            else:
                fp += 1
            i += 1

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / num_gt if num_gt else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        point = {"threshold": threshold, "precision": precision, "recall": recall, "f1": f1}
        curve.append(point)
        if f1 > best["f1"]:
            best = point

    return {"curve": curve, "best_f1": best, "num_gt": num_gt}


def compute_pr_curves(results_dir: str, iou_threshold: float = 0.5) -> dict:
    logger = get_logger(__name__)

    # class -> source -> list of (confidence, bbox, gt_boxes_for_this_image)
    per_class_preds = {source: defaultdict(list) for source in SOURCES}
    # class -> total GT box count (recall denominator), counted once (GT is shared across sources)
    gt_counts = defaultdict(int)

    n_records = 0
    for json_path in sorted(Path(results_dir).rglob("*.json")):
        with open(json_path, "r") as f:
            record = json.load(f)
        annotations = record.get("annotations", {})
        ground_truth = annotations.get("ground_truth", [])
        n_records += 1

        # Group this image's GT boxes by canonical class.
        gt_by_class = defaultdict(list)
        for gt in ground_truth:
            cls = _canonical_class(gt.get("class"))
            gt_by_class[cls].append(gt["bbox"])
            gt_counts[cls] += 1

        for source in SOURCES:
            for pred in annotations.get(source, []):
                cls = _canonical_class(pred.get("class"))
                gt_boxes = gt_by_class.get(cls, [])
                per_class_preds[source][cls].append(
                    (pred.get("confidence", 0.0), pred["bbox"], gt_boxes)
                )

    classes = sorted(gt_counts.keys())
    report = {"samples": n_records, "iou_threshold": iou_threshold, "sources": {}}

    for source in SOURCES:
        report["sources"][source] = {}
        for cls in classes:
            class_preds = per_class_preds[source].get(cls, [])
            matches = match_class_predictions(class_preds, iou_threshold)
            report["sources"][source][cls] = build_pr_curve(matches, gt_counts[cls])

    logger.info(f"Computed PR curves over {n_records} samples, {len(classes)} classes")
    for source in SOURCES:
        for cls in classes:
            best = report["sources"][source][cls]["best_f1"]
            num_gt = report["sources"][source][cls]["num_gt"]
            logger.info(
                f"  {source}/{cls}: best F1={best['f1']:.3f} at threshold={best['threshold']:.3f} "
                f"(P={best['precision']:.3f} R={best['recall']:.3f}, {num_gt} GT boxes)"
            )

    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", default="results", help="Merged results directory (output of merge_annotations.py)")
    parser.add_argument("--report", default="reports/pr_curves.json", help="Output report path")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--log-dir", default="./logs", help="Directory for log files")
    return parser


def main():
    args = build_parser().parse_args()
    setup_logger(log_dir=args.log_dir)

    report = compute_pr_curves(args.results, args.iou_threshold)

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Report written to {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
