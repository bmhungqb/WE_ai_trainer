"""
Compare the new 5-class model (pleat/stain/weaving/hard_pleat/ignore) against
the current production predictions, both scored against worker-corrected
ground truth, to decide whether the new model is ready to replace production.

Reads:
    results/**/*.json   (output of merge_annotations.py, must include a
        "new_model" annotation source - see scripts/inference_new_model.py)

Writes:
    reports/comparison_metrics.json

Two kinds of metrics, both IoU-matched (greedy, same convention as
scripts/compute_metrics.py / src/ai_verify.py::AIVerify._calculate_iou):

1. Per-class + overall precision/recall/F1 for "production" and "new_model"
   against "ground_truth". A predicted box only counts as a true positive if
   it also matches the ground-truth box's class (unlike compute_metrics.py,
   which is location-only).

2. False-alarm suppression: production has no "ignore" concept, so its false
   positives conflate "wrong class" and "should never have fired" into one
   number. Ground truth boxes whose worker-corrected label is the Vietnamese
   no-defect label ("Khong_co_loi", see utils/constants.py MAPPING_CLASSES)
   are the cases where a worker looked and confirmed nothing was there. For
   each source, among its predictions matching one of those no-defect GT
   boxes:
     - "new_model" predicting the "ignore" class there = correct suppression
     - anything else predicting a real defect class there = false alarm
   This isolates the exact behavior the new "ignore" class was trained to
   fix, which precision/recall alone would bury inside "wrong class" errors.

Usage:
    python scripts/compute_comparison_metrics.py --results results --report reports/comparison_metrics.json
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.logger import setup_logger, get_logger
from utils.constants import DEFECT_CLASSES

SOURCES = ("production", "new_model")
NO_DEFECT_LABEL = "Khong_co_loi"
IGNORE_LABEL = "ignore"
DEFECT_CLASS_NAMES = tuple(DEFECT_CLASSES.values())


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


def greedy_match(predicted: list, ground_truth: list, iou_threshold: float):
    """Greedy IoU matching, highest-confidence predictions first. Returns a
    list of (pred, matched_gt_or_None) pairs plus the set of matched GT indices."""
    matched_gt = set()
    pairs = []
    for pred in sorted(predicted, key=lambda p: p.get("confidence", 0.0), reverse=True):
        best_idx, best_iou = None, 0.0
        for i, gt in enumerate(ground_truth):
            if i in matched_gt:
                continue
            score = iou(pred["bbox"], gt["bbox"])
            if score > best_iou:
                best_idx, best_iou = i, score
        if best_idx is not None and best_iou >= iou_threshold:
            matched_gt.add(best_idx)
            pairs.append((pred, ground_truth[best_idx]))
        else:
            pairs.append((pred, None))
    return pairs, matched_gt


def _canonical_class(label: str) -> str:
    """Ground truth's no-defect label ("Khong_co_loi") and the new model's
    no-defect prediction label ("ignore") mean the same thing; normalize
    both to "ignore" so a correct suppression scores as a class match
    instead of an FP+FN pair."""
    return IGNORE_LABEL if label == NO_DEFECT_LABEL else label


def classification_counts(predicted: list, ground_truth: list, iou_threshold: float) -> dict:
    """Per-class TP/FP/FN, requiring both location (IoU) and class match.
    Counts are keyed by canonicalized class name (see _canonical_class)."""
    pairs, matched_gt = greedy_match(predicted, ground_truth, iou_threshold)
    counts = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    for pred, gt in pairs:
        pred_class = _canonical_class(pred.get("class"))
        gt_class = _canonical_class(gt.get("class")) if gt is not None else None
        if gt_class == pred_class:
            counts[pred_class]["tp"] += 1
        else:
            counts[pred_class]["fp"] += 1
            if gt is not None:
                counts[gt_class]["fn"] += 1

    for i, gt in enumerate(ground_truth):
        if i not in matched_gt:
            counts[_canonical_class(gt.get("class"))]["fn"] += 1

    return counts


def prf1(tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def false_alarm_counts(predicted: list, ground_truth: list, iou_threshold: float) -> dict:
    """Among predictions that land on a confirmed no-defect GT box: does the
    source correctly abstain (suppressed) or still raise a real-defect alarm
    (false_alarm)? Predictions with no matching GT box are out of scope here
    (see the main precision/recall metrics for that)."""
    no_defect_gt = [g for g in ground_truth if g.get("class") == NO_DEFECT_LABEL]
    if not no_defect_gt:
        return {"no_defect_gt_boxes": 0, "false_alarms": 0, "suppressed": 0}

    pairs, _ = greedy_match(predicted, no_defect_gt, iou_threshold)
    false_alarms = sum(1 for pred, gt in pairs if gt is not None and pred.get("class") != IGNORE_LABEL)
    suppressed = sum(1 for pred, gt in pairs if gt is not None and pred.get("class") == IGNORE_LABEL)
    return {
        "no_defect_gt_boxes": len(no_defect_gt),
        "false_alarms": false_alarms,
        "suppressed": suppressed,
    }


def compute_comparison_metrics(results_dir: str, iou_threshold: float = 0.5) -> dict:
    logger = get_logger(__name__)

    per_class_totals = {source: defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0}) for source in SOURCES}
    false_alarm_totals = {source: {"no_defect_gt_boxes": 0, "false_alarms": 0, "suppressed": 0} for source in SOURCES}

    samples = 0
    for json_path in sorted(Path(results_dir).rglob("*.json")):
        with open(json_path, "r") as f:
            record = json.load(f)
        annotations = record.get("annotations", {})
        ground_truth = annotations.get("ground_truth", [])

        for source in SOURCES:
            predicted = annotations.get(source, [])

            counts = classification_counts(predicted, ground_truth, iou_threshold)
            for cls, c in counts.items():
                totals = per_class_totals[source][cls]
                totals["tp"] += c["tp"]
                totals["fp"] += c["fp"]
                totals["fn"] += c["fn"]

            fa = false_alarm_counts(predicted, ground_truth, iou_threshold)
            false_alarm_totals[source]["no_defect_gt_boxes"] += fa["no_defect_gt_boxes"]
            false_alarm_totals[source]["false_alarms"] += fa["false_alarms"]
            false_alarm_totals[source]["suppressed"] += fa["suppressed"]

        samples += 1

    report = {"samples": samples, "iou_threshold": iou_threshold, "sources": {}}
    for source in SOURCES:
        classes = per_class_totals[source]
        per_class = {cls: prf1(c["tp"], c["fp"], c["fn"]) for cls, c in sorted(classes.items())}

        overall_tp = sum(c["tp"] for c in classes.values())
        overall_fp = sum(c["fp"] for c in classes.values())
        overall_fn = sum(c["fn"] for c in classes.values())

        fa = false_alarm_totals[source]
        false_alarm_rate = fa["false_alarms"] / fa["no_defect_gt_boxes"] if fa["no_defect_gt_boxes"] else 0.0
        suppression_rate = fa["suppressed"] / fa["no_defect_gt_boxes"] if fa["no_defect_gt_boxes"] else 0.0

        report["sources"][source] = {
            "overall": prf1(overall_tp, overall_fp, overall_fn),
            "per_class": per_class,
            "false_alarm_suppression": {
                **fa,
                "false_alarm_rate": false_alarm_rate,
                "suppression_rate": suppression_rate,
            },
        }

    logger.info(f"Computed comparison metrics over {samples} samples")
    for source in SOURCES:
        overall = report["sources"][source]["overall"]
        fa = report["sources"][source]["false_alarm_suppression"]
        logger.info(
            f"  {source}: P={overall['precision']:.3f} R={overall['recall']:.3f} F1={overall['f1']:.3f} "
            f"| false_alarm_rate={fa['false_alarm_rate']:.3f} (on {fa['no_defect_gt_boxes']} no-defect GT boxes)"
        )

    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", default="results", help="Merged results directory (output of merge_annotations.py)")
    parser.add_argument("--report", default="reports/comparison_metrics.json", help="Output report path")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--log-dir", default="./logs", help="Directory for log files")
    return parser


def main():
    args = build_parser().parse_args()
    setup_logger(log_dir=args.log_dir)

    report = compute_comparison_metrics(args.results, args.iou_threshold)

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Report written to {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
