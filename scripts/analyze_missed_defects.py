"""
For every real-defect ground-truth box that a source (default: new_model)
missed, classify WHY it was missed and dump the details for manual review -
answers "is the model detecting the wrong class, or not detecting the box
at all?" instead of just reporting a single missed_other count.

Reads:
    results/**/*.json   (output of merge_annotations.py)

For each GT box whose class is a real defect (not the no-defect label),
using the same greedy highest-confidence-first IoU matching as
compute_comparison_metrics.py::greedy_match:

    - "wrong_class":       a prediction matched this GT box by location
                            (IoU >= --iou-threshold) but predicted a
                            DIFFERENT class (includes "ignore" - see
                            --iou-threshold's docstring below for how this
                            relates to missed_as_ignore in
                            compute_comparison_metrics.py).
    - "low_iou_near_miss":  the source predicted the same class SOMEWHERE
                            in the image, but its closest same-class box
                            for this GT fell short of --iou-threshold
                            (location is off - box too small/large/shifted).
    - "no_prediction_nearby": the source predicted nothing within
                            --nearby-iou of this GT box at all (a true
                            "didn't see it").

Writes:
    reports/missed_defects.json           - full listing, one entry per
                                             missed GT box, with the reason,
                                             GT box/class, and the source's
                                             nearest same-class prediction
                                             (if any)
    reports/missed_defects_summary.json   - counts per class x reason

Usage:
    python scripts/analyze_missed_defects.py --results results --source new_model

    # Also copy the offending images into a folder for visual review
    python scripts/analyze_missed_defects.py --results results --source new_model \
      --copy-images reports/missed_defects_images
"""

import argparse
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.logger import setup_logger, get_logger

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


def greedy_match(predicted: list, ground_truth: list, iou_threshold: float):
    """Same convention as compute_comparison_metrics.py::greedy_match."""
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


def classify_miss(gt_box: dict, predicted: list, iou_threshold: float, nearby_iou: float) -> dict:
    """gt_box was left unmatched by greedy_match against the full predicted
    list. Figure out why: is there a same-class prediction nearby that just
    missed the IoU bar (low_iou_near_miss), a different-class prediction
    that beat the IoU bar and "stole" this GT via greedy assignment
    elsewhere (wrong_class), or genuinely nothing nearby (no_prediction_nearby)?"""
    gt_class = gt_box.get("class")
    gt_bbox = gt_box["bbox"]

    best_same_class = None  # (iou, pred)
    best_any_class = None   # (iou, pred)
    for pred in predicted:
        score = iou(pred["bbox"], gt_bbox)
        if best_any_class is None or score > best_any_class[0]:
            best_any_class = (score, pred)
        if pred.get("class") == gt_class:
            if best_same_class is None or score > best_same_class[0]:
                best_same_class = (score, pred)

    if best_any_class is not None and best_any_class[0] >= iou_threshold and best_any_class[1].get("class") != gt_class:
        reason = "wrong_class"
        nearest = best_any_class
    elif best_same_class is not None and best_same_class[0] >= nearby_iou:
        reason = "low_iou_near_miss"
        nearest = best_same_class
    else:
        reason = "no_prediction_nearby"
        nearest = best_any_class if (best_any_class and best_any_class[0] >= nearby_iou) else None

    return {
        "reason": reason,
        "nearest_prediction": (
            {"class": nearest[1].get("class"), "confidence": nearest[1].get("confidence"),
             "bbox": nearest[1]["bbox"], "iou": nearest[0]}
            if nearest else None
        ),
    }


def analyze(results_dir: str, source: str, iou_threshold: float, nearby_iou: float) -> tuple:
    logger = get_logger(__name__)

    entries = []
    summary = defaultdict(lambda: defaultdict(int))

    samples = 0
    for json_path in sorted(Path(results_dir).rglob("*.json")):
        with open(json_path, "r") as f:
            record = json.load(f)
        annotations = record.get("annotations", {})
        ground_truth = annotations.get("ground_truth", [])
        predicted = annotations.get(source, [])

        defect_gt = [g for g in ground_truth if g.get("class") != NO_DEFECT_LABEL]
        if not defect_gt:
            samples += 1
            continue

        _, matched_gt = greedy_match(predicted, defect_gt, iou_threshold)

        for i, gt in enumerate(defect_gt):
            if i in matched_gt:
                continue  # caught - not a miss

            classification = classify_miss(gt, predicted, iou_threshold, nearby_iou)
            gt_class = gt.get("class")
            summary[gt_class][classification["reason"]] += 1

            entries.append({
                "image": record.get("image"),
                "gt_class": gt_class,
                "gt_bbox": gt["bbox"],
                **classification,
            })

        samples += 1

    logger.info(f"Analyzed {samples} samples, found {len(entries)} missed defect box(es) for source={source}")
    for cls in sorted(summary.keys()):
        reasons = summary[cls]
        total = sum(reasons.values())
        parts = ", ".join(f"{r}={c}" for r, c in sorted(reasons.items()))
        logger.info(f"  {cls}: {total} missed ({parts})")

    return entries, {cls: dict(reasons) for cls, reasons in summary.items()}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", default="results", help="Merged results directory (output of merge_annotations.py)")
    parser.add_argument("--source", default="new_model", help="Prediction source to analyze (e.g. new_model, production)")
    parser.add_argument("--iou-threshold", type=float, default=0.5, help="Same IoU threshold used for the main TP/FP/FN matching")
    parser.add_argument(
        "--nearby-iou", type=float, default=0.1,
        help="Lower IoU bar used only to decide whether a same-class prediction was 'nearby but "
             "off' (low_iou_near_miss) vs. genuinely absent (no_prediction_nearby)",
    )
    parser.add_argument("--report", default="reports/missed_defects.json", help="Full listing output path")
    parser.add_argument("--summary-report", default="reports/missed_defects_summary.json", help="Per-class summary output path")
    parser.add_argument(
        "--copy-images", default=None,
        help="If set, copy each missed-defect image into this folder (deduplicated) for visual review",
    )
    parser.add_argument("--log-dir", default="./logs", help="Directory for log files")
    return parser


def main():
    args = build_parser().parse_args()
    setup_logger(log_dir=args.log_dir)
    logger = get_logger(__name__)

    entries, summary = analyze(args.results, args.source, args.iou_threshold, args.nearby_iou)

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(entries, f, indent=2)
    logger.info(f"Full listing written to {report_path}")

    summary_path = Path(args.summary_report)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary written to {summary_path}")

    if args.copy_images:
        dest_dir = Path(args.copy_images)
        dest_dir.mkdir(parents=True, exist_ok=True)
        results_path = Path(args.results)
        copied = set()
        for entry in entries:
            rel_image = entry["image"]
            src_path = results_path / rel_image
            if not src_path.exists() or rel_image in copied:
                continue
            dest_path = dest_dir / rel_image.replace("/", "_")
            shutil.copy2(src_path, dest_path)
            copied.add(rel_image)
        logger.info(f"Copied {len(copied)} image(s) to {dest_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
