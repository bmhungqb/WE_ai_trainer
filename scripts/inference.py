"""
Task 3 - Run RFDETR v1 and v2 inference over the downloaded dataset.

Reuses the same SAHI-sliced inference pattern as src/ai_verify.py
(AutoDetectionModel + get_sliced_prediction), but runs standalone against
local images instead of GCS-backed SampleInfo records.

Requires a GPU environment (torch, rfdetr, rfdetr_plus, sahi). This script is
NOT executed in this repository - see docs/RFDETR_COMPARISON.md for how to run
it on the GPU server.

Output:
    reports/predictions_rfdetr_v1.json
    reports/predictions_rfdetr_v2.json

    Each maps a dataset-relative image path ("TPWL/image001.jpg") to a list of
    {"bbox": [x1, y1, x2, y2], "confidence": float, "class": str}.

Usage:
    python scripts/inference.py \
        --dataset dataset \
        --v1-weights weights/weight_checkpoint_png_v6_distill.pth --v1-type rfdetrLarge \
        --v2-weights weights/weight_checkpoint_png_v7_distill.pth --v2-type rfdetrLarge \
        --output-dir reports
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.logger import setup_logger, get_logger
from utils.constants import DEFECT_CLASSES

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def build_model(model_type: str, weight_path: str, confidence_threshold: float):
    from sahi import AutoDetectionModel
    from rfdetr import RFDETRMedium, RFDETRLarge
    from rfdetr_plus import RFDETRXLarge

    if model_type == "rfdetrMedium":
        init_model = RFDETRMedium(pretrain_weights=weight_path)
    elif model_type == "rfdetrLarge":
        init_model = RFDETRLarge(pretrain_weights=weight_path)
    elif model_type == "rfdetrXLarge":
        init_model = RFDETRXLarge(pretrain_weights=weight_path)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    init_model.optimize_for_inference()
    return AutoDetectionModel.from_pretrained(
        model_type="roboflow",
        model=init_model,
        category_mapping=DEFECT_CLASSES,
        confidence_threshold=confidence_threshold,
    )


def predict_image(model, image, slice_size: int) -> list:
    from sahi.predict import get_sliced_prediction

    result = get_sliced_prediction(
        image,
        model,
        slice_height=slice_size,
        slice_width=slice_size,
        verbose=False,
    )

    boxes = []
    for annotation in result.to_coco_annotations():
        x1, y1, w, h = annotation["bbox"]
        boxes.append({
            "bbox": [x1, y1, x1 + w, y1 + h],
            "confidence": annotation["score"],
            "class": DEFECT_CLASSES.get(annotation["category_id"]),
        })
    return boxes


def run_inference(dataset_dir: str, model, slice_size: int) -> dict:
    from PIL import Image

    logger = get_logger(__name__)
    dataset_path = Path(dataset_dir)
    images = sorted(p for p in dataset_path.rglob("*") if p.suffix.lower() in IMAGE_EXTS)

    predictions = {}
    for i, image_path in enumerate(images, 1):
        rel_path = str(image_path.relative_to(dataset_path))
        try:
            image = Image.open(image_path).convert("RGB")
            predictions[rel_path] = predict_image(model, image, slice_size)
            logger.info(f"[{i}/{len(images)}] {rel_path}: {len(predictions[rel_path])} boxes")
        except Exception as e:
            logger.error(f"[{i}/{len(images)}] FAIL {rel_path}: {e}")
            predictions[rel_path] = []

    return predictions


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run RFDETR v1/v2 inference over the dataset")
    parser.add_argument("--dataset", default="dataset", help="Dataset directory")
    parser.add_argument("--output-dir", default="reports", help="Directory for prediction JSONs")
    parser.add_argument("--v1-weights", required=True, help="Path to RFDETR v1 checkpoint")
    parser.add_argument("--v1-type", default="rfdetrLarge", choices=["rfdetrMedium", "rfdetrLarge", "rfdetrXLarge"])
    parser.add_argument("--v2-weights", required=True, help="Path to RFDETR v2 checkpoint")
    parser.add_argument("--v2-type", default="rfdetrLarge", choices=["rfdetrMedium", "rfdetrLarge", "rfdetrXLarge"])
    parser.add_argument("--confidence-threshold", type=float, default=0.5)
    parser.add_argument("--slice-size", type=int, default=512)
    parser.add_argument("--log-dir", default="./logs", help="Directory for log files")
    return parser


def main():
    args = build_parser().parse_args()
    setup_logger(log_dir=args.log_dir)
    logger = get_logger(__name__)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for version, weights, model_type in (
        ("rfdetr_v1", args.v1_weights, args.v1_type),
        ("rfdetr_v2", args.v2_weights, args.v2_type),
    ):
        logger.info(f"Loading {version} ({model_type}) from {weights}")
        model = build_model(model_type, weights, args.confidence_threshold)

        predictions = run_inference(args.dataset, model, args.slice_size)

        out_path = output_dir / f"predictions_{version}.json"
        with open(out_path, "w") as f:
            json.dump(predictions, f, indent=2)
        logger.info(f"Wrote {out_path} ({len(predictions)} images)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
