"""
Run inference with a single new checkpoint (e.g. the June retrain with the
added "ignore" class) over a downloaded dataset directory, the same layout
produced by scripts/download_dataset.py.

Companion to scripts/inference.py (which runs a v1/v2 pair) - this variant
runs exactly one model, since here we're comparing one new model against the
existing production predictions already embedded in each sample's JSON
("pos"/"gt"), not against another RFDETR checkpoint.

Requires a GPU environment (torch, rfdetr, rfdetr_plus, sahi) - not run in
this repo's dev sandbox. See docs/MODEL_COMPARISON_JUNE.md for how to run it
on the GPU/training server.

Output:
    reports/predictions_new_model.json

    Maps a dataset-relative image path ("TPWL/image001.jpg") to a list of
    {"bbox": [x1, y1, x2, y2], "confidence": float, "class": str}.

Usage:
    python scripts/inference_new_model.py \
        --dataset dataset \
        --weights weights/weight_rfdetr_m_ignore_v1.pth \
        --model-type rfdetrMedium \
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


def build_model(model_type: str, weight_path: str, confidence_threshold: float, category_mapping: dict):
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
        category_mapping=category_mapping,
        confidence_threshold=confidence_threshold,
    )


def predict_image(model, image, slice_size: int, category_mapping: dict) -> list:
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
            "class": category_mapping.get(annotation["category_id"]),
        })
    return boxes


def run_inference(dataset_dir: str, model, slice_size: int, category_mapping: dict) -> dict:
    from PIL import Image

    logger = get_logger(__name__)
    dataset_path = Path(dataset_dir)
    images = sorted(p for p in dataset_path.rglob("*") if p.suffix.lower() in IMAGE_EXTS)

    predictions = {}
    for i, image_path in enumerate(images, 1):
        rel_path = str(image_path.relative_to(dataset_path))
        try:
            image = Image.open(image_path).convert("RGB")
            predictions[rel_path] = predict_image(model, image, slice_size, category_mapping)
            logger.info(f"[{i}/{len(images)}] {rel_path}: {len(predictions[rel_path])} boxes")
        except Exception as e:
            logger.error(f"[{i}/{len(images)}] FAIL {rel_path}: {e}")
            predictions[rel_path] = []

    return predictions


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run new-model inference over the dataset")
    parser.add_argument("--dataset", default="dataset", help="Dataset directory")
    parser.add_argument("--output-dir", default="reports", help="Directory for the prediction JSON")
    parser.add_argument("--weights", required=True, help="Path to the new checkpoint")
    parser.add_argument("--model-type", default="rfdetrMedium", choices=["rfdetrMedium", "rfdetrLarge", "rfdetrXLarge"])
    parser.add_argument(
        "--class-names",
        default=None,
        help="Comma-separated class names in the model's output category-id order, e.g. "
             "'pleat,stain,weaving,hard_pleat,ignore'. Defaults to the full DEFECT_CLASSES "
             "mapping if omitted.",
    )
    parser.add_argument("--confidence-threshold", type=float, default=0.5)
    parser.add_argument("--slice-size", type=int, default=576, help="SAHI slice size (match training image size)")
    parser.add_argument("--log-dir", default="./logs", help="Directory for log files")
    return parser


def main():
    args = build_parser().parse_args()
    setup_logger(log_dir=args.log_dir)
    logger = get_logger(__name__)

    if args.class_names:
        category_mapping = {i: name.strip() for i, name in enumerate(args.class_names.split(","))}
    else:
        category_mapping = DEFECT_CLASSES

    logger.info(f"Loading {args.model_type} from {args.weights} (classes: {list(category_mapping.values())})")
    model = build_model(args.model_type, args.weights, args.confidence_threshold, category_mapping)

    predictions = run_inference(args.dataset, model, args.slice_size, category_mapping)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "predictions_new_model.json"
    with open(out_path, "w") as f:
        json.dump(predictions, f, indent=2)
    logger.info(f"Wrote {out_path} ({len(predictions)} images)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
