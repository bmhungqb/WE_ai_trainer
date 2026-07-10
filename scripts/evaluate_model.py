"""Run a trained RFDETR checkpoint over a COCO-format dataset (the layout
produced by scripts/build_train_valid_dataset.py: dataset/train, dataset/valid,
each with _annotations.coco.json) and build an HTML gallery to inspect failure
cases (false positives / false negatives) against ground truth.

Requires a GPU environment (torch, rfdetr, sahi) - not runnable in this repo's
dev sandbox. Run on the training/inference server.

Writes:
    reports/eval_predictions.json  - per-image GT / prediction / TP / FP / FN boxes
    reports/eval_metrics.json      - overall + per-class precision/recall/F1
    <html-output>/data.js
    <html-output>/index.html       - gallery, filterable by split/class, "only failures" toggle
    <html-output>/detail.html      - single image with toggleable GT / TP / FP / FN layers

Usage:
    python scripts/evaluate_model.py \
        --dataset dataset --split valid \
        --weights tmp/rfdetr_tuning/trial_0/checkpoint_best_ema.pth \
        --model-type rfdetrMedium \
        --output reports --html-output html_eval
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.logger import setup_logger, get_logger
from utils.constants import DEFECT_CLASSES

LAYER_COLORS = {
    "ground_truth": "#2ecc71",
    "true_positive": "#3498db",
    "false_positive": "#e74c3c",
    "false_negative": "#f1c40f",
}
LAYER_LABELS = {
    "ground_truth": "Ground Truth",
    "true_positive": "True Positive",
    "false_positive": "False Positive (extra)",
    "false_negative": "False Negative (missed)",
}


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


def match_boxes(predicted: list, ground_truth: list, iou_threshold: float = 0.5):
    """Greedy, class-aware matching. Returns (tp_preds, fp_preds, fn_gts)."""
    matched_gt = set()
    tp_preds, fp_preds = [], []
    for pred in sorted(predicted, key=lambda p: -p["confidence"]):
        best_idx, best_iou = None, 0.0
        for i, gt in enumerate(ground_truth):
            if i in matched_gt or gt["class"] != pred["class"]:
                continue
            score = iou(pred["bbox"], gt["bbox"])
            if score > best_iou:
                best_idx, best_iou = i, score
        if best_idx is not None and best_iou >= iou_threshold:
            matched_gt.add(best_idx)
            tp_preds.append(pred)
        else:
            fp_preds.append(pred)
    fn_gts = [gt for i, gt in enumerate(ground_truth) if i not in matched_gt]
    return tp_preds, fp_preds, fn_gts


def load_split(dataset_dir: Path, split: str):
    coco_path = dataset_dir / split / "_annotations.coco.json"
    with open(coco_path) as f:
        coco = json.load(f)

    cat_names = {c["id"]: c["name"] for c in coco["categories"]}
    gt_by_image = {}
    for anno in coco["annotations"]:
        x, y, w, h = anno["bbox"]
        gt_by_image.setdefault(anno["image_id"], []).append({
            "bbox": [x, y, x + w, y + h],
            "class": cat_names.get(anno["category_id"], str(anno["category_id"])),
        })
    return coco["images"], gt_by_image


def evaluate(dataset_dir: str, splits: list, model, slice_size: int, iou_threshold: float):
    from PIL import Image

    logger = get_logger(__name__)
    dataset_path = Path(dataset_dir)

    manifest = []
    class_totals = {}

    for split in splits:
        images, gt_by_image = load_split(dataset_path, split)
        logger.info(f"[{split}] {len(images)} images")

        for i, img in enumerate(images, 1):
            rel_path = f"{split}/{img['file_name']}"
            local_path = dataset_path / split / img["file_name"]
            ground_truth = gt_by_image.get(img["id"], [])

            try:
                image = Image.open(local_path).convert("RGB")
                predicted = predict_image(model, image, slice_size)
            except Exception as e:
                logger.error(f"[{split}][{i}/{len(images)}] FAIL {rel_path}: {e}")
                predicted = []

            tp, fp, fn = match_boxes(predicted, ground_truth, iou_threshold)

            classes = sorted({b["class"] for b in ground_truth} | {b["class"] for b in predicted})
            for cls in classes:
                totals = class_totals.setdefault(cls, {"tp": 0, "fp": 0, "fn": 0})
                totals["tp"] += sum(1 for b in tp if b["class"] == cls)
                totals["fp"] += sum(1 for b in fp if b["class"] == cls)
                totals["fn"] += sum(1 for b in fn if b["class"] == cls)

            manifest.append({
                "split": split,
                "filename": img["file_name"],
                "image": rel_path,
                "classes": classes,
                "counts": {"tp": len(tp), "fp": len(fp), "fn": len(fn)},
                "is_failure": bool(fp or fn),
                "annotations": {
                    "ground_truth": ground_truth,
                    "true_positive": tp,
                    "false_positive": fp,
                    "false_negative": fn,
                },
            })

            if i % 100 == 0:
                logger.info(f"[{split}] ...{i}/{len(images)}")

    metrics = {"overall": {"tp": 0, "fp": 0, "fn": 0}}
    for cls, t in class_totals.items():
        precision = t["tp"] / (t["tp"] + t["fp"]) if (t["tp"] + t["fp"]) else 0.0
        recall = t["tp"] / (t["tp"] + t["fn"]) if (t["tp"] + t["fn"]) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        metrics[cls] = {"precision": precision, "recall": recall, "f1": f1, **t}
        metrics["overall"]["tp"] += t["tp"]
        metrics["overall"]["fp"] += t["fp"]
        metrics["overall"]["fn"] += t["fn"]

    o = metrics["overall"]
    o["precision"] = o["tp"] / (o["tp"] + o["fp"]) if (o["tp"] + o["fp"]) else 0.0
    o["recall"] = o["tp"] / (o["tp"] + o["fn"]) if (o["tp"] + o["fn"]) else 0.0
    o["f1"] = 2 * o["precision"] * o["recall"] / (o["precision"] + o["recall"]) if (o["precision"] + o["recall"]) else 0.0

    return manifest, metrics


INDEX_HTML = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Model Failure Review</title>
<script src="data.js"></script>
<style>
  body { font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 0; background: #111; color: #eee; }
  header { padding: 16px 24px; background: #1a1a1a; border-bottom: 1px solid #333; position: sticky; top: 0; }
  h1 { font-size: 18px; margin: 0 0 12px; }
  .controls { display: flex; gap: 12px; flex-wrap: wrap; align-items: center; }
  input, select { padding: 8px 10px; border-radius: 6px; border: 1px solid #444; background: #222; color: #eee; font-size: 14px; }
  .field { display: flex; align-items: center; gap: 6px; font-size: 13px; color: #ccc; }
  .count { color: #999; font-size: 13px; margin-left: 4px; }
  .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 12px; padding: 20px; }
  .card { background: #1a1a1a; border-radius: 8px; overflow: hidden; border: 1px solid #2a2a2a; cursor: pointer; transition: transform 0.1s; }
  .card:hover { transform: translateY(-2px); border-color: #555; }
  .card.failure { border-color: #e74c3c; }
  .card img { width: 100%; height: 150px; object-fit: cover; display: block; background: #000; }
  .card .meta { padding: 8px 10px; font-size: 12px; }
  .card .name { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .card .badges { display: flex; gap: 6px; margin-top: 4px; }
  .badge { padding: 1px 6px; border-radius: 4px; font-size: 11px; }
  .badge.fp { background: #e74c3c33; color: #e74c3c; }
  .badge.fn { background: #f1c40f33; color: #f1c40f; }
  .badge.tp { background: #3498db33; color: #3498db; }
</style>
</head>
<body>
<header>
  <h1>Model Failure Review <span class="count" id="count"></span></h1>
  <div class="controls">
    <input id="search" type="text" placeholder="Search by filename...">
    <select id="split-filter"><option value="">All splits</option></select>
    <select id="class-filter"><option value="">All classes</option></select>
    <label class="field"><input id="only-failures" type="checkbox" checked> Only failures (FP/FN)</label>
  </div>
</header>
<div class="grid" id="grid"></div>
<script>
  const splits = [...new Set(EVAL_DATA.map(r => r.split))].sort();
  const splitSelect = document.getElementById('split-filter');
  for (const s of splits) {
    const opt = document.createElement('option');
    opt.value = s; opt.textContent = s;
    splitSelect.appendChild(opt);
  }

  const classes = [...new Set(EVAL_DATA.flatMap(r => r.classes))].sort();
  const classSelect = document.getElementById('class-filter');
  for (const c of classes) {
    const opt = document.createElement('option');
    opt.value = c; opt.textContent = c;
    classSelect.appendChild(opt);
  }

  const searchInput = document.getElementById('search');
  const onlyFailures = document.getElementById('only-failures');

  function render() {
    const q = searchInput.value.toLowerCase();
    const split = splitSelect.value;
    const cls = classSelect.value;
    const failOnly = onlyFailures.checked;
    const grid = document.getElementById('grid');
    grid.innerHTML = '';
    const filtered = EVAL_DATA.filter(r =>
      (!split || r.split === split) &&
      (!q || r.filename.toLowerCase().includes(q)) &&
      (!cls || r.classes.includes(cls)) &&
      (!failOnly || r.is_failure)
    );
    document.getElementById('count').textContent = `(${filtered.length} of ${EVAL_DATA.length})`;
    for (const r of filtered) {
      const card = document.createElement('div');
      card.className = 'card' + (r.is_failure ? ' failure' : '');
      card.onclick = () => { window.location.href = `detail.html?split=${encodeURIComponent(r.split)}&name=${encodeURIComponent(r.filename)}`; };
      card.innerHTML = `
        <img src="../dataset/${r.image}" loading="lazy">
        <div class="meta">
          <div class="name">${r.filename}</div>
          <div class="badges">
            <span class="badge tp">TP ${r.counts.tp}</span>
            <span class="badge fp">FP ${r.counts.fp}</span>
            <span class="badge fn">FN ${r.counts.fn}</span>
          </div>
        </div>`;
      grid.appendChild(card);
    }
  }

  searchInput.addEventListener('input', render);
  splitSelect.addEventListener('change', render);
  classSelect.addEventListener('change', render);
  onlyFailures.addEventListener('change', render);
  render();
</script>
</body>
</html>
"""

DETAIL_HTML = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Failure Detail</title>
<script src="data.js"></script>
<style>
  body { font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 0; background: #111; color: #eee; }
  header { padding: 12px 24px; background: #1a1a1a; border-bottom: 1px solid #333; display: flex; align-items: center; gap: 16px; }
  a.back { color: #9cf; text-decoration: none; font-size: 14px; }
  h1 { font-size: 16px; margin: 0; }
  .layers { display: flex; gap: 16px; padding: 12px 24px; background: #161616; flex-wrap: wrap; }
  .layer { display: flex; align-items: center; gap: 6px; font-size: 14px; cursor: pointer; user-select: none; }
  .swatch { width: 12px; height: 12px; border-radius: 3px; display: inline-block; }
  .stage { position: relative; display: inline-block; margin: 20px; max-width: calc(100% - 40px); }
  .stage img { display: block; max-width: 100%; height: auto; }
  svg.overlay { position: absolute; top: 0; left: 0; width: 100%; height: 100%; }
  svg.overlay rect { fill: none; stroke-width: 2; pointer-events: all; }
  svg.overlay rect:hover { stroke-width: 3; }
  #tooltip { position: fixed; pointer-events: none; background: rgba(0,0,0,0.9); color: #fff; padding: 6px 10px;
             border-radius: 6px; font-size: 12px; display: none; z-index: 10; white-space: nowrap; }
</style>
</head>
<body>
<header>
  <a class="back" href="index.html">&larr; Gallery</a>
  <h1 id="title"></h1>
</header>
<div class="layers" id="layers"></div>
<div class="stage" id="stage">
  <img id="image">
  <svg class="overlay" id="overlay"></svg>
</div>
<div id="tooltip"></div>
<script>
  const LAYER_COLORS = __LAYER_COLORS__;
  const LAYER_LABELS = __LAYER_LABELS__;

  const params = new URLSearchParams(window.location.search);
  const split = params.get('split');
  const name = params.get('name');
  const record = EVAL_DATA.find(r => r.split === split && r.filename === name);

  if (!record) {
    document.getElementById('title').textContent = 'Sample not found';
  } else {
    document.getElementById('title').textContent = `${record.split} / ${record.filename}` +
      ` (TP ${record.counts.tp} / FP ${record.counts.fp} / FN ${record.counts.fn})`;
    const img = document.getElementById('image');
    img.src = `../dataset/${record.image}`;

    const layersDiv = document.getElementById('layers');
    const activeLayers = new Set(Object.keys(LAYER_LABELS));

    for (const key of Object.keys(LAYER_LABELS)) {
      const label = document.createElement('label');
      label.className = 'layer';
      label.innerHTML = `<input type="checkbox" checked data-layer="${key}">
        <span class="swatch" style="background:${LAYER_COLORS[key]}"></span>${LAYER_LABELS[key]}`;
      layersDiv.appendChild(label);
    }

    layersDiv.addEventListener('change', (e) => {
      const key = e.target.dataset.layer;
      if (e.target.checked) activeLayers.add(key); else activeLayers.delete(key);
      draw();
    });

    const svg = document.getElementById('overlay');
    const tooltip = document.getElementById('tooltip');

    function draw() {
      svg.innerHTML = '';
      const scaleX = img.clientWidth / img.naturalWidth;
      const scaleY = img.clientHeight / img.naturalHeight;
      for (const layer of Object.keys(LAYER_LABELS)) {
        if (!activeLayers.has(layer)) continue;
        const boxes = record.annotations[layer] || [];
        for (const box of boxes) {
          const [x1, y1, x2, y2] = box.bbox;
          const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
          rect.setAttribute('x', x1 * scaleX);
          rect.setAttribute('y', y1 * scaleY);
          rect.setAttribute('width', (x2 - x1) * scaleX);
          rect.setAttribute('height', (y2 - y1) * scaleY);
          rect.setAttribute('stroke', LAYER_COLORS[layer]);
          rect.addEventListener('mousemove', (e) => {
            tooltip.style.display = 'block';
            tooltip.style.left = `${e.clientX + 12}px`;
            tooltip.style.top = `${e.clientY + 12}px`;
            const conf = box.confidence !== undefined ? ` | ${(box.confidence * 100).toFixed(1)}%` : '';
            tooltip.textContent = `${box.class}${conf} | ${LAYER_LABELS[layer]}`;
          });
          rect.addEventListener('mouseleave', () => { tooltip.style.display = 'none'; });
          svg.appendChild(rect);
        }
      }
    }

    if (img.complete) draw(); else img.addEventListener('load', draw);
    window.addEventListener('resize', draw);
  }
</script>
</body>
</html>
"""


def build_html(manifest: list, html_output: str):
    output_path = Path(html_output)
    output_path.mkdir(parents=True, exist_ok=True)

    with open(output_path / "data.js", "w") as f:
        f.write("window.EVAL_DATA = ")
        json.dump(manifest, f)
        f.write(";\n")

    with open(output_path / "index.html", "w") as f:
        f.write(INDEX_HTML)

    detail_html = DETAIL_HTML.replace("__LAYER_COLORS__", json.dumps(LAYER_COLORS))
    detail_html = detail_html.replace("__LAYER_LABELS__", json.dumps(LAYER_LABELS))
    with open(output_path / "detail.html", "w") as f:
        f.write(detail_html)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="dataset", help="Dataset dir with train/valid COCO subfolders")
    parser.add_argument("--split", nargs="+", default=["valid"], choices=["train", "valid"], help="Split(s) to evaluate")
    parser.add_argument("--weights", required=True, help="Path to trained RFDETR checkpoint")
    parser.add_argument("--model-type", default="rfdetrMedium", choices=["rfdetrMedium", "rfdetrLarge", "rfdetrXLarge"])
    parser.add_argument("--confidence-threshold", type=float, default=0.5)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--slice-size", type=int, default=576, help="SAHI slice size (match training image size)")
    parser.add_argument("--output", default="reports", help="Directory for prediction/metrics JSON")
    parser.add_argument("--html-output", default="html_eval", help="Directory for the HTML failure gallery")
    parser.add_argument("--log-dir", default="./logs", help="Directory for log files")
    return parser


def main():
    args = build_parser().parse_args()
    setup_logger(log_dir=args.log_dir)
    logger = get_logger(__name__)

    logger.info(f"Loading {args.model_type} from {args.weights}")
    model = build_model(args.model_type, args.weights, args.confidence_threshold)

    manifest, metrics = evaluate(args.dataset, args.split, model, args.slice_size, args.iou_threshold)

    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    with open(output_path / "eval_predictions.json", "w") as f:
        json.dump(manifest, f, indent=2)
    with open(output_path / "eval_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    build_html(manifest, args.html_output)

    n_failures = sum(1 for r in manifest if r["is_failure"])
    logger.info(f"Evaluated {len(manifest)} images, {n_failures} with failures (FP/FN)")
    logger.info(f"Metrics: {json.dumps(metrics, indent=2)}")
    print(f"\nPredictions: {output_path / 'eval_predictions.json'}")
    print(f"Metrics: {output_path / 'eval_metrics.json'}")
    print(f"Gallery: {args.html_output}/index.html")
    print("Serve it together with dataset/, e.g.: python -m http.server --directory .")
    return 0


if __name__ == "__main__":
    sys.exit(main())
