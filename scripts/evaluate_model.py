"""Run a trained RFDETR checkpoint over a COCO-format dataset (the layout
produced by scripts/build_train_valid_dataset.py: dataset/train, dataset/valid,
each with _annotations.coco.json) and build an HTML gallery to inspect
predictions against ground truth.

Requires a GPU environment (torch, rfdetr, sahi) - not runnable in this repo's
dev sandbox. Run on the training/inference server.

Writes:
    reports/eval_predictions.json  - per-image GT / prediction boxes
    <html-output>/data.js
    <html-output>/index.html       - gallery, filterable by split/class
    <html-output>/detail.html      - single image with toggleable GT / Prediction layers

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
    "prediction": "#3498db",
}
LAYER_LABELS = {
    "ground_truth": "Ground Truth",
    "prediction": "Prediction",
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


def evaluate(dataset_dir: str, splits: list, model, slice_size: int):
    from PIL import Image

    logger = get_logger(__name__)
    dataset_path = Path(dataset_dir)

    manifest = []

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

            classes = sorted({b["class"] for b in ground_truth} | {b["class"] for b in predicted})

            manifest.append({
                "split": split,
                "filename": img["file_name"],
                "image": rel_path,
                "classes": classes,
                "counts": {"ground_truth": len(ground_truth), "prediction": len(predicted)},
                "annotations": {
                    "ground_truth": ground_truth,
                    "prediction": predicted,
                },
            })

            if i % 100 == 0:
                logger.info(f"[{split}] ...{i}/{len(images)}")

    return manifest


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
  .card img { width: 100%; height: 150px; object-fit: cover; display: block; background: #000; }
  .card .meta { padding: 8px 10px; font-size: 12px; }
  .card .name { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .card .badges { display: flex; gap: 6px; margin-top: 4px; }
  .badge { padding: 1px 6px; border-radius: 4px; font-size: 11px; }
  .badge.gt { background: #2ecc7133; color: #2ecc71; }
  .badge.pred { background: #3498db33; color: #3498db; }
</style>
</head>
<body>
<header>
  <h1>Model Prediction Review <span class="count" id="count"></span></h1>
  <div class="controls">
    <input id="search" type="text" placeholder="Search by filename...">
    <select id="split-filter"><option value="">All splits</option></select>
    <select id="class-filter"><option value="">All classes</option></select>
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

  function render() {
    const q = searchInput.value.toLowerCase();
    const split = splitSelect.value;
    const cls = classSelect.value;
    const grid = document.getElementById('grid');
    grid.innerHTML = '';
    const filtered = EVAL_DATA.filter(r =>
      (!split || r.split === split) &&
      (!q || r.filename.toLowerCase().includes(q)) &&
      (!cls || r.classes.includes(cls))
    );
    document.getElementById('count').textContent = `(${filtered.length} of ${EVAL_DATA.length})`;
    for (const r of filtered) {
      const card = document.createElement('div');
      card.className = 'card';
      card.onclick = () => { window.location.href = `detail.html?split=${encodeURIComponent(r.split)}&name=${encodeURIComponent(r.filename)}`; };
      card.innerHTML = `
        <img src="../dataset/${r.image}" loading="lazy">
        <div class="meta">
          <div class="name">${r.filename}</div>
          <div class="badges">
            <span class="badge gt">GT ${r.counts.ground_truth}</span>
            <span class="badge pred">Pred ${r.counts.prediction}</span>
          </div>
        </div>`;
      grid.appendChild(card);
    }
  }

  searchInput.addEventListener('input', render);
  splitSelect.addEventListener('change', render);
  classSelect.addEventListener('change', render);
  render();
  window.addEventListener('pageshow', render);
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
      ` (GT ${record.counts.ground_truth} / Pred ${record.counts.prediction})`;
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
    parser.add_argument("--slice-size", type=int, default=576, help="SAHI slice size (match training image size)")
    parser.add_argument("--output", default="reports", help="Directory for prediction JSON")
    parser.add_argument("--html-output", default="html_eval", help="Directory for the HTML gallery")
    parser.add_argument("--log-dir", default="./logs", help="Directory for log files")
    return parser


def main():
    args = build_parser().parse_args()
    setup_logger(log_dir=args.log_dir)
    logger = get_logger(__name__)

    logger.info(f"Loading {args.model_type} from {args.weights}")
    model = build_model(args.model_type, args.weights, args.confidence_threshold)

    manifest = evaluate(args.dataset, args.split, model, args.slice_size)

    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    with open(output_path / "eval_predictions.json", "w") as f:
        json.dump(manifest, f, indent=2)

    build_html(manifest, args.html_output)

    logger.info(f"Evaluated {len(manifest)} images")
    print(f"\nPredictions: {output_path / 'eval_predictions.json'}")
    print(f"Gallery: {args.html_output}/index.html")
    print("Serve it together with dataset/, e.g.: python -m http.server --directory .")
    return 0


if __name__ == "__main__":
    sys.exit(main())
