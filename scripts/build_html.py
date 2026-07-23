"""
Task 5 - Generate a static HTML viewer for qualitative comparison.

Reads:
    results/**/*.json   (output of merge_annotations.py) - any annotation
        source keys present in the merged records can be rendered as a
        layer (ground_truth, production, rfdetr_v1, rfdetr_v2, new_model,
        ...); use --layers to pick which ones, defaulting to the 4-layer
        v1/v2 comparison set below for backward compatibility.

Writes:
    html/data.js     - all merged annotation data, embedded as a JS constant
                        (avoids fetch()/CORS issues when opening files directly
                        from disk, e.g. file:// in a browser)
    html/index.html  - gallery: thumbnails, search by filename, filter by folder
    html/detail.html - single image with toggleable annotation layers (one
                        checkbox per --layers entry)

Images are referenced via a relative path back into results/, so no image
copying is needed. Serve the html/ and results/ directories together, e.g.:
    python -m http.server --directory .

Usage:
    python scripts/build_html.py --results results --output html

    # Just ground truth vs. your new model (e.g. after merge_annotations.py
    # was run with only new_model's predictions file present):
    python scripts/build_html.py --results results --output html \
      --layers ground_truth,new_model

    # Only samples where new_model missed a real defect vs. ground truth:
    python scripts/build_html.py --results results --output html_missed \
      --layers ground_truth,new_model --only-missed-source new_model
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.logger import setup_logger, get_logger

DEFAULT_LAYERS = ["ground_truth", "production", "rfdetr_v1", "rfdetr_v2"]

LAYER_COLORS = {
    "ground_truth": "#2ecc71",
    "production": "#3498db",
    "rfdetr_v1": "#e67e22",
    "rfdetr_v2": "#e74c3c",
    "new_model": "#e74c3c",
}
LAYER_LABELS = {
    "ground_truth": "Ground Truth",
    "production": "Production",
    "rfdetr_v1": "RFDETR v1",
    "rfdetr_v2": "RFDETR v2",
    "new_model": "New Model",
}
# Cycled through for any layer key not covered above (e.g. a custom source
# name), so an unrecognized --layers value never crashes the build.
FALLBACK_COLORS = ["#9b59b6", "#1abc9c", "#f1c40f", "#95a5a6"]

INDEX_HTML = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>RFDETR Model Comparison</title>
<script src="data.js"></script>
<style>
  body { font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 0; background: #111; color: #eee; }
  header { padding: 16px 24px; background: #1a1a1a; border-bottom: 1px solid #333; position: sticky; top: 0; }
  h1 { font-size: 18px; margin: 0 0 12px; }
  .controls { display: flex; gap: 12px; flex-wrap: wrap; align-items: center; }
  input, select { padding: 8px 10px; border-radius: 6px; border: 1px solid #444; background: #222; color: #eee; font-size: 14px; }
  .field { display: flex; align-items: center; gap: 6px; font-size: 12px; color: #999; }
  .count { color: #999; font-size: 13px; margin-left: 4px; }
  .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 12px; padding: 20px; }
  .card { background: #1a1a1a; border-radius: 8px; overflow: hidden; border: 1px solid #2a2a2a; cursor: pointer; transition: transform 0.1s; }
  .card:hover { transform: translateY(-2px); border-color: #555; }
  .card img { width: 100%; height: 150px; object-fit: cover; display: block; background: #000; }
  .card .meta { padding: 8px 10px; font-size: 12px; }
  .card .name { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .card .folder { color: #888; }
</style>
</head>
<body>
<header>
  <h1>RFDETR Model Comparison <span class="count" id="count"></span></h1>
  <div class="controls">
    <input id="search" type="text" placeholder="Search by filename...">
    <select id="folder-filter"><option value="">All folders</option></select>
    <select id="class-filter"><option value="">All classes</option></select>
    <label class="field">From <input id="date-from" type="date"></label>
    <label class="field">To <input id="date-to" type="date"></label>
  </div>
</header>
<div class="grid" id="grid"></div>
<script>
  const folders = [...new Set(RFDETR_DATA.map(r => r.folder))].sort();
  const folderSelect = document.getElementById('folder-filter');
  for (const f of folders) {
    const opt = document.createElement('option');
    opt.value = f; opt.textContent = f;
    folderSelect.appendChild(opt);
  }

  const classes = [...new Set(RFDETR_DATA.flatMap(r => r.classes))].sort();
  const classSelect = document.getElementById('class-filter');
  for (const c of classes) {
    const opt = document.createElement('option');
    opt.value = c; opt.textContent = c;
    classSelect.appendChild(opt);
  }

  const searchInput = document.getElementById('search');
  const dateFrom = document.getElementById('date-from');
  const dateTo = document.getElementById('date-to');

  function render() {
    const q = searchInput.value.toLowerCase();
    const folder = folderSelect.value;
    const cls = classSelect.value;
    const from = dateFrom.value;
    const to = dateTo.value;
    const grid = document.getElementById('grid');
    grid.innerHTML = '';
    const filtered = RFDETR_DATA.filter(r =>
      (!folder || r.folder === folder) &&
      (!q || r.filename.toLowerCase().includes(q)) &&
      (!cls || r.classes.includes(cls)) &&
      (!from || (r.capturedAt && r.capturedAt >= from)) &&
      (!to || (r.capturedAt && r.capturedAt <= to))
    );
    document.getElementById('count').textContent = `(${filtered.length} of ${RFDETR_DATA.length})`;
    for (const r of filtered) {
      const card = document.createElement('div');
      card.className = 'card';
      card.onclick = () => { window.location.href = `detail.html?folder=${encodeURIComponent(r.folder)}&name=${encodeURIComponent(r.filename)}`; };
      card.innerHTML = `
        <img src="../results/${r.image}" loading="lazy">
        <div class="meta">
          <div class="name">${r.filename}</div>
          <div class="folder">${r.folder}${r.capturedAt ? ' &middot; ' + r.capturedAt : ''}</div>
        </div>`;
      grid.appendChild(card);
    }
  }

  searchInput.addEventListener('input', render);
  folderSelect.addEventListener('change', render);
  classSelect.addEventListener('change', render);
  dateFrom.addEventListener('change', render);
  dateTo.addEventListener('change', render);
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
<title>RFDETR Detail View</title>
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
  const folder = params.get('folder');
  const name = params.get('name');
  const record = RFDETR_DATA.find(r => r.folder === folder && r.filename === name);

  if (!record) {
    document.getElementById('title').textContent = 'Sample not found';
  } else {
    document.getElementById('title').textContent = `${record.folder} / ${record.filename}` +
      (record.capturedAt ? ` (${record.capturedAt})` : '');
    const img = document.getElementById('image');
    img.src = `../results/${record.image}`;

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
            tooltip.textContent = `${box.class} | ${(box.confidence * 100).toFixed(1)}% | ${LAYER_LABELS[layer]}`;
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


NO_DEFECT_LABEL = "Khong_co_loi"


def _iou(box1: list, box2: list) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union else 0.0


def has_missed_defect(predicted: list, ground_truth: list, iou_threshold: float) -> bool:
    """Same greedy highest-confidence-first IoU matching convention as
    compute_comparison_metrics.py::greedy_match / analyze_missed_defects.py.
    True if at least one real-defect GT box (i.e. excluding the no-defect
    label) has no matching prediction of ANY class at that location -
    regardless of whether the miss is a location miss or a wrong-class
    match, matching compute_comparison_metrics.py's per-class fn count."""
    defect_gt = [g for g in ground_truth if g.get("class") != NO_DEFECT_LABEL]
    if not defect_gt:
        return False

    matched_gt = set()
    for pred in sorted(predicted, key=lambda p: p.get("confidence", 0.0), reverse=True):
        best_idx, best_iou = None, 0.0
        for i, gt in enumerate(defect_gt):
            if i in matched_gt or gt.get("class") != pred.get("class"):
                continue
            score = _iou(pred["bbox"], gt["bbox"])
            if score > best_iou:
                best_idx, best_iou = i, score
        if best_idx is not None and best_iou >= iou_threshold:
            matched_gt.add(best_idx)

    return len(matched_gt) < len(defect_gt)


def build_manifest(results_dir: str, only_missed_source: str = None, iou_threshold: float = 0.5) -> list:
    manifest = []
    for json_path in sorted(Path(results_dir).rglob("*.json")):
        with open(json_path, "r") as f:
            record = json.load(f)
        annotations = record["annotations"]

        if only_missed_source is not None:
            predicted = annotations.get(only_missed_source, [])
            ground_truth = annotations.get("ground_truth", [])
            if not has_missed_defect(predicted, ground_truth, iou_threshold):
                continue

        folder, filename = record["image"].split("/", 1)
        classes = sorted({
            box["class"]
            for boxes in annotations.values()
            for box in boxes
            if box.get("class")
        })
        manifest.append({
            "folder": folder,
            "filename": filename,
            "image": record["image"],
            "capturedAt": record.get("captured_at"),
            "classes": classes,
            "annotations": annotations,
        })
    return manifest


def resolve_layers(layer_keys: list) -> tuple:
    """Build (colors, labels) dicts scoped to just the requested layer keys,
    falling back to FALLBACK_COLORS/title-cased key name for any key not in
    the known LAYER_COLORS/LAYER_LABELS (e.g. a custom source name)."""
    colors, labels = {}, {}
    next_fallback = 0
    for key in layer_keys:
        if key in LAYER_COLORS:
            colors[key] = LAYER_COLORS[key]
        else:
            colors[key] = FALLBACK_COLORS[next_fallback % len(FALLBACK_COLORS)]
            next_fallback += 1
        labels[key] = LAYER_LABELS.get(key, key.replace("_", " ").title())
    return colors, labels


def build_html(results_dir: str, output_dir: str, layers: list = None, only_missed_source: str = None, iou_threshold: float = 0.5):
    logger = get_logger(__name__)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    layer_keys = layers if layers else DEFAULT_LAYERS
    colors, labels = resolve_layers(layer_keys)

    manifest = build_manifest(results_dir, only_missed_source=only_missed_source, iou_threshold=iou_threshold)
    if only_missed_source is not None:
        logger.info(f"Filtered to samples where '{only_missed_source}' missed at least one ground-truth defect")

    with open(output_path / "data.js", "w") as f:
        f.write("window.RFDETR_DATA = ")
        json.dump(manifest, f)
        f.write(";\n")

    with open(output_path / "index.html", "w") as f:
        f.write(INDEX_HTML)

    detail_html = DETAIL_HTML.replace("__LAYER_COLORS__", json.dumps(colors))
    detail_html = detail_html.replace("__LAYER_LABELS__", json.dumps(labels))
    with open(output_path / "detail.html", "w") as f:
        f.write(detail_html)

    logger.info(f"Generated HTML viewer for {len(manifest)} samples in {output_path} (layers: {', '.join(layer_keys)})")
    return len(manifest)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a static HTML viewer for qualitative comparison")
    parser.add_argument("--results", default="results", help="Merged results directory")
    parser.add_argument("--output", default="html", help="Output directory for the HTML viewer")
    parser.add_argument(
        "--layers", default=None,
        help="Comma-separated annotation source keys to render as toggleable layers, e.g. "
             "'ground_truth,new_model'. Defaults to ground_truth,production,rfdetr_v1,rfdetr_v2 "
             "if omitted.",
    )
    parser.add_argument(
        "--only-missed-source", default=None, metavar="SOURCE",
        help="Only include samples where this annotation source (e.g. 'new_model') missed at "
             "least one real-defect ground-truth box - same greedy IoU-matching convention as "
             "compute_comparison_metrics.py. Omit to include every sample.",
    )
    parser.add_argument("--iou-threshold", type=float, default=0.5, help="IoU threshold used by --only-missed-source")
    parser.add_argument("--log-dir", default="./logs", help="Directory for log files")
    return parser


def main():
    args = build_parser().parse_args()
    setup_logger(log_dir=args.log_dir)

    layers = [k.strip() for k in args.layers.split(",")] if args.layers else None
    build_html(
        args.results, args.output, layers=layers,
        only_missed_source=args.only_missed_source, iou_threshold=args.iou_threshold,
    )
    print(f"Viewer written to {args.output}/index.html")
    print("Serve it together with results/, e.g.: python -m http.server --directory .")
    return 0


if __name__ == "__main__":
    sys.exit(main())
