"""
Task 5 - Generate a static HTML viewer for qualitative comparison.

Reads:
    results/**/*.json   (output of merge_annotations.py)

Writes:
    html/data.js     - all merged annotation data, embedded as a JS constant
                        (avoids fetch()/CORS issues when opening files directly
                        from disk, e.g. file:// in a browser)
    html/index.html  - gallery: thumbnails, search by filename, filter by folder
    html/detail.html - single image with four toggleable annotation layers
                        (Ground Truth, Production, RFDETR v1, RFDETR v2)

Images are referenced via a relative path back into results/, so no image
copying is needed. Serve the html/ and results/ directories together, e.g.:
    python -m http.server --directory .

Usage:
    python scripts/build_html.py --results results --output html
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.logger import setup_logger, get_logger

LAYER_COLORS = {
    "ground_truth": "#2ecc71",
    "production": "#3498db",
    "rfdetr_v1": "#e67e22",
    "rfdetr_v2": "#e74c3c",
}
LAYER_LABELS = {
    "ground_truth": "Ground Truth",
    "production": "Production",
    "rfdetr_v1": "RFDETR v1",
    "rfdetr_v2": "RFDETR v2",
}

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
  .controls { display: flex; gap: 12px; flex-wrap: wrap; }
  input, select { padding: 8px 10px; border-radius: 6px; border: 1px solid #444; background: #222; color: #eee; font-size: 14px; }
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

  function render() {
    const q = document.getElementById('search').value.toLowerCase();
    const folder = folderSelect.value;
    const grid = document.getElementById('grid');
    grid.innerHTML = '';
    const filtered = RFDETR_DATA.filter(r =>
      (!folder || r.folder === folder) &&
      (!q || r.filename.toLowerCase().includes(q))
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
          <div class="folder">${r.folder}</div>
        </div>`;
      grid.appendChild(card);
    }
  }

  document.getElementById('search').addEventListener('input', render);
  folderSelect.addEventListener('change', render);
  render();
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
    document.getElementById('title').textContent = `${record.folder} / ${record.filename}`;
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


def build_manifest(results_dir: str) -> list:
    manifest = []
    for json_path in sorted(Path(results_dir).rglob("*.json")):
        with open(json_path, "r") as f:
            record = json.load(f)
        folder, filename = record["image"].split("/", 1)
        manifest.append({
            "folder": folder,
            "filename": filename,
            "image": record["image"],
            "annotations": record["annotations"],
        })
    return manifest


def build_html(results_dir: str, output_dir: str):
    logger = get_logger(__name__)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    manifest = build_manifest(results_dir)

    with open(output_path / "data.js", "w") as f:
        f.write("window.RFDETR_DATA = ")
        json.dump(manifest, f)
        f.write(";\n")

    with open(output_path / "index.html", "w") as f:
        f.write(INDEX_HTML)

    detail_html = DETAIL_HTML.replace("__LAYER_COLORS__", json.dumps(LAYER_COLORS))
    detail_html = detail_html.replace("__LAYER_LABELS__", json.dumps(LAYER_LABELS))
    with open(output_path / "detail.html", "w") as f:
        f.write(detail_html)

    logger.info(f"Generated HTML viewer for {len(manifest)} samples in {output_path}")
    return len(manifest)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a static HTML viewer for qualitative comparison")
    parser.add_argument("--results", default="results", help="Merged results directory")
    parser.add_argument("--output", default="html", help="Output directory for the HTML viewer")
    parser.add_argument("--log-dir", default="./logs", help="Directory for log files")
    return parser


def main():
    args = build_parser().parse_args()
    setup_logger(log_dir=args.log_dir)

    build_html(args.results, args.output)
    print(f"Viewer written to {args.output}/index.html")
    print("Serve it together with results/, e.g.: python -m http.server --directory .")
    return 0


if __name__ == "__main__":
    sys.exit(main())
