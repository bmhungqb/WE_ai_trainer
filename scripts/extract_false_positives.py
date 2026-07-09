"""
Extract confirmed False Positive OR True Positive detections of a given
model/month into cropped image patches + a manifest, for manual visual
review.

A detection is scored against its best-IoU-matching ground_truth box
(IoU >= iou-threshold), location-only (no class check), same convention as
scripts/compute_metrics.py:
  - confirmed FP: matched GT box's worker-corrected label is "Khong_co_loi"
    (the worker looked at that exact spot and confirmed there was no defect).
  - confirmed TP: matched GT box's label is a real defect (anything other
    than "Khong_co_loi").
Detections with no GT match at all are neither - not extracted.

Reads:
    results/<folder>/<name>.json + <name>.jpg|png   (output of merge_annotations.py)

Writes:
    <output>/crops/<folder>__<name>__<idx>.jpg   - cropped detection patch
    <output>/manifest.json                        - one entry per extracted detection
    <output>/gallery.html                         - self-contained viewer (grid of
        crops, sorted by confidence; click a card to see the box drawn on the
        full source image). References images in <results>/, so serve <output>
        and <results> together, e.g. from the project root:
            python -m http.server --directory .

Usage:
    python scripts/extract_false_positives.py \
      --results results \
      --model rfdetr_v2 \
      --category fp \
      --month 2026-06 \
      --output reports/fp_june_v2

    python scripts/extract_false_positives.py \
      --results results \
      --model rfdetr_v2 \
      --category tp \
      --month 2026-06 \
      --output reports/tp_june_v2
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.logger import setup_logger, get_logger

IMAGE_EXTS = (".jpg", ".jpeg", ".png")
NO_DEFECT_LABEL = "Khong_co_loi"

GALLERY_HTML = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>__TITLE__</title>
<style>
  body { font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 0; background: #111; color: #eee; }
  header { padding: 16px 24px; background: #1a1a1a; border-bottom: 1px solid #333; position: sticky; top: 0; z-index: 5; }
  h1 { font-size: 18px; margin: 0 0 12px; }
  .controls { display: flex; gap: 12px; flex-wrap: wrap; align-items: center; }
  input, select { padding: 8px 10px; border-radius: 6px; border: 1px solid #444; background: #222; color: #eee; font-size: 14px; }
  .field { display: flex; align-items: center; gap: 6px; font-size: 12px; color: #999; }
  .count { color: #999; font-size: 13px; margin-left: 4px; }
  .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 12px; padding: 20px; }
  .card { background: #1a1a1a; border-radius: 8px; overflow: hidden; border: 1px solid #2a2a2a; cursor: pointer; transition: transform 0.1s; }
  .card:hover { transform: translateY(-2px); border-color: #555; }
  .card img { width: 100%; height: 150px; object-fit: cover; display: block; background: #000; }
  .card .meta { padding: 8px 10px; font-size: 12px; }
  .card .conf { font-weight: 600; color: __ACCENT__; }
  .card .name { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; color: #999; }
  .modal { display: none; position: fixed; inset: 0; background: rgba(0,0,0,0.85); z-index: 10; }
  .modal.open { display: flex; align-items: center; justify-content: center; }
  .modal-close { position: absolute; top: 16px; right: 24px; font-size: 28px; color: #fff; cursor: pointer; }
  .stage { position: relative; display: inline-block; max-width: 90vw; max-height: 90vh; }
  .stage img { display: block; max-width: 90vw; max-height: 90vh; height: auto; }
  svg.overlay { position: absolute; top: 0; left: 0; width: 100%; height: 100%; }
  svg.overlay rect { fill: none; stroke: __ACCENT__; stroke-width: 3; }
  .modal-meta { position: absolute; bottom: -36px; left: 0; color: #ccc; font-size: 13px; }
</style>
</head>
<body>
<header>
  <h1>__TITLE__ <span class="count" id="count"></span></h1>
  <div class="controls">
    <input id="search" type="text" placeholder="Search by filename...">
    <select id="class-filter"><option value="">All predicted classes</option></select>
    <label class="field">Min confidence <input id="min-conf" type="number" min="0" max="1" step="0.05" value="0"></label>
  </div>
</header>
<div class="grid" id="grid"></div>
<div class="modal" id="modal">
  <span class="modal-close" id="modal-close">&times;</span>
  <div class="stage">
    <img id="modal-img">
    <svg class="overlay" id="modal-overlay"></svg>
    <div class="modal-meta" id="modal-meta"></div>
  </div>
</div>
<script>
  const DATA = __MANIFEST__;

  const classes = [...new Set(DATA.map(d => d.predicted_class))].sort();
  const classSelect = document.getElementById('class-filter');
  for (const c of classes) {
    const opt = document.createElement('option');
    opt.value = c; opt.textContent = c;
    classSelect.appendChild(opt);
  }

  const searchInput = document.getElementById('search');
  const minConf = document.getElementById('min-conf');
  const grid = document.getElementById('grid');
  const modal = document.getElementById('modal');
  const modalImg = document.getElementById('modal-img');
  const modalOverlay = document.getElementById('modal-overlay');
  const modalMeta = document.getElementById('modal-meta');

  function openModal(item) {
    modal.classList.add('open');
    modalImg.src = item.full_image_rel;
    modalMeta.textContent = `${item.source_image} | ${item.predicted_class} | conf ${(item.confidence*100).toFixed(1)}% | matched GT: ${item.matched_gt_class} | ${item.captured_at || ''}`;
    function draw() {
      modalOverlay.innerHTML = '';
      const scaleX = modalImg.clientWidth / modalImg.naturalWidth;
      const scaleY = modalImg.clientHeight / modalImg.naturalHeight;
      const [x1, y1, x2, y2] = item.bbox;
      const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
      rect.setAttribute('x', x1 * scaleX);
      rect.setAttribute('y', y1 * scaleY);
      rect.setAttribute('width', (x2 - x1) * scaleX);
      rect.setAttribute('height', (y2 - y1) * scaleY);
      modalOverlay.appendChild(rect);
    }
    if (modalImg.complete) draw(); else modalImg.onload = draw;
  }
  document.getElementById('modal-close').onclick = () => modal.classList.remove('open');
  modal.addEventListener('click', (e) => { if (e.target === modal) modal.classList.remove('open'); });

  function render() {
    const q = searchInput.value.toLowerCase();
    const cls = classSelect.value;
    const mc = parseFloat(minConf.value) || 0;
    const filtered = DATA.filter(d =>
      (!q || d.source_image.toLowerCase().includes(q)) &&
      (!cls || d.predicted_class === cls) &&
      d.confidence >= mc
    );
    document.getElementById('count').textContent = `(${filtered.length} of ${DATA.length})`;
    grid.innerHTML = '';
    for (const item of filtered) {
      const card = document.createElement('div');
      card.className = 'card';
      card.onclick = () => openModal(item);
      card.innerHTML = `
        <img src="${item.crop_file}" loading="lazy">
        <div class="meta">
          <div class="conf">${(item.confidence*100).toFixed(1)}% &middot; ${item.predicted_class}</div>
          <div class="name">${item.source_image}</div>
        </div>`;
      grid.appendChild(card);
    }
  }

  searchInput.addEventListener('input', render);
  classSelect.addEventListener('change', render);
  minConf.addEventListener('input', render);
  render();
</script>
</body>
</html>
"""


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


def find_image_path(record_json_path: Path) -> Path | None:
    for ext in IMAGE_EXTS:
        candidate = record_json_path.with_suffix(ext)
        if candidate.exists():
            return candidate
    return None


def classify_detection(det: dict, ground_truth: list, category: str, iou_threshold: float) -> dict | None:
    """Returns the matched GT box if `det` is a confirmed member of `category`
    ("fp" or "tp"), else None. Matched against ANY gt box regardless of class
    (location-only), same convention as scripts/compute_metrics.py."""
    best_iou, best_gt = 0.0, None
    for g in ground_truth:
        score = iou(det["bbox"], g["bbox"])
        if score > best_iou:
            best_iou, best_gt = score, g
    if best_iou < iou_threshold or best_gt is None:
        return None
    is_no_defect = best_gt["class"] == NO_DEFECT_LABEL
    if category == "fp" and is_no_defect:
        return best_gt
    if category == "tp" and not is_no_defect:
        return best_gt
    return None


def crop_with_padding(img, bbox: list, padding: int):
    width, height = img.size
    x1, y1, x2, y2 = bbox
    x1 = max(0, int(x1) - padding)
    y1 = max(0, int(y1) - padding)
    x2 = min(width, int(x2) + padding)
    y2 = min(height, int(y2) + padding)
    return img.crop((x1, y1, x2, y2))


def extract(results_dir: str, model: str, category: str, month: str, output_dir: str,
            iou_threshold: float, padding: int, min_confidence: float) -> list:
    from PIL import Image

    logger = get_logger(__name__)
    output_path = Path(output_dir)
    crops_path = output_path / "crops"
    crops_path.mkdir(parents=True, exist_ok=True)

    manifest = []
    n_records_matched_month = 0
    n_detections_seen = 0

    for json_path in sorted(Path(results_dir).rglob("*.json")):
        with open(json_path, "r") as f:
            record = json.load(f)

        captured_at = record.get("captured_at") or ""
        if month and not captured_at.startswith(month):
            continue
        n_records_matched_month += 1

        annotations = record.get("annotations", {})
        ground_truth = annotations.get("ground_truth", [])
        detections = annotations.get(model, [])
        if not detections:
            continue

        image_path = find_image_path(json_path)
        if image_path is None:
            logger.warning(f"No image found for {json_path}, skipping its detections")
            continue

        img = None
        for idx, det in enumerate(detections):
            n_detections_seen += 1
            if det["confidence"] < min_confidence:
                continue
            matched_gt = classify_detection(det, ground_truth, category, iou_threshold)
            if matched_gt is None:
                continue

            if img is None:
                img = Image.open(image_path)
            crop = crop_with_padding(img, det["bbox"], padding)

            crop_name = f"{record['image'].replace('/', '__')}__{idx}.jpg"
            crop.convert("RGB").save(crops_path / crop_name, quality=95)

            full_image_rel = os.path.relpath(image_path, output_path)

            manifest.append({
                "crop_file": f"crops/{crop_name}",
                "source_image": record["image"],
                "full_image_rel": full_image_rel,
                "captured_at": captured_at,
                "model": model,
                "bbox": det["bbox"],
                "confidence": det["confidence"],
                "predicted_class": det["class"],
                "matched_gt_class": matched_gt["class"],
            })

    manifest.sort(key=lambda m: -m["confidence"])
    with open(output_path / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    title = f"{'False Positives' if category == 'fp' else 'True Positives'} Viewer &mdash; {model} / {month or 'all'}"
    accent = "#e74c3c" if category == "fp" else "#2ecc71"
    gallery_html = GALLERY_HTML.replace("__TITLE__", title)
    gallery_html = gallery_html.replace("__ACCENT__", accent)
    gallery_html = gallery_html.replace("__MANIFEST__", json.dumps(manifest, ensure_ascii=False))
    with open(output_path / "gallery.html", "w") as f:
        f.write(gallery_html)

    logger.info(
        f"Scanned {n_records_matched_month} records in month={month!r}, "
        f"{n_detections_seen} {model} detections seen, "
        f"{len(manifest)} confirmed {category.upper()} extracted to {crops_path}"
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract confirmed FP/TP detections as image crops")
    parser.add_argument("--results", default="results", help="Merged results directory (output of merge_annotations.py)")
    parser.add_argument("--model", default="rfdetr_v2", choices=["production", "rfdetr_v1", "rfdetr_v2"])
    parser.add_argument("--category", default="fp", choices=["fp", "tp"], help="Which confirmed detections to extract")
    parser.add_argument("--month", default="2026-06", help="YYYY-MM prefix to filter captured_at; empty string = all")
    parser.add_argument("--output", default=None, help="Output directory for crops + manifest (default: reports/<category>_<model>_<month>)")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--padding", type=int, default=20, help="Pixels of context padding around each crop")
    parser.add_argument("--min-confidence", type=float, default=0.0, help="Skip detections below this confidence")
    parser.add_argument("--log-dir", default="./logs", help="Directory for log files")
    return parser


def main():
    args = build_parser().parse_args()
    setup_logger(log_dir=args.log_dir)

    output = args.output or f"reports/{args.category}_{args.model}_{args.month or 'all'}"

    manifest = extract(
        args.results, args.model, args.category, args.month, output,
        args.iou_threshold, args.padding, args.min_confidence,
    )

    print(f"Extracted {len(manifest)} confirmed {args.category.upper()} crops to {output}/crops")
    print(f"Manifest: {output}/manifest.json")
    print(f"Viewer: {output}/gallery.html (serve together with --results, e.g. `python -m http.server --directory .`)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
