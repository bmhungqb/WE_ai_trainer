# RFDETR Model Comparison & Visualization

Evaluation pipeline for comparing two RFDETR checkpoints (v1 vs v2) on
production data pulled from GCS bucket `jetson-textile-storage` (folders
`TPWL/`, `TPRL/`).

Note: despite the bucket being colloquially called "S3" in planning docs,
this project stores data in **Google Cloud Storage**, and every script below
uses the existing `google-cloud-storage` / `utils/gcs_utils.py` client, not
AWS S3.

Steps 1, 2, 4, and 5 are CPU-only and can run anywhere. Step 3 requires a GPU
and the `rfdetr` / `rfdetr_plus` / `sahi` / `torch` stack from
`requirements.txt` — run it on the GPU server.

## 0. Setup (GPU server)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in GOOGLE_APPLICATION_CREDENTIALS, LABEL_STUDIO_API_KEY, LABEL_STUDIO_URL
```

## 1. Download dataset

Each sample lives as a flat pair directly under `TPWL/`/`TPRL/` in the bucket:
`<name>.jpg` + `<name>.json` (there's also a `<name>.txt`, unused by this
pipeline). The JSON has the shape:

```json
{"pos": "<class> <cx> <cy> <w> <h> <conf>", "gt": 4}
```

`pos` is the on-device production prediction: one bbox (normalized 0-1,
`cx cy w h`) with its predicted class and confidence, as a single string or a
list of such strings (a fixed-size template — entries with fewer than 6
tokens, e.g. all-`nan` placeholders, mean no confirmed detection). `gt` is
the same bbox's class *after* a worker corrected it — the ground truth.

```bash
python scripts/download_dataset.py \
  --folders TPWL TPRL \
  --start-date 2026-05-01 --end-date 2026-06-30 \
  --output dataset
```

Dates are matched against each blob's GCS `time_created`. Add `--limit N` to
grab just a few samples for a quick smoke test. Output: `dataset/TPWL/*.jpg|.json`, `dataset/TPRL/*.png|.json`.

Each downloaded JSON is stamped with `_captured_at` (the blob's capture
date) for the time filter in step 5. If you have a `dataset/` downloaded
before this field existed, backfill it without re-downloading images:

```bash
python scripts/backfill_capture_time.py --dataset dataset --folders TPWL TPRL
```

## 2. Validate dataset

```bash
python scripts/validate_dataset.py --dataset dataset --report reports/report_dataset.json
```

Checks each image has a parsable JSON sidecar and an openable image; writes
`reports/report_dataset.json`.

## 3. Run RFDETR inference

Runs on CPU too (verified: ~5-6s/image/model with SAHI 512x512 slicing on a
1280x1280 image), just much slower than a GPU server for a large batch.

```bash
python scripts/inference.py \
  --dataset dataset \
  --v1-weights weights/weight_checkpoint_png_v6_distill.pth --v1-type rfdetrLarge \
  --v2-weights weights/weight_checkpoint_png_v7_distill.pth --v2-type rfdetrLarge \
  --output-dir reports
```

Swap `--v1-weights`/`--v2-weights` to point at whichever two checkpoints are
being compared. `--v1-type`/`--v2-type` must match the checkpoint's RFDETR
variant (`rfdetrMedium`, `rfdetrLarge`, `rfdetrXLarge`), same convention as
`src/ai_verify.py`. Writes `reports/predictions_rfdetr_v1.json` and
`reports/predictions_rfdetr_v2.json`.

## 4. Merge annotations

```bash
python scripts/merge_annotations.py --dataset dataset --predictions-dir reports --output results
```

Combines ground truth, production, and both RFDETR prediction sets into
`results/<folder>/<name>.json` (plus a copy of the image), in the format:

```json
{
  "image": "TPWL/image001.jpg",
  "annotations": {
    "production": [...],
    "ground_truth": [...],
    "rfdetr_v1": [...],
    "rfdetr_v2": [...]
  }
}
```

## 5. Build HTML visualization

```bash
python scripts/build_html.py --results results --output html
```

Generates `html/index.html` (gallery, searchable by filename and filterable
by folder, defect class, and capture date range) and `html/detail.html`
(per-image view with independently toggleable layers:
Ground Truth/green, Production/blue, RFDETR v1/orange, RFDETR v2/red; hover a
box to see class, confidence, and source).

Data is embedded in `html/data.js` (not fetched), so the viewer works from
`file://` too — but images are referenced via a relative path into
`results/`, so serve both directories together, e.g. from the project root:

```bash
python -m http.server --directory .
# open http://localhost:8000/html/index.html
```

## (Optional) 6. Compute metrics

```bash
python scripts/compute_metrics.py --results results --report reports/metrics.json
```

IoU-matched precision/recall/F1 for `production`, `rfdetr_v1`, and
`rfdetr_v2` against `ground_truth`.

## Project structure

```
scripts/
├── download_dataset.py       # Task 1
├── backfill_capture_time.py  # backfills _captured_at into older dataset/ downloads
├── validate_dataset.py    # Task 2
├── inference.py            # Task 3 (GPU)
├── merge_annotations.py    # Task 4
├── build_html.py           # Task 5
└── compute_metrics.py      # Optional

dataset/    # Task 1 output
reports/    # validation report, rfdetr predictions, metrics
results/    # Task 4 output (merged per-image annotations)
html/       # Task 5 output (static viewer)
```
