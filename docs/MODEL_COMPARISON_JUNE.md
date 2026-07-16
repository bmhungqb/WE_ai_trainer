# New Model (June retrain) vs Production Comparison

Evaluation pipeline for deciding whether the new 5-class model (trained on
the June dataset: `pleat`, `stain`, `weaving`, `hard_pleat`, and the new
`ignore` class for suppressing false alarms) is ready to replace the current
production model, using production data pulled from GCS bucket
`jetson-textile-storage` (folders `TPWL/`, `TPRL/`).

This reuses most of the pipeline documented in `docs/RFDETR_COMPARISON.md` -
only the inference and metrics steps differ (one new checkpoint instead of a
v1/v2 pair, and metrics scored against both `ground_truth` *and* the
existing `production` predictions already embedded in each sample).

Steps 1, 2, 4, and 6 are CPU-only. Step 3 requires a GPU and the `rfdetr` /
`rfdetr_plus` / `sahi` / `torch` stack from `requirements.txt` - run it on
the training/inference server.

## 0. Setup (GPU server)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in GOOGLE_APPLICATION_CREDENTIALS
```

## 1. Download dataset

Each sample lives as a flat pair directly under `TPWL/`/`TPRL/` in the
bucket: `<name>.jpg` + `<name>.json`. The JSON has the shape:

```json
{"pos": "<class> <cx> <cy> <w> <h> <conf>", "gt": "Khong co loi"}
```

`pos` is the on-device **production** prediction: one bbox (normalized 0-1,
`cx cy w h`) with its predicted class and confidence, as a single string or a
list of such strings. `gt` is the worker-corrected class for that same
bbox - the **ground truth**. A worker-confirmed no-defect spot has
`gt == "Khong co loi"` (mapped to `Khong_co_loi`, see
`utils/constants.py::MAPPING_CLASSES`) - these are the samples the new
model's `ignore` class is meant to catch.

```bash
python scripts/download_dataset.py \
  --folders TPWL TPRL \
  --start-date 2026-06-01 --end-date 2026-06-30 \
  --output dataset
```

Dates are matched against each blob's GCS `time_created`. Output:
`dataset/TPWL/*.jpg|.json`, `dataset/TPRL/*.png|.json`.

## 2. Validate dataset

```bash
python scripts/validate_dataset.py --dataset dataset --report reports/report_dataset.json
```

## 3. Run new-model inference

```bash
python scripts/inference_new_model.py \
  --dataset dataset \
  --weights weights/weight_rfdetr_m_june_ignore.pth \
  --model-type rfdetrMedium \
  --output-dir reports
```

If the checkpoint's output category order differs from
`utils/constants.py::DEFECT_CLASSES` (`pleat,stain,weaving,hard_pleat,ignore`),
pass `--class-names` explicitly, e.g. `--class-names pleat,stain,weaving,hard_pleat,ignore`.
Writes `reports/predictions_new_model.json`.

## 4. Merge annotations

```bash
python scripts/merge_annotations.py --dataset dataset --predictions-dir reports --output results
```

Combines ground truth, production, and the new model's predictions into
`results/<folder>/<name>.json` (plus a copy of the image):

```json
{
  "image": "TPWL/image001.jpg",
  "annotations": {
    "production": [...],
    "ground_truth": [...],
    "new_model": [...]
  }
}
```

(If `reports/predictions_rfdetr_v1.json` / `_v2.json` also exist from a
previous run of `scripts/inference.py`, they're merged in too - the script
tolerates any subset of prediction sources being present.)

## 5. (Optional) Build HTML visualization

```bash
python scripts/build_html.py --results results --output html
```

Same gallery/detail viewer as the RFDETR comparison pipeline - useful for
spot-checking specific false alarms or missed defects before trusting the
aggregate numbers.

## 6. Compute comparison metrics

```bash
python scripts/compute_comparison_metrics.py --results results --report reports/comparison_metrics.json
```

Writes `reports/comparison_metrics.json` with, for both `production` and
`new_model`:

- **Overall + per-class precision/recall/F1**, IoU-matched (default
  threshold 0.5, greedy highest-confidence-first) and requiring the
  predicted class to match the ground-truth class. Ground truth's
  `Khong_co_loi` no-defect label and the new model's `ignore` prediction
  label are treated as the same class here, so a correct suppression scores
  as a true positive instead of being penalized as a class mismatch.
- **`false_alarm_suppression`**: among predictions landing on a
  worker-confirmed no-defect ground-truth box, how many still raised a real
  defect class (`false_alarms`, i.e. would trigger an unnecessary manual
  check on the line) vs. how many correctly abstained
  (`suppressed`) - plus the resulting `false_alarm_rate` /
  `suppression_rate`. Production has no `ignore` class, so its
  `false_alarm_rate` is effectively "how often does production cry wolf on
  a spot workers already confirmed clean" - the number this whole retrain is
  meant to bring down.

Read `comparison_metrics.json`'s `sources.new_model` vs `sources.production`
side by side: the new model is a reasonable candidate to promote to
production if its overall F1 is at or above production's *and* its
`false_alarm_rate` is meaningfully lower, without a large recall drop on any
individual real-defect class (check `per_class` for that - an aggregate F1
gain can hide a regression on a specific defect type).

## Project structure

```
scripts/
├── download_dataset.py            # Task 1
├── validate_dataset.py            # Task 2
├── inference_new_model.py         # Task 3 (GPU) - single new checkpoint
├── merge_annotations.py           # Task 4
├── build_html.py                  # Task 5 (optional)
└── compute_comparison_metrics.py  # Task 6

dataset/    # Task 1 output
reports/    # validation report, new-model predictions, comparison_metrics.json
results/    # Task 4 output (merged per-image annotations)
html/       # Task 5 output (static viewer, optional)
```
