# Agentic AI Textile Defect Detection

An automated AI pipeline for textile defect detection that handles data ingestion from Google Cloud Storage, pseudo-labeling with SAHI-based slicing, human-in-the-loop review via Label Studio, and Optuna-based hyperparameter-optimized model training.

## Architecture

```
GCS (raw images) ──> DataProcessor ──> AIVerify (RF-DETR predictions + SAHI slicing)
        │                                        │
        │                                        v
        │                              Label Studio (human review)
        │                                        │
        v                                        v
   DatasetManager (merge old + new data) ──> AITrainer (Optuna + RF-DETR)
                                                 │
                                                 v
                                          GCS (trained models)
```

## Project Structure

```
.
├── run.py                  # CLI entry point
├── src/
│   ├── agentic_pipeline.py # Main pipeline orchestrator
│   ├── data_processor.py   # Data download and formatting
│   ├── ai_verify.py        # Model inference and verification
│   ├── ai_trainer.py       # Optuna-based model training
│   └── data_manager.py     # Dataset merging and splitting
├── utils/
│   ├── config.py           # Configuration management
│   ├── gcs_utils.py        # Google Cloud Storage utilities
│   ├── label_studio_utils.py # Label Studio API integration
│   ├── schemas.py          # Data schemas
│   ├── logger.py           # Logging setup
│   └── constants.py        # Constants
├── weights/                # Model weight files
└── requirements.txt
```

## Setup

1. **Create and activate a virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables:**
   ```bash
   cp .env.example .env
   ```
   Fill in your credentials:
   - `LABEL_STUDIO_API_KEY` - API key for Label Studio
   - `LABEL_STUDIO_URL` - Label Studio server URL

4. **Authenticate with Google Cloud Storage:**
   ```bash
   gcloud auth application-default login
   ```
   GCS uses [Application Default Credentials (ADC)](https://cloud.google.com/docs/authentication/application-default-credentials) — no API key needed.

## Usage

All pipeline operations are driven through `run.py` with CLI arguments.

### Pipeline Modes

| Mode | Description |
|------|-------------|
| `prepare` | Download data from GCS, run AI verification, push pre-annotations to Label Studio |
| `train` | Pull reviewed data from Label Studio, merge datasets, train with Optuna optimization |
| `all` | Run both pipelines in sequence |

### Examples

**Prepare data for review:**
```bash
python run.py --mode prepare \
  --start-date 2025-01-01 --end-date 2025-01-31 \
  --gcs-bucket jetson-textile-storage \
  --gcs-folder-paths N
```
```bash
python run.py --mode prepare \
  --start-date 2025-01-01 --end-date 2025-01-31 \
  --gcs-bucket jetson-textile-storage \
  --gcs-folder-paths N M K \
  --model 1:rfdetrMedium:weights/weight_rfdetr_m_slice_dinov3_v3.pth \
         2:rfdetrLarge:weights/weight_rfdetr_l_v2.pth

- --gcs-folder-paths accepts multiple space-separated folder names
- --model accepts multiple entries, each in the format id:type:weight_path
```

**Train a model on reviewed data:**
```bash
python run.py --mode train \
  --start-date 2025-01-01 --end-date 2025-01-31 \
  --pretrained-weights weights/weight_rfdetr_m_slice_dinov3_v3.pth \
  --n-trials 10
```

**Run the full pipeline:**
```bash
python run.py --mode all \
  --start-date 2025-01-01 --end-date 2025-01-31
```

**Custom model and image size:**
```bash
python run.py --mode prepare \
  --start-date 2025-01-01 --end-date 2025-01-31 \
  --image-size 576 576 \
  --model 1:rfdetrMedium:weights/weight_rfdetr_m_slice_dinov3_v3.pth
```

### Key CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `--mode` | `prepare` | Pipeline mode: `prepare`, `train`, or `all` |
| `--start-date` | *(required)* | Start date (YYYY-MM-DD) |
| `--end-date` | *(required)* | End date (YYYY-MM-DD) |
| `--gcs-bucket` | `jetson-textile-storage` | Source GCS bucket |
| `--gcs-folder-paths` | `N` | Folder paths inside source bucket |
| `--image-size` | `576 576` | Expected image dimensions (W H) |
| `--model` | `1:rfdetrMedium:...` | Detection models as `id:type:weight_path` |
| `--label-studio-project-id` | `22` | Label Studio project ID |
| `--pretrained-weights` | `weights/weight_rfdetr_m_slice_dinov3_v3.pth` | Pretrained weights for training |
| `--n-trials` | `10` | Number of Optuna hyperparameter trials |
| `--new-data-ratio` | `0.4` | Ratio of new data in merged set |
| `--old-data-ratio` | `0.6` | Ratio of old data in merged set |
| `--split-ratio` | `0.7 0.2 0.1` | Train/val/test split ratios |
| `--log-dir` | `./logs` | Log output directory |

Run `python run.py --help` for the full list of options.
