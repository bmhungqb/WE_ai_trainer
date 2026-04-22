# Agentic AI Textile Defect Detection

An automated AI pipeline for textile defect detection that handles data ingestion from Google Cloud Storage, pseudo-labeling, human-in-the-loop review via Label Studio, and Optuna-based model training.

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
3. **Configure Environment:**
   Copy `.env.example` to `.env` and fill in your API keys (Label Studio, GCS).
4. **Update Configurations:**
   Modify `configs/pipeline_config.json` as needed for your data sources and training parameters.

## Usage

You can run the pipeline stages via the main orchestrator:

```python
from src.agentic_pipeline import AgenticAIPipeline

pipeline = AgenticAIPipeline()

# Phase 1: Prepare data, run AI verification, and push to Label Studio
pipeline.run_prepare_data_pipeline()

# Phase 2: Pull reviewed data, merge with old datasets, and train model
pipeline.run_training_pipeline()
```
