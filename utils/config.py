import os
import json
import logging
from pathlib import Path
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

class AppConfig:
    def __init__(self, config_file: str = "configs/pipeline_config.json"):
        # Resolve the absolute path to ensure we can load it from anywhere
        base_dir = Path(__file__).resolve().parent.parent
        self.config_path = base_dir / config_file
        self.pipeline_config = {}
        self._load_config()

        # Environment variables from .env
        self.LABEL_STUDIO_API_KEY = os.getenv("LABEL_STUDIO_API_KEY")
        self.LABEL_STUDIO_URL = os.getenv("LABEL_STUDIO_URL")
        self.GCS_API_KEY = os.getenv("GCS_API_KEY")

    def _load_config(self):
        """Load configuration from JSON file."""
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    self.pipeline_config = json.load(f)
            except Exception as e:
                logger.error(f"Error loading configuration from {self.config_path}: {e}")
                raise
        else:
            logger.warning(f"Configuration file not found at {self.config_path}")

    @property
    def data_pipeline(self):
        return self.pipeline_config.get("data_pipeline", {})

    @property
    def training_pipeline(self):
        return self.pipeline_config.get("training_pipeline", {})

    @property
    def data_management(self):
        return self.pipeline_config.get("data_management", {})

    @property
    def logging(self):
        return self.pipeline_config.get("logging", {})

    @property
    def ai_trainer_configs(self):
        return self.training_pipeline.get("ai_trainer_configs", {})

    @property
    def evaluation_configs(self):
        return self.training_pipeline.get("evaluation_configs", {})

    @property
    def label_studio_configs(self):
        return self.pipeline_config.get("label_studio_configs", {})

# Global configuration instance
config = AppConfig()
