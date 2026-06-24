import os
import logging
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

load_dotenv()


class AppConfig:
    def __init__(self, pipeline_config: dict = None):
        self.pipeline_config = pipeline_config or {}

        self.LABEL_STUDIO_API_KEY = os.getenv("LABEL_STUDIO_API_KEY")
        self.LABEL_STUDIO_URL = os.getenv("LABEL_STUDIO_URL")

    def configure(self, pipeline_config: dict):
        self.pipeline_config = pipeline_config

    @property
    def date(self):
        return self.pipeline_config.get("date", {})

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


config = AppConfig()
