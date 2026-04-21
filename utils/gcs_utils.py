import logging
from google.cloud import storage

from utils.logger import get_logger

logger = get_logger(__name__)

    
def init_connect_gcs_bucket(self, bucket_name: str = "jetson-textile-storage"):
    """Connect to Google Cloud Storage."""
    logger.info(f"Connecting to Google Cloud Storage bucket: {bucket_name}")
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        return bucket
    except Exception as e:
        logger.error(f"Error connecting to Google Cloud Storage: {str(e)}", exc_info=True)
        raise   

def push_file_to_gcs(local_file_path: Path, gcs_destination: str) -> bool:
    bucket = init_connect_gcs_bucket(gcs_destination.split("/")[0])
    try:
        blob = bucket.blob(gcs_destination)
        blob.upload_from_filename(str(local_file_path))
        logger.info(f"Pushed {local_file_path} to {gcs_destination}")
        return True
    except Exception as e:
        logger.error(f"Error pushing file to Google Cloud Storage: {str(e)}", exc_info=True)
        return False