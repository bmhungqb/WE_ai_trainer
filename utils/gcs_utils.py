from pathlib import Path
from google.cloud import storage
from google.cloud.storage import transfer_manager
import os
from dotenv import load_dotenv
from utils.logger import get_logger

load_dotenv()

logger = get_logger(__name__)

def init_connect_gcs_bucket(bucket_name: str = "jetson-textile-storage"):
    """Connect to Google Cloud Storage."""
    logger.info(f"Connecting to Google Cloud Storage bucket: {bucket_name}")
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        return bucket
    except Exception as e:
        logger.error(f"Error connecting to Google Cloud Storage: {str(e)}", exc_info=True)
        raise   

def push_file_to_gcs(local_file_path: Path, gcs_name: str, gcs_destination: str) -> tuple:
    """Push a single file to Google Cloud Storage."""
    bucket = init_connect_gcs_bucket(gcs_name)
    try:
        gcs_destination_path = f"{gcs_destination}/{local_file_path.name}"
        blob = bucket.blob(gcs_destination_path)
        blob.upload_from_filename(str(local_file_path))
        logger.info(f"Pushed {local_file_path} to {gcs_destination}")
        return True, "success"
    except Exception as e:
        logger.error(f"Error pushing file to Google Cloud Storage: {str(e)}", exc_info=True)
        return False, str(e)

def push_folder_to_gcs(local_folder_path: Path, gcs_destination: str) -> tuple:
    """
    Push entire folder to GCS, preserving folder structure.
    gcs_destination format: gs://bucket/path or bucket/path
    """
    try:
        # Normalize GCS path
        gcs_destination = gcs_destination.replace("gs://", "")
        bucket_name, gcs_prefix = gcs_destination.split("/", 1)

        client = storage.Client()
        bucket = client.bucket(bucket_name)

        # Collect ALL files recursively
        filenames = []
        for root, _, files in os.walk(local_folder_path):
            for f in files:
                full_path = Path(root) / f
                relative_path = full_path.relative_to(local_folder_path)
                filenames.append(str(relative_path))

        # Upload using transfer manager for efficient batch upload
        results = transfer_manager.upload_many_from_filenames(
            bucket,
            filenames=filenames,
            source_directory=str(local_folder_path),
            blob_name_prefix=f"{gcs_prefix}/{local_folder_path.name}/",
        )

        # Check results
        failed = []
        for name, result in zip(filenames, results):
            if isinstance(result, Exception):
                failed.append((name, str(result)))

        if failed:
            return False, f"Failed files: {failed[:5]}"

        return True, "success"

    except Exception as e:
        return False, str(e)