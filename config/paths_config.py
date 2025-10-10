import os 

from pathlib import Path

# Resolve project root (repo root) from this config file's location
PROJECT_ROOT = Path(__file__).resolve().parent.parent

############################### DATA INGESTION ###############################

RAW_DIR = "data/raw"
RAW_FILE_PATH = os.path.join(PROJECT_ROOT, RAW_DIR, "bike_sharing_modified.csv")


if __name__ == "__main__":
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Raw data directory: {RAW_DIR}")
    print(f"Raw data file path: {RAW_FILE_PATH}")