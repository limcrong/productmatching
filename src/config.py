import os
from pathlib import Path

# Root directory of the repository
ROOT_DIR = Path(__file__).resolve().parent.parent

# Data directories
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
THUMBNAILS_DIR = DATA_DIR / "thumbnails"

ZIP_PATH = DATA_DIR / "shopee-product-matching.zip"
DEMO_THUMBNAILS_ZIP = DATA_DIR / "demo_thumbnails.zip"

# Raw file paths
RAW_TRAIN_CSV = RAW_DATA_DIR / "train.csv"
RAW_TEST_CSV = RAW_DATA_DIR / "test.csv"
RAW_TRAIN_IMAGES_DIR = RAW_DATA_DIR / "train_images"

# Processed split CSV paths
TRAIN_SPLIT_CSV = PROCESSED_DATA_DIR / "train_split.csv"
VAL_SPLIT_CSV = PROCESSED_DATA_DIR / "val_split.csv"
TEST_DEMO_SPLIT_CSV = PROCESSED_DATA_DIR / "test_demo_split.csv"

# Experiments directory
EXPERIMENTS_DIR = ROOT_DIR / "experiments"
EXPERIMENTS_RESULTS_JSON = EXPERIMENTS_DIR / "results.json"

# Models and configurations
DEFAULT_TEXT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_IMAGE_MODEL = "efficientnet_b0"

# Target image thumbnail dimensions
THUMBNAIL_SIZE = (224, 224)

# Data split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_DEMO_RATIO = 0.15
RANDOM_SEED = 42
