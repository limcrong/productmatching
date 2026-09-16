import zipfile
import io
import pandas as pd
from PIL import Image
from tqdm import tqdm
from pathlib import Path

import sys
# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import (
    ZIP_PATH,
    RAW_DATA_DIR,
    RAW_TRAIN_CSV,
    PROCESSED_DATA_DIR,
    THUMBNAILS_DIR,
    THUMBNAIL_SIZE,
)
from src.data.splitter import create_splits


def extract_csvs_and_create_splits():
    """
    Extracts train.csv and test.csv from zip and creates train/val/test_demo splits.
    """
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"Opening {ZIP_PATH}...")
    with zipfile.ZipFile(ZIP_PATH, 'r') as z:
        for name in ["train.csv", "test.csv"]:
            if name in z.namelist():
                print(f"Extracting {name} to {RAW_DATA_DIR}...")
                z.extract(name, RAW_DATA_DIR)
                
    if not RAW_TRAIN_CSV.exists():
        raise FileNotFoundError(f"train.csv could not be found in {ZIP_PATH}")
        
    print("Loading train.csv...")
    train_df = pd.read_csv(RAW_TRAIN_CSV)
    print(f"Total listings in train.csv: {len(train_df)}")
    
    # Create disjoint splits by label_group
    train_split, val_split, test_demo_split = create_splits(train_df)
    return test_demo_split


def generate_test_demo_thumbnails(test_demo_df: pd.DataFrame):
    """
    Reads only test/demo images directly from the zip file and saves 224x224 thumbnails.
    Avoids extracting the full 2GB dataset while guaranteeing complete image coverage for demo.
    """
    THUMBNAILS_DIR.mkdir(parents=True, exist_ok=True)
    
    target_images = set(test_demo_df["image"].unique())
    print(f"Generating thumbnails for {len(target_images)} test/demo listings...")
    
    with zipfile.ZipFile(ZIP_PATH, 'r') as z:
        # Pre-filter files in zip
        zip_names = {Path(p).name: p for p in z.namelist() if p.startswith("train_images/")}
        
        count = 0
        for img_name in tqdm(target_images, desc="Generating thumbnails"):
            dest_path = THUMBNAILS_DIR / img_name
            if dest_path.exists():
                count += 1
                continue
                
            zip_internal_path = zip_names.get(img_name)
            if zip_internal_path:
                try:
                    img_data = z.read(zip_internal_path)
                    img = Image.open(io.BytesIO(img_data)).convert("RGB")
                    img.thumbnail(THUMBNAIL_SIZE, Image.Resampling.LANCZOS)
                    img.save(dest_path, "JPEG", quality=85)
                    count += 1
                except Exception as e:
                    print(f"Warning: Failed to process {img_name}: {e}")
            else:
                print(f"Warning: {img_name} not found in zip archive")
                
    print(f"Successfully generated/verified {count} thumbnails in {THUMBNAILS_DIR}")


def main():
    if not ZIP_PATH.exists():
        print(f"Error: {ZIP_PATH} does not exist!")
        return
        
    test_demo_df = extract_csvs_and_create_splits()
    generate_test_demo_thumbnails(test_demo_df)
    print("Data extraction, splitting, and thumbnail generation complete!")


if __name__ == "__main__":
    main()
