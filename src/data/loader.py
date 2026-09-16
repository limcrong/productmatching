import os
import zipfile
from pathlib import Path
import pandas as pd
from typing import Dict, List, Optional
from src.config import (
    RAW_TRAIN_CSV,
    RAW_TRAIN_IMAGES_DIR,
    THUMBNAILS_DIR,
    DEMO_THUMBNAILS_ZIP,
    TRAIN_SPLIT_CSV,
    VAL_SPLIT_CSV,
    TEST_DEMO_SPLIT_CSV,
)


def load_data(split_name: Optional[str] = None) -> pd.DataFrame:
    """
    Loads dataset DataFrame.
    If split_name is None, loads raw train.csv.
    Otherwise loads 'train', 'val', or 'test_demo' processed split.
    """
    if split_name == "train":
        return pd.read_csv(TRAIN_SPLIT_CSV)
    elif split_name in ("val", "validation"):
        return pd.read_csv(VAL_SPLIT_CSV)
    elif split_name in ("test", "demo", "test_demo"):
        return pd.read_csv(TEST_DEMO_SPLIT_CSV)
    else:
        if not RAW_TRAIN_CSV.exists():
            raise FileNotFoundError(f"Raw train CSV not found at {RAW_TRAIN_CSV}")
        return pd.read_csv(RAW_TRAIN_CSV)


def get_ground_truth_map(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Returns mapping from posting_id -> list of all matching posting_ids (including itself)
    sharing the same label_group.
    """
    group_to_postings = df.groupby("label_group")["posting_id"].apply(list).to_dict()
    return df["label_group"].map(group_to_postings).set_axis(df["posting_id"]).to_dict()


_ZIP_CACHE = None


def get_demo_zip() -> Optional[zipfile.ZipFile]:
    global _ZIP_CACHE
    if _ZIP_CACHE is None and DEMO_THUMBNAILS_ZIP.exists():
        try:
            _ZIP_CACHE = zipfile.ZipFile(DEMO_THUMBNAILS_ZIP, "r")
        except Exception:
            _ZIP_CACHE = None
    return _ZIP_CACHE


def get_image_path(image_name: str, prefer_thumbnail: bool = True) -> Optional[Path]:
    """
    Locates image path, preferring thumbnail if requested and available.
    Supports instant on-demand extraction from demo_thumbnails.zip.
    """
    if prefer_thumbnail:
        thumb_path = THUMBNAILS_DIR / image_name
        if thumb_path.exists():
            return thumb_path

        # On-demand extraction from zip: extracts only the rendered image in ~1ms
        if DEMO_THUMBNAILS_ZIP.exists():
            try:
                z = get_demo_zip()
                if z and image_name in z.namelist():
                    THUMBNAILS_DIR.mkdir(parents=True, exist_ok=True)
                    z.extract(image_name, THUMBNAILS_DIR)
                    return thumb_path
            except Exception:
                pass

    raw_path = RAW_TRAIN_IMAGES_DIR / image_name
    if raw_path.exists():
        return raw_path

    return None


def ensure_thumbnails_extracted() -> int:
    """
    Fast initialization check. Extraction is handled on-demand for instant load times.
    """
    THUMBNAILS_DIR.mkdir(parents=True, exist_ok=True)
    return 1 if DEMO_THUMBNAILS_ZIP.exists() else 0
