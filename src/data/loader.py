import os
from pathlib import Path
import pandas as pd
from typing import Dict, List, Optional
from src.config import (
    RAW_TRAIN_CSV,
    RAW_TRAIN_IMAGES_DIR,
    THUMBNAILS_DIR,
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


def get_image_path(image_name: str, prefer_thumbnail: bool = True) -> Optional[Path]:
    """
    Locates image path, preferring thumbnail if requested and available.
    """
    if prefer_thumbnail:
        thumb_path = THUMBNAILS_DIR / image_name
        if thumb_path.exists():
            return thumb_path
            
    raw_path = RAW_TRAIN_IMAGES_DIR / image_name
    if raw_path.exists():
        return raw_path
        
    return None
