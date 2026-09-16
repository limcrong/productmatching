import pandas as pd
import numpy as np
from typing import Tuple
from src.config import (
    RAW_TRAIN_CSV,
    PROCESSED_DATA_DIR,
    TRAIN_SPLIT_CSV,
    VAL_SPLIT_CSV,
    TEST_DEMO_SPLIT_CSV,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_DEMO_RATIO,
    RANDOM_SEED,
)


def create_splits(
    df: pd.DataFrame,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_demo_ratio: float = TEST_DEMO_RATIO,
    seed: int = RANDOM_SEED,
    save: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits the Shopee dataset strictly by label_group to prevent data leakage.
    Ensures that all listings belonging to a given label_group are entirely within
    either train, validation, or test_demo split.
    
    - Train split (~70%): Model fitting, fine-tuning, embeddings
    - Validation split (~15%): Hyperparameter tuning (threshold, fusion weights)
    - Test/Demo split (~15%): Held-out unseen products for unbiased final evaluation & Streamlit demo
    """
    total = train_ratio + val_ratio + test_demo_ratio
    assert np.isclose(total, 1.0), f"Ratios must sum to 1.0, got {total}"

    # Get unique label_groups and their counts to ensure balanced group sizes
    group_counts = df.groupby("label_group").size().reset_index(name="count")
    
    # Shuffle groups deterministically
    group_counts = group_counts.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    
    # Cumulative sum to partition approximately by row count rather than just group count
    group_counts["cum_count"] = group_counts["count"].cumsum()
    total_rows = len(df)
    
    train_cutoff = total_rows * train_ratio
    val_cutoff = total_rows * (train_ratio + val_ratio)
    
    train_groups = set(group_counts[group_counts["cum_count"] <= train_cutoff]["label_group"])
    val_groups = set(
        group_counts[
            (group_counts["cum_count"] > train_cutoff) & (group_counts["cum_count"] <= val_cutoff)
        ]["label_group"]
    )
    test_demo_groups = set(group_counts[group_counts["cum_count"] > val_cutoff]["label_group"])
    
    train_df = df[df["label_group"].isin(train_groups)].copy().reset_index(drop=True)
    val_df = df[df["label_group"].isin(val_groups)].copy().reset_index(drop=True)
    test_demo_df = df[df["label_group"].isin(test_demo_groups)].copy().reset_index(drop=True)
    
    print(f"Total rows: {total_rows}")
    print(f"Train rows: {len(train_df)} ({len(train_groups)} groups)")
    print(f"Validation rows: {len(val_df)} ({len(val_groups)} groups)")
    print(f"Test/Demo rows: {len(test_demo_df)} ({len(test_demo_groups)} groups)")
    
    if save:
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        train_df.to_csv(TRAIN_SPLIT_CSV, index=False)
        val_df.to_csv(VAL_SPLIT_CSV, index=False)
        test_demo_df.to_csv(TEST_DEMO_SPLIT_CSV, index=False)
        print(f"Saved splits to {PROCESSED_DATA_DIR}")
        
    return train_df, val_df, test_demo_df


if __name__ == "__main__":
    if RAW_TRAIN_CSV.exists():
        raw_df = pd.read_csv(RAW_TRAIN_CSV)
        create_splits(raw_df)
    else:
        print(f"Raw file {RAW_TRAIN_CSV} not yet extracted. Please run extraction first.")
