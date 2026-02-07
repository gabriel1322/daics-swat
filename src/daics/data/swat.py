from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd

from daics.utils.logging import get_logger

log = get_logger(__name__)


def load_raw_swat(normal_csv: Path, attack_csv: Path) -> pd.DataFrame:
    """
    Load raw SWaT normal + attack CSV files and concatenate them.

    Paper context:
    - Normal data is used to learn nominal system behaviour
    - Attack data is only used for evaluation
    """
    log.info("Loading SWaT normal data: %s", normal_csv)
    df_normal = pd.read_csv(normal_csv)

    log.info("Loading SWaT attack data: %s", attack_csv)
    df_attack = pd.read_csv(attack_csv)

    df = pd.concat([df_normal, df_attack], ignore_index=True)
    log.info("Raw SWaT shape: %s", df.shape)
    return df


def preprocess_swat(
    df: pd.DataFrame,
    label_col: str,
    drop_timestamp: bool,
    downsample_sec: int | None,
    remove_first_hours: int | None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Clean and preprocess SWaT dataset.

    Paper-aligned choices:
    - Timestamp is not used by DAICS → dropped
    - All features are numeric
    - Downsampling reduces noise and runtime
    """
    df = df.copy()

    # --- Labels ---
    y = (df[label_col] != "Normal").astype(int)
    df = df.drop(columns=[label_col])

    # --- Timestamp ---
    if drop_timestamp:
        df = df.iloc[:, 1:]  # SWaT timestamp is first column

    # --- Force numeric ---
    df = df.apply(pd.to_numeric, errors="coerce")

    # --- Missing values ---
    df = df.ffill().bfill()

    # --- Remove warm-up period ---
    if remove_first_hours:
        samples_per_hour = int(3600 / downsample_sec) if downsample_sec else 3600
        n_drop = remove_first_hours * samples_per_hour
        df = df.iloc[n_drop:].reset_index(drop=True)
        y = y.iloc[n_drop:].reset_index(drop=True)

    # --- Downsampling ---
    if downsample_sec:
        df = df.groupby(df.index // downsample_sec).median()
        y = y.groupby(y.index // downsample_sec).max()

    log.info("Processed SWaT shape: %s", df.shape)
    log.info("Attack ratio: %.3f", y.mean())

    return df, y
