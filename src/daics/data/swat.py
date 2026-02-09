from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SwatPreprocessConfig:
    """
    SWaT preprocessing.

    Raw SWaT CSV columns typically:
      - Timestamp
      - 51 features (sensors+actuators)
      - label column named "Normal/Attack" with values "Normal"/"Attack"

    Paper notes:
      - datasets are sampled at 1Hz in the original release
      - paper normalises later (we do it in dataloaders with MinMax on normal-only)
    """
    drop_timestamp: bool = True
    label_col_raw: str = "Normal/Attack"
    downsample_sec: int = 10
    remove_first_hours: int = 6


def load_swat_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _binary_label_from_raw(series: pd.Series) -> np.ndarray:
    if series.dtype == object:
        s = series.astype(str).str.strip().str.lower()
        return s.str.contains("attack").astype(np.int64).to_numpy()
    y = pd.to_numeric(series, errors="coerce").fillna(0).astype(int).to_numpy()
    return (y != 0).astype(np.int64)


def _remove_first_hours(df: pd.DataFrame, hours: int, sample_rate_hz: int = 1) -> pd.DataFrame:
    if hours <= 0:
        return df
    rows = int(hours * 3600 * sample_rate_hz)
    if rows >= len(df):
        logger.warning("Requested to remove %d rows but df has only %d rows; returning empty.", rows, len(df))
        return df.iloc[0:0].copy()
    return df.iloc[rows:].reset_index(drop=True)


def _downsample_by_stride(df: pd.DataFrame, stride: int) -> pd.DataFrame:
    if stride <= 1:
        return df
    return df.iloc[::stride].reset_index(drop=True)


def preprocess_swat_single_csv(
    csv_path: str,
    cfg: SwatPreprocessConfig,
) -> Tuple[pd.DataFrame, list[str]]:
    """
    Preprocess ONE SWaT CSV into a processed dataframe with:
      - numeric feature columns (float32)
      - 'label' column (int64, 0 normal / 1 attack)

    This function is used for BOTH:
      - Normal CSV  -> should become all label=0 (benign)
      - Merged CSV  -> mixed (normal + attack)
    """
    df = load_swat_csv(csv_path)

    if cfg.label_col_raw not in df.columns:
        raise ValueError(f"Raw label column '{cfg.label_col_raw}' not found in columns of: {csv_path}")

    # Remove initial transient (do it before downsampling to match actual time)
    df = _remove_first_hours(df, cfg.remove_first_hours, sample_rate_hz=1)

    # Downsample (practical; if you want strict 1Hz, set downsample_sec=1)
    df = _downsample_by_stride(df, cfg.downsample_sec)

    # Labels
    y = _binary_label_from_raw(df[cfg.label_col_raw])

    # Drop label + timestamp (keep only features)
    cols_to_drop = [cfg.label_col_raw]
    if cfg.drop_timestamp and "Timestamp" in df.columns:
        cols_to_drop.append("Timestamp")

    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Ensure numeric
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)
    X = X.astype(np.float32)

    feature_cols = list(X.columns)

    out = X.assign(label=y.astype(np.int64))
    logger.info("Preprocessed %s -> shape=%s attack_ratio=%.3f", csv_path, tuple(out.shape), float(out["label"].mean()))
    return out, feature_cols
