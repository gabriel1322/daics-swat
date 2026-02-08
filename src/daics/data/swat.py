from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SwatPreprocessConfig:
    """
    Preprocess config for SWaT.

    Raw SWaT CSV columns typically include:
      - Timestamp (string)
      - 51 features (25 sensors + 26 actuators)
      - label column named "Normal/Attack" with values like "Normal" / "Attack"
    """
    drop_timestamp: bool = True
    label_col_raw: str = "Normal/Attack"  # raw csv label column
    downsample_sec: int = 10
    remove_first_hours: int = 6


def load_swat_csv(path: str) -> pd.DataFrame:
    """
    Load SWaT CSV and normalize column names (strip spaces).
    """
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df


def _binary_label_from_raw(series: pd.Series) -> np.ndarray:
    """
    Convert SWaT raw label column to binary:
      0 = normal
      1 = attack
    Raw values are typically strings containing "Normal" or "Attack".
    """
    if series.dtype == object:
        s = series.astype(str).str.strip().str.lower()
        y = s.str.contains("attack").astype(np.int64).to_numpy()
        return y

    # if already numeric-ish
    y = pd.to_numeric(series, errors="coerce").fillna(0).astype(int).to_numpy()
    return (y != 0).astype(np.int64)


def _downsample_by_stride(df: pd.DataFrame, stride: int) -> pd.DataFrame:
    """
    SWaT sampled at 1 Hz in the original dataset.
    Downsample by taking every `stride` rows (paper: e.g., 10 sec).
    """
    if stride <= 1:
        return df
    return df.iloc[::stride].reset_index(drop=True)


def _remove_first_hours(df: pd.DataFrame, hours: int, sample_rate_hz: int = 1) -> pd.DataFrame:
    """
    Remove first N hours of data (paper uses this to drop transient startup).
    If data is 1Hz, rows_to_drop = hours * 3600.
    If you downsample later, remove first hours *before* downsampling to match time.
    """
    if hours <= 0:
        return df
    rows = int(hours * 3600 * sample_rate_hz)
    if rows >= len(df):
        logger.warning("Requested to remove %d rows but df has only %d rows; returning empty.", rows, len(df))
        return df.iloc[0:0].copy()
    return df.iloc[rows:].reset_index(drop=True)


def preprocess_swat(
    normal_csv: str,
    attack_csv: str,
    cfg: SwatPreprocessConfig,
) -> Tuple[pd.DataFrame, list[str]]:
    """
    Build a single processed dataframe with:
      - numeric feature columns only (float32)
      - a binary label column named 'label' (int64)
    Output is intended to be saved as parquet and consumed by dataloaders.

    Notes:
    - We concatenate (normal part + attack part) to obtain a continuous timeline-like
      dataset (common in many repo implementations). This differs from the paper's
      official split semantics (train normal-only, test mixed). We re-create the paper
      split later in dataloaders by selecting normal-only for train/val and mixed for test.
    """
    logger.info("Loading SWaT normal data: %s", normal_csv)
    df_n = load_swat_csv(normal_csv)
    logger.info("Loading SWaT attack data: %s", attack_csv)
    df_a = load_swat_csv(attack_csv)

    # --- sanity: ensure label col exists
    if cfg.label_col_raw not in df_n.columns:
        raise ValueError(f"Raw label column '{cfg.label_col_raw}' not found in normal CSV columns.")
    if cfg.label_col_raw not in df_a.columns:
        raise ValueError(f"Raw label column '{cfg.label_col_raw}' not found in attack CSV columns.")

    # --- remove first hours (before downsampling, time-consistent)
    df_n = _remove_first_hours(df_n, cfg.remove_first_hours, sample_rate_hz=1)
    df_a = _remove_first_hours(df_a, cfg.remove_first_hours, sample_rate_hz=1)

    # --- downsample
    df_n = _downsample_by_stride(df_n, cfg.downsample_sec)
    df_a = _downsample_by_stride(df_a, cfg.downsample_sec)

    # --- build binary labels
    y_n = _binary_label_from_raw(df_n[cfg.label_col_raw])
    y_a = _binary_label_from_raw(df_a[cfg.label_col_raw])

    # --- drop timestamp and raw label, keep only features
    def _drop_cols(df: pd.DataFrame) -> pd.DataFrame:
        cols_to_drop = [cfg.label_col_raw]
        if cfg.drop_timestamp and "Timestamp" in df.columns:
            cols_to_drop.append("Timestamp")
        return df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    Xn = _drop_cols(df_n)
    Xa = _drop_cols(df_a)

    # align columns
    if list(Xn.columns) != list(Xa.columns):
        # try intersection in same order as normal
        common = [c for c in Xn.columns if c in Xa.columns]
        missing_in_attack = [c for c in Xn.columns if c not in Xa.columns]
        missing_in_normal = [c for c in Xa.columns if c not in Xn.columns]
        if missing_in_attack or missing_in_normal:
            raise ValueError(
                "Normal/Attack feature columns mismatch.\n"
                f"Missing in attack: {missing_in_attack}\n"
                f"Missing in normal: {missing_in_normal}"
            )
        Xn = Xn[common]
        Xa = Xa[common]

    feature_cols = list(Xn.columns)

    # numeric cleanup
    def _to_numeric(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
        out = out.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)
        return out

    Xn = _to_numeric(Xn).astype(np.float32)
    Xa = _to_numeric(Xa).astype(np.float32)

    # merge
    merged = pd.concat(
        [
            Xn.assign(label=y_n.astype(np.int64)),
            Xa.assign(label=y_a.astype(np.int64)),
        ],
        axis=0,
        ignore_index=True,
    )

    attack_ratio = float(merged["label"].mean()) if len(merged) else 0.0
    logger.info("Raw SWaT merged shape: %s", tuple(merged.shape))
    logger.info("Attack ratio: %.3f", attack_ratio)

    return merged, feature_cols
