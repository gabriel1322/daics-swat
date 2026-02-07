from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from daics.data.datasets import WindowDataset
from daics.data.windowing import WindowingConfig


@dataclass(frozen=True)
class SplitConfig:
    """
    We keep splitting logic explicit and reproducible.

    Typical DAICS / ICS evaluation setup:
    - Train/Val: normal-only (unsupervised training on normal behavior)
    - Test: mixed normal+attack (for evaluation)

    We split using contiguous time chunks to preserve temporal structure.
    """
    train_ratio: float = 0.8
    val_ratio: float = 0.1  # remainder is "holdout normal" typically used in test mixing
    seed: int = 42


@dataclass(frozen=True)
class LoaderConfig:
    batch_size: int = 64
    num_workers: int = 0
    pin_memory: bool = True
    drop_last_train: bool = True


def _infer_label_column(df: pd.DataFrame, label_col: Optional[str]) -> str:
    if label_col is not None:
        if label_col not in df.columns:
            raise ValueError(f"label_col='{label_col}' not found in parquet columns.")
        return label_col

    # common candidates
    candidates = ["label", "Label", "is_attack", "attack", "Normal/Attack", "normal_attack"]
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(
        "Could not infer label column. Pass data.label_col in config "
        "or ensure one of these exists: "
        + ", ".join(candidates)
    )


def load_processed_parquet(
    path: str,
    label_col: Optional[str],
) -> Tuple[np.ndarray, np.ndarray, list]:
    df = pd.read_parquet(path)
    df.columns = [c.strip() for c in df.columns]

    lab_col = _infer_label_column(df, label_col)

    # build binary labels
    if df[lab_col].dtype == object:
        raw = df[lab_col].astype(str).str.strip().str.lower()
        y = raw.str.contains("attack").astype(np.int64).to_numpy()
    else:
        y = pd.to_numeric(df[lab_col], errors="coerce").fillna(0).astype(int).to_numpy()
        y = (y != 0).astype(np.int64)

    # features = all numeric columns except label
    feature_cols = [c for c in df.columns if c != lab_col]
    feat_df = df[feature_cols].apply(pd.to_numeric, errors="coerce")

    feat_df = feat_df.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)
    X = feat_df.to_numpy(dtype=np.float32)

    return X, y, feature_cols


def fit_minmax_on_normal(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Paper-style scaling: fit MinMax using normal-only rows (unsupervised training).
    """
    normal_rows = X[y == 0]
    if len(normal_rows) == 0:
        raise ValueError("No normal rows found to fit MinMax scaler.")

    mn = normal_rows.min(axis=0)
    mx = normal_rows.max(axis=0)
    den = np.where((mx - mn) < 1e-12, 1.0, (mx - mn))

    Xn = np.clip((X - mn) / den, 0.0, 1.0).astype(np.float32, copy=False)
    scaler = {"min": mn.astype(np.float32), "den": den.astype(np.float32)}
    return Xn, scaler


def contiguous_normal_splits(
    X: np.ndarray,
    y: np.ndarray,
    split_cfg: SplitConfig,
) -> Dict[str, np.ndarray]:
    """
    Returns indices for:
    - train: normal-only contiguous
    - val  : normal-only contiguous
    - test : remaining (normal tail + all attacks) in original time order
    """
    T = len(y)
    normal_idx = np.where(y == 0)[0]
    if normal_idx.size == 0:
        raise ValueError("No normal indices found.")

    # contiguous by time: use first portion of NORMAL timeline
    n_total = normal_idx.size
    n_train = int(n_total * split_cfg.train_ratio)
    n_val = int(n_total * split_cfg.val_ratio)

    train_idx = normal_idx[:n_train]
    val_idx = normal_idx[n_train : n_train + n_val]

    # everything after val boundary (time-wise) is candidate test region.
    # we take the remaining time indices in order to keep temporal integrity.
    cutoff = int(val_idx[-1] + 1) if val_idx.size > 0 else int(train_idx[-1] + 1)
    test_idx = np.arange(cutoff, T, dtype=np.int64)

    return {"train": train_idx, "val": val_idx, "test": test_idx}


def make_dataloaders(
    parquet_path: str,
    window_cfg: WindowingConfig,
    split_cfg: SplitConfig,
    loader_cfg: LoaderConfig,
    label_col: Optional[str] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, object]]:
    """
    One entrypoint used by training scripts.

    Returns train/val/test loaders and a small 'artifacts' dict
    (feature names, scaler params, etc.) to reuse later.
    """
    X, y, feature_cols = load_processed_parquet(parquet_path, label_col=label_col)
    Xn, scaler = fit_minmax_on_normal(X, y)

    splits = contiguous_normal_splits(Xn, y, split_cfg)

    # Build sequences:
    # For train/val, we use normal-only indices but need contiguous arrays.
    # We'll extract the time region covering those indices, then filter labels as needed.
    # (Kept simple for now; later, we can reproduce exact paper split logic if needed.)

    def _slice_by_time(idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if idx.size == 0:
            raise ValueError("Empty split indices.")
        start = int(idx[0])
        end = int(idx[-1] + 1)
        return Xn[start:end], y[start:end]

    X_tr, y_tr = _slice_by_time(splits["train"])
    X_va, y_va = _slice_by_time(splits["val"])
    X_te, y_te = Xn[splits["test"]], y[splits["test"]]

    # WindowDataset delegates the math to windowing.py (paper notation W/H/S)
    train_ds = WindowDataset(X_tr, y_tr, window_cfg)
    val_ds = WindowDataset(X_va, y_va, window_cfg)
    test_ds = WindowDataset(X_te, y_te, window_cfg)


    train_loader = DataLoader(
        train_ds,
        batch_size=loader_cfg.batch_size,
        shuffle=True,
        num_workers=loader_cfg.num_workers,
        pin_memory=loader_cfg.pin_memory,
        drop_last=loader_cfg.drop_last_train,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=loader_cfg.batch_size,
        shuffle=False,
        num_workers=loader_cfg.num_workers,
        pin_memory=loader_cfg.pin_memory,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=loader_cfg.batch_size,
        shuffle=False,
        num_workers=loader_cfg.num_workers,
        pin_memory=loader_cfg.pin_memory,
        drop_last=False,
    )

    artifacts: Dict[str, object] = {
        "feature_cols": feature_cols,
        "scaler": scaler,
        "splits": {k: v.copy() for k, v in splits.items()},
        "window_cfg": window_cfg,
        "split_cfg": split_cfg,
        "loader_cfg": loader_cfg,
        "parquet_path": parquet_path,
    }
    return train_loader, val_loader, test_loader, artifacts
