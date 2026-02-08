from __future__ import annotations

"""
DAICS dataloaders (paper-aligned)

Paper notation:
  Win  : input window length (seconds / samples)
  Wout : output prediction window length
  H    : horizon gap between input end and output start
  m    : total features per timestep (m = mse + mac)
  mse  : number of sensors
  mac  : number of actuators

Project invariant:
  - processed parquet contains numeric feature columns (SWaT: 51)
  - processed parquet contains 'label' column as {0 normal, 1 attack}
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from daics.data.windowing import SlidingWindowDataset, WindowingConfig


# ---------------------------
# Configs
# ---------------------------

@dataclass(frozen=True)
class SplitConfig:
    """
    One-class split (paper spirit):
      - train/val: normal-only
      - test: remaining timeline (mixed)
    Contiguous split on NORMAL timeline to preserve temporal structure.
    """
    train_ratio: float = 0.8
    val_ratio: float = 0.2
    seed: int = 42


@dataclass(frozen=True)
class LoaderConfig:
    batch_size: int = 64
    num_workers: int = 0
    pin_memory: bool = True
    drop_last_train: bool = True


# ---------------------------
# Parquet loading utilities
# ---------------------------

def _infer_label_column(df: pd.DataFrame, label_col: Optional[str]) -> str:
    """
    Processed parquet should contain 'label' as int {0,1}.
    If someone mistakenly passes raw CSV label ("Normal/Attack"), we fall back to 'label'.
    """
    cols = list(df.columns)

    if "label" in df.columns:
        if label_col is not None and label_col != "label" and label_col not in df.columns:
            return "label"
        if label_col is None or label_col == "label":
            return "label"

    if label_col is not None:
        if label_col in df.columns:
            return label_col
        raise ValueError(
            f"label_col='{label_col}' not found in parquet columns. "
            f"Columns include: {cols[:12]}{'...' if len(cols) > 12 else ''}. "
            "For SWaT parquet produced by our preprocess, use label_col='label'."
        )

    for c in ["Label", "is_attack", "attack", "normal_attack", "Normal/Attack"]:
        if c in df.columns:
            return c

    raise ValueError("Could not infer label column; expected 'label'.")


def load_processed_parquet(
    path: str,
    label_col: Optional[str],
) -> Tuple[np.ndarray, np.ndarray, list]:
    """
    Returns:
      X: float32 (T, m)
      y: int64   (T,)  0 normal, 1 attack
      feature_cols: list[str] (order corresponds to X columns)
    """
    df = pd.read_parquet(path)
    df.columns = [str(c).strip() for c in df.columns]

    lab_col = _infer_label_column(df, label_col)

    # build binary labels
    if df[lab_col].dtype == object:
        raw = df[lab_col].astype(str).str.strip().str.lower()
        y = raw.str.contains("attack").astype(np.int64).to_numpy()
    else:
        y = pd.to_numeric(df[lab_col], errors="coerce").fillna(0).astype(int).to_numpy()
        y = (y != 0).astype(np.int64)

    drop_cols = {lab_col, "__index_level_0__"}
    feature_cols = [c for c in df.columns if c not in drop_cols]

    feat_df = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    feat_df = feat_df.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)

    X = feat_df.to_numpy(dtype=np.float32)
    return X, y, feature_cols


def fit_minmax_on_normal(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Paper (VII-A): normalise readings between 0 and 1.
    Fit MinMax scaler on normal-only points.
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
    y: np.ndarray,
    split_cfg: Any,
) -> Dict[str, np.ndarray]:
    """
    Contiguous split on NORMAL indices:
      train = first train_ratio of normal points
      val   = next val_ratio of normal points
      test  = everything after val cutoff (time order)
    """
    T = len(y)
    normal_idx = np.where(y == 0)[0]
    if normal_idx.size == 0:
        raise ValueError("No normal indices found.")

    n_total = normal_idx.size
    n_train = int(n_total * float(split_cfg.train_ratio))
    n_val = int(n_total * float(split_cfg.val_ratio))

    train_idx = normal_idx[:n_train]
    val_idx = normal_idx[n_train : n_train + n_val]

    cutoff = int(val_idx[-1] + 1) if val_idx.size > 0 else int(train_idx[-1] + 1)
    test_idx = np.arange(cutoff, T, dtype=np.int64)
    return {"train": train_idx, "val": val_idx, "test": test_idx}


# ---------------------------
# Sensor/Actuator partition (SWaT-friendly)
# ---------------------------

_SENSOR_PREFIXES = ("AIT", "FIT", "LIT", "PIT", "DPIT", "UV")


def infer_sensor_actuator_indices(feature_cols: list[str]) -> Tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """
    DAICS WDNN predicts SENSOR states only.
    We therefore need sensor_idx (columns in X corresponding to sensors).

    For SWaT, feature names allow a robust heuristic:
      Sensors: AIT*, FIT*, LIT*, PIT*, DPIT*, UV*
      Actuators: typically MV*, P* (but NOT PIT*)
    """
    sensor_cols: list[str] = []
    actuator_cols: list[str] = []

    for c in feature_cols:
        name = c.strip()
        if name.startswith(_SENSOR_PREFIXES):
            sensor_cols.append(name)
        else:
            actuator_cols.append(name)

    sensor_idx = np.array([feature_cols.index(c) for c in sensor_cols], dtype=np.int64)
    actuator_idx = np.array([feature_cols.index(c) for c in actuator_cols], dtype=np.int64)

    if sensor_idx.size == 0:
        raise ValueError(
            "Could not infer any sensor columns. "
            "Check your feature column names in the parquet."
        )

    return sensor_idx, actuator_idx, sensor_cols, actuator_cols


# ---------------------------
# Main entrypoint
# ---------------------------

def make_dataloaders(
    parquet_path: str,
    window_cfg: WindowingConfig,
    split_cfg: Any,
    loader_cfg: Any,
    label_col: Optional[str] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, object]]:
    """
    Build dataloaders for WDNN-style forecasting.

    SlidingWindowDataset signature (your implementation):
      (X, y_point, sensor_idx, cfg)
    where it internally constructs:
      - x window: (Win, m)
      - y window: (Wout, mse)  (sensors only)
      - lab (window label aggregated): scalar {0,1} using label_agg across Wout
    """
    X, y, feature_cols = load_processed_parquet(parquet_path, label_col=label_col)
    Xn, scaler = fit_minmax_on_normal(X, y)

    sensor_idx, actuator_idx, sensor_cols, actuator_cols = infer_sensor_actuator_indices(feature_cols)

    splits = contiguous_normal_splits(y, split_cfg)

    def _slice_by_time(idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if idx.size == 0:
            raise ValueError("Empty split indices.")
        start = int(idx[0])
        end = int(idx[-1] + 1)
        return Xn[start:end], y[start:end]

    X_tr, y_tr = _slice_by_time(splits["train"])
    X_va, y_va = _slice_by_time(splits["val"])
    X_te, y_te = Xn[splits["test"]], y[splits["test"]]

    train_ds = SlidingWindowDataset(X_tr, y_tr, sensor_idx, window_cfg)
    val_ds = SlidingWindowDataset(X_va, y_va, sensor_idx, window_cfg)
    test_ds = SlidingWindowDataset(X_te, y_te, sensor_idx, window_cfg)

    train_loader = DataLoader(
        train_ds,
        batch_size=int(loader_cfg.batch_size),
        shuffle=True,
        num_workers=int(loader_cfg.num_workers),
        pin_memory=bool(loader_cfg.pin_memory),
        drop_last=bool(getattr(loader_cfg, "drop_last_train", True)),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(loader_cfg.batch_size),
        shuffle=False,
        num_workers=int(loader_cfg.num_workers),
        pin_memory=bool(loader_cfg.pin_memory),
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=int(loader_cfg.batch_size),
        shuffle=False,
        num_workers=int(loader_cfg.num_workers),
        pin_memory=bool(loader_cfg.pin_memory),
        drop_last=False,
    )

    artifacts: Dict[str, object] = {
        # Paper notations / shapes
        "m": int(Xn.shape[1]),
        "mse": int(sensor_idx.size),
        "mac": int(actuator_idx.size),
        "sensor_idx": sensor_idx.copy(),
        "actuator_idx": actuator_idx.copy(),
        "sensor_cols": list(sensor_cols),
        "actuator_cols": list(actuator_cols),

        # Other useful stuff
        "feature_cols": list(feature_cols),
        "scaler": scaler,
        "splits": {k: v.copy() for k, v in splits.items()},
        "window_cfg": window_cfg,
        "parquet_path": parquet_path,
    }
    return train_loader, val_loader, test_loader, artifacts
