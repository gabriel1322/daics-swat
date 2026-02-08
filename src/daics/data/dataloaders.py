from __future__ import annotations

"""
DAICS dataloaders (paper-aligned)

Paper notation:
  Win  : input window length
  Wout : output prediction window length
  H    : horizon gap between input end and output start
  m    : total features per timestep (m = mse + mac)
  mse  : number of sensors (targets for WDNN)
  mac  : number of actuators

Paper-like split for SWaT:
  - Train/Val: from NORMAL dataset only (benign behavior)
  - Test: should contain BOTH normal and attack to compute meaningful TN/FP etc.

In this repo (paper-strict preprocessing outputs):
  - normal_parquet: contains only normal period rows (label=0)
  - attack_parquet: contains only attack period rows (label=1)

We build the test set on-the-fly:
  test = tail(normal, normal_tail_rows) + all(attack)
This yields a mixed test set while keeping preprocessing strict and reproducible.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from daics.data.windowing import SlidingWindowDataset, WindowingConfig


# ---------------------------
# Configs (simple)
# ---------------------------

@dataclass(frozen=True)
class SplitConfig:
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
# Helpers: label + feature loading
# ---------------------------

def _infer_label_column(df: pd.DataFrame, label_col: Optional[str]) -> str:
    """
    Paper invariant for our processed parquets:
      - label column is named 'label' and is integer {0,1}
    """
    df.columns = [str(c).strip() for c in df.columns]

    if label_col is None:
        label_col = "label"

    if label_col not in df.columns:
        # Fail loudly here: processed parquets MUST have 'label'
        raise ValueError(
            f"label_col='{label_col}' not found in parquet columns. "
            "Expected processed label column 'label' (0 normal, 1 attack). "
            f"First columns: {list(df.columns)[:12]}"
        )
    return label_col


def load_processed_parquet(path: str, label_col: Optional[str]) -> Tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Returns:
      X: float32 (T, m)
      y: int64   (T,) in {0,1}
      feature_cols: list[str]
    """
    df = pd.read_parquet(path)
    df.columns = [str(c).strip() for c in df.columns]

    lab_col = _infer_label_column(df, label_col)

    y = pd.to_numeric(df[lab_col], errors="coerce").fillna(0).astype(int).to_numpy()
    y = (y != 0).astype(np.int64)

    drop_cols = {lab_col, "__index_level_0__"}
    feature_cols = [c for c in df.columns if c not in drop_cols]

    feat_df = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    feat_df = feat_df.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)

    X = feat_df.to_numpy(dtype=np.float32)
    return X, y, feature_cols


def fit_minmax_on_normal(X_normal: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Fit MinMax scaler on NORMAL-only points (paper VII-A).
    Returns scaler dict: {"min": ..., "den": ...}
    """
    if len(X_normal) == 0:
        raise ValueError("Normal array is empty; cannot fit scaler.")

    mn = X_normal.min(axis=0)
    mx = X_normal.max(axis=0)
    den = np.where((mx - mn) < 1e-12, 1.0, (mx - mn))
    return {"min": mn.astype(np.float32), "den": den.astype(np.float32)}


def apply_minmax(X: np.ndarray, scaler: Dict[str, np.ndarray]) -> np.ndarray:
    mn = scaler["min"]
    den = scaler["den"]
    return np.clip((X - mn) / den, 0.0, 1.0).astype(np.float32, copy=False)


# ---------------------------
# Sensor/Actuator partition (SWaT heuristic)
# ---------------------------

_SENSOR_PREFIXES = ("AIT", "FIT", "LIT", "PIT", "DPIT", "UV")


def infer_sensor_actuator_indices(feature_cols: list[str]) -> Tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """
    WDNN predicts SENSOR states only.
    SWaT heuristic:
      Sensors: AIT*, FIT*, LIT*, PIT*, DPIT*, UV*
      Actuators: everything else (e.g., MV*, P* (NOT PIT*))
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
        raise ValueError("Could not infer any sensor columns from feature names.")

    return sensor_idx, actuator_idx, sensor_cols, actuator_cols


# ---------------------------
# Normal contiguous split for train/val
# ---------------------------

def contiguous_split_normal(Xn: np.ndarray, split_cfg: Any) -> Tuple[np.ndarray, np.ndarray]:
    """
    Train/Val split on the NORMAL timeline (contiguous).
    Returns:
      idx_train: indices in [0..len(Xn)-1]
      idx_val  : indices in [0..len(Xn)-1]
    """
    n_total = len(Xn)
    n_train = int(n_total * float(split_cfg.train_ratio))
    n_val = int(n_total * float(split_cfg.val_ratio))

    # Ensure we always have something
    n_train = max(1, n_train)
    n_val = max(1, n_val)
    if n_train + n_val > n_total:
        n_val = max(1, n_total - n_train)

    idx_train = np.arange(0, n_train, dtype=np.int64)
    idx_val = np.arange(n_train, n_train + n_val, dtype=np.int64)
    return idx_train, idx_val


# ---------------------------
# Main entrypoint (paper-strict)
# ---------------------------

def make_dataloaders_paper_strict(
    normal_parquet_path: str,
    attack_parquet_path: str,
    window_cfg: WindowingConfig,
    split_cfg: Any,
    loader_cfg: Any,
    label_col: Optional[str] = None,
    test_mode: str = "mixed_tail+attack",
    normal_tail_rows: int = 20000,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, object]]:
    """
    Paper-strict dataloaders:
      - Load normal_parquet and attack_parquet separately
      - Fit scaler on NORMAL only
      - Train/Val: NORMAL only, contiguous split
      - Test: mixed set built on-the-fly:
          tail(normal, normal_tail_rows) + all(attack)

    Dataset yields (x, y_sensors, lab):
      x:   (B, Win, m)
      y:   (B, Wout, mse)    sensors only
      lab: (B,) aggregated label over predicted window
    """
    # ---- Load both parquets
    Xn_raw, y_n, feat_n = load_processed_parquet(normal_parquet_path, label_col=label_col)
    Xa_raw, y_a, feat_a = load_processed_parquet(attack_parquet_path, label_col=label_col)

    if feat_n != feat_a:
        raise ValueError(
            "Feature columns mismatch between normal parquet and attack parquet. "
            "They must be identical and in the same order."
        )

    # Sanity: strict datasets should be pure
    # (you already observed y_n all 0 and y_a all 1; keep a tolerant check)
    if int(y_n.max()) != 0:
        raise ValueError("Normal parquet contains attack labels; expected all zeros.")
    if int(y_a.min()) != 1:
        raise ValueError("Attack parquet contains normal labels; expected all ones.")

    # ---- Fit scaler on NORMAL only, then apply to both
    scaler = fit_minmax_on_normal(Xn_raw)
    Xn = apply_minmax(Xn_raw, scaler)
    Xa = apply_minmax(Xa_raw, scaler)

    # ---- Infer sensor/actuator indices
    sensor_idx, actuator_idx, sensor_cols, actuator_cols = infer_sensor_actuator_indices(feat_n)

    # ---- Train/Val split (normal only)
    idx_tr, idx_va = contiguous_split_normal(Xn, split_cfg)

    X_tr = Xn[idx_tr]
    y_tr = y_n[idx_tr]
    X_va = Xn[idx_va]
    y_va = y_n[idx_va]

    # ---- Test set (paper-like mixed)
    mode = (test_mode or "").strip().lower()
    if mode not in ("mixed_tail+attack", "mixed", "tail+attack"):
        raise ValueError(
            f"Unsupported test_mode='{test_mode}'. "
            "For paper-like evaluation use 'mixed_tail+attack'."
        )

    n_total_normal = len(Xn)
    n_tail = min(int(normal_tail_rows), n_total_normal)
    if n_tail <= 0:
        raise ValueError("normal_tail_rows must be > 0 to build a mixed test set.")

    X_tail = Xn[-n_tail:]
    y_tail = y_n[-n_tail:]

    X_te = np.concatenate([X_tail, Xa], axis=0)
    y_te = np.concatenate([y_tail, y_a], axis=0)

    tail_pct = 100.0 * float(n_tail) / float(max(1, n_total_normal))
    test_note = (
        f"mixed_tail+attack: tail(normal)={n_tail}/{n_total_normal} ({tail_pct:.2f}%) + "
        f"attack={len(Xa)} rows"
    )

    # ---- Window datasets (your signature: (X, y_point, sensor_idx, cfg))
    train_ds = SlidingWindowDataset(X_tr, y_tr, sensor_idx, window_cfg)
    val_ds = SlidingWindowDataset(X_va, y_va, sensor_idx, window_cfg)
    test_ds = SlidingWindowDataset(X_te, y_te, sensor_idx, window_cfg)

    # ---- Loaders
    train_loader = DataLoader(
        train_ds,
        batch_size=int(getattr(loader_cfg, "batch_size", 64)),
        shuffle=True,
        num_workers=int(getattr(loader_cfg, "num_workers", 0)),
        pin_memory=bool(getattr(loader_cfg, "pin_memory", True)),
        drop_last=bool(getattr(loader_cfg, "drop_last_train", True)),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(getattr(loader_cfg, "batch_size", 64)),
        shuffle=False,
        num_workers=int(getattr(loader_cfg, "num_workers", 0)),
        pin_memory=bool(getattr(loader_cfg, "pin_memory", True)),
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=int(getattr(loader_cfg, "batch_size", 64)),
        shuffle=False,
        num_workers=int(getattr(loader_cfg, "num_workers", 0)),
        pin_memory=bool(getattr(loader_cfg, "pin_memory", True)),
        drop_last=False,
    )

    # ---- Artifacts for scripts/README
    artifacts: Dict[str, object] = {
        "m": int(Xn.shape[1]),
        "mse": int(sensor_idx.size),
        "mac": int(actuator_idx.size),
        "sensor_idx": sensor_idx.copy(),
        "actuator_idx": actuator_idx.copy(),
        "sensor_cols": list(sensor_cols),
        "actuator_cols": list(actuator_cols),
        "feature_cols": list(feat_n),
        "scaler": scaler,

        "paths": {
            "normal_parquet": str(normal_parquet_path),
            "attack_parquet": str(attack_parquet_path),
        },

        "splits": {
            "train_normal_rows": int(len(X_tr)),
            "val_normal_rows": int(len(X_va)),
        },

        "test_mode": "mixed_tail+attack",
        "normal_tail_rows": int(normal_tail_rows),
        "test_note": test_note,
        "test_counts": {
            "test_total_rows": int(len(y_te)),
            "test_normal_rows": int((y_te == 0).sum()),
            "test_attack_rows": int((y_te == 1).sum()),
            "tail_normal_rows": int(n_tail),
            "tail_normal_pct_of_normal": float(tail_pct),
        },

        "window_cfg": window_cfg,
    }

    return train_loader, val_loader, test_loader, artifacts
