from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


# ------------------------------------------------------------
# YAML-backed configuration (paper-aligned)
# ------------------------------------------------------------

@dataclass(frozen=True)
class DataConfig:
    # Raw SWaT CSVs
    normal_csv: str = "data/normal.csv"
    merged_csv: str = "data/merged.csv"

    # Processed outputs
    processed_normal_path: str = "data/processed_swat_normal.parquet"
    processed_merged_path: str = "data/processed_swat_merged.parquet"

    # Column name inside processed parquet (0 normal, 1 attack)
    label_col: str = "label"


@dataclass(frozen=True)
class PreprocessConfig:
    drop_timestamp: bool = True

    # Label column to read from raw CSV (SWaT provides Normal/Attack)
    label_col: str = "Normal/Attack"

    downsample_sec: int = 10
    remove_first_hours: int = 6


@dataclass(frozen=True)
class WindowingYamlConfig:
    # Paper notation:
    # Win  = input time window length
    # Wout = output prediction window length
    # H    = horizon gap
    Win: int = 60
    Wout: int = 4
    H: int = 50

    # stride
    S: int = 1

    # label aggregation inside the predicted window Wout
    label_agg: str = "any"


@dataclass(frozen=True)
class DetectionYamlConfig:
    Wanom: int = 30
    Wgrace: int = 0


@dataclass(frozen=True)
class SplitsYamlConfig:
    # Contiguous split on NORMAL timeline (train/val normal-only)
    train_ratio: float = 0.8
    val_ratio: float = 0.2
    seed: int = 42


@dataclass(frozen=True)
class LoaderYamlConfig:
    batch_size: int = 64
    num_workers: int = 0
    pin_memory: bool = True
    drop_last_train: bool = True


# -------------------------
# Paper models configs
# -------------------------

@dataclass(frozen=True)
class WDNNYamlConfig:
    lr: float = 1e-3
    batch_size: int = 32
    epochs: int = 100

    out_dir: str = "runs/wdnn"

    dl1: Optional[int] = None
    dl2: Optional[int] = None
    dl4: int = 80

    cl1_channels: int = 64
    cl1_kernel: int = 2
    cl2_channels: int = 128
    cl2_kernel: int = 2

    leaky_slope: float = 0.01

    weight_decay: float = 0.0
    grad_clip: float = 1.0


@dataclass(frozen=True)
class TTNNYamlConfig:
    lr: float = 1e-2
    batch_size: int = 32
    epochs: int = 1
    median_kernel: int = 59
    out_dir: str = "runs/ttnn"


@dataclass(frozen=True)
class TrainRuntimeConfig:
    seed: int = 42
    device: str = "auto"  # "auto" | "cpu" | "cuda"

    wdnn: WDNNYamlConfig = WDNNYamlConfig()
    ttnn: TTNNYamlConfig = TTNNYamlConfig()


# -------------------------
# Evaluation config (paper-like mixed test)
# -------------------------

@dataclass(frozen=True)
class EvalYamlConfig:
    test_mode: str = "merged_parquet"


@dataclass(frozen=True)
class Config:
    data: DataConfig = DataConfig()
    preprocess: PreprocessConfig = PreprocessConfig()
    windowing: WindowingYamlConfig = WindowingYamlConfig()
    detection: DetectionYamlConfig = DetectionYamlConfig()
    splits: SplitsYamlConfig = SplitsYamlConfig()
    loader: LoaderYamlConfig = LoaderYamlConfig()
    train: TrainRuntimeConfig = TrainRuntimeConfig()
    eval: EvalYamlConfig = EvalYamlConfig()


# ------------------------------------------------------------
# YAML merge / load
# ------------------------------------------------------------

def _deep_update(base: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = v
    return out


def _as_dict(cfg: Config) -> Dict[str, Any]:
    return {
        "data": cfg.data.__dict__,
        "preprocess": cfg.preprocess.__dict__,
        "windowing": cfg.windowing.__dict__,
        "detection": cfg.detection.__dict__,
        "splits": cfg.splits.__dict__,
        "loader": cfg.loader.__dict__,
        "train": {
            "seed": cfg.train.seed,
            "device": cfg.train.device,
            "wdnn": cfg.train.wdnn.__dict__,
            "ttnn": cfg.train.ttnn.__dict__,
        },
        "eval": cfg.eval.__dict__,
    }


def _from_dict(d: Dict[str, Any]) -> Config:
    train_d = d.get("train", {}) or {}
    wdnn_d = train_d.get("wdnn", {}) or {}
    ttnn_d = train_d.get("ttnn", {}) or {}

    eval_d = d.get("eval", {}) or {}

    train_cfg = TrainRuntimeConfig(
        seed=train_d.get("seed", 42),
        device=train_d.get("device", "auto"),
        wdnn=WDNNYamlConfig(**wdnn_d),
        ttnn=TTNNYamlConfig(**ttnn_d),
    )

    return Config(
        data=DataConfig(**(d.get("data", {}) or {})),
        preprocess=PreprocessConfig(**(d.get("preprocess", {}) or {})),
        windowing=WindowingYamlConfig(**(d.get("windowing", {}) or {})),
        detection=DetectionYamlConfig(**(d.get("detection", {}) or {})),
        splits=SplitsYamlConfig(**(d.get("splits", {}) or {})),
        loader=LoaderYamlConfig(**(d.get("loader", {}) or {})),
        train=train_cfg,
        eval=EvalYamlConfig(**eval_d),
    )


def load_config(path: str) -> Config:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    user = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    merged = _deep_update(_as_dict(Config()), user)
    return _from_dict(merged)
