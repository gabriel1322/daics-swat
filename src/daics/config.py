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
    normal_csv: str = "data/SWaT_Dataset_Normal_v1.csv"
    attack_csv: str = "data/SWaT_Dataset_Attack_v0.csv"

    # Output of preprocessing
    processed_path: str = "data/processed_swat.parquet"

    # Column name inside raw CSVs (SWaT uses literally "Normal/Attack")
    raw_label_col: Optional[str] = None

    # Column name inside processed parquet (we store an integer "label": 0 normal, 1 attack)
    label_col: str = "label"


@dataclass(frozen=True)
class PreprocessConfig:
    drop_timestamp: bool = True

    # Label column to read from raw CSV (paper dataset provides Normal/Attack)
    label_col: str = "Normal/Attack"

    downsample_sec: int = 10
    remove_first_hours: int = 6


@dataclass(frozen=True)
class WindowingYamlConfig:
    # Paper notation:
    # Win  = input time window length
    # Wout = output time window length
    # H    = horizon gap
    Win: int = 60
    Wout: int = 4
    H: int = 50

    # stride (not central in paper, keep for implementation)
    S: int = 1

    # label aggregation inside the predicted window Wout
    # "any" => label=1 if any point in the Wout window is anomalous
    label_agg: str = "any"


@dataclass(frozen=True)
class DetectionYamlConfig:
    # Paper hyperparameter
    Wanom: int = 30

    # Optional grace time (paper mentions it later)
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
    # Paper WDNN training hyperparams (Table 4)
    lr: float = 1e-3
    batch_size: int = 32
    epochs: int = 100

    out_dir: str = "runs/wdnn"

    # Optional explicit sizes (paper Table 4)
    dl1: Optional[int] = None
    dl2: Optional[int] = None
    dl4: int = 80

    # Conv params (SWaT Table 4 uses kernel size 2)
    cl1_channels: int = 64
    cl1_kernel: int = 2
    cl2_channels: int = 128
    cl2_kernel: int = 2

    leaky_slope: float = 0.01

    # trainer extras
    weight_decay: float = 0.0
    grad_clip: float = 1.0


@dataclass(frozen=True)
class TTNNYamlConfig:
    # Paper TTNN hyperparams (Table 4)
    lr: float = 1e-2
    batch_size: int = 32
    epochs: int = 1
    median_kernel: int = 59

    out_dir: str = "runs/ttnn"


@dataclass(frozen=True)
class TrainRuntimeConfig:
    seed: int = 42
    device: str = "auto"  # "auto" | "cpu" | "cuda"

    # Nested model configs (THIS matches YAML train.wdnn / train.ttnn)
    wdnn: WDNNYamlConfig = WDNNYamlConfig()
    ttnn: TTNNYamlConfig = TTNNYamlConfig()


@dataclass(frozen=True)
class Config:
    data: DataConfig = DataConfig()
    preprocess: PreprocessConfig = PreprocessConfig()
    windowing: WindowingYamlConfig = WindowingYamlConfig()
    detection: DetectionYamlConfig = DetectionYamlConfig()
    splits: SplitsYamlConfig = SplitsYamlConfig()
    loader: LoaderYamlConfig = LoaderYamlConfig()
    train: TrainRuntimeConfig = TrainRuntimeConfig()


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
    # dataclasses -> nested dict (only what we allow in YAML)
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
    }


def _from_dict(d: Dict[str, Any]) -> Config:
    train_d = d.get("train", {}) or {}
    wdnn_d = train_d.get("wdnn", {}) or {}
    ttnn_d = train_d.get("ttnn", {}) or {}

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
    )


def load_config(path: str) -> Config:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    user = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    merged = _deep_update(_as_dict(Config()), user)
    return _from_dict(merged)
