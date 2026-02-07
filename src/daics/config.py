# daics/config.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(path: str | Path) -> Dict[str, Any]:
    """
    Load a YAML config file and return it as a nested dict.

    Notes:
      - We keep this as a dict (instead of dataclasses) for flexibility while
        the project is still evolving.
      - This is used by scripts/* entrypoints.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p.resolve()}")

    with p.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise ValueError("Config root must be a mapping (YAML dict).")

    return cfg


def get(cfg: Dict[str, Any], key_path: str, default: Any = None) -> Any:
    """
    Safe getter for nested config dicts.

    Example:
      get(cfg, "preprocess.downsample_sec", 10)
    """
    cur: Any = cfg
    for part in key_path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur
