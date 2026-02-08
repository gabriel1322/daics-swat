from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from daics.data.swat import load_swat_csv


@dataclass(frozen=True)
class DatasetPaths:
    normal_csv: str
    attack_csv: str


def load_dataset_csv(name: str, normal_csv: str, attack_csv: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    name = name.strip().lower()
    if name == "swat":
        return load_swat_csv(normal_csv), load_swat_csv(attack_csv)
    raise ValueError(f"Unknown dataset name: {name}")
