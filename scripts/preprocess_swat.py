from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from daics.data.swat import load_raw_swat, preprocess_swat
from daics.utils.logging import get_logger

log = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess SWaT dataset")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]
    prep_cfg = cfg["preprocess"]

    df_raw = load_raw_swat(
        Path(data_cfg["normal_csv"]),
        Path(data_cfg["attack_csv"]),
    )

    X, y = preprocess_swat(
        df_raw,
        label_col=prep_cfg["label_col"],
        drop_timestamp=prep_cfg["drop_timestamp"],
        downsample_sec=prep_cfg["downsample_sec"],
        remove_first_hours=prep_cfg["remove_first_hours"],
    )

    out_path = Path("data/processed_swat.parquet")
    out_path.parent.mkdir(exist_ok=True)

    df_out = X.copy()
    df_out["label"] = y

    df_out.to_parquet(out_path)
    log.info("Saved processed dataset to %s", out_path.resolve())


if __name__ == "__main__":
    main()
