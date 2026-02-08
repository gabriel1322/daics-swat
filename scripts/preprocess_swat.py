from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from daics.config import load_config
from daics.data.swat import SwatPreprocessConfig, preprocess_swat

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="configs/base.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)

    # Paper-aligned invariant for the rest of the repo:
    # processed parquet MUST contain:
    #   - feature columns (numeric)
    #   - 'label' column (0 normal, 1 attack)
    out_path = Path(cfg.data.processed_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pp = SwatPreprocessConfig(
        drop_timestamp=cfg.preprocess.drop_timestamp,
        label_col_raw=cfg.preprocess.label_col,
        downsample_sec=cfg.preprocess.downsample_sec,
        remove_first_hours=cfg.preprocess.remove_first_hours,
    )

    df, feature_cols = preprocess_swat(
        normal_csv=cfg.data.normal_csv,
        attack_csv=cfg.data.attack_csv,
        cfg=pp,
    )

    # Save
    df.to_parquet(out_path, index=False)
    logger.info("Saved processed dataset to %s", out_path.resolve())
    logger.info("Features saved: %d", len(feature_cols))
    logger.info("Columns: %s", ", ".join(list(df.columns)))


if __name__ == "__main__":
    main()
