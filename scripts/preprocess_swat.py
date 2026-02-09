from __future__ import annotations

import argparse
import logging
from pathlib import Path

from daics.config import load_config
from daics.data.swat import SwatPreprocessConfig, preprocess_swat_single_csv

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("preprocess_swat")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="configs/base.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)

    out_normal = Path(cfg.data.processed_normal_path)
    out_merged = Path(cfg.data.processed_merged_path)
    out_normal.parent.mkdir(parents=True, exist_ok=True)
    out_merged.parent.mkdir(parents=True, exist_ok=True)

    pp = SwatPreprocessConfig(
        drop_timestamp=cfg.preprocess.drop_timestamp,
        label_col_raw=cfg.preprocess.label_col,
        downsample_sec=cfg.preprocess.downsample_sec,
        remove_first_hours=cfg.preprocess.remove_first_hours,
    )

    logger.info("=== Paper-aligned preprocessing ===")
    logger.info("Normal CSV : %s", cfg.data.normal_csv)
    logger.info("Merged CSV : %s", cfg.data.merged_csv)
    logger.info("Output normal parquet: %s", out_normal)
    logger.info("Output merged parquet: %s", out_merged)

    df_n, feat_n = preprocess_swat_single_csv(cfg.data.normal_csv, pp)
    df_m, feat_m = preprocess_swat_single_csv(cfg.data.merged_csv, pp)

    if feat_n != feat_m:
        missing_in_merged = [c for c in feat_n if c not in feat_m]
        missing_in_normal = [c for c in feat_m if c not in feat_n]
        raise ValueError(
            "Feature columns mismatch between normal CSV and merged CSV.\n"
            f"Missing in merged: {missing_in_merged}\n"
            f"Missing in normal: {missing_in_normal}\n"
        )

    df_n.to_parquet(out_normal, index=False)
    df_m.to_parquet(out_merged, index=False)

    logger.info("Saved normal parquet to %s", out_normal.resolve())
    logger.info("Saved merged parquet to %s", out_merged.resolve())
    logger.info("Features saved: %d", len(feat_n))


if __name__ == "__main__":
    main()
