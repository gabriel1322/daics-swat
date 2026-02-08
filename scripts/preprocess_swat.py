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
    out_attack = Path(cfg.data.processed_attack_path)
    out_normal.parent.mkdir(parents=True, exist_ok=True)
    out_attack.parent.mkdir(parents=True, exist_ok=True)

    pp = SwatPreprocessConfig(
        drop_timestamp=cfg.preprocess.drop_timestamp,
        label_col_raw=cfg.preprocess.label_col,
        downsample_sec=cfg.preprocess.downsample_sec,
        remove_first_hours=cfg.preprocess.remove_first_hours,
    )

    logger.info("=== Paper-strict preprocessing ===")
    logger.info("Normal CSV : %s", cfg.data.normal_csv)
    logger.info("Attack CSV : %s", cfg.data.attack_csv)
    logger.info("Output normal parquet: %s", out_normal)
    logger.info("Output attack  parquet: %s", out_attack)

    df_n, feat_n = preprocess_swat_single_csv(cfg.data.normal_csv, pp)
    df_a, feat_a = preprocess_swat_single_csv(cfg.data.attack_csv, pp)

    # Enforce same feature set/order
    if feat_n != feat_a:
        # try to align by intersection (in normal order)
        common = [c for c in feat_n if c in feat_a]
        missing_in_attack = [c for c in feat_n if c not in feat_a]
        missing_in_normal = [c for c in feat_a if c not in feat_n]
        raise ValueError(
            "Feature columns mismatch between normal and attack CSV.\n"
            f"Missing in attack: {missing_in_attack}\n"
            f"Missing in normal: {missing_in_normal}\n"
            f"Common count: {len(common)}"
        )

    df_n.to_parquet(out_normal, index=False)
    df_a.to_parquet(out_attack, index=False)

    logger.info("Saved normal parquet to %s", out_normal.resolve())
    logger.info("Saved attack  parquet to %s", out_attack.resolve())
    logger.info("Features saved: %d", len(feat_n))
    logger.info("Columns: %s", ", ".join(list(df_n.columns)))


if __name__ == "__main__":
    main()
