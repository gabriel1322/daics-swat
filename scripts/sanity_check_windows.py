from __future__ import annotations

import argparse

from daics.config import load_config
from daics.data.dataloaders import LoaderConfig, SplitConfig, make_dataloaders
from daics.data.windows import WindowingConfig


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML config, e.g. configs/base.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)

    window_cfg = WindowingConfig(
        W=int(cfg["windowing"]["W"]),
        H=int(cfg["windowing"]["H"]),
        S=int(cfg["windowing"]["S"]),
    )
    split_cfg = SplitConfig(
        train_ratio=float(cfg["splits"]["train_ratio"]),
        val_ratio=float(cfg["splits"]["val_ratio"]),
        seed=int(cfg["splits"]["seed"]),
    )
    loader_cfg = LoaderConfig(
        batch_size=int(cfg["loader"]["batch_size"]),
        num_workers=int(cfg["loader"]["num_workers"]),
        pin_memory=bool(cfg["loader"]["pin_memory"]),
        drop_last_train=bool(cfg["loader"]["drop_last_train"]),
    )

    train_loader, val_loader, test_loader, artifacts = make_dataloaders(
        parquet_path=str(cfg["data"]["processed_path"]),
        window_cfg=window_cfg,
        split_cfg=split_cfg,
        loader_cfg=loader_cfg,
        label_col=cfg["data"].get("label_col", None),
    )

    x, y, lab = next(iter(train_loader))
    print("Train batch shapes:")
    print("  x   :", tuple(x.shape))    # (B, W, N)
    print("  y   :", tuple(y.shape))    # (B, N)
    print("  lab :", tuple(lab.shape))  # (B,)

    print("\nDataset info:")
    print("  features:", len(artifacts["feature_cols"]))
    print("  window  :", artifacts["window_cfg"])
    print("  parquet :", artifacts["parquet_path"])


if __name__ == "__main__":
    main()
