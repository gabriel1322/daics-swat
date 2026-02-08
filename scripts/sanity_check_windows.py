from __future__ import annotations

import argparse

from daics.config import load_config
from daics.data.dataloaders import make_dataloaders


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="configs/base.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)

    print("Building dataloaders...")
    train_loader, val_loader, test_loader, artifacts = make_dataloaders(
        parquet_path=cfg.data.processed_path,
        window_cfg=cfg.windowing,
        split_cfg=cfg.splits,
        loader_cfg=cfg.loader,
        label_col=cfg.data.label_col,
    )

    print("\nExtracting one training batch...")
    x, y, lab = next(iter(train_loader))

    # Paper notation
    B, Win, m = x.shape
    B2, Wout, mse = y.shape

    print("\nBatch shapes (Paper notation):")
    print(f"  x   : {tuple(x.shape)}  -> (B, Win, m)")
    print(f"  y   : {tuple(y.shape)}  -> (B, Wout, mse)  [NOTE: WDNN predicts sensors only]")
    print(f"  lab : {tuple(lab.shape)} -> (B,) aggregated label")

    print("\nDataset info (from artifacts):")
    print(f"  m (features total) : {artifacts['m']}")
    print(f"  mse (sensors)      : {artifacts['mse']}")
    print(f"  mac (actuators)    : {artifacts['mac']}")
    print(f"  Win/Wout/H/S       : {artifacts['window_cfg']}")
    print(f"  parquet            : {artifacts['parquet_path']}")
    print(f"  #sensor_cols       : {len(artifacts['sensor_cols'])}")
    print(f"  #actuator_cols     : {len(artifacts['actuator_cols'])}")

    # --- Paper-aligned assertions ---
    # Input is (Win, m) where m = mse + mac
    assert m == artifacts["m"], "Mismatch: batch input feature dim vs artifacts['m']"
    assert mse == artifacts["mse"], "Mismatch: batch output sensor dim vs artifacts['mse']"
    assert m == artifacts["mse"] + artifacts["mac"], "Mismatch: m != mse + mac (paper invariant)"
    assert Win == cfg.windowing.Win, "Mismatch: Win"
    assert Wout == cfg.windowing.Wout, "Mismatch: Wout"

    # quick iterator availability
    _ = next(iter(val_loader))
    _ = next(iter(test_loader))

    print("\n[OK] sanity_check_windows passed (paper-aligned).")


if __name__ == "__main__":
    main()
