from __future__ import annotations

"""
Sanity check for DAICS windowing + dataloaders (paper-strict SWaT)

Goal:
  - Build loaders with paper-strict split:
      train/val from NORMAL parquet
      test      from ATTACK parquet
  - Pull one batch and print shapes in paper notation:
      x   : (B, Win, m)
      y   : (B, Wout, mse)   [WDNN predicts sensors only]
      lab : (B,)             aggregated label over the predicted window

Why this exists:
  - Catch shape / split bugs early
  - Ensure the rest of the pipeline (WDNN/TTNN/detection) can rely on invariants
"""

import argparse

from daics.config import load_config
from daics.data.dataloaders import make_dataloaders_paper_strict


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="configs/base.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)

    print("Building dataloaders... (paper-strict)")
    train_loader, val_loader, test_loader, artifacts = make_dataloaders_paper_strict(
        normal_parquet_path=cfg.data.processed_normal_path,
        attack_parquet_path=cfg.data.processed_attack_path,
        window_cfg=cfg.windowing,
        split_cfg=cfg.splits,
        loader_cfg=cfg.loader,
        label_col=cfg.data.label_col,
        test_mode=str(cfg.eval.test_mode),
        normal_tail_rows=int(cfg.eval.normal_tail_rows),
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
    print(f"  Wanom/Wgrace       : {cfg.detection.Wanom} / {cfg.detection.Wgrace}")

    # Paper-strict paths
    print("\nPaper-strict parquet paths:")
    paths = artifacts.get("paths", {})
    print(f"  normal parquet     : {paths.get('normal_parquet', cfg.data.processed_normal_path)}")
    print(f"  attack parquet     : {paths.get('attack_parquet', cfg.data.processed_attack_path)}")

    print("\nPaper-strict test construction:")
    if "test_mode" in artifacts:
        print(f"  test_mode          : {artifacts['test_mode']}")
    if "test_counts" in artifacts:
        print(f"  test_counts        : {artifacts['test_counts']}")
    if "test_note" in artifacts:
        print(f"  test_note          : {artifacts['test_note']}")

    # Extra debug (useful when you later explain results)
    print("\nColumns partition (SWaT heuristic):")
    print(f"  #sensor_cols       : {len(artifacts['sensor_cols'])}")
    print(f"  #actuator_cols     : {len(artifacts['actuator_cols'])}")

    # --- Paper-aligned assertions ---
    assert B == B2, "Batch size mismatch between x and y"
    assert m == artifacts["m"], "Mismatch: input feature dim vs artifacts['m']"
    assert mse == artifacts["mse"], "Mismatch: output sensor dim vs artifacts['mse']"
    assert m == artifacts["mse"] + artifacts["mac"], "Mismatch: m != mse + mac"
    assert Win == cfg.windowing.Win, "Mismatch: Win"
    assert Wout == cfg.windowing.Wout, "Mismatch: Wout"

    # quick iterator availability
    _ = next(iter(val_loader))
    _ = next(iter(test_loader))

    print("\n[OK] sanity_check_windows passed (paper-strict).")


if __name__ == "__main__":
    main()
