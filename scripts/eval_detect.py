from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch

from daics.config import load_config
from daics.data.dataloaders import make_dataloaders_paper_strict
from daics.eval.detect import DetectionConfig, detect_from_scores
from daics.eval.mse import compute_section_mse_series
from daics.train.wdnn_trainer import load_wdnn_checkpoint

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("eval_detect")


def _pick_device(device_str: str) -> torch.device:
    d = (device_str or "auto").lower().strip()
    if d == "cpu":
        return torch.device("cpu")
    if d == "cuda":
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


@torch.no_grad()
def _collect_window_labels(loader: torch.utils.data.DataLoader, device: torch.device) -> List[int]:
    """
    Collect aggregated window labels (lab) from the loader in the same order
    as compute_section_mse_series iterates.

    Your dataset yields (x, y, lab) with:
      lab: (B,) aggregated label over the predicted window Wout
    """
    out: List[int] = []
    for _x, _y, lab in loader:
        lab = lab.to(device)
        out.extend([int(v.item()) for v in lab])
    return out


def _confusion(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[int, int, int, int]:
    """
    Returns: (tp, fp, tn, fn)
    """
    yt = y_true.astype(np.int64)
    yp = y_pred.astype(np.int64)
    tp = int(((yt == 1) & (yp == 1)).sum())
    fp = int(((yt == 0) & (yp == 1)).sum())
    tn = int(((yt == 0) & (yp == 0)).sum())
    fn = int(((yt == 1) & (yp == 0)).sum())
    return tp, fp, tn, fn


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b != 0 else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="configs/base.yaml")
    ap.add_argument("--wdnn_ckpt", required=True, help="runs/wdnn/best.pt (or last.pt)")
    ap.add_argument("--thresholds", default="runs/thresholds.json", help="runs/thresholds.json")
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = _pick_device(cfg.train.device)

    # Load thresholds
    thr_path = Path(args.thresholds)
    thr = json.loads(thr_path.read_text(encoding="utf-8"))
    Tg = float(thr["thresholds"]["Tg_section0"])

    Wanom = int(thr["detection"]["Wanom"])
    Wgrace = int(thr["detection"]["Wgrace"])
    det_cfg = DetectionConfig(Wanom=Wanom, Wgrace=Wgrace)

    logger.info("Device: %s", device)
    logger.info("Loaded thresholds: %s", thr_path.resolve())
    logger.info("Using Tg(section0)=%.6f, Wanom=%d, Wgrace=%d", Tg, Wanom, Wgrace)

    # Dataloaders
    logger.info("Building dataloaders...")
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

    if "test_counts" in artifacts:
        tc = artifacts["test_counts"]
        logger.info(
            "Test set composition: total_rows=%d | normal_rows=%d | attack_rows=%d | "
            "tail_normal_rows=%d (%.2f%% of normal parquet)",
            int(tc["test_total_rows"]),
            int(tc["test_normal_rows"]),
            int(tc["test_attack_rows"]),
            int(tc["tail_normal_rows"]),
            float(tc["tail_normal_pct_of_normal"]),
        )
        logger.info("Test note: %s", artifacts.get("test_note", ""))

    # Load WDNN
    wdnn_path = Path(args.wdnn_ckpt)
    wdnn = load_wdnn_checkpoint(str(wdnn_path), device=device)
    logger.info("Loaded WDNN checkpoint: %s", wdnn_path.resolve())

    # Compute test MSE scores
    logger.info("Computing test MSE series (section 0)...")
    mse_test = compute_section_mse_series(model=wdnn, loader=test_loader, device=device)
    scores = np.asarray(mse_test, dtype=np.float32)
    logger.info("Test MSE length: %d", int(len(scores)))
    logger.info("Test MSE mean/std: %.6f / %.6f", float(scores.mean()), float(scores.std(ddof=0)))

    # Collect window-level ground truth labels (aligned with the loader order)
    logger.info("Collecting test window labels (lab)...")
    labs = _collect_window_labels(test_loader, device=device)
    y_true = np.asarray(labs, dtype=np.int64)

    if len(y_true) != len(scores):
        raise RuntimeError(
            f"Length mismatch: scores={len(scores)} vs y_true(lab)={len(y_true)}. "
            "They must be aligned window-by-window."
        )

    # Detection
    raw_pred, alarm = detect_from_scores(scores=scores, Tg=Tg, cfg=det_cfg)

    # Metrics (baseline point-wise on windows)
    tp_raw, fp_raw, tn_raw, fn_raw = _confusion(y_true, raw_pred)
    tp_al, fp_al, tn_al, fn_al = _confusion(y_true, alarm)

    prec_raw = _safe_div(tp_raw, tp_raw + fp_raw)
    rec_raw = _safe_div(tp_raw, tp_raw + fn_raw)
    f1_raw = _safe_div(2 * prec_raw * rec_raw, prec_raw + rec_raw)

    prec_al = _safe_div(tp_al, tp_al + fp_al)
    rec_al = _safe_div(tp_al, tp_al + fn_al)
    f1_al = _safe_div(2 * prec_al * rec_al, prec_al + rec_al)

    # Prints
    print("\n=== Detection evaluation (window-level) ===")
    print(f"WDNN ckpt     : {wdnn_path}")
    print(f"Thresholds    : {thr_path}")
    print(f"Win/Wout/H/S  : {cfg.windowing}")
    print(f"Tg(section0)  : {Tg:.6f}")
    print(f"Wanom/Wgrace  : {Wanom} / {Wgrace}")
    print(f"N windows     : {len(scores)}")

    print("\n--- Raw thresholding (score > Tg) ---")
    print(f"TP={tp_raw} FP={fp_raw} TN={tn_raw} FN={fn_raw}")
    print(f"Precision={prec_raw:.4f} Recall={rec_raw:.4f} F1={f1_raw:.4f}")

    print("\n--- After Wanom (+ Wgrace) ---")
    print(f"TP={tp_al} FP={fp_al} TN={tn_al} FN={fn_al}")
    print(f"Precision={prec_al:.4f} Recall={rec_al:.4f} F1={f1_al:.4f}")

    # Some extra helpful debug info
    n_raw = int(raw_pred.sum())
    n_alarm = int(alarm.sum())
    print("\n--- Debug ---")
    print(f"#raw_pred positives : {n_raw}")
    print(f"#alarms (after Wanom): {n_alarm}")
    if n_alarm > 0:
        first_alarm = int(np.where(alarm == 1)[0][0])
        print(f"First alarm index: {first_alarm} (0-based window index)")
        print(f"Score at first alarm: {float(scores[first_alarm]):.6f} (Tg={Tg:.6f})")

    print("\n--- Summary ---")
    print(
        f"Out of {len(scores)} test windows, the model correctly detected "
        f"{tp_al} attack windows and correctly rejected {tn_al} normal windows "
        f"after Wanom filtering."
    )
    print(
        f"This corresponds to an F1-score of {f1_al:.4f} "
        f"with Precision={prec_al:.4f} and Recall={rec_al:.4f}."
    )

    print("\n[OK] eval_detect finished.")


if __name__ == "__main__":
    main()
