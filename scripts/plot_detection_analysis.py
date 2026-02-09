# scripts/plot_detection_analysis.py
from __future__ import annotations

"""
1) Confusion matrix (RAW):    raw_pred = (score > Tg)
2) Confusion matrix (ALARM):  alarm = apply_wanom(raw_pred, Wanom, Wgrace)
3) Score distributions (normal vs attack windows) + Tg vertical line
4) Zoom around first alarm (if any)

Note:
- Output is saved under runs/plots/<tag>_*.png
"""

import argparse
import json
import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt

from daics.config import load_config
from daics.data.dataloaders import make_dataloaders_paper_strict
from daics.eval.detect import DetectionConfig, detect_from_scores
from daics.eval.mse import compute_section_mse_series
from daics.train.wdnn_trainer import load_wdnn_checkpoint

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("plot_detection_analysis")


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
    Collect the aggregated window labels (lab) in loader order.
    lab is computed over the OUTPUT window (Wout) by SlidingWindowDataset.
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


def _ensure_outdir(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)


def _plot_confusion_matrix(tp: int, fp: int, tn: int, fn: int, out_path: Path, title: str) -> None:
    """
    Confusion matrix layout:
                 pred=0  pred=1
      true=0        TN      FP
      true=1        FN      TP
    """
    M = np.array([[tn, fp], [fn, tp]], dtype=np.int64)

    plt.figure()
    plt.imshow(M, interpolation="nearest")
    plt.title(title)
    plt.xticks([0, 1], ["pred=0", "pred=1"])
    plt.yticks([0, 1], ["true=0", "true=1"])

    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(M[i, j]), ha="center", va="center")

    plt.xlabel("Prediction")
    plt.ylabel("Ground truth")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _plot_score_distributions(
    scores: np.ndarray,
    y_true: np.ndarray,
    Tg: float,
    out_path: Path,
    title: str,
    bins: int = 60,
) -> None:
    """
    Histogram of scores separated by ground truth labels, with Tg as a vertical line.
    """
    s0 = scores[y_true == 0]
    s1 = scores[y_true == 1]

    plt.figure()
    plt.hist(s0, bins=bins, alpha=0.6, label=f"Normal windows (n={len(s0)})")
    plt.hist(s1, bins=bins, alpha=0.6, label=f"Attack windows (n={len(s1)})")
    plt.axvline(float(Tg), linestyle="--", label=f"Tg={Tg:.6f}")
    plt.title(title)
    plt.xlabel("MSE score (per window)")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _plot_zoom_first_alarm(
    scores: np.ndarray,
    y_true: np.ndarray,
    Tg: float,
    alarm: np.ndarray,
    out_path: Path,
    title: str,
    left: int = 200,
    right: int = 400,
) -> None:
    """
    Zoom a region around the first alarm index, if any.
    """
    idx = np.where(alarm == 1)[0]
    if idx.size == 0:
        logger.info("No alarm triggered; skipping zoom plot.")
        return

    first_alarm = int(idx[0])
    start = max(0, first_alarm - int(left))
    end = min(len(scores), first_alarm + int(right))

    x = np.arange(start, end)
    s = scores[start:end]

    plt.figure()
    plt.plot(x, s)
    plt.axhline(float(Tg), linestyle="--")

    # Shade anomaly spans in the zoom region based on ground truth
    in_run = False
    run_start = start
    for i in range(start, end):
        if y_true[i] == 1 and not in_run:
            in_run = True
            run_start = i
        if in_run and (y_true[i] == 0 or i == end - 1):
            run_end = i if y_true[i] == 0 else i + 1
            plt.axvspan(run_start, run_end, alpha=0.15)
            in_run = False

    plt.title(f"{title} (first_alarm_idx={first_alarm})")
    plt.xlabel("Window index")
    plt.ylabel("MSE score (per window)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="configs/base.yaml")
    ap.add_argument("--wdnn_ckpt", required=True, help="runs/wdnn/best.pt")
    ap.add_argument("--thresholds", required=True, help="runs/thresholds.json")

    ap.add_argument("--tag", default="paper_default", help="Tag used in output filenames")
    ap.add_argument("--out_dir", default="runs/plots", help="Output directory for figures")

    # Optional override (useful if you want to test Wanom effect without regenerating thresholds)
    ap.add_argument("--Wanom", type=int, default=None)
    ap.add_argument("--Wgrace", type=int, default=None)

    args = ap.parse_args()

    cfg = load_config(args.config)
    device = _pick_device(cfg.train.device)

    thr = json.loads(Path(args.thresholds).read_text(encoding="utf-8"))
    Tg = float(thr["thresholds"]["Tg_section0"])

    Wanom = int(thr["detection"]["Wanom"])
    Wgrace = int(thr["detection"]["Wgrace"])
    if args.Wanom is not None:
        Wanom = int(args.Wanom)
    if args.Wgrace is not None:
        Wgrace = int(args.Wgrace)

    logger.info("Device: %s", device)
    logger.info("Using Tg=%.6f, Wanom=%d, Wgrace=%d", Tg, Wanom, Wgrace)

    out_dir = Path(args.out_dir)
    _ensure_outdir(out_dir)

    # Dataloaders (paper strict split; test is merged)
    train_loader, val_loader, test_loader, artifacts = make_dataloaders_paper_strict(
        normal_parquet_path=cfg.data.processed_normal_path,
        merged_parquet_path=cfg.data.processed_merged_path,
        window_cfg=cfg.windowing,
        split_cfg=cfg.splits,
        loader_cfg=cfg.loader,
        label_col=cfg.data.label_col,
        test_mode=str(cfg.eval.test_mode),
    )

    # Load WDNN
    wdnn = load_wdnn_checkpoint(args.wdnn_ckpt, device=device)
    wdnn.eval()

    # Scores + ground truth labels (window-level)
    scores = np.asarray(compute_section_mse_series(wdnn, test_loader, device=device), dtype=np.float32)
    labs = np.asarray(_collect_window_labels(test_loader, device=device), dtype=np.int64)

    if len(scores) != len(labs):
        raise RuntimeError(f"Length mismatch: scores={len(scores)} labs={len(labs)}")

    # Detect
    det_cfg = DetectionConfig(Wanom=Wanom, Wgrace=Wgrace)
    raw_pred, alarm = detect_from_scores(scores=scores, Tg=Tg, cfg=det_cfg)

    # Confusions
    tp_r, fp_r, tn_r, fn_r = _confusion(labs, raw_pred)
    tp_a, fp_a, tn_a, fn_a = _confusion(labs, alarm)

    tag = args.tag.strip()
    prefix = out_dir / tag

    # ---- ONLY the 4 plots we want ----
    _plot_confusion_matrix(tp_r, fp_r, tn_r, fn_r, Path(str(prefix) + "_cm_raw.png"), f"Confusion matrix (RAW) [{tag}]")
    _plot_confusion_matrix(tp_a, fp_a, tn_a, fn_a, Path(str(prefix) + "_cm_alarm.png"), f"Confusion matrix (ALARM) [{tag}]")
    _plot_score_distributions(scores, labs, Tg, Path(str(prefix) + "_score_distributions.png"), f"Score distributions [{tag}]")
    _plot_zoom_first_alarm(scores, labs, Tg, alarm, Path(str(prefix) + "_zoom_first_alarm.png"), f"Zoom around first alarm [{tag}]")

    logger.info("[OK] Saved plots with prefix: %s_*", str(prefix))


if __name__ == "__main__":
    main()
