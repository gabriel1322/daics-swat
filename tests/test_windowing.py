import numpy as np

from daics.data.windowing import WindowingConfig, num_windows, build_windows


def test_num_windows_basic():
    cfg = WindowingConfig(W=4, H=1, S=1)
    assert num_windows(5, cfg) == 1
    assert num_windows(6, cfg) == 2
    assert num_windows(10, cfg) == 6


def test_build_windows_shapes():
    T, D = 20, 3
    X = np.random.randn(T, D).astype(np.float32)
    L = np.zeros((T,), dtype=np.int64)
    L[5] = 1  # one anomaly

    cfg = WindowingConfig(W=6, H=1, S=2, label_agg="any")
    x, y, lab = build_windows(X, L, cfg)

    assert x.ndim == 3 and x.shape[1] == 6 and x.shape[2] == 3
    assert y.ndim == 2 and y.shape[1] == 3
    assert lab.ndim == 1
    assert x.shape[0] == y.shape[0] == lab.shape[0]


def test_label_agg_any_marks_window():
    # Window includes timestep 5 => should be anomalous for those windows
    T, D = 12, 2
    X = np.random.randn(T, D).astype(np.float32)
    L = np.zeros((T,), dtype=np.int64)
    L[5] = 1

    cfg = WindowingConfig(W=4, H=1, S=1, label_agg="any")
    x, y, lab = build_windows(X, L, cfg)

    # Window starts 2 => covers [2,3,4,5] includes anomaly
    assert lab[2] == 1
