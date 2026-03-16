import numpy as np


def normalize_volume(vol):
    p1 = np.percentile(vol, 1)
    p99 = np.percentile(vol, 99)
    denom = max(p99 - p1, 1e-6)
    out = (vol - p1) / denom
    return np.clip(out, 0.0, 1.0).astype(np.float32)
