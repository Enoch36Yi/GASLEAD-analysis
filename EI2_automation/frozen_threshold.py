"""Original frozen fit_threshold, retained as a verification oracle."""
import numpy as np
from typing import Any

def fit_threshold(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    x, y = np.asarray(x, float), np.asarray(y, int)
    order = np.argsort(x, kind="mergesort")
    xs, ys = x[order], y[order]
    n = len(xs)
    positive_total = int(np.sum(ys == 1))
    negative_total = int(np.sum(ys == 0))
    if positive_total == 0 or negative_total == 0:
        raise ValueError("Threshold fitting requires both binary classes.")
    # A threshold may occur only between distinct values; never split ties.
    split = np.concatenate(([0], np.flatnonzero(xs[1:] != xs[:-1]) + 1, [n])).astype(int)
    cumulative_positive = np.concatenate(([0], np.cumsum(ys == 1)))
    cumulative_negative = np.concatenate(([0], np.cumsum(ys == 0)))
    pos_below = cumulative_positive[split]
    neg_below = cumulative_negative[split]
    higher_scores = 0.5 * ((positive_total - pos_below) / positive_total + neg_below / negative_total)
    lower_scores = 0.5 * (pos_below / positive_total + (negative_total - neg_below) / negative_total)
    thresholds = np.empty(len(split), float)
    for i, k in enumerate(split):
        if k == 0:
            thresholds[i] = np.nextafter(xs[0], -np.inf)
        elif k == n:
            thresholds[i] = np.nextafter(xs[-1], np.inf)
        else:
            thresholds[i] = xs[k - 1] + (xs[k] - xs[k - 1]) / 2.0
    high_i = int(np.argmax(higher_scores))
    low_i = int(np.argmax(lower_scores))
    if higher_scores[high_i] >= lower_scores[low_i] - 1e-12:
        return {"threshold": float(thresholds[high_i]), "direction": "higher",
                "balanced_accuracy": float(higher_scores[high_i])}
    return {"threshold": float(thresholds[low_i]), "direction": "lower",
            "balanced_accuracy": float(lower_scores[low_i])}
