"""Binary classification metrics, bootstrap confidence intervals and calibration error."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))


def youden_threshold(y: np.ndarray, p: np.ndarray) -> float:
    """Threshold maximising sensitivity + specificity - 1 (chosen on validation data only)."""
    fpr, tpr, thr = roc_curve(y, p)
    best = int(np.argmax(tpr - fpr))
    return float(np.clip(thr[best], 0.0, 1.0))


def threshold_for_sensitivity(y: np.ndarray, p: np.ndarray, target: float = 0.95) -> float:
    """Highest threshold whose sensitivity is at least ``target`` (screening operating point)."""
    fpr, tpr, thr = roc_curve(y, p)
    ok = np.where(tpr >= target)[0]
    return float(np.clip(thr[ok[0]], 0.0, 1.0))


def confusion(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, int]:
    pred = p >= threshold
    y = y.astype(bool)
    return {
        "tp": int(np.sum(pred & y)),
        "fp": int(np.sum(pred & ~y)),
        "tn": int(np.sum(~pred & ~y)),
        "fn": int(np.sum(~pred & y)),
    }


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b else float("nan")


def expected_calibration_error(y: np.ndarray, p: np.ndarray, n_bins: int = 15) -> float:
    """ECE with equal-width bins: sum_b |acc_b - conf_b| * n_b / N on P(pneumonia)."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ids = np.clip(np.digitize(p, bins[1:-1]), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = ids == b
        if mask.any():
            ece += abs(y[mask].mean() - p[mask].mean()) * mask.mean()
    return float(ece)


def binary_metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, float]:
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=np.float64)
    c = confusion(y, p, threshold)
    sens = _safe_div(c["tp"], c["tp"] + c["fn"])
    spec = _safe_div(c["tn"], c["tn"] + c["fp"])
    ppv = _safe_div(c["tp"], c["tp"] + c["fp"])
    npv = _safe_div(c["tn"], c["tn"] + c["fn"])
    two_classes = len(np.unique(y)) == 2
    return {
        "n": int(len(y)),
        "prevalence": float(y.mean()),
        "auroc": float(roc_auc_score(y, p)) if two_classes else float("nan"),
        "auprc": float(average_precision_score(y, p)) if two_classes else float("nan"),
        "accuracy": float((c["tp"] + c["tn"]) / len(y)),
        "sensitivity": sens,
        "specificity": spec,
        "ppv": ppv,
        "npv": npv,
        "f1": _safe_div(2 * ppv * sens, ppv + sens),
        "balanced_accuracy": (sens + spec) / 2,
        "brier": float(np.mean((p - y) ** 2)),
        "ece": expected_calibration_error(y, p),
        "threshold": float(threshold),
        **c,
    }


def bootstrap_ci(
    y: np.ndarray,
    p: np.ndarray,
    threshold: float,
    keys: tuple[str, ...] = ("auroc", "auprc", "sensitivity", "specificity", "accuracy", "f1"),
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
    groups: np.ndarray | None = None,
) -> dict[str, tuple[float, float]]:
    """Percentile bootstrap CIs. If ``groups`` (patient ids) is given, patients are resampled
    as clusters, which respects the correlation between images of the same patient."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=np.float64)
    if groups is not None:
        uniq, inv = np.unique(groups, return_inverse=True)
        members = [np.where(inv == g)[0] for g in range(len(uniq))]
    samples: dict[str, list[float]] = {k: [] for k in keys}
    for _ in range(n_boot):
        if groups is None:
            idx = rng.integers(0, len(y), len(y))
        else:
            idx = np.concatenate([members[g] for g in rng.integers(0, len(members), len(members))])
        if len(np.unique(y[idx])) < 2:
            continue
        m = binary_metrics(y[idx], p[idx], threshold)
        for k in keys:
            samples[k].append(m[k])
    return {
        k: (float(np.nanpercentile(v, 100 * alpha / 2)), float(np.nanpercentile(v, 100 * (1 - alpha / 2))))
        for k, v in samples.items()
    }


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    """Temperature scaling (Guo et al., 2017): T minimising the NLL of sigmoid(logit / T)."""
    logits = np.asarray(logits, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    def nll(log_t: float) -> float:
        z = logits / np.exp(log_t)
        # numerically stable binary cross-entropy with logits
        return float(np.mean(np.maximum(z, 0) - z * y + np.log1p(np.exp(-np.abs(z)))))

    res = minimize_scalar(nll, bounds=(np.log(0.05), np.log(20.0)), method="bounded")
    return float(np.exp(res.x))
