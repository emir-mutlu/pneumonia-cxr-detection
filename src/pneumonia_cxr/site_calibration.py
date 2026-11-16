"""Local validation kit: adapt the released model to a new hospital without retraining.

A model's discrimination may hold up at a new hospital while its probabilities and its
decision threshold do not (see the VinDr-PCXR results: AUROC 0.746, but the stored threshold
finds 7 % of pneumonia cases). A hospital that labels a few hundred of its own X-rays can:

1. measure discrimination locally (AUROC with a bootstrap 95 % CI);
2. recalibrate the probabilities with Platt scaling, ``sigmoid(a * logit + b)``, which corrects
   both over/under-confidence and the local prevalence;
3. choose a local threshold, either for a target sensitivity (default 0.90, a triage-style
   operating point) or Youden's J;
4. get an *honest* estimate of the resulting sensitivity / specificity: two-fold, stratified
   cross-fitting (fit on one half, measure on the other) so the threshold is never scored on
   the images it was tuned on; and
5. save a site profile that :class:`~pneumonia_cxr.inference.OnnxPredictor` applies.

This is not a substitute for a prospective clinical study; it is the minimum check before one.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from .metrics import binary_metrics, bootstrap_ci, sigmoid, threshold_for_sensitivity, youden_threshold
from .site_profile import SiteProfile
from .site_profile import logit as _logit

__all__ = ["SiteProfile", "calibrate_site", "fit_platt", "format_report"]

MIN_PER_CLASS = 30  # below this, CIs are too wide to act on


def fit_platt(logits: np.ndarray, y: np.ndarray, l2: float = 1e-3) -> tuple[float, float]:
    """(a, b) minimising the NLL of sigmoid(a * logit + b), lightly L2-regularised towards (1, 0)."""
    z, y = np.asarray(logits, dtype=np.float64), np.asarray(y, dtype=np.float64)

    def nll(params: np.ndarray) -> float:
        a, b = params
        s = a * z + b
        return float(np.mean(np.logaddexp(0, s) - y * s) + l2 * ((a - 1) ** 2 + b**2))

    res = minimize(nll, x0=np.array([1.0, 0.0]), method="L-BFGS-B")
    return float(res.x[0]), float(res.x[1])


def _choose_threshold(y: np.ndarray, p: np.ndarray, target_sensitivity: float | None) -> float:
    if target_sensitivity is None:
        return youden_threshold(y, p)
    return threshold_for_sensitivity(y, p, target_sensitivity)


def calibrate_site(
    probs: np.ndarray,
    labels: np.ndarray,
    site: str,
    model_threshold: float,
    target_sensitivity: float | None = 0.90,
    seed: int = 42,
    n_boot: int = 1000,
) -> SiteProfile:
    """Fit a site profile on local, labelled model outputs and report what to expect."""
    p = np.clip(np.asarray(probs, dtype=np.float64), 1e-6, 1 - 1e-6)
    y = np.asarray(labels).astype(int)
    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    if min(n_pos, n_neg) < 2:
        raise ValueError("Need both pneumonia and non-pneumonia cases to calibrate a site.")
    z = _logit(p)

    # Honest estimate: two-fold stratified cross-fitting of Platt scaling + threshold.
    rng = np.random.default_rng(seed)
    fold = np.empty(len(y), dtype=int)
    for cls in (0, 1):
        idx = rng.permutation(np.flatnonzero(y == cls))
        fold[idx] = np.arange(len(idx)) % 2
    oof_prob, oof_pred = np.empty(len(y)), np.empty(len(y), dtype=bool)
    for k in (0, 1):
        fit, held = fold != k, fold == k
        a, b = fit_platt(z[fit], y[fit])
        p_fit = sigmoid(a * z[fit] + b)
        thr = _choose_threshold(y[fit], p_fit, target_sensitivity)
        oof_prob[held] = sigmoid(a * z[held] + b)
        oof_pred[held] = oof_prob[held] >= thr
    held_out = binary_metrics(y, oof_pred.astype(float), 0.5)

    a, b = fit_platt(z, y)
    p_local = sigmoid(a * z + b)
    threshold = _choose_threshold(y, p_local, target_sensitivity)
    before = binary_metrics(y, p, model_threshold)
    ci = bootstrap_ci(y, p, model_threshold, n_boot=n_boot) if n_boot else {"auroc": (float("nan"),) * 2}

    keys = ("sensitivity", "specificity", "ppv", "npv")
    warnings = []
    if min(n_pos, n_neg) < MIN_PER_CLASS:
        warnings.append(
            f"Only {n_pos} pneumonia and {n_neg} other cases: estimates are very uncertain. "
            f"Collect at least {MIN_PER_CLASS} of each, ideally several hundred images."
        )
    report = {
        "auroc": before["auroc"],
        "auroc_ci95": ci["auroc"],
        "prevalence": before["prevalence"],
        "before": {"threshold": model_threshold, **{k: before[k] for k in keys}, "ece": before["ece"]},
        "after_held_out_estimate": {**{k: held_out[k] for k in keys}, "ece": binary_metrics(y, oof_prob, 0.5)["ece"]},
        "warnings": warnings,
    }
    return SiteProfile(site, a, b, threshold, target_sensitivity, len(y), n_pos, report=report)


def format_report(profile: SiteProfile) -> str:
    r = profile.report
    lo, hi = r["auroc_ci95"]
    b, h = r["before"], r["after_held_out_estimate"]
    target = f"target sensitivity {profile.target_sensitivity:.0%}" if profile.target_sensitivity else "Youden's J"
    lines = [
        f"Site '{profile.site}': {profile.n} images, {profile.n_positive} pneumonia (prevalence {r['prevalence']:.1%})",
        f"  Discrimination: AUROC {r['auroc']:.3f} (95% CI {lo:.3f}-{hi:.3f})",
        f"  Released threshold {b['threshold']:.3f}: sensitivity {b['sensitivity']:.3f}, specificity "
        f"{b['specificity']:.3f}, ECE {b['ece']:.3f}",
        f"  Local recalibration ({target}), estimated on held-out halves: sensitivity {h['sensitivity']:.3f}, "
        f"specificity {h['specificity']:.3f}, PPV {h['ppv']:.3f}, NPV {h['npv']:.3f}, ECE {h['ece']:.3f}",
        f"  Profile: Platt a={profile.platt_a:.3f}, b={profile.platt_b:.3f}, local threshold {profile.threshold:.3f}",
    ]
    lines += [f"  WARNING: {w}" for w in r["warnings"]]
    return "\n".join(lines)
