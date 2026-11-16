"""How many local X-rays does a new hospital need to recalibrate the released model?

Simulation on VinDr-PCXR, a hospital the released model never saw. A "local sample" of n
images is drawn from the VinDr-PCXR validation split, `calibrate_site` fits Platt scaling and
a threshold for 90 % sensitivity on it, and the resulting operating point is measured on the
untouched VinDr-PCXR test split. Repeated over many random samples per n.

Needs the released model's predictions on both splits:

    pneumo evaluate runs/multi_convnext_tiny --source vindr_pcxr            # test
    python -c "from pneumonia_cxr.evaluate import evaluate_run; \
               evaluate_run('runs/multi_convnext_tiny', ('vindr_pcxr',), 'val')"

Writes reports/site_calibration.json and reports/figures/site_calibration.png (aggregates only).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from pneumonia_cxr.metrics import binary_metrics
from pneumonia_cxr.report import INK_2, MUTED, SERIES, _style, plt
from pneumonia_cxr.site_calibration import calibrate_site

RUN = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/multi_convnext_tiny")
SIZES = (100, 200, 300, 500, 1000)
REPEATS = 200
TARGET = 0.90


def main() -> None:
    val = pd.read_csv(RUN / "preds_vindr_pcxr_val.csv")
    test = pd.read_csv(RUN / "preds_vindr_pcxr_test.csv")
    released = json.loads((RUN / "train_summary.json").read_text())["threshold"]
    y_test, p_test = test["label"].to_numpy(), test["prob"].to_numpy()
    before = binary_metrics(y_test, p_test, released)

    rng = np.random.default_rng(0)
    rows = []
    for n in SIZES:
        for _ in range(REPEATS):
            sample = val.sample(n, random_state=int(rng.integers(1 << 31)))
            if sample["label"].sum() < 5:  # a local sample needs at least a few pneumonia cases
                continue
            profile = calibrate_site(sample["prob"], sample["label"], "vindr", released, TARGET, n_boot=0)
            m = binary_metrics(y_test, profile.apply(p_test), profile.threshold)
            rows.append({"n": n, "sensitivity": m["sensitivity"], "specificity": m["specificity"]})
    df = pd.DataFrame(rows)
    summary = {
        "target_sensitivity": TARGET,
        "repeats_per_size": REPEATS,
        "released_threshold_on_test": {"sensitivity": before["sensitivity"], "specificity": before["specificity"]},
        "test_auroc": before["auroc"],
        "by_local_sample_size": {
            int(n): {
                k: {q: float(g[k].quantile(v)) for q, v in (("p10", 0.1), ("median", 0.5), ("p90", 0.9))}
                for k in ("sensitivity", "specificity")
            }
            | {"runs": int(len(g))}
            for n, g in df.groupby("n")
        },
    }
    Path("reports/site_calibration.json").write_text(json.dumps(summary, indent=2))

    _style()
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    x = np.arange(len(SIZES))
    for color, key, label in ((SERIES[0], "sensitivity", "Sensitivity"), (SERIES[1], "specificity", "Specificity")):
        s = summary["by_local_sample_size"]
        med = [s[n][key]["median"] for n in SIZES]
        ax.fill_between(x, [s[n][key]["p10"] for n in SIZES], [s[n][key]["p90"] for n in SIZES],
                        color=color, alpha=0.15, lw=0)  # fmt: skip
        ax.plot(x, med, color=color, marker="o", ms=8, label=f"{label} (median, 10–90 % band)")
        ax.annotate(f"{med[-1]:.2f}", (x[-1], med[-1]), xytext=(8, 0), textcoords="offset points",
                    va="center", fontsize=8.5, color=INK_2)  # fmt: skip
    ax.axhline(TARGET, color=MUTED, lw=1, ls=":")
    ax.annotate(f"target sensitivity {TARGET:.0%}", (0, TARGET), xytext=(0, 4), textcoords="offset points",
                fontsize=8, color=MUTED)  # fmt: skip
    ax.axhline(before["sensitivity"], color=SERIES[0], lw=1, ls=(0, (4, 2)))
    ax.annotate(f"sensitivity with the released threshold: {before['sensitivity']:.2f}", (0, before["sensitivity"]),
                xytext=(0, 4), textcoords="offset points", fontsize=8, color=INK_2)  # fmt: skip
    ax.set_xticks(x, [str(n) for n in SIZES])
    ax.set(ylim=(0, 1.02), xlabel="Labelled local X-rays used for recalibration", ylabel="On the hospital's test set")
    ax.set_title("New hospital (VinDr-PCXR), released model, no retraining", loc="left")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.08), ncol=2)
    fig.tight_layout()
    fig.savefig("reports/figures/site_calibration.png", dpi=150, bbox_inches="tight")
    print(json.dumps(summary["by_local_sample_size"], indent=1))
    print("before:", summary["released_threshold_on_test"])


if __name__ == "__main__":
    main()
