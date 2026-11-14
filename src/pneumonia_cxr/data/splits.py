"""Patient-level, label-stratified splitting.

Images of the same patient must never be spread across train / validation / test,
otherwise the model can be rewarded for recognising the patient rather than the disease.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def patient_split(
    df: pd.DataFrame,
    fractions: dict[str, float],
    seed: int = 42,
    patient_col: str = "patient_id",
    label_col: str = "label",
) -> pd.Series:
    """Assign each row to a split so that every patient lands in exactly one split.

    Patients are stratified by their patient-level label (positive if any of their images
    is positive), so each split keeps roughly the same class balance.

    Args:
        df: one row per image.
        fractions: e.g. ``{"train": 0.7, "val": 0.1, "test": 0.2}``; must sum to 1.
        seed: RNG seed, the assignment is deterministic for a given seed.

    Returns:
        A Series of split names aligned with ``df.index``.
    """
    if not np.isclose(sum(fractions.values()), 1.0):
        raise ValueError(f"fractions must sum to 1, got {fractions}")

    patient_label = df.groupby(patient_col)[label_col].max()
    rng = np.random.default_rng(seed)
    names = list(fractions)
    assignment: dict[object, str] = {}

    for _, group in patient_label.groupby(patient_label):
        patients = group.index.to_numpy().copy()
        rng.shuffle(patients)
        n = len(patients)
        bounds = np.rint(np.cumsum([fractions[k] for k in names]) * n).astype(int)
        start = 0
        for name, end in zip(names, bounds, strict=True):
            for p in patients[start:end]:
                assignment[p] = name
            start = end

    return df[patient_col].map(assignment).rename("split")


def check_no_patient_leak(df: pd.DataFrame, patient_col: str = "patient_id") -> None:
    """Raise if any patient appears in more than one split."""
    per_patient = df.groupby(patient_col)["split"].nunique()
    leaked = per_patient[per_patient > 1]
    if len(leaked):
        raise AssertionError(f"{len(leaked)} patients appear in several splits, e.g. {leaked.index[:5].tolist()}")
