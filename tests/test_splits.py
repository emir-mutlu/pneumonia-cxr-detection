import numpy as np
import pandas as pd
import pytest

from pneumonia_cxr.data.splits import check_no_patient_leak, patient_split


def _toy(n_patients: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(n_patients):
        label = int(rng.random() < 0.3)
        for _ in range(rng.integers(1, 6)):  # several images per patient
            rows.append({"patient_id": f"p{p}", "label": label})
    return pd.DataFrame(rows)


def test_every_patient_in_exactly_one_split():
    df = _toy()
    df["split"] = patient_split(df, {"train": 0.7, "val": 0.1, "test": 0.2})
    assert df["split"].notna().all()
    check_no_patient_leak(df)


def test_split_fractions_and_stratification():
    df = _toy(2000)
    df["split"] = patient_split(df, {"train": 0.7, "val": 0.1, "test": 0.2})
    patients = df.groupby("patient_id").agg(split=("split", "first"), label=("label", "max"))
    share = patients["split"].value_counts(normalize=True)
    assert share["train"] == pytest.approx(0.7, abs=0.01)
    assert share["test"] == pytest.approx(0.2, abs=0.01)
    prevalence = patients.groupby("split")["label"].mean()
    assert prevalence.max() - prevalence.min() < 0.01


def test_split_is_deterministic():
    df = _toy()
    a = patient_split(df, {"train": 0.8, "val": 0.2}, seed=7)
    b = patient_split(df, {"train": 0.8, "val": 0.2}, seed=7)
    c = patient_split(df, {"train": 0.8, "val": 0.2}, seed=8)
    assert a.equals(b)
    assert not a.equals(c)


def test_leak_is_detected():
    df = pd.DataFrame({"patient_id": ["a", "a", "b"], "label": [1, 1, 0], "split": ["train", "test", "val"]})
    with pytest.raises(AssertionError):
        check_no_patient_leak(df)


def test_fractions_must_sum_to_one():
    with pytest.raises(ValueError):
        patient_split(_toy(), {"train": 0.5, "test": 0.2})
