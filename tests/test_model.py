"""Model helpers, trained on a small synthetic frame (no download needed)."""

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from modelledger.model.credit_model import (
    CATEGORICAL, FEATURES, Scorer, age_group, encode, fit_category_maps,
    sex_from_personal_status,
)

CATS = {
    "checking_status": ["<0", "0<=X<200", "no checking"],
    "credit_history": ["existing paid", "critical/other existing credit"],
    "purpose": ["radio/tv", "new car"],
    "savings_status": ["<100", ">=1000"],
    "employment": ["1<=X<4", ">=7"],
    "other_parties": ["none"],
    "property_magnitude": ["car", "real estate"],
    "other_payment_plans": ["none", "bank"],
    "housing": ["own", "rent"],
    "job": ["skilled"],
    "own_telephone": ["none", "yes"],
}


def _frame(n=200, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({c: rng.choice(v, n) for c, v in CATS.items()})
    df["duration"] = rng.integers(6, 60, n)
    df["credit_amount"] = rng.integers(500, 15000, n)
    df["installment_commitment"] = rng.integers(1, 5, n)
    df["residence_since"] = rng.integers(1, 5, n)
    df["age"] = rng.integers(19, 75, n)
    df["existing_credits"] = rng.integers(1, 4, n)
    df["num_dependents"] = rng.integers(1, 3, n)
    df["personal_status"] = rng.choice(["female div/dep/mar", "male single"], n)
    df["foreign_worker"] = "yes"
    y = (df["duration"] < 30).astype(int)
    return df, y


def test_protected_fields_are_not_model_inputs():
    assert "personal_status" not in FEATURES
    assert "foreign_worker" not in FEATURES


def test_age_and_sex_derivation():
    assert age_group(22) == "18-25"
    assert age_group(60) == "56+"
    assert sex_from_personal_status("female div/dep/mar") == "female"
    assert sex_from_personal_status("male single") == "male"


def test_unseen_category_is_encoded_safely():
    df, _ = _frame()
    maps = fit_category_maps(df)
    df.loc[0, "purpose"] = "spaceship"
    assert encode(df, maps).loc[0, "purpose"] == -1


def test_scorer_returns_real_shap_that_sums_to_the_logit():
    df, y = _frame()
    maps = fit_category_maps(df)
    model = XGBClassifier(n_estimators=20, max_depth=2).fit(encode(df, maps), y)
    scorer = Scorer({"model": model, "category_maps": maps, "threshold": 0.5})
    out = scorer.score(df.head(3))
    base = float(np.ravel(scorer.explainer.expected_value)[0])
    for r in out:
        assert set(r["shap_values"]) == set(FEATURES)
        assert r["sex"] in ("female", "male")
        assert 0 <= r["score"] <= 1
        logit = np.log(r["score"] / (1 - r["score"]))
        assert abs(base + sum(r["shap_values"].values()) - logit) < 1e-3
        assert set(CATEGORICAL) <= set(r["input_features"])
