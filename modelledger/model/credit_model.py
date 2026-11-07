"""
The demo credit model: XGBoost on the German Credit dataset (OpenML "credit-g").

This module is shared by the training script, the demo seeder and the model
service, so all three encode features, score applicants and explain decisions
in exactly the same way.

Conventions
    target 1 = good credit (repaid), 0 = bad credit (defaulted)
    score    = model probability of good credit (approval)
    decision = 1 (approve) when score >= threshold
"""

import hashlib
import os

import numpy as np
import pandas as pd

DATA_CACHE = os.path.join("data", "credit_g.csv")
MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join("models", "credit_model.joblib"))
METADATA_PATH = os.path.join("models", "model_metadata.json")
RANDOM_STATE = 42
TEST_SIZE = 0.3

# All 20 applicant fields in credit-g.
APPLICANT_FIELDS = [
    "checking_status", "duration", "credit_history", "purpose", "credit_amount",
    "savings_status", "employment", "installment_commitment", "personal_status",
    "other_parties", "residence_since", "property_magnitude", "age",
    "other_payment_plans", "housing", "existing_credits", "job", "num_dependents",
    "own_telephone", "foreign_worker",
]
# Excluded from the model: personal_status encodes sex (a protected characteristic
# that a UK lender may not base credit decisions on) and foreign_worker is a
# nationality proxy. They are still used to MONITOR outcomes by group.
EXCLUDED_FROM_MODEL = ["personal_status", "foreign_worker"]
FEATURES = [f for f in APPLICANT_FIELDS if f not in EXCLUDED_FROM_MODEL]
CATEGORICAL = [
    "checking_status", "credit_history", "purpose", "savings_status", "employment",
    "other_parties", "property_magnitude", "other_payment_plans",
    "housing", "job", "own_telephone",
]
AGE_BANDS = [(18, 25, "18-25"), (26, 35, "26-35"), (36, 45, "36-45"),
             (46, 55, "46-55"), (56, 200, "56+")]


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------

def load_credit_g(cache_path: str = DATA_CACHE) -> pd.DataFrame:
    """German Credit (1,000 applicants). Downloads from OpenML once, then caches."""
    if os.path.exists(cache_path):
        return pd.read_csv(cache_path)
    from sklearn.datasets import fetch_openml

    df = fetch_openml(name="credit-g", version=1, as_frame=True, parser="auto").frame
    for col in CATEGORICAL + ["class"]:
        df[col] = df[col].astype(str)
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    df.to_csv(cache_path, index=False)
    return df


def split(df: pd.DataFrame):
    """The one train/test split used everywhere (stratified 70/30)."""
    from sklearn.model_selection import train_test_split

    y = (df["class"] == "good").astype(int)
    return train_test_split(df[APPLICANT_FIELDS], y, test_size=TEST_SIZE,
                            random_state=RANDOM_STATE, stratify=y)


# ----------------------------------------------------------------------
# Protected characteristics, derived for monitoring
# ----------------------------------------------------------------------

def age_group(age) -> str:
    age = int(age)
    for lo, hi, label in AGE_BANDS:
        if lo <= age <= hi:
            return label
    return "unknown"


def sex_from_personal_status(personal_status: str) -> str:
    """credit-g combines sex and marital status in one field, e.g. 'female div/dep/mar'."""
    return "female" if str(personal_status).startswith("female") else "male"


# ----------------------------------------------------------------------
# Encoding
# ----------------------------------------------------------------------

def fit_category_maps(X: pd.DataFrame) -> dict[str, dict[str, int]]:
    return {c: {v: i for i, v in enumerate(sorted(X[c].astype(str).unique()))}
            for c in CATEGORICAL}


def encode(X: pd.DataFrame, category_maps: dict) -> pd.DataFrame:
    """Integer-code categoricals with the training maps. Unseen categories -> -1."""
    out = X[FEATURES].copy()
    for c in CATEGORICAL:
        out[c] = out[c].astype(str).map(category_maps[c]).fillna(-1).astype(int)
    for c in FEATURES:
        if c not in CATEGORICAL:
            out[c] = pd.to_numeric(out[c])
    return out


# ----------------------------------------------------------------------
# Scoring and explanation
# ----------------------------------------------------------------------

def load_bundle(path: str = MODEL_PATH) -> dict:
    import joblib

    if not os.path.exists(path):
        raise FileNotFoundError(f"No trained model at {path}. Run: python scripts/train_model.py")
    return joblib.load(path)


def file_sha256(path: str = MODEL_PATH) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


class Scorer:
    """Scores applicants and returns per-feature SHAP contributions (log-odds of approval)."""

    def __init__(self, bundle: dict):
        import shap

        self.model = bundle["model"]
        self.category_maps = bundle["category_maps"]
        self.threshold = bundle.get("threshold", 0.5)
        self.explainer = shap.TreeExplainer(self.model)

    def score(self, applicants: pd.DataFrame) -> list[dict]:
        """`applicants` must contain FEATURES plus age and personal_status."""
        X = encode(applicants, self.category_maps)
        proba = self.model.predict_proba(X)[:, 1]
        shap_matrix = np.asarray(self.explainer.shap_values(X))
        results = []
        for i, (_, raw) in enumerate(applicants.iterrows()):
            features = {k: (raw[k].item() if hasattr(raw[k], "item") else raw[k])
                        for k in FEATURES}
            results.append({
                "input_features": features,
                "score": float(proba[i]),
                "prediction": int(proba[i] >= self.threshold),
                "shap_values": {f: round(float(shap_matrix[i, j]), 6)
                                for j, f in enumerate(FEATURES)},
                "age_group": age_group(raw["age"]),
                "sex": sex_from_personal_status(raw["personal_status"]),
            })
        return results
