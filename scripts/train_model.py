#!/usr/bin/env python3
"""
Train the demo credit model (XGBoost on German Credit) and record honest metrics.

    python scripts/train_model.py

- Stratified 70/30 split. The 30% test set is never used for training or
  tuning; the demo ledger is later seeded from those test applicants only.
- 5-fold cross-validation on the training set gives out-of-fold scores, used
  both as a performance estimate and as the PSI reference distribution.
- personal_status (encodes sex) and foreign_worker are NOT model inputs.

Writes models/credit_model.joblib and models/model_metadata.json.
"""

import json
import os
import sys

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from xgboost import XGBClassifier

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modelledger.model.credit_model import (  # noqa: E402
    EXCLUDED_FROM_MODEL, FEATURES, METADATA_PATH, MODEL_PATH, RANDOM_STATE,
    encode, file_sha256, fit_category_maps, load_credit_g, split,
)

MODEL_VERSION = "xgboost-credit-v2"
PARAMS = dict(
    n_estimators=300, max_depth=3, learning_rate=0.03, subsample=0.8,
    colsample_bytree=0.8, min_child_weight=5, reg_lambda=1.0,
    eval_metric="logloss", random_state=RANDOM_STATE,
)


def gini(y, p) -> float:
    return 2 * roc_auc_score(y, p) - 1


def main():
    df = load_credit_g()
    X_train, X_test, y_train, y_test = split(df)
    maps = fit_category_maps(X_train)
    Xtr, Xte = encode(X_train, maps), encode(X_test, maps)

    # Out-of-fold scores on the training set (honest estimate, PSI reference)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    oof = cross_val_predict(XGBClassifier(**PARAMS), Xtr, y_train, cv=cv,
                            method="predict_proba")[:, 1]

    model = XGBClassifier(**PARAMS).fit(Xtr, y_train)
    p_train = model.predict_proba(Xtr)[:, 1]
    p_test = model.predict_proba(Xte)[:, 1]

    metrics = {
        "train_auc": roc_auc_score(y_train, p_train),
        "cv_auc": roc_auc_score(y_train, oof),
        "test_auc": roc_auc_score(y_test, p_test),
        "test_gini": gini(y_test, p_test),
        "test_accuracy": accuracy_score(y_test, (p_test >= 0.5).astype(int)),
        "test_brier": brier_score_loss(y_test, p_test),
        "majority_class_accuracy": max(y_test.mean(), 1 - y_test.mean()),
    }
    metrics = {k: round(float(v), 4) for k, v in metrics.items()}

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump({"model": model, "category_maps": maps, "threshold": 0.5,
                 "features": FEATURES, "model_version": MODEL_VERSION}, MODEL_PATH)

    # PSI reference: decile edges and shares of the out-of-fold score distribution
    edges = np.unique(np.quantile(oof, np.linspace(0, 1, 11)))
    metadata = {
        "model_version": MODEL_VERSION,
        "model_hash": file_sha256(MODEL_PATH),
        "dataset": "German Credit (OpenML credit-g, 1,000 applicants)",
        "features": FEATURES,
        "excluded_from_model": EXCLUDED_FROM_MODEL,
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "params": {k: v for k, v in PARAMS.items() if k != "eval_metric"},
        "metrics": metrics,
        "psi_reference_scores": [round(float(s), 6) for s in oof],
        "psi_reference_bin_edges": [round(float(e), 6) for e in edges],
    }
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Model saved to {MODEL_PATH} (sha256 {metadata['model_hash'][:16]}...)")
    for k, v in metrics.items():
        print(f"  {k:24s} {v}")


if __name__ == "__main__":
    main()
