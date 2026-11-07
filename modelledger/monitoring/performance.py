"""
ModelLedger: ongoing model performance monitoring.

  Gini   2 x AUC - 1, on decisions whose outcome is known (discrimination)
  KS     maximum gap between the score distributions of repaid and defaulted
  PSI    Population Stability Index of live scores against the development
         (out-of-fold training) score distribution stored with the model

PSI rule of thumb: below 0.10 stable, 0.10-0.25 investigate, above 0.25
significant shift. These are conventions, not regulatory limits.
"""

import json
import os
from datetime import date

import numpy as np
from sklearn.metrics import roc_auc_score

from modelledger.core.database import LedgerDB

MIN_OUTCOMES = 30


def gini(y, score) -> float:
    return 2 * roc_auc_score(y, score) - 1


def ks_statistic(y, score) -> float:
    y, score = np.asarray(y), np.asarray(score)
    order = np.argsort(score)
    ys = y[order]
    cum_pos = np.cumsum(ys) / ys.sum()
    cum_neg = np.cumsum(1 - ys) / (1 - ys).sum()
    return float(np.max(np.abs(cum_pos - cum_neg)))


def psi(reference, actual, bins: int = 10) -> float:
    reference, actual = np.asarray(reference, float), np.asarray(actual, float)
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.clip(np.histogram(reference, edges)[0] / len(reference), 1e-6, None)
    a = np.clip(np.histogram(actual, edges)[0] / len(actual), 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


def psi_status(value: float | None) -> str | None:
    if value is None:
        return None
    return "stable" if value < 0.10 else "investigate" if value < 0.25 else "significant shift"


def load_reference(metadata_path: str = os.path.join("models", "model_metadata.json")):
    if not os.path.exists(metadata_path):
        return None
    with open(metadata_path) as f:
        return json.load(f).get("psi_reference_scores")


class PerformanceMonitor:
    def __init__(self, db: LedgerDB, reference_scores=None):
        self.db = db
        self.reference = reference_scores if reference_scores is not None else load_reference()

    def compute(self, date_from: str | None = None, date_to: str | None = None) -> dict:
        df = self.db.get_decisions_df(date_from=date_from, date_to=date_to)
        out = {"record_count": int(len(df)), "date_from": date_from, "date_to": date_to}

        known = df.dropna(subset=["true_label"])
        out["n_with_outcome"] = int(len(known))
        if len(known) >= MIN_OUTCOMES and known["true_label"].nunique() == 2:
            out["gini"] = round(gini(known["true_label"], known["score"]), 4)
            out["ks"] = round(ks_statistic(known["true_label"].astype(int), known["score"]), 4)
        else:
            out["gini"] = out["ks"] = None
            out["discrimination_note"] = (f"Needs at least {MIN_OUTCOMES} decisions with "
                                          "known outcomes, including both repaid and defaulted")

        if self.reference is not None and len(df) >= MIN_OUTCOMES:
            out["psi"] = round(psi(self.reference, df["score"]), 4)
        else:
            out["psi"] = None
        out["psi_status"] = psi_status(out["psi"])
        return out

    def record_snapshot(self, snapshot_date: str | None = None, **kwargs) -> dict:
        m = self.compute(**kwargs)
        self.db.insert_performance_record({
            "date": snapshot_date or date.today().isoformat(),
            "n_with_outcome": m["n_with_outcome"], "gini": m["gini"],
            "ks": m["ks"], "psi": m["psi"],
        })
        return m
