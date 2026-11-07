"""Fairness and performance metrics."""

import numpy as np
import pytest

from modelledger.fairness.monitor import FairnessMonitor
from modelledger.monitoring.performance import PerformanceMonitor, gini, ks_statistic, psi


def _log(fp, n, sex, approve_share, repaid_share, score=0.7):
    """n decisions for one group with given approval and repayment shares."""
    for i in range(n):
        rec = fp.log_decision({"i": i}, int(i < approve_share * n), score, {},
                              age_group="26-35", sex=sex)
        fp.log_outcome(rec["decision_id"], int(i < repaid_share * n))


def test_parity_gap_is_computed_on_known_numbers(fp, db):
    _log(fp, 40, "female", 0.5, 0.5)
    _log(fp, 40, "male", 0.8, 0.5)
    m = FairnessMonitor(db).compute_all_metrics(group_column="sex")
    dp = m["demographic_parity"]
    assert dp["groups"]["female"]["approval_rate"] == 0.5
    assert dp["groups"]["male"]["approval_rate"] == 0.8
    assert dp["max_disparity"] == pytest.approx(0.3)
    assert dp["breached"]


def test_equal_opportunity_uses_repaid_applicants_only(fp, db):
    _log(fp, 60, "female", 0.5, 0.5)   # approved = first half = the repaid ones -> TPR 1.0
    _log(fp, 60, "male", 0.5, 1.0)     # half of repaid approved -> TPR 0.5
    eo = FairnessMonitor(db).compute_all_metrics(group_column="sex")["equal_opportunity"]
    assert eo["groups"]["female"]["true_positive_rate"] == 1.0
    assert eo["groups"]["male"]["true_positive_rate"] == 0.5


def test_small_groups_are_excluded_from_the_gap(fp, db):
    _log(fp, 50, "female", 0.5, 0.5)
    _log(fp, 5, "male", 1.0, 0.5)        # tiny group with an extreme rate
    dp = FairnessMonitor(db).compute_all_metrics(group_column="sex")["demographic_parity"]
    assert dp["available"] is False
    assert dp["groups"]["male"]["included"] is False


def test_computing_metrics_has_no_side_effects(fp, db):
    _log(fp, 40, "female", 0.5, 0.5)
    FairnessMonitor(db).compute_all_metrics()
    assert db.get_fairness_history() == []
    FairnessMonitor(db).record_snapshot("2026-01-01")
    assert len(db.get_fairness_history()) == 2        # one row per group column


def test_unknown_group_column_is_rejected(db):
    with pytest.raises(ValueError):
        FairnessMonitor(db).compute_all_metrics(group_column="postcode")


def test_gini_ks_psi_on_known_inputs():
    y = np.array([0, 0, 1, 1])
    perfect = np.array([0.1, 0.2, 0.8, 0.9])
    assert gini(y, perfect) == pytest.approx(1.0)
    assert ks_statistic(y, perfect) == pytest.approx(1.0)
    x = np.random.default_rng(0).uniform(size=2000)
    assert psi(x, x) == pytest.approx(0.0, abs=1e-9)
    assert psi(x, np.clip(x + 0.4, 0, 1)) > 0.25


def test_performance_needs_enough_outcomes(fp, db):
    _log(fp, 10, "female", 0.5, 0.5)
    p = PerformanceMonitor(db, reference_scores=[0.5] * 100).compute()
    assert p["gini"] is None


def test_performance_monitor_end_to_end(fp, db):
    rng = np.random.default_rng(1)
    for _ in range(200):
        repaid = int(rng.random() < 0.7)
        score = float(np.clip(rng.normal(0.65 if repaid else 0.45, 0.15), 0, 1))
        rec = fp.log_decision({}, int(score > 0.5), score, {}, sex="male", age_group="26-35")
        fp.log_outcome(rec["decision_id"], repaid)
    reference = np.clip(rng.normal(0.6, 0.17, 1000), 0, 1)
    p = PerformanceMonitor(db, reference_scores=reference).compute()
    assert 0.3 < p["gini"] < 0.9
    assert p["psi_status"] == "stable"
