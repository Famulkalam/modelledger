"""
ModelLedger: fairness monitoring of logged credit decisions.

Metrics (prediction 1 = approve, true_label 1 = repaid)
  Demographic parity gap   max - min approval rate across groups
  Equal opportunity gap    max - min approval rate among applicants who went on
                           to repay (true positive rate), across groups

Small groups make these gaps noisy, so groups with fewer than
`min_group_size` decisions (or repaid applicants, for equal opportunity) are
reported but left out of the gap.

On choosing a metric: parity and equal opportunity can disagree when groups
have different repayment rates, and well-known results (Kleinberg, Mullainathan
and Raghavan 2016; Chouldechova 2017) show that calibration and equal error
rates cannot generally all hold at once in that situation. Which gap matters,
and what threshold triggers review, is a policy decision for risk, compliance
and legal teams. The 0.10 default here is illustrative.

Computing metrics has no side effects. Snapshots are written to history only
by record_snapshot() (called by the daily job and the demo seeder).
"""

import logging
from datetime import date

import pandas as pd

from modelledger.core.database import LedgerDB
from modelledger.utils.alerts import dispatch_alerts

logger = logging.getLogger(__name__)

GROUP_COLUMNS = ("sex", "age_group")


class FairnessMonitor:
    PARITY_THRESHOLD = 0.10
    EO_THRESHOLD = 0.10
    MIN_GROUP_SIZE = 30

    def __init__(self, db: LedgerDB, min_group_size: int | None = None):
        self.db = db
        if min_group_size is not None:
            self.MIN_GROUP_SIZE = min_group_size

    def compute_all_metrics(self, date_from: str | None = None, date_to: str | None = None,
                            group_column: str = "sex") -> dict:
        if group_column not in GROUP_COLUMNS:
            raise ValueError(f"group_column must be one of {GROUP_COLUMNS}")
        df = self.db.get_decisions_df(date_from=date_from, date_to=date_to)
        base = {"group_column": group_column, "record_count": int(len(df)),
                "min_group_size": self.MIN_GROUP_SIZE,
                "date_from": date_from, "date_to": date_to}
        if df.empty or df[group_column].isna().all():
            reason = "No decisions in range" if df.empty else f"No '{group_column}' values"
            return {**base,
                    "demographic_parity": {"available": False, "reason": reason},
                    "equal_opportunity": {"available": False, "reason": reason}}
        return {**base,
                "demographic_parity": self._gap(df, group_column, self.PARITY_THRESHOLD,
                                                "approval_rate"),
                "equal_opportunity": self._equal_opportunity(df, group_column)}

    def _gap(self, df: pd.DataFrame, group_col: str, threshold: float, rate_name: str) -> dict:
        stats = df.groupby(group_col)["prediction"].agg(["mean", "count"])
        groups = {str(g): {rate_name: round(float(r["mean"]), 4), "n": int(r["count"]),
                           "included": int(r["count"]) >= self.MIN_GROUP_SIZE}
                  for g, r in stats.iterrows()}
        included = [v[rate_name] for v in groups.values() if v["included"]]
        if len(included) < 2:
            return {"available": False, "groups": groups,
                    "reason": f"Fewer than two groups with at least {self.MIN_GROUP_SIZE} cases"}
        gap = max(included) - min(included)
        return {"available": True, "groups": groups, "max_disparity": round(gap, 4),
                "threshold": threshold, "breached": gap > threshold}

    def _equal_opportunity(self, df: pd.DataFrame, group_col: str) -> dict:
        repaid = df[df["true_label"] == 1]
        if repaid.empty:
            return {"available": False,
                    "reason": "Needs outcomes. Record them with PATCH /decisions/{id}/outcome."}
        return self._gap(repaid, group_col, self.EO_THRESHOLD, "true_positive_rate")

    def record_snapshot(self, snapshot_date: str | None = None,
                        date_from: str | None = None, date_to: str | None = None) -> list[dict]:
        """Compute metrics for every group column, store them, and raise alerts."""
        snapshot_date = snapshot_date or date.today().isoformat()
        results = []
        for col in GROUP_COLUMNS:
            m = self.compute_all_metrics(date_from=date_from, date_to=date_to, group_column=col)
            dp, eo = m["demographic_parity"], m["equal_opportunity"]
            self.db.insert_fairness_record({
                "date": snapshot_date, "group_column": col, "record_count": m["record_count"],
                "parity_gap": dp.get("max_disparity"), "eo_gap": eo.get("max_disparity"),
                "parity_breached": int(dp.get("breached", False)) if dp.get("available") else None,
                "eo_breached": int(eo.get("breached", False)) if eo.get("available") else None,
            })
            self._alert(m)
            results.append(m)
        return results

    def _alert(self, m: dict) -> None:
        alerts = []
        for key, label in (("demographic_parity", "approval rate"),
                           ("equal_opportunity", "true positive rate")):
            r = m[key]
            if r.get("available") and r.get("breached"):
                alerts.append({
                    "severity": "HIGH", "metric": f"{key}:{m['group_column']}",
                    "message": (f"{label} gap by {m['group_column']} is {r['max_disparity']:.3f}, "
                                f"above the {r['threshold']} review threshold"),
                })
        if alerts:
            dispatch_alerts(alerts)
