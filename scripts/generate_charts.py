#!/usr/bin/env python3
"""
Plot the monitoring history stored in the ledger database.

    python scripts/generate_charts.py

Uses ONLY real snapshots from fairness_history and performance_history
(written by the daily job or the demo seeder). If there is no history it
stops; it never invents data.
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modelledger.core.database import LedgerDB  # noqa: E402

DB_PATH = os.path.join("data", "modelledger.db")
OUTPUT_PATH = os.path.join("assets", "monitoring_history.png")


def main():
    if not os.path.exists(DB_PATH):
        sys.exit("No ledger database. Run scripts/seed_demo_data.py first.")
    db = LedgerDB(DB_PATH)
    fair = pd.DataFrame(db.get_fairness_history())
    perf = pd.DataFrame(db.get_performance_history())
    db.close()
    if fair.empty or perf.empty:
        sys.exit("No monitoring history recorded yet; nothing to plot.")

    fair["date"] = pd.to_datetime(fair["date"])
    perf["date"] = pd.to_datetime(perf["date"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2), sharex=True)
    styles = {"sex": ("#1f77b4", "Sex"), "age_group": ("#d62728", "Age group")}
    for col, (colour, label) in styles.items():
        d = fair[fair["group_column"] == col]
        ax1.plot(d["date"], d["parity_gap"], color=colour, lw=2, label=f"{label}: approval-rate gap")
        ax1.plot(d["date"], d["eo_gap"], color=colour, lw=1.5, ls="--",
                 label=f"{label}: equal-opportunity gap")
    ax1.axhline(0.10, color="grey", ls=":", lw=1, label="Review threshold (illustrative)")
    ax1.set_title("Fairness gaps (cumulative decisions to date)")
    ax1.set_ylabel("Gap between highest and lowest group")
    ax1.set_ylim(0, None)
    ax1.text(0.02, 0.95, "Gaps appear once at least two groups\nhave 30 or more decisions",
             transform=ax1.transAxes, fontsize=8, color="#555", va="top")
    ax1.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, frameon=False)

    ax2.plot(perf["date"], perf["gini"], color="#2ca02c", lw=2, label="Gini")
    ax2.plot(perf["date"], perf["psi"], color="#9467bd", lw=2, label="PSI vs development sample")
    ax2.axhline(0.10, color="#9467bd", ls=":", lw=1, label="PSI 0.10 (investigate)")
    ax2.set_title("Model performance and stability")
    ax2.text(0.98, 0.95, "Early PSI is noisy: few decisions logged yet",
             transform=ax2.transAxes, fontsize=8, color="#555", va="top", ha="right")
    ax2.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3, frameon=False)
    ax2.set_ylim(0, None)

    for ax in (ax1, ax2):
        ax.grid(alpha=0.3)
        ax.tick_params(axis="x", rotation=30, labelsize=8)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("ModelLedger demo: 300 held-out German Credit applicants over 30 days "
                 "(groups under 30 cases excluded)", fontsize=10)
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=150)
    print(f"Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
