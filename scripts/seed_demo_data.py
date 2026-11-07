#!/usr/bin/env python3
"""
Seed the demo ledger with REAL decisions.

    python scripts/seed_demo_data.py                     # clean ledger
    python scripts/seed_demo_data.py --attack naive      # edit one decision via SQL
    python scripts/seed_demo_data.py --attack recompute  # edit + recompute every hash
    python scripts/seed_demo_data.py --attack outcome    # rewrite a repayment outcome

What it does
  1. Takes the 300 held-out TEST applicants from German Credit (never seen in
     training), scores them with the trained model and real SHAP values, and
     logs them as decisions spread over the last 30 days.
  2. Sex and age group come from each applicant's own record.
  3. Logs each applicant's actual outcome (repaid / defaulted) as an outcome
     event, and signs anchors along the way.
  4. Records a daily fairness and performance snapshot (cumulative to that day).

Demo simplification: in real lending, outcomes arrive months after the
decision. Here the dataset's known outcomes are back-filled, so every daily
snapshot uses all outcomes. That is stated in the README and the report.
"""

import argparse
import os
import shutil
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from modelledger.core.anchor import maybe_write_anchor, verify_all_anchors, write_anchor  # noqa: E402
from modelledger.core.database import LedgerDB  # noqa: E402
from modelledger.core.fingerprinter import DecisionFingerprinter, hash_decision  # noqa: E402
from modelledger.fairness.monitor import FairnessMonitor  # noqa: E402
from modelledger.model.credit_model import (  # noqa: E402
    Scorer, file_sha256, load_bundle, load_credit_g, split,
)
from modelledger.monitoring.performance import PerformanceMonitor  # noqa: E402

DB_PATH = os.path.join("data", "modelledger.db")
ANCHOR_DIR = os.environ.get("ANCHOR_DIR", "anchors")
ANCHOR_INTERVAL = int(os.environ.get("ANCHOR_INTERVAL", "100"))


def seed(days: int = 30) -> tuple[LedgerDB, DecisionFingerprinter]:
    if not os.environ.get("ANCHOR_SIGNING_KEY"):
        sys.exit("ANCHOR_SIGNING_KEY is not set. Run `python scripts/generate_keys.py` first.")

    bundle = load_bundle()
    scorer = Scorer(bundle)
    df = load_credit_g()
    _, X_test, _, y_test = split(df)
    X_test = X_test.sample(frac=1.0, random_state=7)          # random arrival order
    y_test = y_test.loc[X_test.index]

    for path in (DB_PATH, DB_PATH + "-wal", DB_PATH + "-shm"):
        if os.path.exists(path):
            os.remove(path)
    shutil.rmtree(ANCHOR_DIR, ignore_errors=True)

    db = LedgerDB(DB_PATH)
    fp = DecisionFingerprinter(db, bundle["model_version"], file_sha256())

    start = (datetime.now(timezone.utc) - timedelta(days=days)).replace(
        hour=9, minute=0, second=0, microsecond=0)
    n = len(X_test)
    scored = scorer.score(X_test)
    decision_ids = []
    for i, s in enumerate(scored):
        ts = start + timedelta(days=days * i / n) + timedelta(minutes=7 * (i % 60))
        rec = fp.log_decision(timestamp=ts.isoformat(), **s)
        decision_ids.append(rec["decision_id"])
        maybe_write_anchor(db, ANCHOR_INTERVAL, ANCHOR_DIR)
    print(f"Logged {n} decisions from held-out test applicants over {days} days")

    for decision_id, label in zip(decision_ids, y_test.tolist()):
        fp.log_outcome(decision_id, int(label))
    write_anchor(db, ANCHOR_DIR)
    print(f"Logged {len(decision_ids)} outcome events; {len(db.get_all_anchors())} anchors signed")

    fairness, performance = FairnessMonitor(db), PerformanceMonitor(db)
    for d in range(days + 1):
        day = (start + timedelta(days=d)).date().isoformat()
        fairness.record_snapshot(snapshot_date=day, date_to=day)
        performance.record_snapshot(snapshot_date=day, date_to=day)
    print(f"Recorded {days + 1} daily fairness and performance snapshots")
    return db, fp


def attack(db: LedgerDB, kind: str, position: int = 150) -> None:
    print(f"\nSimulating attack '{kind}' on record #{position}")
    if kind in ("naive", "recompute"):
        db.raw_execute("UPDATE audit_ledger SET prediction = 1 - prediction WHERE id = ?",
                       (position,))
    if kind == "recompute":
        previous = "0" * 64
        for r in db.get_all_records():
            r["previous_hash"] = previous
            h = hash_decision(r)
            db.raw_execute("UPDATE audit_ledger SET previous_hash=?, decision_hash=? WHERE id=?",
                           (previous, h, r["id"]))
            previous = h
        print("  Every hash recomputed, so the chain itself looks intact")
    if kind == "outcome":
        db.raw_execute("UPDATE outcome_log SET true_label = 1 - true_label WHERE id = ?",
                       (position,))


def report(db: LedgerDB, fp: DecisionFingerprinter) -> None:
    chain, outcomes = fp.verify_chain(), fp.verify_outcomes()
    anchors = verify_all_anchors(db, ANCHOR_DIR)
    print("\nVerification")
    print(f"  decision chain intact: {chain['intact']}")
    print(f"  outcome chain intact:  {outcomes['intact']}")
    print(f"  anchors intact:        {anchors['intact']} ({anchors['checked']} checked)")
    for r in anchors["results"]:
        for p in r["problems"]:
            print(f"    anchor @ {r['record_count']}: {p}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--attack", choices=["naive", "recompute", "outcome"])
    args = parser.parse_args()
    db, fp = seed(args.days)
    if args.attack:
        attack(db, args.attack)
    report(db, fp)
    db.close()


if __name__ == "__main__":
    main()
