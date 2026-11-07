"""Decision and outcome chains."""

import pytest

from conftest import make_decisions
from modelledger.core.database import GENESIS_HASH
from modelledger.core.fingerprinter import hash_decision


def test_genesis_and_links(fp, db):
    recs = make_decisions(fp, 5)
    assert recs[0]["previous_hash"] == GENESIS_HASH
    for prev, cur in zip(recs, recs[1:]):
        assert cur["previous_hash"] == prev["decision_hash"]
    assert fp.verify_chain()["intact"]


def test_hash_is_deterministic_and_covers_every_stored_field(fp, db):
    rec = db.get_all_records()[0] if make_decisions(fp, 1) else None
    assert hash_decision(rec) == rec["decision_hash"]
    for field, new in [("prediction", 1 - rec["prediction"]), ("score", 0.123),
                       ("sex", "other"), ("age_group", "56+"), ("timestamp", "2000-01-01"),
                       ("model_hash", "b" * 64), ("input_features", "{}"), ("shap_values", "{}")]:
        assert hash_decision({**rec, field: new}) != rec["decision_hash"], field


@pytest.mark.parametrize("sql", [
    "UPDATE audit_ledger SET prediction = 1 - prediction WHERE id = 4",
    "UPDATE audit_ledger SET sex = 'male' WHERE id = 4",          # protected attribute
    "UPDATE audit_ledger SET age_group = '56+' WHERE id = 4",     # protected attribute
    "UPDATE audit_ledger SET score = 0.99 WHERE id = 4",
    "DELETE FROM audit_ledger WHERE id = 4",                      # deletion mid-chain
])
def test_any_edit_breaks_the_chain(fp, db, sql):
    make_decisions(fp, 8)
    db.raw_execute(sql)
    assert not fp.verify_chain()["intact"]


def test_backfilled_timestamp_is_kept_and_hashed(fp):
    rec = fp.log_decision({"x": 1}, 1, 0.8, {}, timestamp="2026-01-01T09:00:00+00:00")
    assert rec["timestamp"] == "2026-01-01T09:00:00+00:00"
    assert fp.verify_chain()["intact"]


def test_input_validation(fp):
    with pytest.raises(ValueError):
        fp.log_decision({}, 2, 0.5, {})
    with pytest.raises(ValueError):
        fp.log_decision({}, 1, 1.5, {})


def test_outcomes_are_events_not_overwrites(fp, db):
    rec = make_decisions(fp, 1)[0]
    fp.log_outcome(rec["decision_id"], 0)
    fp.log_outcome(rec["decision_id"], 1)          # a correction is a new event
    assert db.get_outcome_count() == 2
    assert db.get_latest_outcomes()[rec["decision_id"]] == 1
    assert fp.verify_outcomes()["intact"]
    assert fp.verify_chain()["intact"]             # decisions untouched


def test_editing_an_outcome_is_detected(fp, db):
    recs = make_decisions(fp, 3)
    for r in recs:
        fp.log_outcome(r["decision_id"], 1)
    db.raw_execute("UPDATE outcome_log SET true_label = 0 WHERE id = 2")
    assert not fp.verify_outcomes()["intact"]


def test_outcome_for_unknown_decision_is_rejected(fp):
    with pytest.raises(KeyError):
        fp.log_outcome("no-such-decision", 1)
