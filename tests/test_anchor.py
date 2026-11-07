"""Signed anchors: the defence against full-chain rewrites."""

import json
import os

from conftest import PUBLIC_HEX, make_decisions
from modelledger.core.anchor import (
    generate_keypair, load_anchor_files, verify_all_anchors, write_anchor,
)
from modelledger.core.fingerprinter import hash_decision, hash_outcome


def _recompute_decisions(db):
    """What an insider with DB access but no signing key can do."""
    previous = "0" * 64
    for r in db.get_all_records():
        r["previous_hash"] = previous
        h = hash_decision(r)
        db.raw_execute("UPDATE audit_ledger SET previous_hash=?, decision_hash=? WHERE id=?",
                       (previous, h, r["id"]))
        previous = h


def test_clean_ledger_verifies(fp, db, anchor_dir):
    make_decisions(fp, 10)
    write_anchor(db, anchor_dir)
    result = verify_all_anchors(db, anchor_dir)
    assert result["intact"] and result["checked"] == 1


def test_anchor_still_valid_after_ledger_grows(fp, db, anchor_dir):
    """Regression: the old check compared against the whole current ledger and failed."""
    make_decisions(fp, 10)
    write_anchor(db, anchor_dir)
    make_decisions(fp, 15, start=10)
    result = verify_all_anchors(db, anchor_dir)
    assert result["intact"]
    assert result["decisions_after_last_anchor"] == 15


def test_full_rewrite_attack_is_caught(fp, db, anchor_dir):
    """Regression: edit + recompute every hash used to pass every check."""
    make_decisions(fp, 10)
    write_anchor(db, anchor_dir)
    db.raw_execute("UPDATE audit_ledger SET prediction = 1 - prediction WHERE id = 3")
    _recompute_decisions(db)
    assert fp.verify_chain()["intact"]                    # the chain alone is fooled
    assert not verify_all_anchors(db, anchor_dir)["intact"]  # the anchor is not


def test_outcome_rewrite_with_recomputed_hashes_is_caught(fp, db, anchor_dir):
    recs = make_decisions(fp, 4)
    for r in recs:
        fp.log_outcome(r["decision_id"], 1)
    write_anchor(db, anchor_dir)
    db.raw_execute("UPDATE outcome_log SET true_label = 0 WHERE id = 2")
    previous = "0" * 64
    for e in db.get_all_outcomes():
        e["previous_hash"] = previous
        h = hash_outcome(e)
        db.raw_execute("UPDATE outcome_log SET previous_hash=?, event_hash=? WHERE id=?",
                       (previous, h, e["id"]))
        previous = h
    assert fp.verify_outcomes()["intact"]
    assert not verify_all_anchors(db, anchor_dir)["intact"]


def test_deleting_records_is_caught(fp, db, anchor_dir):
    make_decisions(fp, 10)
    write_anchor(db, anchor_dir)
    db.raw_execute("DELETE FROM audit_ledger WHERE id > 7")
    result = verify_all_anchors(db, anchor_dir)
    assert not result["intact"]
    assert "deleted" in result["results"][0]["problems"][0]


def test_edited_anchor_file_fails_signature(fp, db, anchor_dir):
    make_decisions(fp, 5)
    write_anchor(db, anchor_dir)
    path = os.path.join(anchor_dir, os.listdir(anchor_dir)[0])
    with open(path) as f:
        anchor = json.load(f)
    anchor["record_count"] = 3
    with open(path, "w") as f:
        json.dump(anchor, f)
    result = verify_all_anchors(db, anchor_dir)
    assert not result["intact"]
    assert any("signature" in p for p in result["results"][0]["problems"])


def test_anchor_signed_with_another_key_is_rejected(fp, db, anchor_dir):
    make_decisions(fp, 5)
    other_private, _ = generate_keypair()
    write_anchor(db, anchor_dir, private_hex=other_private)   # forged by someone else
    assert not verify_all_anchors(db, anchor_dir, public_hex=PUBLIC_HEX)["intact"]


def test_verification_uses_files_not_database_table(fp, db, anchor_dir):
    make_decisions(fp, 5)
    write_anchor(db, anchor_dir)
    db.raw_execute("DELETE FROM anchor_log")       # insider wipes the DB copy
    assert len(load_anchor_files(anchor_dir)) == 1
    assert verify_all_anchors(db, anchor_dir)["checked"] == 1
