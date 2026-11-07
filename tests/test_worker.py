"""Queue worker: events published by the model service land in the ledger."""

import json

from modelledger.core.fingerprinter import DecisionFingerprinter
from modelledger.worker import handle_event


def test_worker_logs_queued_event(db, tmp_path, monkeypatch):
    monkeypatch.setenv("ANCHOR_DIR", str(tmp_path / "anchors"))
    event = {"timestamp": "2026-11-01T10:00:00+00:00", "model_version": "m", "model_hash": "h",
             "input_features": {"duration": 24}, "prediction": 1, "score": 0.81,
             "shap_values": {"duration": 0.1}, "age_group": "26-35", "sex": "female"}
    rec = handle_event(db, json.dumps(event))
    stored = db.get_record_by_id(rec["decision_id"])
    assert stored["sex"] == "female" and stored["timestamp"] == event["timestamp"]
    assert DecisionFingerprinter(db, "", "").verify_chain()["intact"]
