"""API behaviour, including the attacks and the rate limit."""

import os

from conftest import HEADERS
from modelledger.api import server
from modelledger.core.fingerprinter import hash_decision


def _decision(i, sex="female"):
    return {"input_features": {"duration": 12 + i}, "prediction": i % 2,
            "score": 0.4 + (i % 5) / 10, "shap_values": {"duration": -0.1},
            "age_group": "26-35", "sex": sex}


def test_auth_required(client):
    assert client.get("/verify").status_code == 422          # header missing
    assert client.get("/verify", headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.get("/health").status_code == 200


def test_decision_round_trip(client):
    r = client.post("/decisions", json=_decision(1), headers=HEADERS)
    assert r.status_code == 201
    got = client.get(f"/decisions/{r.json()['decision_id']}", headers=HEADERS).json()
    assert got["input_features"] == {"duration": 13}
    assert got["sex"] == "female"


def test_outcome_is_logged_as_event(client):
    did = client.post("/decisions", json=_decision(1), headers=HEADERS).json()["decision_id"]
    r = client.patch(f"/decisions/{did}/outcome", json={"true_label": 1}, headers=HEADERS)
    assert r.status_code == 200 and r.json()["event_hash"]
    events = client.get(f"/decisions/{did}", headers=HEADERS).json()["outcome_events"]
    assert len(events) == 1
    assert client.patch("/decisions/nope/outcome", json={"true_label": 1},
                        headers=HEADERS).status_code == 404


def test_anchors_written_at_interval(client):
    for i in range(5):                     # ANCHOR_INTERVAL=5 in the fixture
        r = client.post("/decisions", json=_decision(i), headers=HEADERS)
    assert r.json()["anchor_written"] is True
    assert client.get("/anchors", headers=HEADERS).json()["checked"] == 1


def test_verify_catches_full_rewrite_attack(client):
    for i in range(5):
        client.post("/decisions", json=_decision(i), headers=HEADERS)
    assert client.get("/verify", headers=HEADERS).status_code == 200
    db = server.get_db()
    db.raw_execute("UPDATE audit_ledger SET prediction = 1 - prediction WHERE id = 2")
    previous = "0" * 64
    for rec in db.get_all_records():
        rec["previous_hash"] = previous
        h = hash_decision(rec)
        db.raw_execute("UPDATE audit_ledger SET previous_hash=?, decision_hash=? WHERE id=?",
                       (previous, h, rec["id"]))
        previous = h
    r = client.get("/verify", headers=HEADERS)
    assert r.status_code == 409
    body = r.json()
    assert body["decisions"]["intact"] is True and body["anchors"]["intact"] is False


def test_rate_limit_is_enforced(client, monkeypatch):
    monkeypatch.setenv("RATE_LIMIT", "3/minute")
    codes = [client.post("/decisions", json=_decision(i), headers=HEADERS).status_code
             for i in range(5)]
    assert codes == [201, 201, 201, 429, 429]


def test_fairness_endpoint_validates_group_and_has_no_side_effects(client):
    for i in range(40):
        client.post("/decisions", json=_decision(i, "female" if i % 2 else "male"),
                    headers=HEADERS)
    assert client.get("/fairness?group_by=postcode", headers=HEADERS).status_code == 422
    r = client.get("/fairness?group_by=sex", headers=HEADERS)
    assert r.status_code == 200
    assert server.get_db().get_fairness_history() == []
    assert client.post("/monitoring/snapshot", headers=HEADERS).status_code == 201
    assert len(server.get_db().get_fairness_history()) == 2


def test_report_job(client):
    for i in range(3):
        client.post("/decisions", json=_decision(i), headers=HEADERS)
    job = client.post("/report", headers=HEADERS).json()["job_id"]
    server.generate_report_sync(job)      # make the background job deterministic
    r = client.get(f"/reports/{job}", headers=HEADERS)
    assert r.status_code == 200 and r.headers["content-type"] == "application/pdf"
    assert os.path.getsize(server.get_db().get_report_job(job)["output_path"]) > 2000
