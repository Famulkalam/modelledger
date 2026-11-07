"""
ModelLedger: hash-chained decision and outcome records.

Every credit decision becomes a record whose SHA-256 hash covers ALL of its
stored fields, including the protected-characteristic fields used for
fairness monitoring, plus the hash of the previous record. Changing any field
of any record changes its hash and breaks the chain.

Outcomes (whether a borrower actually repaid) arrive later. They are never
written back into the decision record. Each outcome is a new event in a
second hash chain, so a corrected outcome is a new event, and the history of
outcome changes is itself tamper-evident.

What hash chaining alone does NOT stop: someone with write access to the
database can edit a record and recompute every later hash, because SHA-256
needs no secret. Signed anchors (core/anchor.py) close that gap for every
record written before the most recent anchor.
"""

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone

from modelledger.core.database import GENESIS_HASH, LedgerDB, dumps

logger = logging.getLogger(__name__)

DECISION_HASH_FIELDS = (
    "decision_id", "timestamp", "model_version", "model_hash", "input_features",
    "prediction", "score", "shap_values", "age_group", "sex", "previous_hash",
)
OUTCOME_HASH_FIELDS = ("event_id", "decision_id", "true_label", "timestamp", "previous_hash")


def _sha256(payload: dict) -> str:
    return hashlib.sha256(dumps(payload).encode("utf-8")).hexdigest()


def hash_decision(record: dict) -> str:
    return _sha256({k: record[k] for k in DECISION_HASH_FIELDS})


def hash_outcome(event: dict) -> str:
    return _sha256({k: event[k] for k in OUTCOME_HASH_FIELDS})


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class DecisionFingerprinter:
    """Creates chained records and verifies both chains."""

    def __init__(self, db: LedgerDB, model_version: str, model_hash: str):
        self.db = db
        self.model_version = model_version
        self.model_hash = model_hash

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    def log_decision(
        self,
        input_features: dict,
        prediction: int,
        score: float,
        shap_values: dict,
        age_group: str | None = None,
        sex: str | None = None,
        timestamp: str | None = None,
    ) -> dict:
        """Append one decision. `score` is the model's probability of approval.

        `timestamp` may be supplied when back-filling historical decisions
        (as the demo seeder does); it is part of the hash either way.
        """
        if prediction not in (0, 1):
            raise ValueError("prediction must be 0 or 1")
        if not 0.0 <= float(score) <= 1.0:
            raise ValueError("score must be a probability between 0 and 1")

        def build(previous_hash: str) -> dict:
            record = {
                "decision_id": str(uuid.uuid4()),
                "timestamp": timestamp or _now(),
                "model_version": self.model_version,
                "model_hash": self.model_hash,
                "input_features": dumps(input_features),
                "prediction": int(prediction),
                "score": round(float(score), 6),
                "shap_values": dumps(shap_values),
                "age_group": age_group,
                "sex": sex,
                "previous_hash": previous_hash,
            }
            record["decision_hash"] = hash_decision(record)
            return record

        record = self.db.append_decision(build)
        logger.info("Decision logged: %s (hash ...%s)", record["decision_id"],
                    record["decision_hash"][-8:])
        return record

    def log_outcome(self, decision_id: str, true_label: int,
                    timestamp: str | None = None) -> dict:
        """Append an outcome event (1 = repaid / good, 0 = defaulted / bad)."""
        if true_label not in (0, 1):
            raise ValueError("true_label must be 0 or 1")
        if self.db.get_record_by_id(decision_id) is None:
            raise KeyError(decision_id)

        def build(previous_hash: str) -> dict:
            event = {
                "event_id": str(uuid.uuid4()),
                "decision_id": decision_id,
                "true_label": int(true_label),
                "timestamp": timestamp or _now(),
                "previous_hash": previous_hash,
            }
            event["event_hash"] = hash_outcome(event)
            return event

        return self.db.append_outcome(build)

    # ------------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------------

    @staticmethod
    def _verify(rows: list[dict], hash_fn, hash_col: str, id_col: str) -> dict:
        result = {"total": len(rows), "breaks": [], "intact": True}
        expected_prev = GENESIS_HASH
        for i, row in enumerate(rows):
            if row["previous_hash"] != expected_prev:
                result["breaks"].append({
                    "position": i, "id": row[id_col],
                    "reason": "previous_hash does not match the preceding record",
                })
            if hash_fn(row) != row[hash_col]:
                result["breaks"].append({
                    "position": i, "id": row[id_col],
                    "reason": "content does not match its stored hash",
                })
            expected_prev = row[hash_col]
        result["intact"] = not result["breaks"]
        return result

    def verify_chain(self) -> dict:
        """Verify the decision chain (content hashes and links)."""
        result = self._verify(self.db.get_all_records(), hash_decision,
                              "decision_hash", "decision_id")
        if not result["intact"]:
            logger.error("Decision chain broken at %s",
                         [b["position"] for b in result["breaks"]])
        return result

    def verify_outcomes(self) -> dict:
        """Verify the outcome chain, and that each event refers to a real decision."""
        events = self.db.get_all_outcomes()
        result = self._verify(events, hash_outcome, "event_hash", "event_id")
        known = {r["decision_id"] for r in self.db.get_all_records()}
        for i, ev in enumerate(events):
            if ev["decision_id"] not in known:
                result["breaks"].append({
                    "position": i, "id": ev["event_id"],
                    "reason": "outcome refers to a decision that is not in the ledger",
                })
        result["intact"] = not result["breaks"]
        return result


def parse_json_fields(record: dict) -> dict:
    """Return a copy of a stored decision with JSON fields decoded (for API output)."""
    out = dict(record)
    out["input_features"] = json.loads(out["input_features"])
    out["shap_values"] = json.loads(out["shap_values"])
    return out
