"""
ModelLedger: signed anchors.

The problem anchors solve
    SHA-256 chaining needs no secret, so anyone with database write access can
    edit a record and recompute every later hash. The chain then verifies as
    "intact". (The original version of this project was vulnerable to exactly
    this.)

What an anchor is
    Every N decisions the service signs a small statement with an Ed25519
    private key:
        "At <time> the ledger held <record_count> decisions whose latest hash
         was <head_hash>, and <outcome_count> outcome events whose latest hash
         was <outcome_head_hash>."
    Because each hash commits to everything before it, the head hash pins the
    entire history up to that point. The statement is written to a file
    outside the database (in production: write-once storage owned by a
    different team, e.g. S3 Object Lock).

How tampering is caught
    Verification recomputes the chains and checks that record number
    <record_count> still has hash <head_hash>. Edit any record before that
    point, even with every later hash recomputed, and the head hash changes, so
    the check fails. Deleting records (count below the anchored count) also
    fails. Forging a new anchor requires the private key.

Why Ed25519 rather than an HMAC
    An HMAC uses one shared secret for signing and verifying, so anyone able to
    verify could also forge. With Ed25519, auditors verify using only the
    public key.

Limits (stated honestly)
    - Records written after the latest anchor are protected only by the hash
      chain until the next anchor is written.
    - If an attacker holds the private key AND can rewrite the anchor storage,
      anchors prove nothing. Keep the key in a KMS/HSM and the anchors on WORM
      storage under separate control. This repository keeps both locally for
      the demo.
"""

import glob
import json
import logging
import os
from datetime import datetime, timezone

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from modelledger.core.database import GENESIS_HASH, LedgerDB, dumps

logger = logging.getLogger(__name__)

DEFAULT_ANCHOR_INTERVAL = int(os.environ.get("ANCHOR_INTERVAL", "100"))
DEFAULT_ANCHOR_DIR = os.environ.get("ANCHOR_DIR", "anchors")
SIGNED_FIELDS = ("timestamp", "record_count", "head_hash", "outcome_count", "outcome_head_hash")


# ----------------------------------------------------------------------
# Keys
# ----------------------------------------------------------------------

def generate_keypair() -> tuple[str, str]:
    """Return (private_key_hex, public_key_hex)."""
    private = Ed25519PrivateKey.generate()
    return private.private_bytes_raw().hex(), _public_hex(private)


def _public_hex(private: Ed25519PrivateKey) -> str:
    return private.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw).hex()


def load_private_key(private_hex: str | None = None) -> Ed25519PrivateKey:
    private_hex = private_hex or os.environ.get("ANCHOR_SIGNING_KEY")
    if not private_hex:
        raise EnvironmentError(
            "ANCHOR_SIGNING_KEY is not set. Run `python scripts/generate_keys.py` "
            "and put the values in your .env file."
        )
    return Ed25519PrivateKey.from_private_bytes(bytes.fromhex(private_hex))


def load_public_key(public_hex: str | None = None) -> Ed25519PublicKey:
    """Public key from argument, ANCHOR_PUBLIC_KEY, or derived from the signing key."""
    public_hex = public_hex or os.environ.get("ANCHOR_PUBLIC_KEY")
    if public_hex:
        return Ed25519PublicKey.from_public_bytes(bytes.fromhex(public_hex))
    return load_private_key().public_key()


# ----------------------------------------------------------------------
# Writing
# ----------------------------------------------------------------------

def _head(rows: list[dict], hash_col: str) -> str:
    return rows[-1][hash_col] if rows else GENESIS_HASH


def write_anchor(db: LedgerDB, output_dir: str = DEFAULT_ANCHOR_DIR,
                 private_hex: str | None = None) -> dict:
    """Sign the current state of both chains and write it to a file."""
    private = load_private_key(private_hex)
    records = db.get_all_records()
    outcomes = db.get_all_outcomes()

    anchor = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "record_count": len(records),
        "head_hash": _head(records, "decision_hash"),
        "outcome_count": len(outcomes),
        "outcome_head_hash": _head(outcomes, "event_hash"),
    }
    payload = dumps({k: anchor[k] for k in SIGNED_FIELDS}).encode("utf-8")
    anchor["public_key"] = _public_hex(private)
    anchor["signature"] = private.sign(payload).hex()

    os.makedirs(output_dir, exist_ok=True)
    safe_ts = anchor["timestamp"].replace(":", "-")
    path = os.path.join(output_dir, f"anchor_{anchor['record_count']:08d}_{safe_ts}.json")
    with open(path, "w") as f:
        json.dump(anchor, f, indent=2)
    db.insert_anchor(anchor, file_path=path)
    logger.info("Anchor written: %d decisions, %d outcomes -> %s",
                anchor["record_count"], anchor["outcome_count"], path)
    return anchor


def maybe_write_anchor(db: LedgerDB, interval: int = DEFAULT_ANCHOR_INTERVAL,
                       output_dir: str = DEFAULT_ANCHOR_DIR) -> bool:
    """Write an anchor when the decision count reaches a multiple of `interval`."""
    count = db.get_record_count()
    if count == 0 or count % interval:
        return False
    try:
        write_anchor(db, output_dir)
        return True
    except EnvironmentError as e:
        logger.warning("Anchor not written: %s", e)
        return False


# ----------------------------------------------------------------------
# Verification
# ----------------------------------------------------------------------

def verify_signature(anchor: dict, public_key: Ed25519PublicKey) -> bool:
    payload = dumps({k: anchor[k] for k in SIGNED_FIELDS}).encode("utf-8")
    try:
        public_key.verify(bytes.fromhex(anchor["signature"]), payload)
        return True
    except (InvalidSignature, ValueError):
        return False


def verify_anchor_against_ledger(anchor: dict, records: list[dict], outcomes: list[dict],
                                 public_key: Ed25519PublicKey) -> dict:
    """Check one anchor against the ledger AS IT WAS when the anchor was written.

    record_count N means: the first N decisions must still end in head_hash.
    Records added after the anchor do not affect the check.
    """
    problems = []
    if not verify_signature(anchor, public_key):
        problems.append("signature is not valid for the trusted public key")

    n, m = anchor["record_count"], anchor["outcome_count"]
    if len(records) < n:
        problems.append(f"ledger has {len(records)} decisions, anchor recorded {n} "
                        "(records deleted)")
    elif _head(records[:n], "decision_hash") != anchor["head_hash"]:
        problems.append(f"decision history up to record {n} has changed since it was anchored")

    if len(outcomes) < m:
        problems.append(f"outcome log has {len(outcomes)} events, anchor recorded {m} "
                        "(events deleted)")
    elif _head(outcomes[:m], "event_hash") != anchor["outcome_head_hash"]:
        problems.append(f"outcome history up to event {m} has changed since it was anchored")

    return {
        "timestamp": anchor["timestamp"],
        "record_count": n,
        "outcome_count": m,
        "valid": not problems,
        "problems": problems,
    }


def load_anchor_files(anchor_dir: str = DEFAULT_ANCHOR_DIR) -> list[dict]:
    anchors = []
    for path in sorted(glob.glob(os.path.join(anchor_dir, "anchor_*.json"))):
        with open(path) as f:
            anchors.append(json.load(f))
    return sorted(anchors, key=lambda a: (a["record_count"], a["timestamp"]))


def verify_all_anchors(db: LedgerDB, anchor_dir: str = DEFAULT_ANCHOR_DIR,
                       public_hex: str | None = None) -> dict:
    """Verify every anchor file against the current ledger.

    The files (not the anchor_log table) are treated as the trusted copy,
    because someone who can edit the database can also edit that table.
    """
    try:
        public_key = load_public_key(public_hex)
    except EnvironmentError as e:
        return {"checked": 0, "intact": False, "results": [], "error": str(e)}

    records, outcomes = db.get_all_records(), db.get_all_outcomes()
    results = [verify_anchor_against_ledger(a, records, outcomes, public_key)
               for a in load_anchor_files(anchor_dir)]
    covered = max((r["record_count"] for r in results if r["valid"]), default=0)
    return {
        "checked": len(results),
        "intact": all(r["valid"] for r in results),
        "decisions_covered_by_anchors": covered,
        "decisions_after_last_anchor": max(len(records) - covered, 0),
        "results": results,
    }
