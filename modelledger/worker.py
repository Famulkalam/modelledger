"""
ModelLedger queue worker.

Consumes decision events published by the credit-model service (Redis list
`modelledger:decisions`), appends each to the hash-chained ledger and signs an
anchor every ANCHOR_INTERVAL decisions.

Run: python -m modelledger.worker
"""

import json
import logging
import os
import signal
import sys

from dotenv import load_dotenv

load_dotenv()

from modelledger.core.anchor import maybe_write_anchor  # noqa: E402
from modelledger.core.database import LedgerDB  # noqa: E402
from modelledger.core.fingerprinter import DecisionFingerprinter  # noqa: E402

logger = logging.getLogger("modelledger.worker")
QUEUE = "modelledger:decisions"
_shutdown = False


def _stop(signum, frame):
    global _shutdown
    logger.info("Shutdown requested; finishing current event")
    _shutdown = True


def handle_event(db: LedgerDB, payload: bytes | str) -> dict:
    """Append one queued decision to the ledger. Separated out for testing."""
    event = json.loads(payload)
    fp = DecisionFingerprinter(db, event["model_version"], event["model_hash"])
    record = fp.log_decision(
        input_features=event["input_features"],
        prediction=event["prediction"],
        score=event["score"],
        shap_values=event["shap_values"],
        age_group=event.get("age_group"),
        sex=event.get("sex"),
        timestamp=event.get("timestamp"),
    )
    maybe_write_anchor(db, int(os.environ.get("ANCHOR_INTERVAL", "100")),
                       os.environ.get("ANCHOR_DIR", "anchors"))
    return record


def run_worker():
    import redis

    url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
    r = redis.Redis.from_url(url)
    try:
        r.ping()
    except redis.ConnectionError as e:
        logger.error("Cannot connect to Redis at %s: %s", url, e)
        sys.exit(1)

    db = LedgerDB()
    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)
    logger.info("Worker listening on %s", QUEUE)

    while not _shutdown:
        item = r.blpop(QUEUE, timeout=1)
        if item is None:
            continue
        _, payload = item
        try:
            record = handle_event(db, payload)
            logger.info("Logged decision %s", record["decision_id"])
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.error("Rejected malformed event (%s): %s", e, payload[:200])
            r.rpush(QUEUE + ":dead_letter", payload)
        except Exception:
            logger.exception("Failed to log event; returning it to the queue")
            r.lpush(QUEUE, payload)
            break  # stop rather than loop on a persistent failure
    db.close()
    logger.info("Worker stopped")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    run_worker()
