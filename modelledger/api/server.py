"""
ModelLedger API.

Security (demo level, see README for production notes)
  - X-API-Key header on every endpoint except /health
  - Rate limit on POST /decisions, keyed on the API key (falls back to client IP)

Integrity
  - GET /verify checks the decision chain, the outcome chain and every signed
    anchor file. It returns 409 if any check fails.

Monitoring
  - GET /fairness and GET /performance compute metrics without side effects.
  - A daily job (01:00 UTC) and POST /monitoring/snapshot store snapshots.
"""

import asyncio
import json
import logging
import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import date, datetime, timedelta, timezone
from functools import lru_cache
from typing import Literal

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from pythonjsonlogger import jsonlogger
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from modelledger import __version__
from modelledger.compliance.report import ComplianceReportGenerator
from modelledger.core.anchor import maybe_write_anchor, verify_all_anchors, write_anchor
from modelledger.core.database import LedgerDB
from modelledger.core.fingerprinter import DecisionFingerprinter, parse_json_fields
from modelledger.fairness.monitor import FairnessMonitor
from modelledger.monitoring.performance import PerformanceMonitor

load_dotenv()

_handler = logging.StreamHandler()
_handler.setFormatter(jsonlogger.JsonFormatter("%(asctime)s %(name)s %(levelname)s %(message)s"))
logging.root.handlers = [_handler]
logging.root.setLevel(logging.INFO)
logger = logging.getLogger("modelledger.api")


# ---------------------------------------------------------------------------
# Configuration (read at call time so tests and deployments can override)
# ---------------------------------------------------------------------------

def anchor_dir() -> str:
    return os.environ.get("ANCHOR_DIR", "anchors")


def anchor_interval() -> int:
    return int(os.environ.get("ANCHOR_INTERVAL", "100"))


def reports_dir() -> str:
    return os.environ.get("REPORTS_DIR", "reports")


@lru_cache
def model_identity() -> tuple[str, str]:
    """Model version and hash: env vars first, then models/model_metadata.json."""
    version, model_hash = os.environ.get("MODEL_VERSION"), os.environ.get("MODEL_HASH")
    meta_path = os.path.join("models", "model_metadata.json")
    if (not version or not model_hash) and os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        version = version or meta.get("model_version")
        model_hash = model_hash or meta.get("model_hash")
    return version or "unknown", model_hash or "unknown"


@lru_cache
def get_db() -> LedgerDB:
    return LedgerDB(os.environ.get("DB_PATH", os.path.join("data", "modelledger.db")))


def get_fingerprinter() -> DecisionFingerprinter:
    version, model_hash = model_identity()
    return DecisionFingerprinter(get_db(), version, model_hash)


def reset_state() -> None:
    """Drop cached singletons (used by tests)."""
    get_db.cache_clear()
    model_identity.cache_clear()
    limiter.reset()


# ---------------------------------------------------------------------------
# Auth and rate limiting
# ---------------------------------------------------------------------------

async def verify_api_key(x_api_key: str = Header(..., alias="X-API-Key")):
    expected = os.environ.get("MODELLEDGER_API_KEY")
    if not expected or x_api_key != expected:
        logger.warning("Rejected request with invalid API key")
        raise HTTPException(status_code=401, detail="Invalid API key")
    return x_api_key


def _rate_key(request: Request) -> str:
    return request.headers.get("X-API-Key") or get_remote_address(request)


def _rate_limit() -> str:
    return os.environ.get("RATE_LIMIT", "1000/minute")


limiter = Limiter(key_func=_rate_key)


# ---------------------------------------------------------------------------
# Lifespan: startup checks and the daily monitoring job
# ---------------------------------------------------------------------------

def daily_monitoring_job() -> None:
    """Snapshot fairness and performance for all decisions up to yesterday."""
    yesterday = (date.today() - timedelta(days=1)).isoformat()
    db = get_db()
    try:
        FairnessMonitor(db).record_snapshot(snapshot_date=yesterday, date_to=yesterday)
        PerformanceMonitor(db).record_snapshot(snapshot_date=yesterday, date_to=yesterday)
        logger.info("Daily monitoring snapshot recorded for %s", yesterday)
    except Exception:
        logger.exception("Daily monitoring job failed")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # The rate limiter and the scheduler keep state in-process, so multiple
    # uvicorn workers would each run the daily job and each allow the full
    # rate. Ledger writes themselves are safe across processes.
    if int(os.environ.get("UVICORN_WORKERS", "1")) > 1:
        raise RuntimeError("Run the API with a single uvicorn worker (UVICORN_WORKERS=1).")
    if not os.environ.get("MODELLEDGER_API_KEY"):
        raise RuntimeError("MODELLEDGER_API_KEY is not set.")

    scheduler = None
    if os.environ.get("DISABLE_SCHEDULER") != "1":
        from apscheduler.schedulers.asyncio import AsyncIOScheduler

        scheduler = AsyncIOScheduler(timezone="UTC")
        scheduler.add_job(daily_monitoring_job, "cron", hour=1, minute=0)
        scheduler.start()
    logger.info("ModelLedger %s started", __version__)
    yield
    if scheduler:
        scheduler.shutdown(wait=False)


app = FastAPI(
    title="ModelLedger",
    version=__version__,
    description=("Tamper-evident audit ledger and monitoring for credit-decision models. "
                 "Designed to support record-keeping and monitoring expectations; it does "
                 "not by itself make a firm compliant with any regulation."),
    lifespan=lifespan,
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
_pdf_executor = ThreadPoolExecutor(max_workers=2)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class DecisionRequest(BaseModel):
    input_features: dict
    prediction: int = Field(..., ge=0, le=1, description="1 = approve, 0 = decline")
    score: float = Field(..., ge=0.0, le=1.0, description="Model probability of approval")
    shap_values: dict
    age_group: str | None = None
    sex: str | None = None


class OutcomeRequest(BaseModel):
    true_label: int = Field(..., ge=0, le=1, description="1 = repaid, 0 = defaulted")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    db = get_db()
    last = db.get_last_anchor()
    version, model_hash = model_identity()
    return {
        "status": "ok",
        "version": __version__,
        "decisions": db.get_record_count(),
        "outcome_events": db.get_outcome_count(),
        "last_anchor_at": last["timestamp"] if last else None,
        "last_anchor_record_count": last["record_count"] if last else 0,
        "fairness_breach_on_record": db.get_latest_fairness_breach() is not None,
        "model_version": version,
        "model_hash": model_hash,
    }


@app.post("/decisions", status_code=201, dependencies=[Depends(verify_api_key)])
@limiter.limit(_rate_limit)
def log_decision(request: Request, req: DecisionRequest):
    record = get_fingerprinter().log_decision(**req.model_dump())
    anchored = maybe_write_anchor(get_db(), anchor_interval(), anchor_dir())
    return {
        "decision_id": record["decision_id"],
        "decision_hash": record["decision_hash"],
        "previous_hash": record["previous_hash"],
        "anchor_written": anchored,
    }


@app.get("/decisions/{decision_id}", dependencies=[Depends(verify_api_key)])
def get_decision(decision_id: str):
    record = get_db().get_record_by_id(decision_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Decision not found")
    out = parse_json_fields(record)
    out["outcome_events"] = [e for e in get_db().get_all_outcomes()
                             if e["decision_id"] == decision_id]
    return out


@app.patch("/decisions/{decision_id}/outcome", dependencies=[Depends(verify_api_key)])
def record_outcome(decision_id: str, req: OutcomeRequest):
    """Record the repayment outcome as a new chained event (never an overwrite).

    Sending a corrected outcome adds another event; the latest event is used
    for monitoring and the earlier one stays in the audit trail.
    """
    try:
        event = get_fingerprinter().log_outcome(decision_id, req.true_label)
    except KeyError:
        raise HTTPException(status_code=404, detail="Decision not found")
    return {"decision_id": decision_id, "event_id": event["event_id"],
            "event_hash": event["event_hash"], "true_label": event["true_label"]}


@app.get("/verify", dependencies=[Depends(verify_api_key)])
def verify():
    fp = get_fingerprinter()
    decisions, outcomes = fp.verify_chain(), fp.verify_outcomes()
    anchors = verify_all_anchors(get_db(), anchor_dir())
    result = {
        "intact": decisions["intact"] and outcomes["intact"] and anchors["intact"],
        "decisions": decisions,
        "outcomes": outcomes,
        "anchors": anchors,
    }
    return JSONResponse(content=result, status_code=200 if result["intact"] else 409)


@app.get("/anchors", dependencies=[Depends(verify_api_key)])
def list_anchors():
    return verify_all_anchors(get_db(), anchor_dir())


@app.post("/anchors", status_code=201, dependencies=[Depends(verify_api_key)])
def create_anchor():
    try:
        anchor = write_anchor(get_db(), anchor_dir())
    except EnvironmentError as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {k: anchor[k] for k in ("timestamp", "record_count", "outcome_count", "head_hash")}


@app.get("/fairness", dependencies=[Depends(verify_api_key)])
def fairness(group_by: Literal["sex", "age_group"] = "sex",
             date_from: str | None = None, date_to: str | None = None):
    return FairnessMonitor(get_db()).compute_all_metrics(date_from, date_to, group_by)


@app.get("/performance", dependencies=[Depends(verify_api_key)])
def performance(date_from: str | None = None, date_to: str | None = None):
    return PerformanceMonitor(get_db()).compute(date_from=date_from, date_to=date_to)


@app.post("/monitoring/snapshot", status_code=201, dependencies=[Depends(verify_api_key)])
def monitoring_snapshot():
    db = get_db()
    return {"fairness": FairnessMonitor(db).record_snapshot(),
            "performance": PerformanceMonitor(db).record_snapshot()}


@app.post("/report", status_code=202, dependencies=[Depends(verify_api_key)])
async def generate_report():
    job_id = str(uuid.uuid4())
    get_db().create_report_job(job_id, datetime.now(timezone.utc).isoformat())
    asyncio.get_running_loop().run_in_executor(_pdf_executor, generate_report_sync, job_id)
    return {"job_id": job_id, "status": "pending",
            "message": "Poll GET /reports/{job_id} for the PDF."}


def generate_report_sync(job_id: str) -> None:
    db = get_db()
    job = db.get_report_job(job_id)
    if job and job["status"] == "complete":
        return
    try:
        path = os.path.join(reports_dir(), f"modelledger_report_{job_id}.pdf")
        ComplianceReportGenerator(db, anchor_dir=anchor_dir()).generate(path)
        db.update_report_job(job_id, "complete", path, datetime.now(timezone.utc).isoformat())
    except Exception:
        logger.exception("Report job %s failed", job_id)
        db.update_report_job(job_id, "failed")


@app.get("/reports/{job_id}", dependencies=[Depends(verify_api_key)])
def get_report(job_id: str):
    job = get_db().get_report_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Report job not found")
    if job["status"] == "pending":
        return {"job_id": job_id, "status": "pending"}
    if job["status"] == "failed":
        raise HTTPException(status_code=500, detail="Report generation failed")
    if not job["output_path"] or not os.path.exists(job["output_path"]):
        raise HTTPException(status_code=404, detail="Report file missing")
    return FileResponse(job["output_path"], media_type="application/pdf",
                        filename=os.path.basename(job["output_path"]))
