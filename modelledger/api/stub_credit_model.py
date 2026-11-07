"""
Demo credit-model service (stands in for a bank's existing decision engine).

POST /predict scores an applicant with the trained XGBoost model, computes
real SHAP values, returns the decision, and publishes the decision event to a
Redis queue. The ModelLedger worker consumes the queue and writes the ledger,
so the decision path never waits on the audit system.

The model must be trained first (python scripts/train_model.py). If it is
missing the service returns 503; it never falls back to made-up predictions.
"""

import json
import logging
import os
from datetime import datetime, timezone
from typing import Literal

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from modelledger.model.credit_model import MODEL_PATH, Scorer, file_sha256, load_bundle

logger = logging.getLogger(__name__)
QUEUE = "modelledger:decisions"

app = FastAPI(title="Demo credit model", version="2.0.0",
              description="XGBoost credit model that publishes each decision to Redis.")

_state: dict = {}


def get_scorer() -> Scorer:
    if "scorer" not in _state:
        try:
            bundle = load_bundle()
        except FileNotFoundError as e:
            raise HTTPException(status_code=503, detail=str(e))
        _state["scorer"] = Scorer(bundle)
        _state["model_version"] = bundle["model_version"]
        _state["model_hash"] = file_sha256(MODEL_PATH)
    return _state["scorer"]


def get_redis():
    if "redis" not in _state:
        import redis

        client = redis.from_url(os.environ.get("REDIS_URL", "redis://localhost:6379/0"))
        client.ping()
        _state["redis"] = client
    return _state["redis"]


class CreditApplication(BaseModel):
    """German Credit applicant fields (OpenML credit-g category labels)."""

    checking_status: Literal["<0", "0<=X<200", ">=200", "no checking"] = "0<=X<200"
    duration: int = Field(24, ge=1, le=120, description="Loan duration in months")
    credit_history: Literal["no credits/all paid", "all paid", "existing paid",
                            "delayed previously", "critical/other existing credit"] = "existing paid"
    purpose: Literal["new car", "used car", "furniture/equipment", "radio/tv",
                     "domestic appliance", "repairs", "education", "vacation", "retraining",
                     "business", "other"] = "radio/tv"
    credit_amount: float = Field(2500, gt=0)
    savings_status: Literal["<100", "100<=X<500", "500<=X<1000", ">=1000",
                            "no known savings"] = "<100"
    employment: Literal["unemployed", "<1", "1<=X<4", "4<=X<7", ">=7"] = "1<=X<4"
    installment_commitment: int = Field(3, ge=1, le=4)
    personal_status: Literal["male div/sep", "female div/dep/mar", "male single",
                             "male mar/wid", "female single"] = "female div/dep/mar"
    other_parties: Literal["none", "co applicant", "guarantor"] = "none"
    residence_since: int = Field(2, ge=1, le=4)
    property_magnitude: Literal["real estate", "life insurance", "car",
                                "no known property"] = "car"
    age: int = Field(35, ge=18, le=100)
    other_payment_plans: Literal["bank", "stores", "none"] = "none"
    housing: Literal["rent", "own", "for free"] = "own"
    existing_credits: int = Field(1, ge=1, le=4)
    job: Literal["unemp/unskilled non res", "unskilled resident", "skilled",
                 "high qualif/self emp/mgmt"] = "skilled"
    num_dependents: int = Field(1, ge=1, le=2)
    own_telephone: Literal["none", "yes"] = "none"
    foreign_worker: Literal["yes", "no"] = "yes"


@app.get("/health")
def health():
    model_ready = os.path.exists(MODEL_PATH)
    try:
        get_redis()
        redis_ok = True
    except Exception:
        redis_ok = False
    return {"service": "credit-model", "model_loaded": model_ready, "redis_connected": redis_ok}


@app.post("/predict")
def predict(application: CreditApplication):
    scorer = get_scorer()
    result = scorer.score(pd.DataFrame([application.model_dump()]))[0]
    event = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_version": _state["model_version"],
        "model_hash": _state["model_hash"],
        **result,
    }
    try:
        get_redis().rpush(QUEUE, json.dumps(event))
        queued = True
    except Exception as e:  # the decision is still returned; the gap is visible in /health
        logger.error("Could not publish decision to Redis: %s", e)
        queued = False
    return {
        "decision": "approve" if result["prediction"] == 1 else "decline",
        "score": round(result["score"], 4),
        "shap_values": result["shap_values"],
        "queued_for_audit": queued,
    }
