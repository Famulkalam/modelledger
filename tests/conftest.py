"""Shared fixtures. Tests never touch the network or the real data/ folder."""

import os

import pytest

from modelledger.core.anchor import generate_keypair

PRIVATE_HEX, PUBLIC_HEX = generate_keypair()
os.environ.update({
    "MODELLEDGER_API_KEY": "test-api-key",
    "ANCHOR_SIGNING_KEY": PRIVATE_HEX,
    "ANCHOR_PUBLIC_KEY": PUBLIC_HEX,
    "UVICORN_WORKERS": "1",
    "DISABLE_SCHEDULER": "1",
    "MODEL_VERSION": "test-model",
    "MODEL_HASH": "a" * 64,
})

from modelledger.core.database import LedgerDB  # noqa: E402
from modelledger.core.fingerprinter import DecisionFingerprinter  # noqa: E402


@pytest.fixture
def db():
    database = LedgerDB(":memory:")
    yield database
    database.close()


@pytest.fixture
def fp(db):
    return DecisionFingerprinter(db, "test-model", "a" * 64)


@pytest.fixture
def anchor_dir(tmp_path):
    return str(tmp_path / "anchors")


def make_decisions(fp, n=10, start=0):
    """n decisions alternating sex and age group, with simple features."""
    out = []
    for i in range(start, start + n):
        out.append(fp.log_decision(
            input_features={"duration": 12 + i, "credit_amount": 1000 + 100 * i},
            prediction=1 if i % 3 else 0,
            score=0.3 + (i % 7) / 10,
            shap_values={"duration": -0.01 * i, "credit_amount": 0.02},
            age_group=["18-25", "26-35", "36-45"][i % 3],
            sex="female" if i % 2 else "male",
        ))
    return out


@pytest.fixture
def client(tmp_path, monkeypatch):
    """API client on a fresh temporary database and anchor folder."""
    from fastapi.testclient import TestClient

    from modelledger.api import server

    monkeypatch.setenv("DB_PATH", str(tmp_path / "ledger.db"))
    monkeypatch.setenv("ANCHOR_DIR", str(tmp_path / "anchors"))
    monkeypatch.setenv("REPORTS_DIR", str(tmp_path / "reports"))
    monkeypatch.setenv("ANCHOR_INTERVAL", "5")
    server.reset_state()
    with TestClient(server.app) as c:
        yield c
    server.reset_state()


HEADERS = {"X-API-Key": "test-api-key"}
