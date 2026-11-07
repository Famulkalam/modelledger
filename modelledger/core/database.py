"""
ModelLedger: SQLite storage layer.

Tables
  audit_ledger         hash-chained credit decisions (append-only)
  outcome_log          hash-chained outcome events (actual repaid/defaulted labels)
  anchor_log           index of signed anchors (the authoritative copies are files)
  fairness_history     daily fairness snapshots
  performance_history  daily performance snapshots (Gini, KS, PSI)
  report_jobs          background PDF jobs

Concurrency
  Every append reads the previous hash and inserts the new record inside one
  BEGIN IMMEDIATE transaction, so concurrent writers (the API and the queue
  worker are separate processes in the Docker stack) cannot both chain onto
  the same parent. SQLite's file lock serialises writers across processes;
  a re-entrant lock serialises threads sharing this connection.
"""

import json
import logging
import os
import sqlite3
import threading
from typing import Callable

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = os.environ.get("DB_PATH", os.path.join("data", "modelledger.db"))
GENESIS_HASH = "0" * 64

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS audit_ledger (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    decision_id     TEXT UNIQUE NOT NULL,
    timestamp       TEXT NOT NULL,
    model_version   TEXT NOT NULL,
    model_hash      TEXT NOT NULL,
    input_features  TEXT NOT NULL,
    prediction      INTEGER NOT NULL,
    score           REAL NOT NULL,
    shap_values     TEXT NOT NULL,
    age_group       TEXT,
    sex             TEXT,
    previous_hash   TEXT NOT NULL,
    decision_hash   TEXT UNIQUE NOT NULL
);

CREATE TABLE IF NOT EXISTS outcome_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id        TEXT UNIQUE NOT NULL,
    decision_id     TEXT NOT NULL,
    true_label      INTEGER NOT NULL,
    timestamp       TEXT NOT NULL,
    previous_hash   TEXT NOT NULL,
    event_hash      TEXT UNIQUE NOT NULL
);

CREATE TABLE IF NOT EXISTS anchor_log (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp           TEXT NOT NULL,
    record_count        INTEGER NOT NULL,
    head_hash           TEXT NOT NULL,
    outcome_count       INTEGER NOT NULL,
    outcome_head_hash   TEXT NOT NULL,
    public_key          TEXT NOT NULL,
    signature           TEXT NOT NULL,
    file_path           TEXT
);

CREATE TABLE IF NOT EXISTS fairness_history (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    date             TEXT NOT NULL,
    group_column     TEXT NOT NULL,
    record_count     INTEGER,
    parity_gap       REAL,
    eo_gap           REAL,
    parity_breached  INTEGER,
    eo_breached      INTEGER
);

CREATE TABLE IF NOT EXISTS performance_history (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    date             TEXT NOT NULL,
    n_with_outcome   INTEGER,
    gini             REAL,
    ks               REAL,
    psi              REAL
);

CREATE TABLE IF NOT EXISTS report_jobs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id          TEXT UNIQUE NOT NULL,
    status          TEXT NOT NULL DEFAULT 'pending',
    output_path     TEXT,
    created_at      TEXT NOT NULL,
    completed_at    TEXT
);
"""

DECISION_COLUMNS = [
    "decision_id", "timestamp", "model_version", "model_hash", "input_features",
    "prediction", "score", "shap_values", "age_group", "sex",
    "previous_hash", "decision_hash",
]
OUTCOME_COLUMNS = [
    "event_id", "decision_id", "true_label", "timestamp", "previous_hash", "event_hash",
]


class LedgerDB:
    """SQLite-backed ledger. Use ":memory:" for tests."""

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = db_path
        if db_path != ":memory:":
            os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        # isolation_level=None: we manage transactions explicitly.
        self.conn = sqlite3.connect(
            db_path, check_same_thread=False, isolation_level=None, timeout=30
        )
        self.conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        with self._lock:
            if db_path != ":memory:":
                self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.execute("PRAGMA busy_timeout=30000")
            self.conn.executescript(SCHEMA_SQL)

    # ------------------------------------------------------------------
    # Atomic chained appends
    # ------------------------------------------------------------------

    def _append(self, table: str, hash_col: str, columns: list[str],
                build: Callable[[str], dict]) -> dict:
        """Read the chain head and insert the next record in ONE transaction."""
        with self._lock:
            self.conn.execute("BEGIN IMMEDIATE")
            try:
                row = self.conn.execute(
                    f"SELECT {hash_col} FROM {table} ORDER BY id DESC LIMIT 1"
                ).fetchone()
                previous_hash = row[0] if row else GENESIS_HASH
                record = build(previous_hash)
                placeholders = ",".join("?" for _ in columns)
                self.conn.execute(
                    f"INSERT INTO {table} ({','.join(columns)}) VALUES ({placeholders})",
                    [record[c] for c in columns],
                )
                self.conn.execute("COMMIT")
                return record
            except Exception:
                self.conn.execute("ROLLBACK")
                raise

    def append_decision(self, build: Callable[[str], dict]) -> dict:
        return self._append("audit_ledger", "decision_hash", DECISION_COLUMNS, build)

    def append_outcome(self, build: Callable[[str], dict]) -> dict:
        return self._append("outcome_log", "event_hash", OUTCOME_COLUMNS, build)

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def _all(self, sql: str, params: tuple = ()) -> list[dict]:
        with self._lock:
            return [dict(r) for r in self.conn.execute(sql, params).fetchall()]

    def _one(self, sql: str, params: tuple = ()) -> dict | None:
        with self._lock:
            row = self.conn.execute(sql, params).fetchone()
            return dict(row) if row else None

    def get_latest_hash(self) -> str:
        row = self._one("SELECT decision_hash FROM audit_ledger ORDER BY id DESC LIMIT 1")
        return row["decision_hash"] if row else GENESIS_HASH

    def get_all_records(self) -> list[dict]:
        return self._all("SELECT * FROM audit_ledger ORDER BY id ASC")

    def get_record_by_id(self, decision_id: str) -> dict | None:
        return self._one("SELECT * FROM audit_ledger WHERE decision_id = ?", (decision_id,))

    def get_record_count(self) -> int:
        return self._one("SELECT COUNT(*) AS n FROM audit_ledger")["n"]

    def get_all_outcomes(self) -> list[dict]:
        return self._all("SELECT * FROM outcome_log ORDER BY id ASC")

    def get_outcome_count(self) -> int:
        return self._one("SELECT COUNT(*) AS n FROM outcome_log")["n"]

    def get_latest_outcomes(self) -> dict[str, int]:
        """decision_id -> most recent true_label (later events supersede earlier ones)."""
        latest: dict[str, int] = {}
        for ev in self.get_all_outcomes():
            latest[ev["decision_id"]] = ev["true_label"]
        return latest

    def get_decisions_df(self, date_from: str | None = None, date_to: str | None = None):
        """Decisions as a DataFrame with the latest known outcome in `true_label`.

        date_from / date_to are ISO dates or timestamps; date_to is inclusive of
        the whole day when a bare date (YYYY-MM-DD) is given.
        """
        import pandas as pd

        query, params, conds = "SELECT * FROM audit_ledger", [], []
        if date_from:
            conds.append("timestamp >= ?")
            params.append(date_from)
        if date_to:
            if len(date_to) == 10:  # bare date: include the whole day
                conds.append("substr(timestamp, 1, 10) <= ?")
            else:
                conds.append("timestamp <= ?")
            params.append(date_to)
        if conds:
            query += " WHERE " + " AND ".join(conds)
        query += " ORDER BY id ASC"
        with self._lock:
            df = pd.read_sql_query(query, self.conn, params=params)
        outcomes = self.get_latest_outcomes()
        df["true_label"] = df["decision_id"].map(outcomes)
        return df

    # ------------------------------------------------------------------
    # Anchors
    # ------------------------------------------------------------------

    def insert_anchor(self, anchor: dict, file_path: str | None = None) -> None:
        with self._lock:
            self.conn.execute(
                """INSERT INTO anchor_log (timestamp, record_count, head_hash, outcome_count,
                   outcome_head_hash, public_key, signature, file_path)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (anchor["timestamp"], anchor["record_count"], anchor["head_hash"],
                 anchor["outcome_count"], anchor["outcome_head_hash"],
                 anchor["public_key"], anchor["signature"], file_path),
            )

    def get_all_anchors(self) -> list[dict]:
        return self._all("SELECT * FROM anchor_log ORDER BY id ASC")

    def get_last_anchor(self) -> dict | None:
        return self._one("SELECT * FROM anchor_log ORDER BY id DESC LIMIT 1")

    # ------------------------------------------------------------------
    # Monitoring history
    # ------------------------------------------------------------------

    def insert_fairness_record(self, rec: dict) -> None:
        with self._lock:
            self.conn.execute(
                """INSERT INTO fairness_history (date, group_column, record_count, parity_gap,
                   eo_gap, parity_breached, eo_breached) VALUES (?,?,?,?,?,?,?)""",
                (rec["date"], rec["group_column"], rec.get("record_count"), rec.get("parity_gap"),
                 rec.get("eo_gap"), rec.get("parity_breached"), rec.get("eo_breached")),
            )

    def get_fairness_history(self, group_column: str | None = None) -> list[dict]:
        if group_column:
            return self._all(
                "SELECT * FROM fairness_history WHERE group_column = ? ORDER BY date, id",
                (group_column,),
            )
        return self._all("SELECT * FROM fairness_history ORDER BY date, id")

    def get_latest_fairness_breach(self) -> dict | None:
        return self._one(
            "SELECT * FROM fairness_history WHERE parity_breached = 1 OR eo_breached = 1 "
            "ORDER BY id DESC LIMIT 1"
        )

    def insert_performance_record(self, rec: dict) -> None:
        with self._lock:
            self.conn.execute(
                "INSERT INTO performance_history (date, n_with_outcome, gini, ks, psi) "
                "VALUES (?,?,?,?,?)",
                (rec["date"], rec.get("n_with_outcome"), rec.get("gini"), rec.get("ks"),
                 rec.get("psi")),
            )

    def get_performance_history(self) -> list[dict]:
        return self._all("SELECT * FROM performance_history ORDER BY date, id")

    # ------------------------------------------------------------------
    # Report jobs
    # ------------------------------------------------------------------

    def create_report_job(self, job_id: str, created_at: str) -> None:
        with self._lock:
            self.conn.execute(
                "INSERT INTO report_jobs (job_id, status, created_at) VALUES (?,?,?)",
                (job_id, "pending", created_at),
            )

    def update_report_job(self, job_id: str, status: str, output_path: str | None = None,
                          completed_at: str | None = None) -> None:
        with self._lock:
            self.conn.execute(
                "UPDATE report_jobs SET status = ?, output_path = ?, completed_at = ? "
                "WHERE job_id = ?",
                (status, output_path, completed_at, job_id),
            )

    def get_report_job(self, job_id: str) -> dict | None:
        return self._one("SELECT * FROM report_jobs WHERE job_id = ?", (job_id,))

    # ------------------------------------------------------------------
    # Demo / test helpers
    # ------------------------------------------------------------------

    def raw_execute(self, sql: str, params: tuple = ()) -> None:
        """Direct SQL, bypassing the chain. Used ONLY to simulate tampering."""
        with self._lock:
            self.conn.execute(sql, params)

    def close(self) -> None:
        with self._lock:
            self.conn.close()


def dumps(obj) -> str:
    """Canonical JSON used everywhere a value is hashed."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
