"""Several processes appending at once (the API and the worker share one file)."""

import multiprocessing as mp

from modelledger.core.database import LedgerDB
from modelledger.core.fingerprinter import DecisionFingerprinter


def _writer(path: str, tag: str, n: int) -> None:
    db = LedgerDB(path)
    fp = DecisionFingerprinter(db, "m", "h")
    for i in range(n):
        fp.log_decision({"writer": tag, "i": i}, 1, 0.7, {})
    db.close()


def test_concurrent_writers_keep_one_chain(tmp_path):
    path = str(tmp_path / "ledger.db")
    LedgerDB(path).close()
    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=_writer, args=(path, t, 60)) for t in "abcd"]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
        assert p.exitcode == 0
    db = LedgerDB(path)
    result = DecisionFingerprinter(db, "m", "h").verify_chain()
    assert result["total"] == 240
    assert result["intact"]
