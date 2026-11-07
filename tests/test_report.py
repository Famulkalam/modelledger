"""PDF evidence pack content."""

import pdfplumber

from conftest import make_decisions
from modelledger.compliance.report import ComplianceReportGenerator
from modelledger.core.anchor import write_anchor


def _text(path):
    with pdfplumber.open(path) as pdf:
        return " ".join(" ".join((p.extract_text() or "").split()) for p in pdf.pages)


def test_report_content(fp, db, anchor_dir, tmp_path):
    for r in make_decisions(fp, 60):
        fp.log_outcome(r["decision_id"], 1)
    write_anchor(db, anchor_dir)
    path = ComplianceReportGenerator(db, anchor_dir=anchor_dir).generate(
        str(tmp_path / "report.pdf"))
    text = _text(path)
    assert "Overall integrity: PASSED" in text
    assert "PRA SS1/23" in text and "FCA SS1/23" not in text
    assert "a" * 32 in text.replace(" ", "")           # full model hash, not a placeholder
    assert "not a compliance certification" in text


def test_report_flags_tampering(fp, db, anchor_dir, tmp_path):
    make_decisions(fp, 10)
    write_anchor(db, anchor_dir)
    db.raw_execute("UPDATE audit_ledger SET sex = 'other' WHERE id = 3")
    text = _text(ComplianceReportGenerator(db, anchor_dir=anchor_dir).generate(
        str(tmp_path / "r.pdf")))
    assert "Overall integrity: FAILED" in text
    assert "content does not match its stored hash" in text


def test_table_text_wraps_instead_of_overlapping(fp, db, anchor_dir, tmp_path):
    """Regression: long cells used to run into the next column and off the page."""
    make_decisions(fp, 5)
    text = _text(ComplianceReportGenerator(db, anchor_dir=anchor_dir).generate(
        str(tmp_path / "r.pdf")))
    assert "Tamper-evident, per-decision log." in text
