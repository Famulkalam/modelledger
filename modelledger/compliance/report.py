"""
ModelLedger: PDF evidence pack for model risk, validation and compliance reviewers.

Sections
  1. Summary              integrity status, volumes, period covered
  2. Model                version, file hash, development metrics
  3. Integrity            decision chain, outcome chain, signed anchors
  4. Performance          Gini, KS and PSI on logged decisions
  5. Fairness             approval-rate and equal-opportunity gaps by sex and age
  6. Regulatory context   what the evidence can SUPPORT (not certify)
  7. Governance           from config/governance.yaml
"""

import json
import logging
import os
from datetime import datetime, timezone

import yaml
from pydantic import BaseModel
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from modelledger import __version__
from modelledger.core.anchor import verify_all_anchors
from modelledger.core.database import LedgerDB
from modelledger.core.fingerprinter import DecisionFingerprinter
from modelledger.fairness.monitor import GROUP_COLUMNS, FairnessMonitor
from modelledger.monitoring.performance import PerformanceMonitor

logger = logging.getLogger(__name__)

PAGE_WIDTH = A4[0] - 30 * mm  # usable width with 15 mm margins


class AlertReviewConfig(BaseModel):
    reviewer: str
    response_time_days: int
    escalation_path: str


class LogRetentionConfig(BaseModel):
    minimum_years: int
    basis: str
    archive: str


class AccessControlsConfig(BaseModel):
    read_access: list[str]
    write_access: list[str]
    db_admin_access: str


class ReviewCycleConfig(BaseModel):
    ledger_integrity: str
    fairness_metrics: str
    performance_metrics: str
    regulatory_mapping: str


class GovernanceConfig(BaseModel):
    alert_review: AlertReviewConfig
    log_retention: LogRetentionConfig
    access_controls: AccessControlsConfig
    review_cycle: ReviewCycleConfig


REGULATORY_CONTEXT = [
    ("PRA SS1/23 (UK model risk management)",
     "Model inventory, use and performance monitoring, independent validation",
     "Each decision records the model version and file hash; Gini, KS and PSI are tracked "
     "over time; this report is an evidence pack for validators. SS1/23 applies directly to "
     "banks with internal-model approval and is widely used as a benchmark by others."),
    ("FCA Consumer Duty",
     "Monitor customer outcomes, including for different groups of customers",
     "Approval and equal-opportunity gaps by sex and age group, with history and alerts."),
    ("Equality Act 2010",
     "Avoid direct and indirect discrimination (age and sex are protected characteristics)",
     "Sex is excluded from the model; outcomes are still monitored by sex and age to help "
     "spot possible indirect effects. The metrics are not a legal test."),
    ("EU AI Act Art. 12 (record-keeping)",
     "High-risk systems must allow automatic logging of events",
     "Tamper-evident, per-decision log. Creditworthiness assessment is high-risk under "
     "Annex III; those obligations apply from 2 December 2027 after the Digital Omnibus "
     "delay, and only to firms within the Act's scope."),
    ("EU AI Act Art. 13 and 72",
     "Transparency to deployers; post-market monitoring",
     "Per-decision SHAP values; ongoing performance and fairness monitoring."),
]


def _default_governance() -> GovernanceConfig:
    return GovernanceConfig(
        alert_review=AlertReviewConfig(reviewer="Model owner (illustrative)",
                                       response_time_days=5,
                                       escalation_path="Model risk committee"),
        log_retention=LogRetentionConfig(minimum_years=5, basis="Firm retention policy",
                                         archive="Write-once object storage"),
        access_controls=AccessControlsConfig(read_access=["Model validation", "Internal audit"],
                                             write_access=["ModelLedger service account"],
                                             db_admin_access="Named administrators, logged"),
        review_cycle=ReviewCycleConfig(ledger_integrity="Daily (automated)",
                                       fairness_metrics="Monthly",
                                       performance_metrics="Monthly",
                                       regulatory_mapping="Annually"),
    )


class ComplianceReportGenerator:
    def __init__(self, db: LedgerDB, governance_config_path: str = "config/governance.yaml",
                 anchor_dir: str = "anchors",
                 metadata_path: str = os.path.join("models", "model_metadata.json")):
        self.db = db
        self.anchor_dir = anchor_dir
        if os.path.exists(governance_config_path):
            with open(governance_config_path) as f:
                self.governance = GovernanceConfig(**yaml.safe_load(f))
        else:
            self.governance = _default_governance()
        self.metadata = {}
        if os.path.exists(metadata_path):
            with open(metadata_path) as f:
                self.metadata = json.load(f)

        ss = getSampleStyleSheet()
        self.h1, self.h2, self.body = ss["Heading1"], ss["Heading2"], ss["BodyText"]
        self.small = ParagraphStyle("small", parent=ss["BodyText"], fontSize=8, leading=10)
        self.note = ParagraphStyle("note", parent=ss["BodyText"], fontSize=8, leading=10,
                                   textColor=colors.HexColor("#555555"))
        self.cell = ParagraphStyle("cell", parent=ss["BodyText"], fontSize=8, leading=10)
        self.head = ParagraphStyle("head", parent=self.cell, textColor=colors.white,
                                   fontName="Helvetica-Bold")

    # ------------------------------------------------------------------

    def _table(self, header: list[str], rows: list[list], widths: list[float]) -> Table:
        """Table whose cells wrap: every cell is a Paragraph and widths sum to the page."""
        total = sum(widths)
        widths = [w * PAGE_WIDTH / total for w in widths]
        data = [[Paragraph(h, self.head) for h in header]]
        data += [[Paragraph(str(c), self.cell) for c in row] for row in rows]
        t = Table(data, colWidths=widths, repeatRows=1)
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2C5F8A")),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#999999")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F2F5F8")]),
        ]))
        return t

    def generate(self, output_path: str) -> str:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        doc = SimpleDocTemplate(output_path, pagesize=A4, leftMargin=15 * mm,
                                rightMargin=15 * mm, topMargin=18 * mm, bottomMargin=18 * mm,
                                title="ModelLedger evidence pack")
        fp = DecisionFingerprinter(self.db, "", "")
        self.chain, self.outcomes = fp.verify_chain(), fp.verify_outcomes()
        self.anchors = verify_all_anchors(self.db, self.anchor_dir)
        self.records = self.db.get_all_records()

        story = []
        for section in (self._summary, self._model, self._integrity, self._performance,
                        self._fairness, self._regulatory, self._governance):
            story += section()
            story.append(Spacer(1, 4 * mm))
        doc.build(story)
        logger.info("Report written to %s", output_path)
        return os.path.abspath(output_path)

    # ------------------------------------------------------------------

    def _summary(self) -> list:
        intact = self.chain["intact"] and self.outcomes["intact"] and self.anchors["intact"]
        period = (f"{self.records[0]['timestamp'][:10]} to {self.records[-1]['timestamp'][:10]}"
                  if self.records else "no decisions logged")
        return [
            Paragraph("ModelLedger evidence pack", self.styles_title()),
            Paragraph(f"Generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC by "
                      f"ModelLedger {__version__}", self.note),
            Spacer(1, 3 * mm),
            Paragraph("1. Summary", self.h1),
            Paragraph(f"Decisions: <b>{len(self.records)}</b> ({period}). "
                      f"Outcome events: <b>{self.db.get_outcome_count()}</b>. "
                      f"Signed anchors checked: <b>{self.anchors['checked']}</b>.", self.body),
            Paragraph(f"Overall integrity: <b>{'PASSED' if intact else 'FAILED'}</b>.", self.body),
            Paragraph("This report supports review by model risk, validation and compliance "
                      "teams. It is not a compliance certification or legal advice.", self.note),
        ]

    def styles_title(self):
        return ParagraphStyle("title", parent=self.h1, fontSize=18, leading=22)

    def _model(self) -> list:
        version = self.records[-1]["model_version"] if self.records else "n/a"
        model_hash = self.records[-1]["model_hash"] if self.records else "n/a"
        out = [Paragraph("2. Model", self.h1),
               Paragraph(f"Version: {version}", self.body),
               Paragraph(f"File SHA-256: <font size=7>{model_hash}</font>", self.body)]
        m = self.metadata.get("metrics")
        if m:
            out.append(Paragraph(
                f"Development metrics ({self.metadata.get('dataset', '')}): test AUC "
                f"{m['test_auc']}, test Gini {m['test_gini']}, 5-fold CV AUC {m['cv_auc']}, "
                f"train AUC {m['train_auc']}. Test accuracy {m['test_accuracy']} against "
                f"{m['majority_class_accuracy']} for always predicting the majority class.",
                self.body))
            excluded = ", ".join(self.metadata.get("excluded_from_model", []))
            if excluded:
                out.append(Paragraph(f"Fields excluded from the model: {excluded}.", self.body))
        return out

    def _integrity(self) -> list:
        def status(r):
            return "Intact" if r["intact"] else f"{len(r['breaks'])} problem(s)"
        rows = [
            ["Decision chain", f"{self.chain['total']} records", status(self.chain)],
            ["Outcome chain", f"{self.outcomes['total']} events", status(self.outcomes)],
            ["Signed anchors", f"{self.anchors['checked']} files",
             "Intact" if self.anchors["intact"] else "FAILED"],
        ]
        out = [Paragraph("3. Integrity", self.h1),
               self._table(["Check", "Scope", "Result"], rows, [2, 2, 3])]
        problems = [f"Decision #{b['position'] + 1}: {b['reason']}" for b in self.chain["breaks"]]
        problems += [f"Outcome #{b['position'] + 1}: {b['reason']}" for b in self.outcomes["breaks"]]
        problems += [f"Anchor at {r['record_count']} decisions: {p}"
                     for r in self.anchors["results"] for p in r["problems"]]
        if self.anchors.get("error"):
            problems.append(self.anchors["error"])
        for p in problems[:20]:
            out.append(Paragraph(p, self.small))
        out.append(Paragraph(
            f"{self.anchors.get('decisions_after_last_anchor', 0)} decision(s) were logged after "
            "the last valid anchor; they are protected by the hash chain only until the next "
            "anchor is signed.", self.note))
        return out

    def _performance(self) -> list:
        p = PerformanceMonitor(self.db).compute()
        rows = [["Decisions with known outcome", p["n_with_outcome"]],
                ["Gini", p["gini"] if p["gini"] is not None else "insufficient outcomes"],
                ["KS", p["ks"] if p["ks"] is not None else "insufficient outcomes"],
                ["PSI vs development sample",
                 f"{p['psi']} ({p['psi_status']})" if p["psi"] is not None else "n/a"]]
        return [Paragraph("4. Performance", self.h1),
                self._table(["Measure", "Value"], rows, [3, 3]),
                Paragraph("PSI below 0.10 is conventionally read as stable, 0.10 to 0.25 as "
                          "worth investigating, and above 0.25 as a significant shift.", self.note)]

    def _fairness(self) -> list:
        monitor = FairnessMonitor(self.db)
        out = [Paragraph("5. Fairness", self.h1)]
        for col in GROUP_COLUMNS:
            m = monitor.compute_all_metrics(group_column=col)
            label = "Sex" if col == "sex" else "Age group"
            out.append(Paragraph(label, self.h2))
            for key, title, rate in (("demographic_parity", "Approval rate", "approval_rate"),
                                     ("equal_opportunity", "Approval rate among repaid",
                                      "true_positive_rate")):
                r = m[key]
                groups = r.get("groups", {})
                rows = [[g, f"{v[rate]:.3f}", v["n"], "yes" if v["included"] else "no (too few)"]
                        for g, v in groups.items()]
                if rows:
                    out.append(self._table(["Group", title, "n", "In gap?"], rows, [2, 2, 1, 2]))
                if r.get("available"):
                    out.append(Paragraph(
                        f"Gap {r['max_disparity']:.3f} against an illustrative review threshold "
                        f"of {r['threshold']}: <b>{'REVIEW' if r['breached'] else 'within'}</b>.",
                        self.small))
                else:
                    out.append(Paragraph(f"Not available: {r['reason']}", self.small))
                out.append(Spacer(1, 2 * mm))
        out.append(Paragraph(
            f"Groups with fewer than {monitor.MIN_GROUP_SIZE} cases are shown but excluded from "
            "the gap. Approval-rate parity and equal opportunity can conflict when repayment "
            "rates differ between groups; which to prioritise, and the threshold, is a policy "
            "decision.", self.note))
        return out

    def _regulatory(self) -> list:
        return [Paragraph("6. Regulatory context", self.h1),
                Paragraph("What this evidence can support. It does not establish compliance on "
                          "its own.", self.small),
                self._table(["Framework", "Relevant expectation", "Evidence ModelLedger provides"],
                            [list(r) for r in REGULATORY_CONTEXT], [2.2, 3, 4.8])]

    def _governance(self) -> list:
        g = self.governance
        rows = [
            ["Alert review", f"{g.alert_review.reviewer}; respond within "
                             f"{g.alert_review.response_time_days} business days; escalate to "
                             f"{g.alert_review.escalation_path}"],
            ["Log retention", f"At least {g.log_retention.minimum_years} years "
                              f"({g.log_retention.basis}); archive: {g.log_retention.archive}"],
            ["Access", f"Read: {', '.join(g.access_controls.read_access)}. Write: "
                       f"{', '.join(g.access_controls.write_access)}. Admin: "
                       f"{g.access_controls.db_admin_access}"],
            ["Review cycle", f"Integrity: {g.review_cycle.ledger_integrity}; fairness: "
                             f"{g.review_cycle.fairness_metrics}; performance: "
                             f"{g.review_cycle.performance_metrics}; regulatory mapping: "
                             f"{g.review_cycle.regulatory_mapping}"],
        ]
        return [Paragraph("7. Governance", self.h1),
                Paragraph("Illustrative settings from config/governance.yaml.", self.small),
                self._table(["Area", "Setting"], rows, [1.5, 6])]
