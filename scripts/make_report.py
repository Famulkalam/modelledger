#!/usr/bin/env python3
"""Write a PDF evidence pack for the current ledger to reports/.

    python scripts/make_report.py
"""

import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from modelledger.compliance.report import ComplianceReportGenerator  # noqa: E402
from modelledger.core.database import LedgerDB  # noqa: E402

if __name__ == "__main__":
    db = LedgerDB(os.environ.get("DB_PATH", os.path.join("data", "modelledger.db")))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    path = os.path.join("reports", f"modelledger_report_{stamp}.pdf")
    print(ComplianceReportGenerator(db, anchor_dir=os.environ.get("ANCHOR_DIR", "anchors"))
          .generate(path))
