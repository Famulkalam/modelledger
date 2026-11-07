#!/usr/bin/env python3
"""Export the API's OpenAPI spec to docs/openapi.json.

    python scripts/export_openapi.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("MODELLEDGER_API_KEY", "spec-export")

from modelledger.api.server import app  # noqa: E402

if __name__ == "__main__":
    os.makedirs("docs", exist_ok=True)
    spec = app.openapi()
    with open(os.path.join("docs", "openapi.json"), "w") as f:
        json.dump(spec, f, indent=2)
    print(f"Wrote docs/openapi.json ({len(spec['paths'])} paths)")
