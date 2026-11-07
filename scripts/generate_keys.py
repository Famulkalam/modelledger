#!/usr/bin/env python3
"""
Generate an Ed25519 key pair for signing anchors.

    python scripts/generate_keys.py

Paste the output into your .env file. Keep ANCHOR_SIGNING_KEY secret (in
production: a KMS or HSM). ANCHOR_PUBLIC_KEY can be shared with auditors so
they can verify anchors without being able to create them.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modelledger.core.anchor import generate_keypair  # noqa: E402

if __name__ == "__main__":
    private_hex, public_hex = generate_keypair()
    print(f"ANCHOR_SIGNING_KEY={private_hex}")
    print(f"ANCHOR_PUBLIC_KEY={public_hex}")
