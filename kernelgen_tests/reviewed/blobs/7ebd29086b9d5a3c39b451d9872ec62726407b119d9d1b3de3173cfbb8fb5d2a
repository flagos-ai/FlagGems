"""Explicit, evidence-backed benchmark skips, never runtime error suppression."""

import hashlib
import json
from functools import lru_cache
from pathlib import Path

import pytest


@lru_cache(maxsize=1)
def _rules():
    rows = json.loads(Path(__file__).with_suffix(".json").read_text())
    return {
        (vendor, case_id, digest): reason for vendor, case_id, digest, reason in rows
    }


def skip_verified_case(vendor, case):
    metadata = {"dtype": str(case.dtype), "shape": case.shape, "params": case.params}
    digest = hashlib.sha256(
        json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    reason = _rules().get((vendor, case.case_id, digest))
    if reason:
        pytest.skip(f"Verified {reason}: {vendor} {case.case_id}")
