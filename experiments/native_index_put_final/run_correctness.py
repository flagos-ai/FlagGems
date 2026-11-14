#!/usr/bin/env python3
"""Run the official correctness suite against this checkout and count Native hits."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "native_ext"))
sys.path.insert(0, str(ROOT / "src"))

parser = argparse.ArgumentParser()
parser.add_argument("state", choices=("on", "off"))
args = parser.parse_args()
native_on = args.state == "on"
os.environ["FLAGGEMS_MTHREADS_NATIVE_INDEX_PUT"] = "1" if native_on else "0"

import torch_musa  # noqa: E402,F401

calls = {
    "native_index_put_": 0,
    "try_native_index_put_fullrank_": 0,
    "try_native_index_put_accumulate_": 0,
    "try_native_index_put_bool_": 0,
    "try_native_index_put_row_clone_": 0,
}
try:
    import native_index_put_ext as extension  # noqa: E402

    for name in calls:
        original = getattr(extension, name, None)
        if original is None:
            continue

        def counted(*call_args, _name=name, _original=original, **kwargs):
            calls[_name] += 1
            return _original(*call_args, **kwargs)

        setattr(extension, name, counted)
except (ImportError, OSError):
    extension = None

import pytest  # noqa: E402

result = pytest.main(["-q", str(ROOT / "tests" / "test_index_put.py")])
print(
    "Native extension calls:",
    ", ".join(f"{name}={count}" for name, count in calls.items()),
)
if native_on and sum(calls.values()) == 0:
    raise SystemExit("Native mode was requested but no Native extension call occurred")
raise SystemExit(int(result))
