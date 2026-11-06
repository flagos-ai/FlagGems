#!/usr/bin/env python3
"""Run the official comprehensive Operator matrix and trace Native dispatch."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "experiments" / "native_ext"))
sys.path.insert(0, str(ROOT / "src"))

parser = argparse.ArgumentParser()
parser.add_argument("state", choices=("on", "off"))
parser.add_argument("--output", type=Path)
parser.add_argument("--warmup", type=int, default=10)
parser.add_argument("--iter", type=int, default=10)
args = parser.parse_args()
os.environ["FLAGGEMS_MTHREADS_NATIVE_INDEX_PUT"] = "1" if args.state == "on" else "0"

import pytest  # noqa: E402
import torch_musa  # noqa: E402,F401

output = args.output or (OUT / "results" / f"official_{args.state}.json")
output = output if output.is_absolute() else ROOT / output
output.parent.mkdir(parents=True, exist_ok=True)
pytest_args = [
    "-q",
    str(ROOT / "benchmark" / "test_index_put.py"),
    "--mode",
    "operator",
    "--level",
    "comprehensive",
    "--warmup",
    str(args.warmup),
    "--iter",
    str(args.iter),
    "--record",
    "json",
    "--output",
    str(output),
    "--parallel",
    "0",
]
result = pytest.main(pytest_args)
print(f"benchmark report: {output}")
raise SystemExit(int(result))
