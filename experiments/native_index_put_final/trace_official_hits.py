#!/usr/bin/env python3
"""Count actual Native launches during the official preflight workload."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "experiments" / "native_ext"))
sys.path.insert(0, str(ROOT / "src"))

parser = argparse.ArgumentParser()
parser.add_argument(
    "--output", type=Path, default=OUT / "results" / "native_preflight.json"
)
args = parser.parse_args()
args.output.parent.mkdir(parents=True, exist_ok=True)
os.environ["FLAGGEMS_MTHREADS_NATIVE_INDEX_PUT"] = "1"

import torch  # noqa: E402
import torch_musa  # noqa: E402, F401

counts = Counter()
import native_index_put_ext as extension  # noqa: E402


def api_from_stack():
    frame = sys._getframe()
    accumulate = None
    while frame is not None:
        if accumulate is None and isinstance(frame.f_locals.get("accumulate"), bool):
            accumulate = frame.f_locals["accumulate"]
        owner = frame.f_locals.get("self")
        operator = getattr(owner, "op_name", None)
        if operator in ("index_put", "index_put_"):
            return operator, accumulate
        frame = frame.f_back
    return "unknown", accumulate


def wrap(symbol):
    original = getattr(extension, symbol)

    def traced(input_tensor, index_arg, value_tensor, *rest, **kwargs):
        api, accumulate = api_from_stack()
        if torch.is_tensor(index_arg):
            index_shapes = [tuple(index_arg.shape)]
        else:
            index_shapes = [tuple(index.shape) for index in index_arg]
        signature = (
            api,
            accumulate,
            symbol,
            str(input_tensor.dtype),
            tuple(input_tensor.shape),
            tuple(index_shapes),
            tuple(value_tensor.shape),
        )
        counts[(signature, "attempted")] += 1
        result = original(input_tensor, index_arg, value_tensor, *rest, **kwargs)
        if result is not None:
            counts[(signature, "launched")] += 1
        return result

    setattr(extension, symbol, traced)


for _symbol in (
    "native_index_put_",
    "try_native_index_put_fullrank_",
    "try_native_index_put_accumulate_",
    "try_native_index_put_bool_",
    "try_native_index_put_row_clone_",
):
    if hasattr(extension, _symbol):
        wrap(_symbol)

import pytest  # noqa: E402

result = pytest.main(
    [
        "-q",
        str(ROOT / "benchmark" / "test_index_put.py"),
        "--mode",
        "operator",
        "--level",
        "comprehensive",
        "--warmup",
        "1",
        "--iter",
        "1",
        "--record",
        "json",
        "--output",
        str(args.output),
        "--parallel",
        "0",
    ]
)

trace_path = args.output.with_name("native_hit_cases.csv")
with trace_path.open("w", newline="") as file:
    fields = [
        "api",
        "extension_symbol",
        "dtype",
        "input_shape",
        "index_shapes",
        "values_shape",
        "accumulate",
        "attempts",
        "successful_dispatches",
        "kernel_launches",
    ]
    writer = csv.DictWriter(file, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    signatures = sorted({signature for signature, _ in counts}, key=repr)
    for signature in signatures:
        api, accumulate, symbol, dtype, input_shape, index_shapes, values_shape = (
            signature
        )
        dispatches = counts[(signature, "launched")]
        kernels_per_dispatch = 3 if symbol == "try_native_index_put_bool_" else 1
        writer.writerow(
            {
                "api": api,
                "accumulate": accumulate,
                "extension_symbol": symbol,
                "dtype": dtype,
                "input_shape": repr(input_shape),
                "index_shapes": repr(index_shapes),
                "values_shape": repr(values_shape),
                "attempts": counts[(signature, "attempted")],
                "successful_dispatches": dispatches,
                "kernel_launches": dispatches * kernels_per_dispatch,
            }
        )
launched = sum(counts[(sig, "launched")] for sig in signatures)
print(f"instrumented official-workload Native trace: {trace_path}")
print(f"Native launched calls during timing: {launched}")
raise SystemExit(int(result))
