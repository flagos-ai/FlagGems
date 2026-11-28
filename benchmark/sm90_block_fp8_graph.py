# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Correctness-gated, alternating full-wrapper CUDA Graph benchmarks."""

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

import torch
import triton

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flag_gems.ops._w8a8_block_fp8_configs import (  # noqa: E402
    _H100_MIDRANGE_FP8_CONFIGS,
)
from flag_gems.ops.w8a8_block_fp8_matmul import w8a8_block_fp8_matmul  # noqa: E402
from tests.sm90_fp8_helpers import (  # noqa: E402
    assert_exact,
    fp8_inputs,
    native_block_fp8_baseline,
)


def cases(quick):
    return (
        [
            (6, 1792, 5120),
            (24, 1792, 5120),
            (48, 1792, 5120),
            (96, 5120, 4096),
            (24, 4608, 5120),
            (48, 5120, 2304),
        ]
        if quick
        else sorted(_H100_MIDRANGE_FP8_CONFIGS)
    )


def prepare(shape):
    xs = fp8_inputs(*shape)
    return lambda: native_block_fp8_baseline(*xs), lambda: w8a8_block_fp8_matmul(
        *xs, [32, 32]
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument("--rep-ms", type=int, default=100)
    a = p.parse_args()
    out = Path(a.out)
    if out.exists():
        raise FileExistsError(out)
    matrix = list(cases(a.quick))
    rows = []
    meta = dict(
        gpu=torch.cuda.get_device_name(),
        torch=torch.__version__,
        triton=triton.__version__,
        rounds=a.rounds,
        rep_ms=a.rep_ms,
        warmup=25,
        method=(
            "median CUDA Graph; alternate order and reverse shapes in second round; "
            "native test helper versus public wrapper, both incl. reductions"
        ),
        measurements=rows,
    )
    for rnd in range(a.rounds):
        for idx, shape in enumerate(matrix if rnd % 2 == 0 else list(reversed(matrix))):
            baseline, candidate = prepare(shape)
            assert_exact(candidate(), baseline())
            for _ in range(25):
                baseline()
                candidate()
            torch.cuda.synchronize()
            times = {}
            funcs = [("baseline", baseline), ("candidate", candidate)]
            if (rnd + idx) % 2:
                funcs.reverse()
            for name, fn in funcs:
                times[name] = 1000 * triton.testing.do_bench_cudagraph(
                    fn, rep=a.rep_ms, return_mode="median"
                )
            row = dict(
                shape=shape,
                round=rnd + 1,
                baseline_us=times["baseline"],
                candidate_us=times["candidate"],
                byte_exact=True,
            )
            rows.append(row)
            out.write_text(json.dumps(meta, indent=2))
            print(json.dumps(row), flush=True)
    grouped = []
    for shape in matrix:
        part = [r for r in rows if r["shape"] == shape]
        before = statistics.mean(r["baseline_us"] for r in part)
        after = statistics.mean(r["candidate_us"] for r in part)
        grouped.append(
            dict(
                shape=shape,
                baseline_us=before,
                candidate_us=after,
                speedup=before / after,
            )
        )
    ratios = [r["speedup"] for r in grouped]
    meta["summary"] = dict(
        cases=len(grouped),
        geometric_mean=math.exp(statistics.mean(map(math.log, ratios))),
        minimum=min(ratios),
        maximum=max(ratios),
        rows=grouped,
    )
    meta["completed_epoch"] = time.time()
    out.write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta["summary"]))


if __name__ == "__main__":
    main()
