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
from flag_gems.fused.dsv41 import mhc
from tests.dsv41_helpers import assert_exact, mhc_inputs


def cases(quick):
    sizes = sorted(
        {6 * b for b in [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20]}
        | {5 * b for b in [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20]}
    )
    return [(m,) for m in ([12, 24, 40, 48, 60, 96] if quick else sizes)]


def prepare(shape):
    x, w, s, b = mhc_inputs(*shape)

    def base():
        return mhc.hc_split_sinkhorn(mhc.hc_mix_stats(x, w, 1e-6, tuned=False), s, b)

    cand = (
        (lambda: mhc.hc_mix_stats_sinkhorn_exact(x, w, s, b, 4, 20, 1e-6, 1e-6))
        if shape[0] in [36, 40, 42, 48, 50, 60, 80, 84, 96]
        else (lambda: mhc.hc_split_sinkhorn(mhc.hc_mix_stats(x, w, 1e-6), s, b))
    )
    return base, cand


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
        method="median CUDA Graph; alternate order and reverse shapes in second round; full public wrapper incl. reductions",
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
