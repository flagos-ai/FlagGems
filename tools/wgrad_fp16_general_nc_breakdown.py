#!/usr/bin/env python3
# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

"""Break down fp16/bf16 general-NC overhead for wgrad_gemm_accum_fp16.

Run inside Docker GPU env:
  CUDA_VISIBLE_DEVICES=4 PYTHONPATH=/workspace/FlagGems/src \
  python tools/wgrad_fp16_general_nc_breakdown.py
"""

from __future__ import annotations

import argparse
import time
from typing import Callable

import torch

from flag_gems.ops import wgrad_gemm_accum_fp16


Shape = tuple[int, int, int]  # (K, N, M)


def _as_transpose_nc(contig_2d: torch.Tensor) -> torch.Tensor:
    rows, cols = contig_2d.shape
    nc = torch.empty(cols, rows, dtype=contig_2d.dtype, device=contig_2d.device).t()
    nc.copy_(contig_2d)
    assert not nc.is_contiguous()
    assert nc.t().is_contiguous()
    return nc


def _as_general_nc(contig_2d: torch.Tensor) -> torch.Tensor:
    rows, cols = contig_2d.shape
    padded = torch.empty(rows, cols * 2, dtype=contig_2d.dtype, device=contig_2d.device)
    nc = padded[:, :cols]
    nc.copy_(contig_2d)
    assert not nc.is_contiguous()
    assert not nc.t().is_contiguous()
    return nc


def _bench_ms(fn: Callable[[], None], warmup: int, reps: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0 / reps


def _parse_shapes(arg: str) -> list[Shape]:
    shapes: list[Shape] = []
    for item in arg.split(";"):
        k_str, n_str, m_str = item.strip().split(",")
        shapes.append((int(k_str), int(n_str), int(m_str)))
    return shapes


def run_one(dtype: torch.dtype, shape: Shape, warmup: int, reps: int) -> None:
    k_dim, n_dim, m_dim = shape
    x = torch.randn(k_dim, n_dim, device="cuda", dtype=dtype)
    g = torch.randn(k_dim, m_dim, device="cuda", dtype=dtype)
    seed = torch.randn(m_dim, n_dim, device="cuda", dtype=dtype)

    main_contig = seed.clone()
    main_tnc = _as_transpose_nc(seed)
    main_gnc = _as_general_nc(seed)

    t_contig = _bench_ms(lambda: wgrad_gemm_accum_fp16(x, g, main_contig), warmup, reps)
    t_tnc = _bench_ms(lambda: wgrad_gemm_accum_fp16(x, g, main_tnc), warmup, reps)
    t_gnc = _bench_ms(lambda: wgrad_gemm_accum_fp16(x, g, main_gnc), warmup, reps)

    # Approximate slow-path decomposition for general-NC:
    # weight = main_gnc.contiguous(); addmm(weight, g.T, x, out=weight); main_gnc.copy_(weight)
    g_t = g.t()
    x_c = x
    weight_buf = seed.clone()
    t_densify = _bench_ms(lambda: main_gnc.contiguous(), warmup, reps)
    t_addmm = _bench_ms(
        lambda: torch.addmm(weight_buf, g_t, x_c, beta=1, alpha=1, out=weight_buf),
        warmup,
        reps,
    )
    t_copyback = _bench_ms(lambda: main_gnc.copy_(weight_buf), warmup, reps)

    print(
        f"{str(dtype):<14} ({k_dim:>4},{n_dim:>4},{m_dim:>4}) "
        f"contig={t_contig:>7.3f}ms tnc={t_tnc:>7.3f}ms gnc={t_gnc:>7.3f}ms "
        f"gnc/contig={t_gnc / t_contig:>5.2f} "
        f"[densify={t_densify:>6.3f}, addmm={t_addmm:>6.3f}, copy={t_copyback:>6.3f}]"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shapes",
        type=str,
        default="256,2048,4096;1024,1024,1024;2048,2048,2048;8192,4096,4096",
        help="Semicolon-separated K,N,M list. Example: 256,2048,4096;1024,1024,1024",
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--reps", type=int, default=80)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    dtypes = [torch.float16]
    if torch.cuda.is_bf16_supported():
        dtypes.append(torch.bfloat16)

    shapes = _parse_shapes(args.shapes)
    print("dtype          shape(K,N,M)   latencies and general-NC breakdown")
    for dtype in dtypes:
        for shape in shapes:
            run_one(dtype, shape, args.warmup, args.reps)


if __name__ == "__main__":
    main()
