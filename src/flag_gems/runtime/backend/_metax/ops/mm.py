# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""MetaX MM: classify the workload, then launch a tuned plan.

Dispatch reads shapes, strides, dtypes and pointer alignment. It selects one
family -- vector forwarding to mv.py, a SIMT reduction in either orientation,
a tiled dot, a dual dot sharing its LHS, or a triangular self-transpose
product -- plus an optional K partition and an optional RHS repacking. The
self-transpose route requires identical input pointers and reversed strides.
Dispatch never benchmarks candidates; standard libtuner selects kernel tiles.

Strides reach the kernels as constexpr, so a padded or sliced view compiles to
the same specialized code a dense one gets instead of falling back to a slow
general form. Workload decisions and compiled code are cached; partial results
are call-local.
"""

import copy
import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, NamedTuple

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, libtuner

from .mv import mv

logger = logging.getLogger(__name__)
EXPAND_CONFIG_FILENAME = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "mm_metax_expand.yaml")
)

# MACA MMA rejects, or silently miscomputes, a tl.dot whose M/N/K tile is
# smaller than 16. YAML candidates already sit on this floor; AABS must not
# shrink them further.
_MMA_MIN = 16
# SIMT only wins when both output axes fit inside one MMA tile. A long axis
# lets a masked 16-wide GEMM reuse the other operand; reloading it per SIMT
# CTA is the losing traffic.
_SIMT_EXTENT = 8
_SIMT_WIDE = 32
# Nominal SIMT tile over the wide axis, used only to estimate CTA counts.
_SIMT_TILE = 32
# Each K partition must run at least this many BK steps to amortize its
# prologue and the separate reduction pass that follows it.
_SPLIT_MIN_K_TILES = 2
_SPLIT_MAX = 32
# One reduction step of the tiled kernels.
_K_TILE = 64
# C550's native FP16/BF16 GEMM uses a 128x128x128, four-stage async pipeline.
# Four Triton warps is the measured mapping for this shape; the generic
# shared-memory budget below is intentionally bypassed for this exact dense-NN
# candidate.
_MMA_NATIVE_TILE = (128, 128, 128)
_MMA_NATIVE_WARPS = 4
_MMA_NATIVE_STAGES = 4


# Workload description. These records hold no tensors.


@dataclass(frozen=True)
class _Features:
    """Exact call metadata plus the derived quantities dispatch reasons about."""

    m: int
    n: int
    k: int
    a_strides: tuple
    b_strides: tuple
    c_strides: tuple
    dtype: torch.dtype
    out_dtype: torch.dtype
    aligned: bool
    self_transpose: bool
    sm_count: int
    shared_bytes: int
    l2_bytes: int

    @property
    def half(self):
        return self.dtype in (torch.float16, torch.bfloat16)

    @property
    def element_size(self):
        return 2 if self.half else 4

    @property
    def reduction_tiles(self):
        return triton.cdiv(self.k, _K_TILE)

    @property
    def rhs_k_contiguous(self):
        return self.b_strides[0] == 1

    @property
    def dense_operands(self):
        return self.a_strides == (self.k, 1) and self.b_strides in (
            (self.n, 1),
            (1, self.k),
        )

    @property
    def dense_output(self):
        return self.c_strides == (self.n, 1)

    @property
    def vector_aligned(self):
        # Eight half elements form the 16-byte vector load these tiles issue.
        return self.aligned and all(
            stride == 1 or stride % 8 == 0
            for stride in self.a_strides + self.b_strides + self.c_strides
        )

    def tiles(self, rows, columns):
        return triton.cdiv(self.m, rows) * triton.cdiv(self.n, columns)

    def wave_work(self, rows, columns, occupancy=1):
        capacity = self.sm_count * occupancy
        return triton.cdiv(self.tiles(rows, columns), capacity) * rows * columns

    def tile_utilization(self, rows, columns):
        return self.m * self.n / (self.tiles(rows, columns) * rows * columns)

    def wave_utilization(self, rows, columns):
        tiles = self.tiles(rows, columns)
        slots = triton.cdiv(tiles, self.sm_count) * self.sm_count
        return tiles / slots


class _MmCall(NamedTuple):
    """Tensors and metadata for one invocation."""

    a: torch.Tensor
    b: torch.Tensor
    c: torch.Tensor
    f: _Features


class _MmPlan(NamedTuple):
    """An immutable execution choice; never owns tensors or workspace."""

    launch: Callable
    split_k: int = 1
    pack_rhs: bool = False


# Triton arithmetic families.


@triton.jit
def mm_kernel(
    A,
    B,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    SAM: tl.constexpr,
    SAK: tl.constexpr,
    SBK: tl.constexpr,
    SBN: tl.constexpr,
    SCM: tl.constexpr,
    SCN: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    GROUP_M: tl.constexpr = 1,
    SPLIT_K: tl.constexpr = 1,
    TRANSPOSE: tl.constexpr = False,
    STATIC_K: tl.constexpr = False,
):
    """One output tile per CTA, optionally with a deterministic K partition."""
    if TRANSPOSE:
        # Compute C^T = B^T A^T. Swapping the dot operands changes which
        # operand feeds which MMA port without materializing a transpose.
        A, B = B, A
        M, N = N, M
        SAM, SAK, SBK, SBN = SBN, SBK, SAK, SAM
        SCM, SCN = SCN, SCM
    nm, nn = tl.cdiv(M, BM), tl.cdiv(N, BN)
    split = tl.program_id(0) // (nm * nn) % SPLIT_K
    pid = tl.program_id(0) % (nm * nn)
    group = pid // (GROUP_M * nn)
    first_m = group * GROUP_M
    group_m = tl.minimum(nm - first_m, GROUP_M)
    local = pid % (GROUP_M * nn)
    pm = first_m + local % group_m
    pn = local // group_m
    mi = pm * BM + tl.arange(0, BM)
    ni = pn * BN + tl.arange(0, BN)
    iterations = tl.cdiv(K, BK * SPLIT_K)
    ki = tl.arange(0, BK) + split * iterations * BK
    # Tell the vectorizer each axis is a dense power-of-two tile. The values
    # do not change; masked tails still compare against M/N/K below.
    mi = tl.max_contiguous(tl.multiple_of(mi, BM), BM)
    ni = tl.max_contiguous(tl.multiple_of(ni, BN), BN)
    ki = tl.max_contiguous(tl.multiple_of(ki, BK), BK)
    ap = A + mi[:, None].to(tl.int64) * SAM + ki[None, :].to(tl.int64) * SAK
    bp = B + ki[:, None].to(tl.int64) * SBK + ni[None, :].to(tl.int64) * SBN
    acc = tl.zeros((BM, BN), tl.float32)
    if STATIC_K:
        # Short reductions cannot amortize a pipelined loop's prologue and
        # epilogue. Keep addresses in int64 here too, including sliced views.
        for k in tl.static_range((K + BK * SPLIT_K - 1) // (BK * SPLIT_K)):
            if K % (BK * SPLIT_K) == 0:
                if M % BM == 0:
                    ak = tl.load(ap)
                else:
                    ak = tl.load(ap, mi[:, None] < M, other=0)
                if N % BN == 0:
                    bk = tl.load(bp)
                else:
                    bk = tl.load(bp, ni[None, :] < N, other=0)
            else:
                ak = tl.load(
                    ap, (mi[:, None] < M) & (ki[None, :] + k * BK < K), other=0
                )
                bk = tl.load(
                    bp, (ki[:, None] + k * BK < K) & (ni[None, :] < N), other=0
                )
            acc = tl.dot(ak, bk, acc, out_dtype=tl.float32, allow_tf32=False)
            ap += BK * SAK
            bp += BK * SBK
    else:
        for k in range(iterations):
            if K % (BK * SPLIT_K) == 0:
                if M % BM == 0:
                    ak = tl.load(ap)
                else:
                    ak = tl.load(ap, mi[:, None] < M, other=0)
                if N % BN == 0:
                    bk = tl.load(bp)
                else:
                    bk = tl.load(bp, ni[None, :] < N, other=0)
            else:
                ak = tl.load(
                    ap, (mi[:, None] < M) & (ki[None, :] + k * BK < K), other=0
                )
                bk = tl.load(
                    bp, (ki[:, None] + k * BK < K) & (ni[None, :] < N), other=0
                )
            acc = tl.dot(ak, bk, acc, out_dtype=tl.float32, allow_tf32=False)
            ap += BK * SAK
            bp += BK * SBK
    cp = (
        C
        + split.to(tl.int64) * M * N
        + mi[:, None].to(tl.int64) * SCM
        + ni[None, :].to(tl.int64) * SCN
    )
    tl.store(cp, acc, (mi[:, None] < M) & (ni[None, :] < N))


@triton.jit
def mm_kernel_nn(
    A,
    B,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    GROUP_M: tl.constexpr = 1,
    TRANSPOSE: tl.constexpr = False,
    STATIC_K: tl.constexpr = False,
):
    """Mask-free dense-NN kernel for dimensions divisible by the selected tile."""
    nm, nn = tl.cdiv(M, BM), tl.cdiv(N, BN)
    pid = tl.program_id(0)
    group = pid // (GROUP_M * nn)
    first_m = group * GROUP_M
    group_m = tl.minimum(nm - first_m, GROUP_M)
    local = pid % (GROUP_M * nn)
    pm = first_m + local % group_m
    pn = local // group_m
    mi = pm * BM + tl.arange(0, BM)
    ni = pn * BN + tl.arange(0, BN)
    ki = tl.arange(0, BK)
    mi = tl.max_contiguous(tl.multiple_of(mi, BM), BM)
    ni = tl.max_contiguous(tl.multiple_of(ni, BN), BN)
    ki = tl.max_contiguous(tl.multiple_of(ki, BK), BK)
    ap = A + mi[:, None].to(tl.int64) * K + ki[None, :].to(tl.int64)
    bp = B + ki[:, None].to(tl.int64) * N + ni[None, :].to(tl.int64)
    acc = tl.zeros((BM, BN), tl.float32)
    # Carry the four-stage depth into the loop so MetaX can overlap K panels.
    for _ in tl.range(0, tl.cdiv(K, BK), 1, num_stages=4):
        acc = tl.dot(
            tl.load(ap),
            tl.load(bp),
            acc,
            out_dtype=tl.float32,
            allow_tf32=False,
        )
        ap += BK
        bp += BK * N
    cp = C + mi[:, None].to(tl.int64) * N + ni[None, :].to(tl.int64)
    tl.store(cp, acc)


@triton.jit
def _syrk_kernel(
    A,
    C,
    M: tl.constexpr,
    K: tl.constexpr,
    SAM: tl.constexpr,
    SAK: tl.constexpr,
    SCM: tl.constexpr,
    SCN: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
):
    """Compute the lower triangle of A @ A.T and mirror off-diagonal tiles."""
    pid = tl.program_id(0).to(tl.int64)
    row = ((tl.sqrt(8.0 * pid.to(tl.float32) + 1.0) - 1.0) * 0.5).to(tl.int64)
    # Correct either direction of rounding at a triangular-number boundary.
    row = tl.where(row * (row + 1) // 2 > pid, row - 1, row)
    row = tl.where((row + 1) * (row + 2) // 2 <= pid, row + 1, row)
    column = pid - row * (row + 1) // 2
    mi = row * BT + tl.arange(0, BT)
    ni = column * BT + tl.arange(0, BT)
    mi = tl.max_contiguous(tl.multiple_of(mi, BT), BT)
    ni = tl.max_contiguous(tl.multiple_of(ni, BT), BT)
    ki = tl.arange(0, BK)
    ap = A + mi[:, None] * SAM + ki[None, :] * SAK
    bp = A + ki[:, None] * SAK + ni[None, :] * SAM
    acc = tl.zeros((BT, BT), tl.float32)
    for _ in range(tl.cdiv(K, BK)):
        # Dispatch guarantees complete K tiles; M tails remain masked.
        if M % BT == 0:
            av = tl.load(ap)
            bv = tl.load(bp)
        else:
            av = tl.load(ap, mi[:, None] < M, other=0)
            bv = tl.load(bp, ni[None, :] < M, other=0)
        acc = tl.dot(av, bv, acc, out_dtype=tl.float32, allow_tf32=False)
        ap += BK * SAK
        bp += BK * SAK
    mask = (mi[:, None] < M) & (ni[None, :] < M)
    tl.store(C + mi[:, None] * SCM + ni[None, :] * SCN, acc, mask)
    if row != column:
        tl.store(C + ni[None, :] * SCM + mi[:, None] * SCN, acc, mask)


@triton.jit
def _dual_gemm_kernel(
    A,
    B,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    SAM: tl.constexpr,
    SAK: tl.constexpr,
    SBK: tl.constexpr,
    SBN: tl.constexpr,
    SCM: tl.constexpr,
    SCN: tl.constexpr,
    BM: tl.constexpr,
    B0: tl.constexpr,
    B1: tl.constexpr,
    BK: tl.constexpr,
    SWAP: tl.constexpr = False,
    GROUP_M: tl.constexpr = 1,
):
    """Two power-of-two dots share A, forming a narrower non-power-of-two tile.

    SWAP computes C^T = B^T A^T, storing through the original output strides.
    """
    if SWAP:
        A, B = B, A
        M, N = N, M
        SAM, SAK, SBK, SBN = SBN, SBK, SAK, SAM
        SCM, SCN = SCN, SCM
    nn = tl.cdiv(N, B0 + B1)
    nm = tl.cdiv(M, BM)
    pid = tl.program_id(0)
    group = pid // (GROUP_M * nn)
    first_m = group * GROUP_M
    group_m = tl.minimum(nm - first_m, GROUP_M)
    local = pid % (GROUP_M * nn)
    pm = first_m + local % group_m
    pn = local // group_m
    m = pm * BM + tl.arange(0, BM)
    n0 = pn * (B0 + B1) + tl.arange(0, B0)
    n1 = pn * (B0 + B1) + B0 + tl.arange(0, B1)
    k = tl.arange(0, BK)
    m = tl.max_contiguous(tl.multiple_of(m, BM), BM)
    n0 = tl.max_contiguous(tl.multiple_of(n0, B0), B0)
    n1 = tl.max_contiguous(tl.multiple_of(n1, B1), B1)
    k = tl.max_contiguous(tl.multiple_of(k, BK), BK)
    ap = A + m[:, None].to(tl.int64) * SAM + k[None, :].to(tl.int64) * SAK
    bp0 = B + k[:, None].to(tl.int64) * SBK + n0[None, :].to(tl.int64) * SBN
    bp1 = B + k[:, None].to(tl.int64) * SBK + n1[None, :].to(tl.int64) * SBN
    c0 = tl.zeros((BM, B0), tl.float32)
    c1 = tl.zeros((BM, B1), tl.float32)
    for it in range(tl.cdiv(K, BK)):
        av = tl.load(ap, (m[:, None] < M) & (k[None, :] + it * BK < K), 0)
        b0 = tl.load(bp0, (n0[None, :] < N) & (k[:, None] + it * BK < K), 0)
        b1 = tl.load(bp1, (n1[None, :] < N) & (k[:, None] + it * BK < K), 0)
        c0 = tl.dot(av, b0, c0, out_dtype=tl.float32, allow_tf32=False)
        c1 = tl.dot(av, b1, c1, out_dtype=tl.float32, allow_tf32=False)
        ap += BK * SAK
        bp0 += BK * SBK
        bp1 += BK * SBK
    p0 = C + m[:, None].to(tl.int64) * SCM + n0[None, :].to(tl.int64) * SCN
    p1 = C + m[:, None].to(tl.int64) * SCM + n1[None, :].to(tl.int64) * SCN
    tl.store(p0, c0, (m[:, None] < M) & (n0[None, :] < N))
    tl.store(p1, c1, (m[:, None] < M) & (n1[None, :] < N))


@triton.jit
def _simt_row_kernel(
    A,
    B,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    SAM: tl.constexpr,
    SAK: tl.constexpr,
    SBK: tl.constexpr,
    SBN: tl.constexpr,
    SCM: tl.constexpr,
    SCN: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    SPLIT_K: tl.constexpr = 1,
):
    """One row of A per CTA: columns are the inner axis, so A broadcasts."""
    nn = tl.cdiv(N, BN)
    split = tl.program_id(0) // (M * nn) % SPLIT_K
    m = (tl.program_id(0) // nn % M).to(tl.int64)
    n = (tl.program_id(0) % nn * BN + tl.arange(0, BN)).to(tl.int64)
    iterations = tl.cdiv(K, BK * SPLIT_K)
    k = tl.arange(0, BK) + split.to(tl.int64) * iterations * BK
    acc = tl.zeros((BN, BK), tl.float32)
    for start in range(iterations):
        ks = k + start * BK
        a = tl.load(A + m * SAM + ks * SAK, ks < K, other=0).to(tl.float32)
        b = tl.load(
            B + n[:, None] * SBN + ks[None, :] * SBK,
            (n[:, None] < N) & (ks[None, :] < K),
            other=0,
        ).to(tl.float32)
        acc = tl.fma(a[None, :], b, acc)
    tl.store(
        C + split.to(tl.int64) * M * N + m * SCM + n * SCN,
        tl.sum(acc, axis=1),
        n < N,
    )


@triton.jit
def mm_kernel_small_n_partial(
    A,
    B,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    SAM: tl.constexpr,
    SAK: tl.constexpr,
    SBK: tl.constexpr,
    SBN: tl.constexpr,
    SCM: tl.constexpr,
    SCN: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
    SPLIT_K: tl.constexpr = 1,
):
    """One column of C per CTA: a BM tile of A shares each loaded B column."""
    nm = tl.cdiv(M, BM)
    split = tl.program_id(0) // (N * nm) % SPLIT_K
    n = (tl.program_id(0) // nm % N).to(tl.int64)
    m = (tl.program_id(0) % nm * BM + tl.arange(0, BM)).to(tl.int64)
    iterations = tl.cdiv(K, BK * SPLIT_K)
    k = tl.arange(0, BK) + split.to(tl.int64) * iterations * BK
    acc = tl.zeros((BM, BK), tl.float32)
    for start in range(iterations):
        ks = k + start * BK
        a = tl.load(
            A + m[:, None] * SAM + ks[None, :] * SAK,
            (m[:, None] < M) & (ks[None, :] < K),
            other=0,
        ).to(tl.float32)
        b = tl.load(B + ks * SBK + n * SBN, ks < K, other=0).to(tl.float32)
        acc = tl.fma(b[None, :], a, acc)
    tl.store(
        C + split.to(tl.int64) * M * N + m * SCM + n * SCN,
        tl.sum(acc, axis=1),
        m < M,
    )


@triton.jit
def _pack_rhs_kernel(
    B,
    T,
    R: tl.constexpr,
    C: tl.constexpr,
    SR: tl.constexpr,
    SC: tl.constexpr,
    BR: tl.constexpr,
    BC: tl.constexpr,
):
    """Materialize a K-contiguous RHS using a tiled Triton copy."""
    row = (tl.program_id(0) // tl.cdiv(C, BC) * BR + tl.arange(0, BR)).to(tl.int64)
    col = (tl.program_id(0) % tl.cdiv(C, BC) * BC + tl.arange(0, BC)).to(tl.int64)
    mask = (row[:, None] < R) & (col[None, :] < C)
    value = tl.load(B + row[:, None] * SR + col[None, :] * SC, mask, other=0)
    tl.store(T + row[:, None] + col[None, :] * R, value, mask)


@triton.jit
def mm_kernel_splitk_reduce(
    P,
    C,
    N: tl.constexpr,
    TOTAL: tl.constexpr,
    SCM: tl.constexpr,
    SCN: tl.constexpr,
    SPLIT_K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Each lane owns one output; add the FP32 partials without atomics."""
    i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    result = tl.zeros((BLOCK,), tl.float32)
    for s in tl.static_range(SPLIT_K):
        result += tl.load(P + s * TOTAL + i, i < TOTAL, other=0)
    tl.store(C + i // N * SCM + i % N * SCN, result, i < TOTAL)


# Per-family tuning and compiler constraints.


def _is_native_mma_candidate(config, args):
    return (
        _is_dense_mma_config(config)
        and args.get("SPLIT_K", 1) == 1
        and args["A"].dtype in (torch.float16, torch.bfloat16)
        and args["C"].dtype == args["A"].dtype
        and args["SAM"] == args["K"]
        and args["SAK"] == 1
        and args["SBK"] == args["N"]
        and args["SBN"] == 1
        and args["SCM"] == args["N"]
        and args["SCN"] == 1
    )


def _is_nt_mma_candidate(config, args):
    """Allow the native-shaped tile for K-contiguous NT partials."""
    return (
        _is_dense_mma_config(config)
        and args["A"].dtype in (torch.float16, torch.bfloat16)
        and args["SAK"] == 1
        and args["SBK"] == 1
        and args["SBN"] == args["K"]
        and args["SCN"] == 1
        and args.get("SPLIT_K", 1) > 1
    )


def _is_dense_mma_config(config):
    """Return the one native-shaped config accepted by the dense kernel."""
    meta = config.kwargs
    return (
        (meta["BM"], meta["BN"], meta["BK"]) == _MMA_NATIVE_TILE
        and meta["GROUP_M"] == 8
        and not meta["TRANSPOSE"]
        and not meta["STATIC_K"]
        and meta["pipeline"] == "cpasync"
        and meta["scenario"] == ""
        and config.num_warps == _MMA_NATIVE_WARPS
        and config.num_stages == _MMA_NATIVE_STAGES
    )


def _prune_dense(configs, named_args, **kwargs):
    args = {**named_args, **kwargs}
    if not (
        args["A"].dtype in (torch.float16, torch.bfloat16)
        and args["C"].dtype == args["A"].dtype
        and args["M"] == args["N"] == args["K"]
        and args["A"].stride() == (args["K"], 1)
        and args["B"].stride() == (args["N"], 1)
        and args["C"].stride() == (args["N"], 1)
    ):
        return []
    return [
        copy.deepcopy(config)
        for config in configs
        if _is_dense_mma_config(config)
        and all(
            args[axis] % config.kwargs[tile] == 0
            for axis, tile in (("M", "BM"), ("N", "BN"), ("K", "BK"))
        )
    ]


def _prune_gemm(configs, named_args, **kwargs):
    args = {**named_args, **kwargs}
    m, n, k = args["M"], args["N"], args["K"]
    size = args["A"].element_size()
    result = []
    for config in configs:
        meta = config.kwargs
        if meta["STATIC_K"] and (
            k > 128 or min(m, n) < 64 or args.get("SPLIT_K", 1) != 1
        ):
            continue
        mt, nt = (n, m) if meta["TRANSPOSE"] else (m, n)
        bm, bn, bk = meta["BM"], meta["BN"], meta["BK"]
        if bm < _MMA_MIN or bn < _MMA_MIN or bk < _MMA_MIN:
            continue
        if meta["scenario"] == "unprefetch" and (mt % bm or nt % bn or k % bk):
            # Masked edges make this layout spill heavily with deeper stages.
            continue
        if (
            size == 4
            and args.get("SPLIT_K", 1) > 1
            and k % (bk * args["SPLIT_K"]) != 0
            and meta["pipeline"].startswith("cpasync")
        ):
            # This FlagTree backend's genSwiMask asserts on the masked FP32
            # split-K async pipeline. The basic pipeline handles the same tail.
            continue
        if meta["STATIC_K"] and bm != bn and size != 4:
            continue
        if bm > max(32, triton.next_power_of_2(mt)) or bn > max(
            32, triton.next_power_of_2(nt)
        ):
            continue
        if bk > max(32, triton.next_power_of_2(k)):
            continue
        # FlagTree still allocates a second K buffer unless num_stages is 1.
        # unprefetch does not shrink a 256-wide tile; it doubles it to 128 KiB.
        # The native C550 MMA pipeline is a compiler-recognized exception:
        # its four-stage 128x128x128 candidate uses the async layout without
        # the generic two-buffer estimate. The NT partial uses the same
        # compiler allocation when B is K-contiguous and split-K is active.
        native_mma = _is_native_mma_candidate(config, args)
        nt_mma = _is_nt_mma_candidate(config, args)
        buffers = 1 if config.num_stages <= 1 else min(config.num_stages, 2)
        if not (native_mma or nt_mma) and (bm + bn) * bk * size * buffers > 65536:
            continue
        if config.num_stages > 2 and k <= bk:
            continue
        if bm >= 256 and bn >= 256:
            # The default NT accumulator conversion needs 128 KiB of scratch.
            # reduceSmemUsage tiles that conversion so the 256x256x32 basic
            # pipeline fits in 64 KiB. Row masks also fit, provided most
            # lanes remain useful. Keep N/K complete: a masked N panel or
            # FP32 partial changes conversion traffic and register pressure.
            complete_tiles = not (mt % bm or nt % bn or k % bk)
            masked_rows = (
                args.get("nt_row_masks", False)
                and mt >= 1024
                and k >= 512
                and nt % bn == 0
                and k % bk == 0
                and mt * 8 >= triton.cdiv(mt, bm) * bm * 7
            )
            reduced_scratch = meta["scenario"] == "reduceSmemUsage"
            if reduced_scratch and not (
                size == 2
                and args["C"].dtype == args["A"].dtype
                and args["SAK"] == 1
                and args["SBK"] == 1
                and args["SCN"] == 1
                and (complete_tiles or masked_rows)
                and args.get("SPLIT_K", 1) == 1
                and not meta["TRANSPOSE"]
                and meta["pipeline"] == "basic"
            ):
                continue
            if (
                (args["SBK"] == 1 and not reduced_scratch)
                or bk > 32
                or config.num_stages > 2
                or meta["scenario"] == "unprefetch"
            ):
                continue
        if bm * bn >= 128 * 128:
            # 1024² has 16 CTAs of 256x256 and loses to a 64 tile. 2048² has 64
            # CTAs, half a wave, but that 256 tile still beats a filled 128 wave.
            sm = _device_limits(args["A"].device)[0]
            ctas = triton.cdiv(mt, bm) * triton.cdiv(nt, bn) * args.get("SPLIT_K", 1)
            if ctas < sm // 2:
                continue
        # 8-warp 128x128 needs a long K. Pipelined unprefetch on a strided RHS
        # is 5-10x slower than a 256 tile; the basic 8-warp form compiles and
        # stays in the pool so autotune can keep it when it actually wins.
        if bm == 128 and bn == 128 and config.num_warps == 8:
            if k < 512:
                continue
            if (
                meta["scenario"] == "unprefetch"
                and config.num_stages > 1
                and (args["SAK"] != 1 or args["SBK"] != 1)
            ):
                continue
        if meta["TRANSPOSE"]:
            if meta["STATIC_K"]:
                if size != 2 or args["SAK"] != 1 or args["SBK"] != 1:
                    continue
            elif not (
                args.get("wide_transpose", False)
                or args["SAM"] == 1
                or (args["SBN"] == 1 and k <= 256 and min(m, n) >= 64)
            ):
                continue
        # FlagTree's AABS mutates a config while benchmarking it. Keep the YAML
        # candidate pool intact so a tiny first call cannot shrink later GEMMs.
        result.append(copy.deepcopy(config))
    return result


def _prune_wide(configs, named_args, **kwargs):
    return _prune_gemm(configs, named_args, wide_transpose=True, **kwargs)


def _prune_nt_rows(configs, named_args, **kwargs):
    return _prune_gemm(configs, named_args, nt_row_masks=True, **kwargs)


def _prune_dual(configs, named_args, **kwargs):
    args = {**named_args, **kwargs}
    f = _features(args["A"], args["B"], args["C"])
    return [
        copy.deepcopy(config)
        for config in configs
        if _dual_tile_profitable(
            f, config.kwargs["B0"] + config.kwargs["B1"], config.kwargs["BM"]
        )
    ]


def _prune_syrk(configs, named_args, **kwargs):
    args = {**named_args, **kwargs}
    sm = _device_limits(args["A"].device)[0]
    result = []
    for config in configs:
        tile = config.kwargs["BT"]
        if args["A"].dtype == torch.float32 and tile > 64:
            # Larger FP32 accumulators spill; retaining the input layout and
            # smaller tiles is faster than packing to the half-precision path.
            continue
        tiles = triton.cdiv(args["M"], tile)
        if tile >= 256 and (args["M"] % tile or tiles * (tiles + 1) // 2 < sm // 2):
            continue
        if config.num_stages > 2 and args["K"] < 512:
            continue
        result.append(copy.deepcopy(config))
    return result


def _simt_candidates(configs, tile, extent, k, split):
    fitting = [
        config
        for config in configs
        if config.kwargs[tile] <= max(16, triton.next_power_of_2(extent))
    ]
    # Output-axis and K-axis limits must be applied jointly: the BK=64
    # candidates use 32-wide output tiles, so a short, narrow reduction used
    # to prune every candidate. Retain the smallest K tile that fits the
    # output axis; the kernel masks the extra reduction lanes.
    k_limit = max(
        min(config.kwargs["BK"] for config in fitting),
        triton.next_power_of_2(triton.cdiv(k, split)),
    )
    return [
        copy.deepcopy(config) for config in fitting if config.kwargs["BK"] <= k_limit
    ]


def _prune_simt_row(configs, named_args, **kwargs):
    args = {**named_args, **kwargs}
    return _simt_candidates(configs, "BN", args["N"], args["K"], args.get("SPLIT_K", 1))


def _prune_simt_column(configs, named_args, **kwargs):
    args = {**named_args, **kwargs}
    return _simt_candidates(configs, "BM", args["M"], args["K"], args.get("SPLIT_K", 1))


def _keep_all(configs, named_args, **kwargs):
    return copy.deepcopy(configs)


def _tune(
    kernel,
    name,
    key,
    prune,
    *,
    additional_configs=(),
    config_filter=None,
    expand_name=None,
):
    configs = runtime.get_tuned_config(name) + list(additional_configs)
    if config_filter is not None:
        configs = [config for config in configs if config_filter(config)]
    return libentry()(
        libtuner(
            configs=configs,
            key=key,
            prune_configs_by={"early_config_prune": prune},
            flagtune_op_name="mm",
            flagtune_expand_op_name=expand_name or name,
            flagtune_yaml_path=EXPAND_CONFIG_FILENAME,
            use_cuda_graph=True,
            rep=20,
        )(kernel)
    )


_STRIDE_KEY = ["M", "N", "K", "SAM", "SAK", "SBK", "SBN", "SCM", "SCN", "SPLIT_K"]

_gemm_tuned = _tune(mm_kernel, "mm_gemm", _STRIDE_KEY, _prune_gemm)
_gemm_dense_tuned = _tune(
    mm_kernel_nn,
    "mm_gemm",
    ["M", "N", "K"],
    _prune_dense,
    config_filter=_is_dense_mma_config,
    expand_name="mm_dense",
)
# This family has a long K and row-major A / column-major B. Static-K and
# transposed candidates are never legal here. A distinct candidate pool also
# gives libtuner a separate persistent best-config cache from the old pruning.
_nt_rows_tuned = _tune(
    mm_kernel,
    "mm_gemm",
    _STRIDE_KEY,
    _prune_nt_rows,
    expand_name="mm_nt_rows",
    config_filter=lambda config: not (
        config.kwargs["STATIC_K"] or config.kwargs["TRANSPOSE"]
    ),
)
_wide_tuned = _tune(
    mm_kernel,
    "mm_wide",
    _STRIDE_KEY,
    _prune_wide,
    additional_configs=_gemm_tuned.fn.configs,
)
_syrk_tuned = _tune(
    _syrk_kernel,
    "mm_syrk",
    ["M", "K", "SAM", "SAK", "SCM", "SCN"],
    _prune_syrk,
)
_dual_tuned = _tune(_dual_gemm_kernel, "mm_dual", _STRIDE_KEY[:-1], _prune_dual)
_simt_row_tuned = _tune(_simt_row_kernel, "mm_simt_row", _STRIDE_KEY, _prune_simt_row)
_simt_column_tuned = _tune(
    mm_kernel_small_n_partial, "mm_simt_column", _STRIDE_KEY, _prune_simt_column
)
_pack_tuned = _tune(_pack_rhs_kernel, "mm_pack", ["R", "C", "SR", "SC"], _keep_all)
_reduce_tuned = _tune(
    mm_kernel_splitk_reduce,
    "mm_reduce",
    ["N", "TOTAL", "SCM", "SCN", "SPLIT_K"],
    _keep_all,
)

_DUAL_TILES = tuple(
    sorted(
        {
            (c.kwargs["B0"] + c.kwargs["B1"], c.kwargs["BM"])
            for c in _dual_tuned.fn.configs
        }
    )
)

# Dispatch policy. Every predicate reads metadata only.


@lru_cache(None)
def _device_limits(device):
    props = runtime.torch_device_fn.get_device_properties(device)
    return (
        props.multi_processor_count,
        props.shared_memory_per_block,
        props.L2_cache_size,
    )


def _features(a, b, c) -> _Features:
    return _Features(
        a.shape[0],
        b.shape[1],
        a.shape[1],
        tuple(a.stride()),
        tuple(b.stride()),
        tuple(c.stride()),
        a.dtype,
        c.dtype,
        all(t.data_ptr() % 16 == 0 for t in (a, b, c)),
        a.data_ptr() == b.data_ptr()
        and a.shape == b.shape[::-1]
        and a.stride() == b.stride()[::-1],
        *_device_limits(a.device),
    )


def _floor_power_of_two(value):
    return 1 << (int(value).bit_length() - 1) if value >= 1 else 0


def _ceil_power_of_two(value):
    return 1 << (int(value) - 1).bit_length() if value > 1 else 1


@lru_cache(None)
def _reference_tile(element_size, shared_bytes):
    """The coarsest pooled tile whose pipeline leaves room for a second CTA.

    Split-K is sized against this tile. A coarser reference reports fewer CTAs
    than the autotuner will really run and over-partitions K; a finer one
    reports more and suppresses partitions the device needs to stay busy.
    Read declared defaults so FlagTune mode and call order cannot change it.
    """
    fits = [
        (config.kwargs["BM"], config.kwargs["BN"])
        for config in runtime.get_tuned_config("mm_gemm")
        if (config.kwargs["BM"] + config.kwargs["BN"])
        * config.kwargs["BK"]
        * element_size
        * config.num_stages
        <= shared_bytes // 2
    ]
    return max(fits, key=lambda tile: tile[0] * tile[1]) if fits else (32, 32)


def _nt_mma_suitable(f: _Features) -> bool:
    """Return whether the NT MMA candidate has enough work to amortize split-K."""
    return (
        f.half
        and f.out_dtype == f.dtype
        and 128 <= min(f.m, f.n) <= 512
        and f.k >= 2048
        and f.k % _MMA_NATIVE_TILE[2] == 0
        and f.a_strides == (f.k, 1)
        and f.b_strides == (1, f.k)
        and f.dense_output
        and f.vector_aligned
        and not f.self_transpose
    )


def _nt_mma_split_count(f: _Features) -> int:
    """Choose a power-of-two K split for the 128-wide NT MMA tile."""
    output_tiles = f.tiles(_MMA_NATIVE_TILE[0], _MMA_NATIVE_TILE[1])
    if output_tiles < 4:
        return 1
    target = max(1, f.sm_count * 3 // 4)
    wanted = _ceil_power_of_two(triton.cdiv(target, output_tiles))
    affordable = max(
        1, _floor_power_of_two(f.k // (_MMA_NATIVE_TILE[2] * _SPLIT_MIN_K_TILES))
    )
    budgeted = max(1, _floor_power_of_two(f.l2_bytes // max(1, 4 * f.m * f.n)))
    split = max(1, min(wanted, affordable, budgeted, _SPLIT_MAX))
    # Async K panels need complete partitions. Lower the split until it divides
    # the reduction; the generic masked path remains the fallback otherwise.
    while split > 1 and f.k % (_MMA_NATIVE_TILE[2] * split):
        split //= 2
    # A split adds an FP32 reduction pass. One output row tile has enough
    # parallel work to hide it; multiple output row tiles need more than four K panels
    # per partition or the reduction costs more than the MMA gain.
    if (
        split > 1
        and triton.cdiv(f.m, _MMA_NATIVE_TILE[0]) > 1
        and f.k // _MMA_NATIVE_TILE[2] <= 4 * split
    ):
        return 1
    return split if split > 1 else 1


def _split_count(f: _Features) -> int:
    """Partition K when the preferred output tiling cannot fill the device."""
    # For medium half-precision outputs with a short reduction, the extra
    # FP32 workspace and reduction launch cost more than the saved GEMM time.
    # Longer K and FP32 still benefit from the occupancy-based policy below.
    if (
        f.half
        and f.out_dtype == f.dtype
        and 256 <= min(f.m, f.n) <= max(f.m, f.n) <= 512
        and 256 <= f.k <= 512
        and f.m % 64 == f.n % 64 == f.k % 64 == 0
        and f.dense_operands
        and f.dense_output
        and f.vector_aligned
    ):
        return 1
    target = max(1, f.sm_count * 3 // 4)
    tiles = f.tiles(*_reference_tile(f.element_size, f.shared_bytes))
    if tiles >= target:
        return _long_k_split_count(f)
    if (
        f.half
        and f.out_dtype == f.dtype
        and 2 <= f.m < _MMA_MIN
        and tiles >= f.sm_count // 2
        and 2048 <= f.k <= 4096
        and f.n % _K_TILE == f.k % _K_TILE == 0
        and f.dense_operands
        and f.b_strides[1] == 1
        and f.dense_output
        and f.vector_aligned
    ):
        # The four-stage NN tiles with masked rows can stream RHS panels with
        # half a device wave. Splitting that wave adds FP32 partial traffic
        # and a reduction launch without improving the complete call. Keep
        # the occupancy fallback for narrower outputs and long-K scheduling
        # above for outputs that already meet the original CTA target.
        # Complete row tiles and deeper K still benefit from partitioning.
        return 1
    wanted = _ceil_power_of_two(triton.cdiv(target, tiles))
    # Each partition needs enough reduction work to pay for the second pass,
    # and the FP32 partial buffer is written once and read once: keep it in L2.
    affordable = _floor_power_of_two(f.reduction_tiles // _SPLIT_MIN_K_TILES)
    budgeted = _floor_power_of_two(f.l2_bytes // max(1, 4 * f.m * f.n))
    split = max(1, min(wanted, affordable, budgeted, _SPLIT_MAX))
    return _long_k_split_count(f) if split == 1 else split


def _long_k_split_count(f: _Features) -> int:
    """Balance CTA waves for long NN reductions with a small/medium M axis."""
    if not (
        f.half
        and f.out_dtype == f.dtype
        and 2 <= f.m <= 512
        and f.n >= 2048
        and (f.m >= 128 or f.n >= 8192)
        and 4096 <= f.k <= 8192
        and f.k % _K_TILE == 0
        and f.dense_operands
        and f.b_strides[1] == 1
        and f.dense_output
        and f.vector_aligned
    ):
        return 1
    # The small reference used for the occupancy fallback above can already
    # fill the device while a faster, larger tile leaves a partial last wave.
    # This is a scheduling estimate; libtuner still selects the actual tile.
    bm, bn = (min(128, max(32, triton.next_power_of_2(x))) for x in (f.m, f.n))
    ctas = f.tiles(bm, bn)
    budget = min(
        4,
        _floor_power_of_two(f.k // 1024),
        _floor_power_of_two(2 * f.l2_bytes // (4 * f.m * f.n)),
    )

    def wave_work(split):
        return triton.cdiv(ctas * split, f.sm_count) * triton.cdiv(f.k, _K_TILE * split)

    candidates = [1] + [split for split in (2, 4) if split <= budget]
    best = min(candidates, key=wave_work)
    # Require headroom for the extra workspace and reduction launch. At
    # K=2048 these costs can erase the predicted gain, so that range is kept
    # on the existing policy even when the wave count looks favorable.
    return best if wave_work(1) >= wave_work(best) * 1.15 else 1


def _dual_tile_profitable(f: _Features, rows: int, columns: int) -> bool:
    # Reducing per-CTA work helps only if enough CTAs run concurrently. This is
    # a scheduling estimate, not a measured hardware occupancy counter.
    return (
        f.wave_work(128, 128) / f.wave_work(rows, columns) >= 1.15
        and f.tile_utilization(rows, columns) >= 0.85
        and f.wave_utilization(rows, columns) >= 0.85
    )


def _dual_suitable(f: _Features) -> bool:
    # Deep half-precision pipelines need contiguous vectors and enough K work.
    # Keep non-vectorizable tails on the tiled kernel; logical M tails stay
    # valid when physical strides are aligned and output padding is modest.
    if not (
        f.half
        and f.out_dtype == f.dtype
        and f.dense_operands
        and f.dense_output
        and f.vector_aligned
        and f.k % _K_TILE == 0
        and f.reduction_tiles >= 32
        and min(f.m, f.n) >= 4 * 128
        and f.tiles(128, 128) >= f.sm_count * 4
        and f.shared_bytes >= 52 * 1024
    ):
        return False
    # Each dual candidate needs more than half an SM's shared memory, so assume
    # one resident CTA and require both output lanes and CTA wave slots to stay
    # at least 85% used while wave work drops by at least 15%.
    return any(_dual_tile_profitable(f, rows, columns) for rows, columns in _DUAL_TILES)


def _pack_profitable(f: _Features, split: int) -> bool:
    # Packing is amortized by repeated use of each B panel over many M tiles.
    if not (f.half and f.dense_operands and not f.rhs_k_contiguous):
        return False
    if (
        f.m // 128 >= 8
        and f.n // 128 >= 8
        and f.reduction_tiles >= 256
        and f.m % 128 == 0
        and f.n % 128 == 0
        and f.k % _K_TILE == 0
    ):
        return True
    # Long reductions with a narrow RHS benefit from contiguous K loads.
    # Limit packing to a reusable L2-sized RHS and enough M reuse to cover
    # the full per-call transpose. The output is still computed normally.
    return (
        split == 1
        and f.out_dtype == f.dtype
        and f.m >= 1024
        and 256 <= f.n <= 512
        and f.n % 128 == 0
        and 4096 <= f.k <= 8192
        and f.k % _K_TILE == 0
        and f.k * f.n * f.element_size <= f.l2_bytes
        and f.dense_output
        and f.vector_aligned
    )


def _dense_mma_suitable(f: _Features) -> bool:
    """Use the mask-free native-shaped kernel for exact dense squares."""
    return (
        f.half
        and f.out_dtype == f.dtype
        and f.m == f.n == f.k
        and f.m >= 8 * _MMA_NATIVE_TILE[0]
        and f.m % _MMA_NATIVE_TILE[0] == 0
        and f.dense_operands
        and f.a_strides == (f.k, 1)
        and f.b_strides == (f.n, 1)
        and f.dense_output
        and f.vector_aligned
        and not f.self_transpose
    )


@lru_cache(maxsize=4096)
def _dispatch_mm(f: _Features) -> _MmPlan:
    """Select the algorithm from metadata; never launch or benchmark candidates."""
    # An empty reduction produces zeros; zero_ is a no-op for empty outputs.
    if not f.m or not f.n or not f.k:
        return _MmPlan(_launch_zero)
    if f.n == 1:
        return _MmPlan(_launch_mv)
    if f.m == 1:
        return _MmPlan(_launch_mv_transposed)
    if _dense_mma_suitable(f):
        return _MmPlan(_launch_dense)
    if _nt_mma_suitable(f):
        split = _nt_mma_split_count(f)
        if split > 1:
            return _MmPlan(_launch_gemm, split)
    if _simt_suitable(f):
        if f.n < f.m:
            programs = f.n * triton.cdiv(f.m, _SIMT_TILE)
            return _MmPlan(_launch_simt_column, _simt_split_count(f, programs))
        programs = f.m * triton.cdiv(f.n, _SIMT_TILE)
        return _MmPlan(_launch_simt_row, _simt_split_count(f, programs))
    if (
        f.self_transpose
        and f.out_dtype == f.dtype
        and f.m >= 1024
        # At two K panels, triangular indexing and the mirrored store cost
        # more than the saved dot work. Keep those short reductions tiled.
        and f.k >= 256
        and f.m % 128 == 0
        and f.k % 64 == 0
        and f.aligned
        and f.dense_output
        and f.a_strides in ((f.k, 1), (1, f.m))
    ):
        return _MmPlan(_launch_syrk, pack_rhs=f.half and f.a_strides[1] != 1)
    if _dual_suitable(f):
        return _MmPlan(_launch_dual)
    if (
        f.half
        and f.out_dtype == f.dtype
        and 2 <= f.m <= 64
        and f.n >= 2048
        and f.k >= 512
        and (
            (f.m >= 16 and f.n % 128 == 0)
            # Long rows amortize partial N panels. Smaller N with a tail can
            # select a tile whose full split/reduce call is slower.
            or (f.n >= 8192 and f.n % 32 == 0)
        )
        and f.k % 64 == 0
        and f.dense_operands
        and f.b_strides[1] == 1
        and f.dense_output
        and f.vector_aligned
    ):
        return _MmPlan(_launch_wide, _split_count(f))
    split = _split_count(f)
    pack = _pack_profitable(f, split)
    if (
        split == 1
        and f.half
        and f.out_dtype == f.dtype
        and f.dense_operands
        and (f.rhs_k_contiguous or pack)
        and f.dense_output
        and f.vector_aligned
        and f.m >= 1024
        and f.m % 256 != 0
        and f.n % 256 == 0
        and f.k >= 512
        and f.k % 32 == 0
        and f.m * 8 >= triton.cdiv(f.m, 256) * 256 * 7
        and f.tiles(256, 256) >= f.sm_count // 2
    ):
        return _MmPlan(_launch_nt_rows, pack_rhs=pack)
    return _MmPlan(_launch_gemm, split, pack)


def _simt_suitable(f: _Features) -> bool:
    return min(f.m, f.n) <= _SIMT_EXTENT and max(f.m, f.n) <= _SIMT_WIDE


def _simt_split_count(f: _Features, programs: int) -> int:
    """A SIMT CTA is tiny, so parallelism past the output comes from K alone."""
    wanted = _ceil_power_of_two(triton.cdiv(f.sm_count, max(1, programs)))
    affordable = _floor_power_of_two(f.reduction_tiles // _SPLIT_MIN_K_TILES)
    budgeted = _floor_power_of_two(f.l2_bytes // max(1, 4 * f.m * f.n))
    return max(1, min(wanted, affordable, budgeted, _SPLIT_MAX))


# Execution.


def _partial_target(call, split_k):
    """The FP32 workspace a partitioned launch writes, or the output itself."""
    if split_k == 1:
        return call.c, call.f.c_strides
    f = call.f
    target = torch.empty((split_k, f.m, f.n), device=call.a.device, dtype=torch.float32)
    return target, (f.n, 1)


def _reduce(call, target, split_k):
    total = call.f.m * call.f.n
    _reduce_tuned[lambda cfg: (triton.cdiv(total, cfg["BLOCK"]),)](
        target,
        call.c,
        call.f.n,
        total,
        *call.f.c_strides,
        SPLIT_K=split_k,
    )


def _launch_zero(call, plan):
    call.c.zero_()


def _launch_mv(call, plan):
    mv(call.a, call.b[:, 0], out=call.c[:, 0])


def _launch_mv_transposed(call, plan):
    mv(call.b.transpose(0, 1), call.a[0], out=call.c[0])


def _launch_simt_row(call, plan):
    f = call.f
    target, strides = _partial_target(call, plan.split_k)
    _simt_row_tuned[lambda cfg: (plan.split_k * f.m * triton.cdiv(f.n, cfg["BN"]),)](
        call.a,
        call.b,
        target,
        f.m,
        f.n,
        f.k,
        *f.a_strides,
        *f.b_strides,
        *strides,
        SPLIT_K=plan.split_k,
    )
    if plan.split_k > 1:
        _reduce(call, target, plan.split_k)


def _launch_simt_column(call, plan):
    f = call.f
    target, strides = _partial_target(call, plan.split_k)
    _simt_column_tuned[lambda cfg: (plan.split_k * f.n * triton.cdiv(f.m, cfg["BM"]),)](
        call.a,
        call.b,
        target,
        f.m,
        f.n,
        f.k,
        *f.a_strides,
        *f.b_strides,
        *strides,
        SPLIT_K=plan.split_k,
    )
    if plan.split_k > 1:
        _reduce(call, target, plan.split_k)


def _dual_grid(f, cfg):
    columns = cfg["B0"] + cfg["B1"]
    # SWAP computes the transposed problem, so its CTA count is the same tiling
    # read the other way round.
    return (
        (f.tiles(columns, cfg["BM"]),)
        if cfg["SWAP"]
        else (f.tiles(cfg["BM"], columns),)
    )


def _launch_dual(call, plan):
    f = call.f
    _dual_tuned[lambda cfg: _dual_grid(f, cfg)](
        call.a,
        call.b,
        call.c,
        f.m,
        f.n,
        f.k,
        *f.a_strides,
        *f.b_strides,
        *f.c_strides,
    )


def _launch_dense(call, plan):
    f = call.f
    _gemm_dense_tuned[
        lambda cfg: (triton.cdiv(f.m, cfg["BM"]) * triton.cdiv(f.n, cfg["BN"]),)
    ](call.a, call.b, call.c, f.m, f.n, f.k)


def _launch_gemm(call, plan):
    _launch_tiled(call, plan, _gemm_tuned)


def _launch_wide(call, plan):
    _launch_tiled(call, plan, _wide_tuned)


def _launch_nt_rows(call, plan):
    _launch_tiled(call, plan, _nt_rows_tuned)


def _launch_tiled(call, plan, tuned):
    f = call.f
    b, b_strides = call.b, f.b_strides
    if plan.pack_rhs:
        b = torch.empty_strided((f.k, f.n), (1, f.k), dtype=b.dtype, device=b.device)
        _pack_tuned[
            lambda cfg: (triton.cdiv(f.k, cfg["BR"]) * triton.cdiv(f.n, cfg["BC"]),)
        ](call.b, b, f.k, f.n, *f.b_strides)
        b_strides = (1, f.k)
    target, strides = _partial_target(call, plan.split_k)

    def grid(cfg):
        rows, columns = (f.n, f.m) if cfg["TRANSPOSE"] else (f.m, f.n)
        return (
            plan.split_k
            * triton.cdiv(rows, cfg["BM"])
            * triton.cdiv(columns, cfg["BN"]),
        )

    tuned[grid](
        call.a,
        b,
        target,
        f.m,
        f.n,
        f.k,
        *f.a_strides,
        *b_strides,
        *strides,
        SPLIT_K=plan.split_k,
    )
    if plan.split_k > 1:
        _reduce(call, target, plan.split_k)


def _launch_syrk(call, plan):
    f = call.f
    a = call.a
    if plan.pack_rhs:
        # Packing B = A.T also provides a row-major A as a view of that
        # call-local buffer. Neither input data nor workspace is cached.
        b = torch.empty_strided(
            (f.k, f.n), (1, f.k), device=call.b.device, dtype=call.b.dtype
        )
        _pack_tuned[
            lambda cfg: (triton.cdiv(f.k, cfg["BR"]) * triton.cdiv(f.n, cfg["BC"]),)
        ](call.b, b, f.k, f.n, *f.b_strides)
        a = b.transpose(0, 1)

    def grid(cfg):
        tiles = triton.cdiv(f.m, cfg["BT"])
        return (tiles * (tiles + 1) // 2,)

    _syrk_tuned[grid](a, call.c, f.m, f.k, *a.stride(), *f.c_strides)


# Public API.


def mm(a, b, *, out=None):
    """Dispatch the workload from tensor metadata, then launch the selected plan."""
    logger.debug("GEMS METAX MM")
    if out is None:
        out = torch.empty((a.shape[0], b.shape[1]), device=a.device, dtype=a.dtype)
    call = _MmCall(a, b, out, _features(a, b, out))
    with torch_device_fn.device(a.device):
        plan = _dispatch_mm(call.f)
        plan.launch(call, plan)
        return call.c


def mm_out(a, b, *, out):
    # Registration filters use the function name, so keep a distinct out entry.
    return mm(a, b, out=out)
