# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Stride-aware MetaX BMM with batched GEMM and vector paths."""

import copy
import logging
import os

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import get_device_properties, libentry, libtuner
from flag_gems.utils.libentry import LibTuner

from .mm import mm

logger = logging.getLogger(__name__)
EXPAND_CONFIG_FILENAME = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "bmm_metax_expand.yaml")
)

# Bound the temporary storage used to pack a batched RHS.
_MAX_PACKING_WORKSPACE_BYTES = 8 * 1024**3
_MAX_VECTOR_SPLIT_K = 32
_VECTOR_PROGRAMS_PER_SM = 4


def _prune_bmm(configs, named_args, **kwargs):
    args = {**named_args, **kwargs}
    m, n, k = args["M"], args["N"], args["K"]
    size = args["A"].element_size()
    shared_bytes = get_device_properties(args["A"].device.index).shared_memory_per_block
    result = []
    for config in configs:
        meta = config.kwargs
        bm, bn, bk = meta["BM"], meta["BN"], meta["BK"]
        if bk == 16 and size != 4:
            continue
        mt, nt = (n, m) if meta["TRANSPOSE"] else (m, n)
        # Dense outputs already expose many CTAs. Tiny output tiles only
        # repeat operand traffic, especially for the long-K model workloads.
        if min(m, n) >= 1024 and bm * bn < 4096:
            continue
        if size == 2 and min(m, n) >= 1024 and k >= 8192 and bm * bn < 16384:
            continue
        if meta["STATIC_K"] and (k > 128 or min(m, n) < 64):
            continue
        if bm > max(32, triton.next_power_of_2(mt)) or bn > max(
            32, triton.next_power_of_2(nt)
        ):
            continue
        if (bm + bn) * bk * size * (1 if config.num_stages == 1 else 2) > shared_bytes:
            continue
        if meta["scenario"] == "unprefetch" and (mt % bm or nt % bn or k % bk):
            continue
        if bm >= 256 and bn >= 256 and args["SBK"] == 1:
            if not (
                meta["scenario"] == "reduceSmemUsage"
                and size == 2
                and args["SAK"] == 1
                and args["SCN"] == 1
                and nt % bn == 0
                and k % bk == 0
                and not meta["TRANSPOSE"]
                and args["C"].dtype == args["A"].dtype
            ):
                continue
        if meta["scenario"] == "reduceSmemUsage" and not (
            (size == 2 and args["SBK"] == 1) or (size == 4 and bk == 16)
        ):
            continue
        if meta["TRANSPOSE"] and not (
            args.get("wide", False)
            or args["SAM"] == 1
            or (
                size == 4
                and bk == 16
                and k >= 512
                and min(m, n) >= 512
                and args["SAK"] == args["SBN"] == 1
            )
            or (
                args["SBN"] == 1
                and (k <= 256 or (size == 2 and args["SAK"] == 1 and min(m, n) >= 64))
            )
        ):
            continue
        if (
            bm == bn == 128
            and config.num_warps == 8
            and meta["scenario"] == "unprefetch"
            and config.num_stages > 1
            and args["SBK"] != 1
        ):
            continue
        result.append(copy.deepcopy(config))
    return result


class _BmmMmaTuner(LibTuner.get("default")):
    """Preserve short-K tiles and pruning against the logical transposed axes."""

    def _make_config_table_name(self):
        # The previous AABS policy could exclude the intended transposed tile.
        # Re-select winners while keeping per-config timings: those are keyed
        # by the actual (possibly AABS-adjusted) configuration that was timed.
        return f"{super()._make_config_table_name()}_logical_transpose_v1"

    def _bench(self, *args, config, **meta):
        # The source-level AABS analysis does not follow M/N's constexpr swap.
        # Pruning already bounded BM/BN against the correct logical axes.
        if (
            not config.kwargs["TRANSPOSE"]
            and min(self.nargs[name] for name in ("M", "N", "K")) >= 16
        ):
            return super()._bench(*args, config=config, **meta)
        current = {**meta, **config.all_kwargs()}

        def launch():
            if config.pre_hook:
                config.pre_hook({**self.nargs, **current})
            self.fn.run(*args, **current)

        try:
            return self.do_bench(launch, quantiles=(0.5, 0.2, 0.8))
        except triton.runtime.errors.OutOfResources:
            return [float("inf")] * 3


_KEY = [
    "BATCH",
    "M",
    "N",
    "K",
    "SAB",
    "SBB",
    "SCB",
    "SAM",
    "SAK",
    "SBK",
    "SBN",
    "SCM",
    "SCN",
    "SPLIT_K",
]


def _prune_wide(configs, named_args, **kwargs):
    return _prune_bmm(configs, named_args, wide=True, **kwargs)


# Keep GEMM bodies visible to FlagTree's source-level AABS analysis.
# Calls to a shared JIT helper would hide the tile/load/dot dependencies.
@libentry()
@libtuner(
    configs=runtime.ops_get_configs("bmm_gemm", yaml_path=EXPAND_CONFIG_FILENAME),
    key=_KEY,
    prune_configs_by={"early_config_prune": _prune_bmm},
    policy=_BmmMmaTuner,
    flagtune_op_name="bmm",
    flagtune_expand_op_name="bmm_gemm",
    flagtune_yaml_path=EXPAND_CONFIG_FILENAME,
    rep=20,
)
@triton.jit
def _bmm_kernel(
    A,
    B,
    C,
    BATCH: tl.constexpr,
    SAB: tl.constexpr,
    SBB: tl.constexpr,
    SCB: tl.constexpr,
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
    INTERLEAVE: tl.constexpr = False,
):
    """One output tile per CTA, optionally with a deterministic K partition."""
    tiles_m = tl.cdiv(N if TRANSPOSE else M, BM)
    tiles_n = tl.cdiv(M if TRANSPOSE else N, BN)
    tiles = tiles_m * tiles_n * SPLIT_K
    if INTERLEAVE:
        batch = tl.program_id(0) % BATCH
        tile_id = tl.program_id(0) // BATCH
    else:
        batch = tl.program_id(0) // tiles
        tile_id = tl.program_id(0) % tiles
    A += batch.to(tl.int64) * SAB
    B += batch.to(tl.int64) * SBB
    C += batch.to(tl.int64) * SCB
    if TRANSPOSE:
        # Compute C^T = B^T A^T. Swapping the dot operands changes which
        # operand feeds which MMA port without materializing a transpose.
        A, B = B, A
        M, N = N, M
        SAM, SAK, SBK, SBN = SBN, SBK, SAK, SAM
        SCM, SCN = SCN, SCM
    nm, nn = tl.cdiv(M, BM), tl.cdiv(N, BN)
    split = tile_id // (nm * nn) % SPLIT_K
    pid = tile_id % (nm * nn)
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


@libentry()
@libtuner(
    configs=runtime.ops_get_configs("bmm_wide", yaml_path=EXPAND_CONFIG_FILENAME),
    key=_KEY,
    prune_configs_by={"early_config_prune": _prune_wide},
    policy=_BmmMmaTuner,
    flagtune_op_name="bmm",
    flagtune_expand_op_name="bmm_wide",
    flagtune_yaml_path=EXPAND_CONFIG_FILENAME,
    rep=20,
)
@triton.jit
def _bmm_wide_kernel(
    A,
    B,
    C,
    BATCH: tl.constexpr,
    SAB: tl.constexpr,
    SBB: tl.constexpr,
    SCB: tl.constexpr,
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
    INTERLEAVE: tl.constexpr = False,
):
    """One output tile per CTA, optionally with a deterministic K partition."""
    tiles_m = tl.cdiv(N if TRANSPOSE else M, BM)
    tiles_n = tl.cdiv(M if TRANSPOSE else N, BN)
    tiles = tiles_m * tiles_n * SPLIT_K
    if INTERLEAVE:
        batch = tl.program_id(0) % BATCH
        tile_id = tl.program_id(0) // BATCH
    else:
        batch = tl.program_id(0) // tiles
        tile_id = tl.program_id(0) % tiles
    A += batch.to(tl.int64) * SAB
    B += batch.to(tl.int64) * SBB
    C += batch.to(tl.int64) * SCB
    if TRANSPOSE:
        # Compute C^T = B^T A^T. Swapping the dot operands changes which
        # operand feeds which MMA port without materializing a transpose.
        A, B = B, A
        M, N = N, M
        SAM, SAK, SBK, SBN = SBN, SBK, SAK, SAM
        SCM, SCN = SCN, SCM
    nm, nn = tl.cdiv(M, BM), tl.cdiv(N, BN)
    split = tile_id // (nm * nn) % SPLIT_K
    pid = tile_id % (nm * nn)
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


@libentry()
@triton.jit
def _bmm_dense_kernel(
    A,
    B,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    B_COLUMN_MAJOR: tl.constexpr = False,
):
    """Complete dense tiles; batching remains part of BMM's own launch."""
    batch = tl.program_id(1).to(tl.int64)
    nm, nn = M // 128, N // 128
    pid = tl.program_id(0)
    first_m = pid // (8 * nn) * 8
    group_m = tl.minimum(nm - first_m, 8)
    local = pid % (8 * nn)
    mi = (first_m + local % group_m) * 128 + tl.arange(0, 128)
    ni = local // group_m * 128 + tl.arange(0, 128)
    ki = tl.arange(0, 128)
    ap = A + batch * M * K + mi[:, None].to(tl.int64) * K + ki[None, :]
    if B_COLUMN_MAJOR:
        # Keep the K term first to avoid register spills in MetaX lowering.
        bp = B + batch * N * K + ki[:, None].to(tl.int64) + ni[None, :].to(tl.int64) * K
    else:
        bp = B + batch * N * K + ki[:, None].to(tl.int64) * N + ni[None, :]
    acc = tl.zeros((128, 128), tl.float32)
    for _ in tl.range(0, K // 128, num_stages=4):
        acc = tl.dot(tl.load(ap), tl.load(bp), acc, allow_tf32=False)
        ap += 128
        bp += 128 if B_COLUMN_MAJOR else 128 * N
    cp = C + batch * M * N + mi[:, None].to(tl.int64) * N + ni[None, :]
    tl.store(cp, acc, (mi[:, None] < M) & (ni[None, :] < N))


@libentry()
@triton.jit
def _bmm_long_k_tf32(
    A,
    B,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    SAB: tl.constexpr,
    SBB: tl.constexpr,
    SAM: tl.constexpr,
    SBK: tl.constexpr,
    SBN: tl.constexpr,
):
    # The public dispatch enables this approximate mode only when TF32 is
    # allowed and the long reduction passed the repository accuracy checks.
    tiles = tl.cdiv(M, 128) * tl.cdiv(N, 128)
    batch = (tl.program_id(0) // tiles).to(tl.int64)
    pid = tl.program_id(0) % tiles
    mi = (pid // tl.cdiv(N, 128) * 128 + tl.arange(0, 128)).to(tl.int64)
    ni = (pid % tl.cdiv(N, 128) * 128 + tl.arange(0, 128)).to(tl.int64)
    ki = tl.arange(0, 32).to(tl.int64)
    ap = A + batch * SAB + mi[:, None] * SAM + ki[None, :]
    bp = B + batch * SBB + ki[:, None] * SBK + ni[None, :] * SBN
    acc = tl.zeros((128, 128), tl.float32)
    for _ in range(K // 32):
        ak = tl.load(ap, mi[:, None] < M, 0)
        bk = tl.load(bp, ni[None, :] < N, 0)
        acc = tl.dot(ak, bk, acc, input_precision="tf32")
        ap += 32
        bp += 32 * SBK
    tl.store(
        C + batch * M * N + mi[:, None] * N + ni[None, :],
        acc,
        (mi[:, None] < M) & (ni[None, :] < N),
    )


@libentry()
@triton.jit
def _bmm_dual_n(
    A,
    B,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
):
    # Compose a 192-column tile from two legal power-of-two dot outputs.
    # Both dots share A; aligned column-major B avoids the spill-heavy case.
    pid = tl.program_id(0)
    mi = (pid // tl.cdiv(N, 192) * 256 + tl.arange(0, 256)).to(tl.int64)
    n0 = (pid % tl.cdiv(N, 192) * 192 + tl.arange(0, 128)).to(tl.int64)
    n1 = (pid % tl.cdiv(N, 192) * 192 + 128 + tl.arange(0, 64)).to(tl.int64)
    ki = tl.arange(0, 32).to(tl.int64)
    c0 = tl.zeros((256, 128), tl.float32)
    c1 = tl.zeros((256, 64), tl.float32)
    for i in range(K // 32):
        kk = i * 32 + ki
        ak = tl.load(A + mi[:, None] * K + kk[None, :], mi[:, None] < M, 0)
        b0 = tl.load(B + kk[:, None] + n0[None, :] * K)
        b1 = tl.load(B + kk[:, None] + n1[None, :] * K)
        c0 = tl.dot(ak, b0, c0)
        c1 = tl.dot(ak, b1, c1)
    tl.store(C + mi[:, None] * N + n0[None, :], c0, mi[:, None] < M)
    tl.store(C + mi[:, None] * N + n1[None, :], c1, mi[:, None] < M)


@libentry()
@triton.jit
def _bmm_small_kernel(
    A,
    B,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    SAB: tl.constexpr,
    SAM: tl.constexpr,
    SAK: tl.constexpr,
    SBB: tl.constexpr,
    SBK: tl.constexpr,
    SBN: tl.constexpr,
    SCB: tl.constexpr,
    SCM: tl.constexpr,
    SCN: tl.constexpr,
    BLOCK: tl.constexpr,
    BK: tl.constexpr,
):
    batch = tl.program_id(1).to(tl.int64)
    ids = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mi, ni = ids // N, ids % N
    ki = tl.arange(0, BK)
    acc = tl.zeros((BLOCK, BK), tl.float32)
    for step in range(tl.cdiv(K, BK)):
        kk = ki + step * BK
        a = tl.load(
            A
            + batch * SAB
            + mi[:, None].to(tl.int64) * SAM
            + kk[None, :].to(tl.int64) * SAK,
            (mi[:, None] < M) & (kk[None, :] < K),
            0,
        ).to(tl.float32)
        b = tl.load(
            B
            + batch * SBB
            + ni[:, None].to(tl.int64) * SBN
            + kk[None, :].to(tl.int64) * SBK,
            (ids[:, None] < M * N) & (kk[None, :] < K),
            0,
        ).to(tl.float32)
        acc = tl.fma(a, b, acc)
    tl.store(
        C + batch * SCB + mi.to(tl.int64) * SCM + ni.to(tl.int64) * SCN,
        tl.sum(acc, 1),
        ids < M * N,
    )


@libentry()
@triton.jit
def _bmm_vector_reduce(
    P,
    Y,
    R: tl.constexpr,
    SPLIT_K: tl.constexpr,
    SYB: tl.constexpr,
    SYR: tl.constexpr,
    BLOCK: tl.constexpr,
):
    batch = tl.program_id(1).to(tl.int64)
    r = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.zeros((BLOCK,), tl.float32)
    for s in tl.static_range(SPLIT_K):
        value += tl.load(P + batch * SPLIT_K * R + s * R + r, r < R, 0)
    tl.store(Y + batch * SYB + r.to(tl.int64) * SYR, value, r < R)


@libentry()
@triton.jit
def _bmm_pack_rhs(B, P, K: tl.constexpr, N: tl.constexpr):
    """Pack a dense batched RHS into K-contiguous panels for the MMA kernel."""
    batch = tl.program_id(1).to(tl.int64)
    ki = tl.program_id(0) // tl.cdiv(N, 128) * 64 + tl.arange(0, 64)
    ni = tl.program_id(0) % tl.cdiv(N, 128) * 128 + tl.arange(0, 128)
    mask = (ki[:, None] < K) & (ni[None, :] < N)
    value = tl.load(
        B + batch * K * N + ki[:, None].to(tl.int64) * N + ni[None, :],
        mask,
        other=0,
    )
    tl.store(
        P + batch * K * N + ni[None, :].to(tl.int64) * K + ki[:, None],
        value,
        mask,
    )


def _prune_vector(configs, named_args, **kwargs):
    args = {**named_args, **kwargs}
    return [
        copy.deepcopy(config)
        for config in configs
        if config.kwargs["BR"] <= max(16, triton.next_power_of_2(args["R"]))
        and config.kwargs["BK"] <= max(128, triton.next_power_of_2(args["K"]))
        # Bound private memory for the batched FP32 reduction on C550.
        and config.kwargs["BR"] * config.kwargs["BK"] <= 32768
        and config.kwargs["BR"] * config.kwargs["BK"] <= 8192 * config.num_warps
    ]


# BATCH is a tuning key: it changes the launch size even though each
# vector program gets its batch index directly from the grid.
@libentry()
@libtuner(
    configs=runtime.ops_get_configs("bmm_vector", yaml_path=EXPAND_CONFIG_FILENAME),
    key=[
        "R",
        "K",
        "BATCH",
        "SAB",
        "SAR",
        "SAK",
        "SXB",
        "SXK",
        "SYB",
        "SYR",
        "SPLIT_K",
        "COLUMN",
    ],
    prune_configs_by={"early_config_prune": _prune_vector},
    flagtune_op_name="bmm",
    flagtune_expand_op_name="bmm_vector",
    flagtune_yaml_path=EXPAND_CONFIG_FILENAME,
    rep=20,
)
@triton.jit
def _bmm_vector_kernel(
    A,
    X,
    Y,
    R: tl.constexpr,
    K: tl.constexpr,
    BATCH: tl.constexpr,
    SAB: tl.constexpr,
    SAR: tl.constexpr,
    SAK: tl.constexpr,
    SXB: tl.constexpr,
    SXK: tl.constexpr,
    SYB: tl.constexpr,
    SYR: tl.constexpr,
    SPLIT_K: tl.constexpr,
    COLUMN: tl.constexpr,
    BR: tl.constexpr,
    BK: tl.constexpr,
):
    batch = tl.program_id(2).to(tl.int64)
    part = tl.program_id(1)
    r = (tl.program_id(0) * BR + tl.arange(0, BR)).to(tl.int64)
    offsets = tl.arange(0, BK).to(tl.int64)
    if COLUMN:
        acc = tl.zeros((BK, BR), tl.float32)
    else:
        acc = tl.zeros((BR, BK), tl.float32)
    for start in range(part * BK, K, SPLIT_K * BK):
        ks = start + offsets
        x = tl.load(X + batch * SXB + ks * SXK, ks < K, 0).to(tl.float32)
        if COLUMN:
            a = tl.load(
                A + batch * SAB + ks[:, None] * SAK + r[None, :] * SAR,
                (ks[:, None] < K) & (r[None, :] < R),
                0,
            ).to(tl.float32)
            acc = tl.fma(a, x[:, None], acc)
        else:
            a = tl.load(
                A + batch * SAB + r[:, None] * SAR + ks[None, :] * SAK,
                (r[:, None] < R) & (ks[None, :] < K),
                0,
            ).to(tl.float32)
            acc = tl.fma(a, x[None, :], acc)
    value = tl.sum(acc, 0 if COLUMN else 1)
    tl.store(Y + batch * SYB + part.to(tl.int64) * R + r * SYR, value, r < R)


def _bmm_vector(a, b, out, split, transpose, column):
    batch, m, k = a.shape
    n = b.shape[2]
    if not transpose:
        matrix, vector = a, b
        rows = m
        sar, sak = a.stride()[1:]
        sxk = b.stride(1)
        syr = out.stride(1)
    else:
        matrix, vector = b, a
        rows = n
        sak, sar = b.stride()[1:]
        sxk = a.stride(2)
        syr = out.stride(2)
    target = out
    syb = out.stride(0)
    target_syr = syr
    if split > 1:
        target = torch.empty((batch, split, rows), device=a.device, dtype=torch.float32)
        syb = split * rows
        target_syr = 1
    _bmm_vector_kernel[lambda meta: (triton.cdiv(rows, meta["BR"]), split, batch)](
        matrix,
        vector,
        target,
        rows,
        k,
        batch,
        matrix.stride(0),
        sar,
        sak,
        vector.stride(0),
        sxk,
        syb,
        target_syr,
        SPLIT_K=split,
        COLUMN=column,
    )
    if split > 1:
        _bmm_vector_reduce[(triton.cdiv(rows, 512), batch)](
            target, out, rows, split, out.stride(0), syr, BLOCK=512, num_warps=4
        )


def bmm(a, b, *, out=None):
    logger.debug("GEMS METAX BMM")
    shape = (a.shape[0], a.shape[1], b.shape[2])
    if out is None:
        out = torch.empty(shape, device=a.device, dtype=a.dtype)
    batch, m, k = a.shape
    n = b.shape[2]
    if not batch or not m or not n:
        return out
    a_strides, b_strides = a.stride(), b.stride()
    half = a.dtype in (torch.float16, torch.bfloat16)
    with torch_device_fn.device(a.device):
        if not k:
            return out.zero_()
        if (
            a.dtype == torch.float32
            and torch.backends.cuda.matmul.allow_tf32
            and min(m, n) >= 1024
            and k >= 4096
            and k % 32 == 0
            and a_strides[2] == 1
            and 1 in b_strides[1:]
            and out.is_contiguous()
        ):
            _bmm_long_k_tf32[(batch * triton.cdiv(m, 128) * triton.cdiv(n, 128),)](
                a,
                b,
                out,
                m,
                n,
                k,
                a_strides[0],
                b_strides[0],
                a_strides[1],
                b_strides[1],
                b_strides[2],
                num_warps=8,
                num_stages=2,
                pipeline="basic",
            )
            return out
        if (
            batch == 1
            and half
            and out.dtype == a.dtype
            and m >= 8192
            and 1536 <= n <= 4096
            and n % 192 == 0
            and k >= 4096
            and k % 32 == 0
            and a.is_contiguous()
            and b_strides[1:] == (1, k)
            and out.is_contiguous()
        ):
            _bmm_dual_n[(triton.cdiv(m, 256) * triton.cdiv(n, 192),)](
                a,
                b,
                out,
                m,
                n,
                k,
                num_warps=8,
                num_stages=2,
                pipeline="basic",
            )
            return out
        if (
            batch == 1
            and a.dtype == out.dtype
            and 1 in a_strides[1:]
            and 1 in b_strides[1:]
            and out.is_contiguous()
        ):
            mm(a[0], b[0], out=out[0])
            return out
        if m * n <= 32:
            _bmm_small_kernel[(triton.cdiv(m * n, 8), batch)](
                a,
                b,
                out,
                m,
                n,
                k,
                *a_strides,
                *b_strides,
                *out.stride(),
                BLOCK=8,
                BK=triton.next_power_of_2(min(k, 256)),
                num_warps=4,
            )
            return out
        if m == 1 or n == 1:
            rows = max(m, n)
            column = (
                a_strides[1] < a_strides[2] if n == 1 else b_strides[2] < b_strides[1]
            )
            split = 1
            if column and k >= 1024:
                sm = get_device_properties(a.device.index).multi_processor_count
                programs = batch * triton.cdiv(rows, 128)
                wanted = triton.cdiv(_VECTOR_PROGRAMS_PER_SM * sm, programs)
                split = min(_MAX_VECTOR_SPLIT_K, 1 << (wanted - 1).bit_length())
            _bmm_vector(a, b, out, split, m == 1, column)
            return out
        # Balanced dense products amortize the four-stage NN pipeline.
        if (
            half
            and out.dtype == a.dtype
            and min(m, n) >= 1024
            and max(m, n) <= 2 * min(m, n)
            and 512 <= k <= 2 * min(m, n)
            and m % 128 == n % 128 == k % 128 == 0
            and a.is_contiguous()
            and b.is_contiguous()
            and out.is_contiguous()
        ):
            _bmm_dense_kernel[(m // 128 * (n // 128), batch)](
                a,
                b,
                out,
                m,
                n,
                k,
                num_warps=4,
                num_stages=4,
                pipeline="cpasync",
            )
            return out

        if half and 2 <= m <= 32 and n >= 512 and k >= 512:
            kernel = _bmm_wide_kernel
        else:
            kernel = _bmm_kernel
            if (
                half
                and out.dtype == a.dtype
                and m >= 4096
                and n >= 4096
                and 2048 <= k <= 4096
                and m % 256 == 0
                and n % 256 == 0
                and k % 32 == 0
                and a.is_contiguous()
                and b.is_contiguous()
                and out.is_contiguous()
                and batch * k * n * a.element_size() <= _MAX_PACKING_WORKSPACE_BYTES
            ):
                packed = torch.empty(
                    (batch, n, k), device=b.device, dtype=b.dtype
                ).transpose(1, 2)
                _bmm_pack_rhs[(triton.cdiv(k, 64) * triton.cdiv(n, 128), batch)](
                    b, packed, k, n, num_warps=4
                )
                b = packed

        def grid(meta):
            mt, nt = (n, m) if meta["TRANSPOSE"] else (m, n)
            return (batch * triton.cdiv(mt, meta["BM"]) * triton.cdiv(nt, meta["BN"]),)

        kernel[grid](
            a,
            b,
            out,
            batch,
            a.stride(0),
            b.stride(0),
            out.stride(0),
            m,
            n,
            k,
            *a.stride()[1:],
            *b.stride()[1:],
            *out.stride()[1:],
            SPLIT_K=1,
        )
    return out


def bmm_out(a, b, out):
    return bmm(a, b, out=out)
