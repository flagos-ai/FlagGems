# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import copy
import logging
import os
from functools import lru_cache

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import get_device_properties, libentry, libtuner

from .mv import mv

logger = logging.getLogger(__name__)
EXPAND_CONFIG_FILENAME = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "mm_metax_expand.yaml")
)

# Constraints of this MM implementation, not queried device capacities.
_VECTOR_ALIGNMENT_BYTES = 16
_HALF_VECTOR_ELEMENTS = _VECTOR_ALIGNMENT_BYTES // 2
# MACA MMA rejects or miscomputes dot tiles smaller than this floor.
_MMA_MIN = 16
# Measured C550 FP16/BF16 async configuration and compiler allocation.
# This allocation is a kernel requirement, not a device capacity fallback.
_MMA_NATIVE_TILE = (128, 128, 128)
_MMA_NATIVE_WARPS = 4
_MMA_NATIVE_STAGES = 4
_MMA_NATIVE_SHARED_BYTES = 64 * 1024

# Workload policy.
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


_STRIDE_KEY = ["M", "N", "K", "SAM", "SAK", "SBK", "SBN", "SCM", "SCN", "SPLIT_K"]


def _is_native_mma_config(config):
    """Match the async MM configuration with a measured SMEM allocation."""
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


def _gemm_shared_bytes(config, element_size, *, native_mma=False):
    """Estimate this GEMM implementation's compiler-managed shared memory."""
    if native_mma:
        return _MMA_NATIVE_SHARED_BYTES
    # FlagTree keeps a second K buffer unless num_stages is 1. The
    # recognized native MMA configuration is the measured exception.
    buffers = 1 if config.num_stages <= 1 else min(config.num_stages, 2)
    meta = config.kwargs
    return (meta["BM"] + meta["BN"]) * meta["BK"] * element_size * buffers


def _is_nn_mma_candidate(config, args):
    """The 64 KiB async NN tile permits M tails, but needs complete N/K panels."""
    return (
        _is_native_mma_config(config)
        and args.get("SPLIT_K", 1) == 1
        and args["A"].dtype in (torch.float16, torch.bfloat16)
        and args["C"].dtype == args["A"].dtype
        and args["SAM"] == args["K"]
        and args["SAK"] == 1
        and args["SBK"] == args["N"]
        and args["SBN"] == 1
        and args["SCM"] == args["N"]
        and args["SCN"] == 1
        and args["N"] % _MMA_NATIVE_TILE[1] == 0
        and args["K"] % _MMA_NATIVE_TILE[2] == 0
    )


def _is_nt_mma_candidate(config, args):
    """Allow the native-shaped tile for K-contiguous NT partials."""
    return (
        _is_native_mma_config(config)
        and args["A"].dtype in (torch.float16, torch.bfloat16)
        and args["SAK"] == 1
        and args["SBK"] == 1
        and args["SBN"] == args["K"]
        and args["SCN"] == 1
        and args.get("SPLIT_K", 1) > 1
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
        if _is_native_mma_config(config)
        and all(
            args[axis] % config.kwargs[tile] == 0
            for axis, tile in (("M", "BM"), ("N", "BN"), ("K", "BK"))
        )
    ]


def _prune_gemm(configs, named_args, **kwargs):
    args = {**named_args, **kwargs}
    m, n, k = args["M"], args["N"], args["K"]
    size = args["A"].element_size()
    properties = get_device_properties(args["A"].device.index)
    sm_count = properties.multi_processor_count
    shared_bytes = properties.shared_memory_per_block
    result = []
    for config in configs:
        meta = config.kwargs
        if meta["STATIC_K"] and (
            k > 128 or min(m, n) < 64 or args.get("SPLIT_K", 1) != 1
        ):
            continue
        mt, nt = (n, m) if meta["TRANSPOSE"] else (m, n)
        bm, bn, bk = meta["BM"], meta["BN"], meta["BK"]
        if bk == 16 and size != 4:
            continue
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
        # Apply the measured allocation exception only when eligible.
        native_mma = _is_nn_mma_candidate(config, args) or _is_nt_mma_candidate(
            config, args
        )
        shared = _gemm_shared_bytes(config, size, native_mma=native_mma)
        if shared > shared_bytes:
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
            ctas = triton.cdiv(mt, bm) * triton.cdiv(nt, bn) * args.get("SPLIT_K", 1)
            if ctas < sm_count // 2:
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
                or (
                    size == 4
                    and bk == 16
                    and k >= 512
                    and min(m, n) >= 512
                    and args["SAK"] == args["SBN"] == 1
                )
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


def _prune_syrk(configs, named_args, **kwargs):
    args = {**named_args, **kwargs}
    sm = get_device_properties(args["A"].device.index).multi_processor_count
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


def _keep_all(configs, _named_args, **_kwargs):
    return copy.deepcopy(configs)


# Keep GEMM bodies visible to FlagTree's source-level AABS analysis.
# Calls to a shared JIT helper would hide the tile/load/dot dependencies.
@libentry()
@libtuner(
    configs=runtime.get_tuned_config("mm_gemm"),
    key=_STRIDE_KEY,
    prune_configs_by={"early_config_prune": _prune_gemm},
    flagtune_op_name="mm",
    flagtune_expand_op_name="mm_gemm",
    flagtune_yaml_path=EXPAND_CONFIG_FILENAME,
    rep=20,
)
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
    """Masked tiled GEMM with optional operand transpose and K partitions."""
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


# This family has a long K and row-major A / column-major B. Static-K and
# transposed candidates are never legal here. A distinct candidate pool also
# gives libtuner a separate persistent best-config cache from the old pruning.
@libentry()
@libtuner(
    configs=[
        config
        for config in runtime.get_tuned_config("mm_gemm")
        if not (config.kwargs["STATIC_K"] or config.kwargs["TRANSPOSE"])
    ],
    key=_STRIDE_KEY,
    prune_configs_by={"early_config_prune": _prune_nt_rows},
    flagtune_op_name="mm",
    flagtune_expand_op_name="mm_nt_rows",
    flagtune_yaml_path=EXPAND_CONFIG_FILENAME,
    rep=20,
)
@triton.jit
def mm_kernel_nt_rows(
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
    """Masked tiled GEMM with optional operand transpose and K partitions."""
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


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("mm_wide") + mm_kernel.fn.configs,
    key=_STRIDE_KEY,
    prune_configs_by={"early_config_prune": _prune_wide},
    flagtune_op_name="mm",
    flagtune_expand_op_name="mm_wide",
    flagtune_yaml_path=EXPAND_CONFIG_FILENAME,
    rep=20,
)
@triton.jit
def mm_kernel_wide(
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
    """Masked tiled GEMM with optional operand transpose and K partitions."""
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


# Accept the shared GEMM config schema; the dense config filter fixes
# TRANSPOSE and STATIC_K to False before this kernel is launched.
@libentry()
@libtuner(
    configs=[
        config
        for config in runtime.get_tuned_config("mm_gemm")
        if _is_native_mma_config(config)
    ],
    key=["M", "N", "K"],
    prune_configs_by={"early_config_prune": _prune_dense},
    flagtune_op_name="mm",
    flagtune_expand_op_name="mm_dense",
    flagtune_yaml_path=EXPAND_CONFIG_FILENAME,
    rep=20,
)
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


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("mm_syrk"),
    key=["M", "K", "SAM", "SAK", "SCM", "SCN"],
    prune_configs_by={"early_config_prune": _prune_syrk},
    flagtune_op_name="mm",
    flagtune_expand_op_name="mm_syrk",
    flagtune_yaml_path=EXPAND_CONFIG_FILENAME,
    rep=20,
)
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
    # why 8.0? explain the formula
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


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("mm_simt_row"),
    key=_STRIDE_KEY,
    prune_configs_by={"early_config_prune": _prune_simt_row},
    flagtune_op_name="mm",
    flagtune_expand_op_name="mm_simt_row",
    flagtune_yaml_path=EXPAND_CONFIG_FILENAME,
    rep=20,
)
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


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("mm_simt_column"),
    key=_STRIDE_KEY,
    prune_configs_by={"early_config_prune": _prune_simt_column},
    flagtune_op_name="mm",
    flagtune_expand_op_name="mm_simt_column",
    flagtune_yaml_path=EXPAND_CONFIG_FILENAME,
    rep=20,
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


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("mm_pack"),
    key=["R", "C", "SR", "SC"],
    prune_configs_by={"early_config_prune": _keep_all},
    flagtune_op_name="mm",
    flagtune_expand_op_name="mm_pack",
    flagtune_yaml_path=EXPAND_CONFIG_FILENAME,
    rep=20,
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


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("mm_reduce"),
    key=["N", "TOTAL", "SCM", "SCN", "SPLIT_K"],
    prune_configs_by={"early_config_prune": _keep_all},
    flagtune_op_name="mm",
    flagtune_expand_op_name="mm_reduce",
    flagtune_yaml_path=EXPAND_CONFIG_FILENAME,
    rep=20,
)
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


def _split_count(
    m,
    n,
    k,
    element_size,
    dense_half,
    rhs_n_contiguous,
    sm_count,
    shared_bytes,
    l2_bytes,
):
    # Short, medium-sized GEMMs do not amortize a separate reduction pass.
    if (
        (dense_half or (element_size == 4 and rhs_n_contiguous))
        and 256 <= min(m, n) <= max(m, n) <= 512
        and 256 <= k <= 512
        and m % 64 == n % 64 == k % 64 == 0
    ):
        return 1
    target = max(1, sm_count * 3 // 4)
    bm, bn = _reference_tile(element_size, shared_bytes)
    tiles = triton.cdiv(m, bm) * triton.cdiv(n, bn)
    if tiles < target:
        if (
            dense_half
            and rhs_n_contiguous
            and 2 <= m < _MMA_MIN
            and tiles >= sm_count // 2
            and 2048 <= k <= 4096
            and n % _K_TILE == k % _K_TILE == 0
        ):
            # Masked NN rows can stream RHS panels with half a device wave.
            return 1
        wanted = _ceil_power_of_two(triton.cdiv(target, tiles))
        affordable = _floor_power_of_two(triton.cdiv(k, _K_TILE) // _SPLIT_MIN_K_TILES)
        budgeted = _floor_power_of_two(l2_bytes // max(1, 4 * m * n))
        split = max(1, min(wanted, affordable, budgeted, _SPLIT_MAX))
        if split > 1:
            return split

    # Balance CTA waves for long NN reductions after the occupancy fallback.
    if not (
        dense_half
        and rhs_n_contiguous
        and 2 <= m <= 512
        and n >= 2048
        and (m >= 128 or n >= 8192)
        and 4096 <= k <= 8192
        and k % _K_TILE == 0
    ):
        return 1
    bm, bn = (min(128, max(32, triton.next_power_of_2(x))) for x in (m, n))
    ctas = triton.cdiv(m, bm) * triton.cdiv(n, bn)
    budget = min(
        4,
        _floor_power_of_two(k // 1024),
        _floor_power_of_two(2 * l2_bytes // (4 * m * n)),
    )

    def wave_work(split):
        return triton.cdiv(ctas * split, sm_count) * triton.cdiv(k, _K_TILE * split)

    candidates = [1] + [split for split in (2, 4) if split <= budget]
    best = min(candidates, key=wave_work)
    # Leave headroom for the FP32 workspace and reduction launch.
    return best if wave_work(1) >= wave_work(best) * 1.15 else 1


@lru_cache(maxsize=4096)
def _dispatch_mm(
    m,
    n,
    k,
    a_strides,
    b_strides,
    c_strides,
    dtype,
    out_dtype,
    aligned,
    self_transpose,
    sm_count,
    shared_bytes,
    l2_bytes,
):
    """Select a tuned kernel, K partitions and RHS packing from call metadata."""
    half = dtype in (torch.float16, torch.bfloat16)
    element_size = 2 if half else 4
    reduction_tiles = triton.cdiv(k, _K_TILE)
    dense_operands = a_strides == (k, 1) and b_strides in ((n, 1), (1, k))
    dense_output = c_strides == (n, 1)
    vector_aligned = aligned and all(
        stride == 1 or stride % _HALF_VECTOR_ELEMENTS == 0
        for stride in a_strides + b_strides + c_strides
    )
    dense_half = (
        half
        and out_dtype == dtype
        and dense_operands
        and dense_output
        and vector_aligned
    )

    # Exact dense squares use the native, mask-free MMA kernel.
    if (
        dense_half
        and m == n == k
        and m >= 8 * _MMA_NATIVE_TILE[0]
        and m % _MMA_NATIVE_TILE[0] == 0
        and b_strides == (n, 1)
        and not self_transpose
    ):
        return mm_kernel_nn, 1, False

    if (
        dense_half
        and 128 <= min(m, n) <= 512
        and k >= 2048
        and k % _MMA_NATIVE_TILE[2] == 0
        and b_strides == (1, k)
        and not self_transpose
    ):
        output_tiles = triton.cdiv(m, _MMA_NATIVE_TILE[0]) * triton.cdiv(
            n, _MMA_NATIVE_TILE[1]
        )
        if output_tiles >= 4:
            target = max(1, sm_count * 3 // 4)
            wanted = _ceil_power_of_two(triton.cdiv(target, output_tiles))
            affordable = max(
                1, _floor_power_of_two(k // (_MMA_NATIVE_TILE[2] * _SPLIT_MIN_K_TILES))
            )
            budgeted = max(1, _floor_power_of_two(l2_bytes // max(1, 4 * m * n)))
            split = max(1, min(wanted, affordable, budgeted, _SPLIT_MAX))
            # Async K panels require complete partitions.
            while split > 1 and k % (_MMA_NATIVE_TILE[2] * split):
                split //= 2
            if (
                split > 1
                and triton.cdiv(m, _MMA_NATIVE_TILE[0]) > 1
                and k // _MMA_NATIVE_TILE[2] <= 4 * split
            ):
                split = 1
            if split > 1:
                return mm_kernel, split, False

    if min(m, n) <= _SIMT_EXTENT and max(m, n) <= _SIMT_WIDE:
        if n < m:
            tuned = mm_kernel_small_n_partial
            programs = n * triton.cdiv(m, _SIMT_TILE)
        else:
            tuned = _simt_row_kernel
            programs = m * triton.cdiv(n, _SIMT_TILE)
        wanted = _ceil_power_of_two(triton.cdiv(sm_count, max(1, programs)))
        affordable = _floor_power_of_two(reduction_tiles // _SPLIT_MIN_K_TILES)
        budgeted = _floor_power_of_two(l2_bytes // max(1, 4 * m * n))
        split = max(1, min(wanted, affordable, budgeted, _SPLIT_MAX))
        return tuned, split, False

    if (
        self_transpose
        and out_dtype == dtype
        and m >= 1024
        and k >= 256
        and m % 128 == 0
        and k % 64 == 0
        and aligned
        and dense_output
        and a_strides in ((k, 1), (1, m))
    ):
        return _syrk_kernel, 1, half and a_strides[1] != 1

    split = _split_count(
        m,
        n,
        k,
        element_size,
        dense_half,
        b_strides[1] == 1,
        sm_count,
        shared_bytes,
        l2_bytes,
    )
    if (
        dense_half
        and 2 <= m <= 64
        and n >= 2048
        and k >= 512
        and ((m >= 16 and n % 128 == 0) or (n >= 8192 and n % 32 == 0))
        and k % 64 == 0
        and b_strides[1] == 1
    ):
        return mm_kernel_wide, split, False

    pack = False
    if half and dense_operands and b_strides[0] != 1:
        pack = (
            m // 128 >= 8
            and n // 128 >= 8
            and reduction_tiles >= 256
            and m % 128 == 0
            and n % 128 == 0
            and k % _K_TILE == 0
        ) or (
            split == 1
            and out_dtype == dtype
            and m >= 1024
            and 256 <= n <= 512
            and n % 128 == 0
            and 4096 <= k <= 8192
            and k % _K_TILE == 0
            and k * n * element_size <= l2_bytes
            and dense_output
            and vector_aligned
        )
    if (
        split == 1
        and dense_half
        and (b_strides[0] == 1 or pack)
        and m >= 1024
        and m % 256 != 0
        and n % 256 == 0
        and k >= 512
        and k % 32 == 0
        and m * 8 >= triton.cdiv(m, 256) * 256 * 7
        and triton.cdiv(m, 256) * triton.cdiv(n, 256) >= sm_count // 2
    ):
        return mm_kernel_nt_rows, 1, pack
    return mm_kernel, split, pack


def mm(a, b, *, out=None):
    logger.debug("GEMS METAX MM")
    m, k = a.shape
    n = b.shape[1]
    if out is None:
        out = torch.empty((m, n), device=a.device, dtype=a.dtype)
    a_strides, b_strides, c_strides = a.stride(), b.stride(), out.stride()

    with torch_device_fn.device(a.device):
        if not m or not n or not k:
            out.zero_()
            return out
        if n == 1:
            mv(a, b[:, 0], out=out[:, 0])
            return out
        if m == 1:
            mv(b.transpose(0, 1), a[0], out=out[0])
            return out

        properties = get_device_properties(a.device.index)
        tuned, split_k, pack_rhs = _dispatch_mm(
            m,
            n,
            k,
            a_strides,
            b_strides,
            c_strides,
            a.dtype,
            out.dtype,
            all(t.data_ptr() % _VECTOR_ALIGNMENT_BYTES == 0 for t in (a, b, out)),
            a.data_ptr() == b.data_ptr()
            and a.shape == b.shape[::-1]
            and a_strides == b_strides[::-1],
            properties.multi_processor_count,
            properties.shared_memory_per_block,
            properties.L2_cache_size,
        )
        if pack_rhs:
            packed = torch.empty_strided((k, n), (1, k), dtype=b.dtype, device=b.device)
            _pack_rhs_kernel[
                lambda cfg: (triton.cdiv(k, cfg["BR"]) * triton.cdiv(n, cfg["BC"]),)
            ](b, packed, k, n, *b_strides)
            b, b_strides = packed, (1, k)
            if tuned is _syrk_kernel:
                a = b.transpose(0, 1)
                a_strides = a.stride()

        if tuned is mm_kernel_nn:
            grid = lambda cfg: (triton.cdiv(m, cfg["BM"]) * triton.cdiv(n, cfg["BN"]),)
            tuned[grid](a, b, out, m, n, k)
            return out
        if tuned is _syrk_kernel:
            grid = lambda cfg: (
                triton.cdiv(m, cfg["BT"]) * (triton.cdiv(m, cfg["BT"]) + 1) // 2,
            )
            tuned[grid](a, out, m, k, *a_strides, *c_strides)
            return out

        target = out
        target_strides = c_strides
        if split_k > 1:
            target = torch.empty((split_k, m, n), device=a.device, dtype=torch.float32)
            target_strides = (n, 1)

        if tuned is _simt_row_kernel:
            grid = lambda cfg: (split_k * m * triton.cdiv(n, cfg["BN"]),)
        elif tuned is mm_kernel_small_n_partial:
            grid = lambda cfg: (split_k * n * triton.cdiv(m, cfg["BM"]),)
        else:

            def grid(cfg):
                rows, columns = (n, m) if cfg["TRANSPOSE"] else (m, n)
                return (
                    split_k
                    * triton.cdiv(rows, cfg["BM"])
                    * triton.cdiv(columns, cfg["BN"]),
                )

        tuned[grid](
            a,
            b,
            target,
            m,
            n,
            k,
            *a_strides,
            *b_strides,
            *target_strides,
            SPLIT_K=split_k,
        )
        if split_k > 1:
            mm_kernel_splitk_reduce[lambda cfg: (triton.cdiv(m * n, cfg["BLOCK"]),)](
                target, out, n, m * n, *c_strides, SPLIT_K=split_k
            )
    return out


def mm_out(a, b, *, out):
    # Registration filters use the function name, so keep a distinct out entry.
    return mm(a, b, out=out)
