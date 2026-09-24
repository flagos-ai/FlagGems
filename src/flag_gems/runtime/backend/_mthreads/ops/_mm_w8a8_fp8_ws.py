# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""TLE FP8 GEMM; requires the MUSA SQMMA fixes in flagos-ai/FlagTree#1267."""

from copy import deepcopy
from pathlib import Path

import triton
import triton.experimental.tle.language as tle
import triton.language as tl

from flag_gems import runtime
from flag_gems.utils import libentry, libtuner

_CONFIG_YAML = str(Path(__file__).resolve().parent.parent / "tune_configs.yaml")


def _set_blocks(args):
    args["A"].block_shape = [args["BM"], args["BK"]]
    args["B"].block_shape = [args["BN"], args["BK"]]


def _prune_ws_configs(configs, named_args, **kwargs):
    args = {**named_args, **kwargs}
    m, n, k = args["M"], args["N"], args["K"]
    result = []
    for config in configs:
        p = config.kwargs
        bm, bn, bk = p["BM"], p["BN"], p["BK"]
        if bm > m or bn * p["NC"] > n:
            continue
        # Avoid fine-grained long-K grids after MP31 repeated-launch failures.
        if k >= 2048 and bm * bn < 16384:
            continue
        # Large-M multi-consumer tiles hit read/write barrier errors at long K.
        if k >= 4096 and bm >= 512 and p["NC"] > 1:
            continue
        if p["NC"] > 1 and bm * bn > 64 * p["CW"] * 32:
            continue
        if p["NC"] > 1 and n % (bn * p["NC"]):
            continue
        steps = triton.cdiv(k, bk)
        if p["STAGES"] > steps + 1:
            continue
        # Avoid single-slot producer/consumer reuse when K spans multiple tiles.
        if steps > 1 and p["STAGES"] == 1:
            continue
        result.append(deepcopy(config))
    return result


@triton.jit
def _tile_index(
    tile,
    M: tl.constexpr,
    N: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    GM: tl.constexpr,
):
    grid_m = tl.cdiv(M, BM)
    grid_n = tl.cdiv(N, BN)
    group = tile // (GM * grid_n)
    group_m = tl.minimum(GM, grid_m - group * GM)
    pm = group * GM + tile % group_m
    pn = tile % (GM * grid_n) // group_m
    return pm, pn


@triton.jit
def _produce(
    writer,
    A,
    B,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    NC: tl.constexpr,
    GM: tl.constexpr,
):
    pm, pn = _tile_index(tl.program_id(0), M, N, BM, BN * NC, GM)
    for block in tl.range(0, tl.cdiv(K, BK), num_stages=1):
        slot = writer.acquire(block)
        tle.gpu.copy(A, slot.a, (BM, BK), (pm * BM, block * BK))
        tle.gpu.copy(B, slot.b0, (BN, BK), (pn * BN * NC, block * BK))
        if NC >= 2:
            tle.gpu.copy(B, slot.b1, (BN, BK), (pn * BN * NC + BN, block * BK))
        if NC >= 3:
            tle.gpu.copy(B, slot.b2, (BN, BK), (pn * BN * NC + 2 * BN, block * BK))
        if NC >= 4:
            tle.gpu.copy(B, slot.b3, (BN, BK), (pn * BN * NC + 3 * BN, block * BK))
        writer.commit(block)


@triton.jit
def _consume(
    reader,
    C,
    SA,
    SB,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    NC: tl.constexpr,
    GM: tl.constexpr,
    ROLE: tl.constexpr,
):
    pm, pn = _tile_index(tl.program_id(0), M, N, BM, BN * NC, GM)
    acc = tl.zeros((BM, BN), tl.float32)
    for block in tl.range(0, tl.cdiv(K, BK), num_stages=1):
        ready = reader.wait(block)
        if ROLE == 0:
            bb = ready.slot.b0
        elif ROLE == 1:
            bb = ready.slot.b1
        elif ROLE == 2:
            bb = ready.slot.b2
        else:
            bb = ready.slot.b3
        acc = tle.gpu.wgmma(ready.slot.a, bb, acc, trans_b=True)
        acc = tle.gpu.wgmma_wait(0, acc)
        reader.release(block)
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN * NC + ROLE * BN + tl.arange(0, BN)
    acc *= tl.load(SA)
    acc *= tl.load(SB)
    tl.store(
        C + rm[:, None].to(tl.int64) * N + rn[None, :],
        acc,
        (rm[:, None] < M) & (rn[None, :] < N),
    )


@libentry()
@libtuner(
    configs=runtime.ops_get_configs(
        "mm_w8a8_fp8_musa_ws_default", yaml_path=_CONFIG_YAML, pre_hook=_set_blocks
    ),
    key=["M", "N", "K"],
    strategy=["default"] * 3,
    warmup=5,
    rep=10,
    prune_configs_by={"early_config_prune": _prune_ws_configs},
    flagtune_op_name="mm_w8a8_fp8",
    flagtune_expand_op_name="mm_w8a8_fp8_musa_ws",
    flagtune_yaml_path=_CONFIG_YAML,
    flagtune_pre_hook=_set_blocks,
)
@triton.jit
def ws_multi_kernel(
    A,
    B,
    C,
    SA,
    SB,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    NC: tl.constexpr,
    CW: tl.constexpr,
    PW: tl.constexpr,
    STAGES: tl.constexpr,
    GM: tl.constexpr,
):
    aa = tle.gpu.alloc(
        (STAGES, BM, BK), A.dtype, scope=tle.gpu.smem, nv_mma_shared_layout=True
    )
    b0 = tle.gpu.alloc(
        (STAGES, BN, BK), B.dtype, scope=tle.gpu.smem, nv_mma_shared_layout=True
    )
    if NC >= 2:
        b1 = tle.gpu.alloc(
            (STAGES, BN, BK), B.dtype, scope=tle.gpu.smem, nv_mma_shared_layout=True
        )
    if NC >= 3:
        b2 = tle.gpu.alloc(
            (STAGES, BN, BK), B.dtype, scope=tle.gpu.smem, nv_mma_shared_layout=True
        )
    if NC >= 4:
        b3 = tle.gpu.alloc(
            (STAGES, BN, BK), B.dtype, scope=tle.gpu.smem, nv_mma_shared_layout=True
        )
    if NC == 1:
        pipe = tle.pipe(
            capacity=STAGES, scope="cta", name="ab", readers=("r0",), a=aa, b0=b0
        )
    elif NC == 2:
        pipe = tle.pipe(
            capacity=STAGES,
            scope="cta",
            name="ab",
            readers=("r0", "r1"),
            a=aa,
            b0=b0,
            b1=b1,
        )
    elif NC == 3:
        pipe = tle.pipe(
            capacity=STAGES,
            scope="cta",
            name="ab",
            readers=("r0", "r1", "r2"),
            a=aa,
            b0=b0,
            b1=b1,
            b2=b2,
        )
    else:
        pipe = tle.pipe(
            capacity=STAGES,
            scope="cta",
            name="ab",
            readers=("r0", "r1", "r2", "r3"),
            a=aa,
            b0=b0,
            b1=b1,
            b2=b2,
            b3=b3,
        )
    if NC == 1:
        tle.gpu.warp_specialize(
            [
                (
                    _consume,
                    (
                        pipe.reader("r0", fields=("a", "b0")),
                        C,
                        SA,
                        SB,
                        M,
                        N,
                        K,
                        BM,
                        BN,
                        BK,
                        NC,
                        GM,
                        0,
                    ),
                ),
                (_produce, (pipe.writer(), A, B, M, N, K, BM, BN, BK, NC, GM)),
            ],
            worker_num_warps=[PW],
            worker_num_regs=[24],
        )
    elif NC == 2:
        tle.gpu.warp_specialize(
            [
                (
                    _consume,
                    (
                        pipe.reader("r0", fields=("a", "b0")),
                        C,
                        SA,
                        SB,
                        M,
                        N,
                        K,
                        BM,
                        BN,
                        BK,
                        NC,
                        GM,
                        0,
                    ),
                ),
                (
                    _consume,
                    (
                        pipe.reader("r1", fields=("a", "b1")),
                        C,
                        SA,
                        SB,
                        M,
                        N,
                        K,
                        BM,
                        BN,
                        BK,
                        NC,
                        GM,
                        1,
                    ),
                ),
                (_produce, (pipe.writer(), A, B, M, N, K, BM, BN, BK, NC, GM)),
            ],
            worker_num_warps=[CW, PW],
            worker_num_regs=[128, 24],
        )
    elif NC == 3:
        tle.gpu.warp_specialize(
            [
                (
                    _consume,
                    (
                        pipe.reader("r0", fields=("a", "b0")),
                        C,
                        SA,
                        SB,
                        M,
                        N,
                        K,
                        BM,
                        BN,
                        BK,
                        NC,
                        GM,
                        0,
                    ),
                ),
                (
                    _consume,
                    (
                        pipe.reader("r1", fields=("a", "b1")),
                        C,
                        SA,
                        SB,
                        M,
                        N,
                        K,
                        BM,
                        BN,
                        BK,
                        NC,
                        GM,
                        1,
                    ),
                ),
                (
                    _consume,
                    (
                        pipe.reader("r2", fields=("a", "b2")),
                        C,
                        SA,
                        SB,
                        M,
                        N,
                        K,
                        BM,
                        BN,
                        BK,
                        NC,
                        GM,
                        2,
                    ),
                ),
                (_produce, (pipe.writer(), A, B, M, N, K, BM, BN, BK, NC, GM)),
            ],
            worker_num_warps=[CW, CW, PW],
            worker_num_regs=[128, 128, 24],
        )
    else:
        tle.gpu.warp_specialize(
            [
                (
                    _consume,
                    (
                        pipe.reader("r0", fields=("a", "b0")),
                        C,
                        SA,
                        SB,
                        M,
                        N,
                        K,
                        BM,
                        BN,
                        BK,
                        NC,
                        GM,
                        0,
                    ),
                ),
                (
                    _consume,
                    (
                        pipe.reader("r1", fields=("a", "b1")),
                        C,
                        SA,
                        SB,
                        M,
                        N,
                        K,
                        BM,
                        BN,
                        BK,
                        NC,
                        GM,
                        1,
                    ),
                ),
                (
                    _consume,
                    (
                        pipe.reader("r2", fields=("a", "b2")),
                        C,
                        SA,
                        SB,
                        M,
                        N,
                        K,
                        BM,
                        BN,
                        BK,
                        NC,
                        GM,
                        2,
                    ),
                ),
                (
                    _consume,
                    (
                        pipe.reader("r3", fields=("a", "b3")),
                        C,
                        SA,
                        SB,
                        M,
                        N,
                        K,
                        BM,
                        BN,
                        BK,
                        NC,
                        GM,
                        3,
                    ),
                ),
                (_produce, (pipe.writer(), A, B, M, N, K, BM, BN, BK, NC, GM)),
            ],
            worker_num_warps=[CW, CW, CW, PW],
            worker_num_regs=[128, 128, 128, 24],
        )


@triton.jit
def _produce_fragmented(
    writer,
    A,
    B,
    Bsmall,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    GM: tl.constexpr,
):
    TN: tl.constexpr = BN + BN // 4
    pm, pn = _tile_index(tl.program_id(0), M, N, BM, TN, GM)
    for block in range(tl.cdiv(K, BK)):
        slot = writer.acquire(block)
        tle.gpu.copy(A, slot.a, (BM, BK), (pm * BM, block * BK))
        tle.gpu.copy(B, slot.b0, (BN, BK), (pn * TN, block * BK))
        tle.gpu.copy(Bsmall, slot.b1, (BN // 4, BK), (pn * TN + BN, block * BK))
        writer.commit(block)


@triton.jit
def _consume_fragmented(
    reader,
    C,
    SA,
    SB,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    GM: tl.constexpr,
    ROLE: tl.constexpr,
):
    TN: tl.constexpr = BN + BN // 4
    CN: tl.constexpr = BN if ROLE == 0 else BN // 4
    pm, pn = _tile_index(tl.program_id(0), M, N, BM, TN, GM)
    acc = tl.zeros((BM, CN), tl.float32)
    for block in range(tl.cdiv(K, BK)):
        ready = reader.wait(block)
        if ROLE == 0:
            b = ready.slot.b0
        else:
            b = ready.slot.b1
        acc = tle.gpu.wgmma(ready.slot.a, b, acc, trans_b=True)
        acc = tle.gpu.wgmma_wait(0, acc)
        reader.release(block)
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * TN + ROLE * BN + tl.arange(0, CN)
    tl.store(
        C + rm[:, None] * N + rn[None, :],
        acc * tl.load(SA) * tl.load(SB),
        (rm[:, None] < M) & (rn[None, :] < N),
    )


_BASE_WS_JIT = ws_multi_kernel.fn.fn


def _set_fragmented_blocks(args):
    args["A"].block_shape = [args["BM"], args["BK"]]
    args["B"].block_shape = [args["BN"], args["BK"]]
    args["Bsmall"].block_shape = [max(32, args["BN"] // 4), args["BK"]]


def _prune_fragmented_configs(configs, named_args, **kwargs):
    args = {**named_args, **kwargs}
    m, n, k = args["M"], args["N"], args["K"]
    result = []
    for config in configs:
        p = config.kwargs
        if p["FRAGMENTED"]:
            if p["BM"] > m or p["BN"] + p["BN"] // 4 > n:
                continue
            if p["STAGES"] > triton.cdiv(k, p["BK"]) + 1:
                continue
            result.append(deepcopy(config))
        else:
            result.extend(_prune_ws_configs([config], args))
    return result


@libentry()
@libtuner(
    configs=runtime.ops_get_configs(
        "mm_w8a8_fp8_musa_ws_fragmented_default",
        yaml_path=_CONFIG_YAML,
        pre_hook=_set_fragmented_blocks,
    ),
    key=["M", "N", "K"],
    strategy=["default"] * 3,
    warmup=5,
    rep=10,
    prune_configs_by={"early_config_prune": _prune_fragmented_configs},
    flagtune_op_name="mm_w8a8_fp8",
    flagtune_expand_op_name="mm_w8a8_fp8_musa_ws_fragmented",
    flagtune_yaml_path=_CONFIG_YAML,
    flagtune_pre_hook=_set_fragmented_blocks,
)
@triton.jit
def fragmented_kernel(
    A,
    B,
    Bsmall,
    C,
    SA,
    SB,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    NC: tl.constexpr,
    CW: tl.constexpr,
    PW: tl.constexpr,
    STAGES: tl.constexpr,
    GM: tl.constexpr,
    FRAGMENTED: tl.constexpr,
):
    if not FRAGMENTED:
        _BASE_WS_JIT(A, B, C, SA, SB, M, N, K, BM, BN, BK, NC, CW, PW, STAGES, GM)
    else:
        aa = tle.gpu.alloc(
            (STAGES, BM, BK), A.dtype, scope=tle.gpu.smem, nv_mma_shared_layout=True
        )
        b0 = tle.gpu.alloc(
            (STAGES, BN, BK), B.dtype, scope=tle.gpu.smem, nv_mma_shared_layout=True
        )
        b1 = tle.gpu.alloc(
            (STAGES, BN // 4, BK),
            B.dtype,
            scope=tle.gpu.smem,
            nv_mma_shared_layout=True,
        )
        pipe = tle.pipe(
            capacity=STAGES,
            scope="cta",
            name="ab",
            readers=("r0", "r1"),
            a=aa,
            b0=b0,
            b1=b1,
        )
        tle.gpu.warp_specialize(
            [
                (
                    _consume_fragmented,
                    (
                        pipe.reader("r0", fields=("a", "b0")),
                        C,
                        SA,
                        SB,
                        M,
                        N,
                        K,
                        BM,
                        BN,
                        BK,
                        GM,
                        0,
                    ),
                ),
                (
                    _consume_fragmented,
                    (
                        pipe.reader("r1", fields=("a", "b1")),
                        C,
                        SA,
                        SB,
                        M,
                        N,
                        K,
                        BM,
                        BN,
                        BK,
                        GM,
                        1,
                    ),
                ),
                (
                    _produce_fragmented,
                    (pipe.writer(), A, B, Bsmall, M, N, K, BM, BN, BK, GM),
                ),
            ],
            worker_num_warps=[CW, PW],
            worker_num_regs=[128, 24],
        )
