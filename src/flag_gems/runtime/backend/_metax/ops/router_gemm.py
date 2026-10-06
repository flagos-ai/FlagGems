# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import copy
import logging
import os

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import get_device_properties, libentry, libtuner

logger = logging.getLogger(__name__)
EXPAND_CONFIG_FILENAME = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "router_gemm_metax_expand.yaml")
)

# Split-K search limits for partition size, workspace, and program count.
_SPLIT_K_CANDIDATES = (1, 2, 4, 8, 16, 32)
_MIN_K_PER_SPLIT = 256
_MAX_SPLIT_K_WORKSPACE_BYTES = 32 * 1024**2
_MAX_SPLIT_K_PROGRAMS_PER_SM = 4


@triton.jit
def _router_simt(
    X,
    W,
    Y,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    SXM: tl.constexpr,
    SXK: tl.constexpr,
    SWN: tl.constexpr,
    SWK: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    # Coalesced K reads use the C550's 64-thread warp. One token is reused
    # across BN expert dot products without materializing input copies.
    nm, nn = tl.cdiv(M, BM), tl.cdiv(N, BN)
    pid = tl.program_id(0)
    split = pid // (nm * nn) % SPLIT_K
    iterations = tl.cdiv(K, BK * SPLIT_K)
    ki = tl.arange(0, BK).to(tl.int64) + split * iterations * BK
    mi = (pid // nn % nm).to(tl.int64)
    ni = (pid % nn * BN + tl.arange(0, BN)).to(tl.int64)
    xp = X + mi * SXM
    wp = W + ni[:, None] * SWN
    acc = tl.zeros((BN, BK), tl.float32)
    for i in range(iterations):
        ks = ki + i * BK
        x = tl.load(xp + ks * SXK, ks < K, 0).to(tl.float32)
        w = tl.load(
            wp + ks[None, :] * SWK, (ni[:, None] < N) & (ks[None, :] < K), 0
        ).to(tl.float32)
        acc = tl.fma(x[None, :], w, acc)
    result = tl.sum(acc, 1)
    tl.store(Y + split.to(tl.int64) * M * N + mi * N + ni, result, ni < N)


@triton.jit
def _router_mma(
    X,
    W,
    Y,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    SXM: tl.constexpr,
    SXK: tl.constexpr,
    SWN: tl.constexpr,
    SWK: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    SPLIT_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    nm, nn = tl.cdiv(M, BM), tl.cdiv(N, BN)
    split = tl.program_id(0) // (nm * nn) % SPLIT_K
    pid = tl.program_id(0) % (nm * nn)
    group = pid // (GROUP_M * nn)
    first_m = group * GROUP_M
    group_m = tl.minimum(nm - first_m, GROUP_M)
    local = pid % (GROUP_M * nn)
    mi = (first_m + local % group_m) * BM + tl.arange(0, BM)
    ni = local // group_m * BN + tl.arange(0, BN)
    iterations = tl.cdiv(K, BK * SPLIT_K)
    ki = tl.arange(0, BK) + split * iterations * BK
    mi = tl.max_contiguous(tl.multiple_of(mi, BM), BM)
    ni = tl.max_contiguous(tl.multiple_of(ni, BN), BN)
    ki = tl.max_contiguous(tl.multiple_of(ki, BK), BK)
    xp = X + mi[:, None].to(tl.int64) * SXM + ki[None, :].to(tl.int64) * SXK
    wp = W + ni[None, :].to(tl.int64) * SWN + ki[:, None].to(tl.int64) * SWK
    acc = tl.zeros((BM, BN), tl.float32)
    for i in range(iterations):
        if K % (BK * SPLIT_K) == 0:
            if M % BM == 0:
                x = tl.load(xp)
            else:
                x = tl.load(xp, mi[:, None] < M, 0)
            if N % BN == 0:
                w = tl.load(wp)
            else:
                w = tl.load(wp, ni[None, :] < N, 0)
        else:
            x = tl.load(xp, (mi[:, None] < M) & (ki[None, :] + i * BK < K), 0)
            w = tl.load(wp, (ni[None, :] < N) & (ki[:, None] + i * BK < K), 0)
        acc = tl.dot(x, w, acc)
        xp += BK * SXK
        wp += BK * SWK
    tl.store(
        Y + split.to(tl.int64) * M * N + mi[:, None].to(tl.int64) * N + ni[None, :],
        acc,
        (mi[:, None] < M) & (ni[None, :] < N),
    )


@libentry()
@triton.jit
def _router_reduce_kernel(
    P,
    Y,
    SIZE: tl.constexpr,
    SPLIT_K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    acc = tl.zeros((BLOCK,), tl.float32)
    for split in tl.static_range(SPLIT_K):
        acc += tl.load(P + split * SIZE + i, i < SIZE, 0)
    tl.store(Y + i, acc, i < SIZE)


# Measured compiler allocation for this kernel, not a device capacity fallback.
_NT128_SHARED_BYTES = 64 * 1024


def _router_bench_post_hook(args, exception):
    # Autotuning must include the reduction in each split-K candidate's timing.
    if exception is None and args["SPLIT_K"] > 1:
        size = args["M"] * args["N"]
        _router_reduce_kernel[(triton.cdiv(size, 256),)](
            args["P"], args["Y"], size, args["SPLIT_K"], 256, num_warps=4
        )


def _prune_router(configs, named_args, **kwargs):
    args = {**named_args, **kwargs}
    m, n, k = args["M"], args["N"], args["K"]
    properties = get_device_properties(args["X"].device.index)
    sm = properties.multi_processor_count
    shared_bytes = properties.shared_memory_per_block
    result = []
    for config in configs:
        meta = config.kwargs
        bm, bn, bk = meta["BM"], meta["BN"], meta["BK"]
        nt_128 = (
            not meta["SIMT"]
            and bm == bn == bk == 128
            and config.num_warps == 4
            and meta["pipeline"] == "cpasync"
            and args["SXK"] == args["SWK"] == 1
        )
        if meta["SIMT"]:
            if (
                not args["ALLOW_SIMT"]
                or bm != 1
                or bm > triton.next_power_of_2(m)
                or bn > triton.next_power_of_2(n)
            ):
                continue
            if bm * bn * bk > 8192:
                continue
        elif bm > max(16, triton.next_power_of_2(m)) or bn > max(
            16, triton.next_power_of_2(n)
        ):
            continue
        elif (_NT128_SHARED_BYTES if nt_128 else (bm + bn) * bk * 4) > shared_bytes:
            continue
        tiles = triton.cdiv(m, bm) * triton.cdiv(n, bn)
        for split in _SPLIT_K_CANDIDATES:
            if split > args["MAX_SPLIT"]:
                break
            if nt_128 and k % (bk * split):
                continue
            if split > 1 and (
                k < split * _MIN_K_PER_SPLIT
                or tiles * split > _MAX_SPLIT_K_PROGRAMS_PER_SM * sm
            ):
                continue
            candidate = copy.deepcopy(config)
            candidate.kwargs["SPLIT_K"] = split
            result.append(candidate)
    return result


# MAX_SPLIT, ALLOW_SIMT and *_ALIGNMENT are host pruning/cache metadata.
# They remain kernel parameters so LibEntry and LibTuner key them correctly.
@libentry()
@libtuner(
    configs=runtime.ops_get_configs("router_gemm", yaml_path=EXPAND_CONFIG_FILENAME),
    key=[
        "M",
        "N",
        "K",
        "SXM",
        "SXK",
        "SWN",
        "SWK",
        "MAX_SPLIT",
        "ALLOW_SIMT",
        "X_ALIGNMENT",
        "W_ALIGNMENT",
    ],
    prune_configs_by={"early_config_prune": _prune_router},
    post_hook=_router_bench_post_hook,
    flagtune_op_name="router_gemm",
    flagtune_expand_op_name="router_gemm",
    flagtune_yaml_path=EXPAND_CONFIG_FILENAME,
    rep=30,
)
@triton.jit
def _router_kernel(
    X,
    W,
    Y,
    P,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    SXM: tl.constexpr,
    SXK: tl.constexpr,
    SWN: tl.constexpr,
    SWK: tl.constexpr,
    MAX_SPLIT: tl.constexpr,
    ALLOW_SIMT: tl.constexpr,
    X_ALIGNMENT: tl.constexpr,
    W_ALIGNMENT: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    SPLIT_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    SIMT: tl.constexpr,
):
    if SPLIT_K > 1:
        Y = P
    if SIMT:
        _router_simt(X, W, Y, M, N, K, SXM, SXK, SWN, SWK, BM, BN, BK, SPLIT_K)
    else:
        _router_mma(X, W, Y, M, N, K, SXM, SXK, SWN, SWK, BM, BN, BK, SPLIT_K, GROUP_M)


def router_gemm(input, weight):
    logger.debug("GEMS METAX ROUTER GEMM")
    m, k = input.shape
    n = weight.shape[0]
    sxm, sxk = input.stride()
    swn, swk = weight.stride()
    out = torch.empty((m, n), dtype=torch.float32, device=input.device)
    bytes_per_split = m * n * out.element_size()
    split_limit = min(
        _SPLIT_K_CANDIDATES[-1],
        max(1, k // _MIN_K_PER_SPLIT),
        max(1, _MAX_SPLIT_K_WORKSPACE_BYTES // bytes_per_split),
    )
    max_split = max(split for split in _SPLIT_K_CANDIDATES if split <= split_limit)
    allow_simt = m <= 32 or n <= 8
    partial = torch.empty(
        (max_split * m * n if max_split > 1 else 0,),
        device=input.device,
        dtype=torch.float32,
    )
    grid = lambda meta: (
        triton.cdiv(m, meta["BM"]) * triton.cdiv(n, meta["BN"]) * meta["SPLIT_K"],
    )
    with torch_device_fn.device(input.device):
        # SPLIT_K=1 writes out directly; larger splits write FP32 partials.
        _, meta = _router_kernel[grid](
            input,
            weight,
            out,
            partial,
            m,
            n,
            k,
            sxm,
            sxk,
            swn,
            swk,
            max_split,
            allow_simt,
            input.data_ptr() % 16,
            weight.data_ptr() % 16,
        )
        if meta["SPLIT_K"] > 1:
            # The next launch waits for all partials, then sums them in a fixed
            # order without atomic updates to the output.
            _router_reduce_kernel[(triton.cdiv(m * n, 256),)](
                partial, out, m * n, meta["SPLIT_K"], 256, num_warps=4
            )
    return out
