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
# See the License for the specific language governing permissions and
# limitations under the License.

"""Hygon Triton/FlagTree implementation of the MoE router GEMM.

Compute ``x @ weight.T`` for FP16/BF16/FP32 inputs with FP32 output.
Weights remain in their native [N, K] layout. GEMM and reduction use Triton;
PyTorch only allocates buffers; all nonempty computation uses Triton.
The host dispatch selects GEMV, shared-row GEMV, two-stage Split-K, or GEMM.
FlagTune searches a bounded candidate set within the selected kernel family.
"""

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, libtuner
from flag_gems.utils import triton_lang_extension as ext
from flag_gems.utils.libentry import LibTuner

__all__ = ["router_gemm"]


class _RouterTuner(LibTuner.get("default")):
    """Use LibTuner's FlagTune mode, search, and persistent dtype-aware cache.

    Shape-aware pruning selects a bounded space. Split-K tuning measures
    the producer and reduction kernels together.
    """

    def _flagtune_configs_for_mode(self, op_name, mode):
        family = self.base_fn.__name__.split("_")[2]
        family = "matrix" if family == "kernel" else family
        configs = _router_config_pool(family, mode is runtime.TuningMode.EXPANDED)
        return configs, self._flagtune_default_strategy

    def _bench(self, *args, config, **kwargs):
        if not self.base_fn.__name__.startswith("router_gemm_partial"):
            return super()._bench(*args, config=config, **kwargs)
        # Select Split-K using the complete device pipeline. Compiled handles
        # here are local to the tuning trial; normal dispatch uses LibEntry.
        x, weight, partials, m, n, k = args[:6]
        out = torch.empty((m, n), device=x.device, dtype=torch.float32)
        bound = dict(zip(self.arg_names, args))
        bound.update(config.kwargs)
        grid = kwargs["grid"](bound) if callable(kwargs["grid"]) else kwargs["grid"]
        grid = tuple(grid) + (1,) * (3 - len(grid))
        main = self.fn.run(*args, **kwargs, **config.all_kwargs())
        bm, bn, bk, sk = (config.kwargs[name] for name in ("BM", "BN", "BK", "SK"))
        rgrid = (triton.cdiv(m * n, 256), 1, 1)
        reduction, _ = router_gemm_reduce_kernel[rgrid](partials, out, m * n, sk, 256)

        def run_pair():
            main[grid](*args, bm, bn, bk, sk)
            reduction[rgrid](partials, out, m * n, sk, 256)

        return triton.testing.do_bench(
            run_pair, warmup=2, rep=8, quantiles=[0.5, 0.2, 0.8]
        )

    def prune_configs(self, kwargs):
        family = self.base_fn.__name__.split("_")[2]
        family = "matrix" if family == "kernel" else family
        args = self.nargs
        tensor = next(
            value for value in args.values() if isinstance(value, torch.Tensor)
        )
        seed = _router_seed(
            family, args.get("M", 0), args.get("N", 0), args.get("K", 0), tensor.dtype
        )
        if self._flagtune_mode is not runtime.TuningMode.EXPANDED:
            return [seed]
        return _router_config_space(family, seed, args.get("M"))


def _router_config(meta, warps=4, stages=2):
    return triton.Config(meta, num_warps=warps, num_stages=stages)


def _router_config_space(family, seed, m=None):
    configs = [seed]
    if family == "vector":
        configs += [
            _router_config({"BN": bn, "BK": bk}, warps)
            for bn, bk, warps in [(1, 512, 4), (2, 2048, 4), (8, 512, 4), (1, 2048, 2)]
        ]
    elif family == "shared":
        configs += [
            _router_config({"BM": bm, "BN": bn, "BK": bk})
            for bm, bn, bk in [(1, 4, 1024), (2, 4, 1024), (2, 4, 2048), (2, 8, 512)]
        ]
    elif family == "partial":
        # Bound the search by the output-tile occupancy regime.
        if m is not None and m <= 16:
            candidates = [(32, 32, 128, sk) for sk in (8, 16)]
            candidates += [(16, 64, 128, sk) for sk in (4, 8, 16)]
            candidates += [(16, 16, 256, 8)]
        elif m is not None and m <= 64:
            candidates = [
                (bm, bn, bk, sk)
                for bm, bn, bk in (
                    (16, 64, 128),
                    (32, 32, 128),
                    (64, 64, 64),
                    (32, 128, 64),
                )
                for sk in (8, 16)
            ]
        else:
            candidates = [(32, 128, 64, sk) for sk in (8, 16)]
            candidates += [(64, 64, 64, sk) for sk in (2, 4, 8)]
            candidates += [(32, 64, 128, sk) for sk in (4, 8)]
        configs += [
            _router_config({"BM": bm, "BN": bn, "BK": bk, "SK": sk})
            for bm, bn, bk, sk in candidates
        ]
    elif family == "matrix":
        configs += [
            _router_matrix_tile(*tile)
            for tile in (
                (16, 16, 256, 4, 2, 1, "none"),
                (32, 32, 128, 4, 2, 4, "none"),
                (64, 64, 32, 4, 2, 4, "none"),
                (128, 256, 64, 8, 1, 8, "none"),
            )
        ]
    # Always retain the previously measured default as a candidate.
    unique = {}
    for config in configs:
        unique[str(config)] = config
    return list(unique.values())


def _router_split_seed(m, n):
    # Seeds cover occupancy regimes; Expanded also measures nearby choices.
    if m <= 16:
        return 32, 32, 128, 16
    if m <= 64:
        return (16, 64, 128, 16) if n <= 512 else (64, 64, 64, 8)
    return (32, 128, 64, 16) if n <= 512 else (64, 64, 64, 4)


# Default seeds, also retained in FlagTune's Expanded candidate space.
_ROUTER_MATRIX_CONFIGS = {
    (8, 2048, 4096): (16, 16, 256, 4, 2, 4, "none"),
    (100, 2048, 4096): (32, 64, 128, 8, 2, 1, "none"),
    (1028, 256, 4096): (32, 32, 128, 4, 2, 4, "none"),
    (2048, 256, 4096): (64, 64, 128, 4, 2, 4, "mmac5-ds6"),
    (3805, 256, 4096): (64, 64, 64, 4, 2, 4, "none"),
    (7884, 512, 4096): (128, 256, 32, 8, 2, 8, "none"),
    (16384, 2048, 4096): (128, 256, 64, 8, 1, 8, "none"),
}


def _router_matrix_config(m, n, k):
    return _ROUTER_MATRIX_CONFIGS.get(
        (m, n, k),
        (
            (128, 256, 64, 8, 1, 8, "none")
            if m >= 1024
            else (32, 64, 128, 4, 2, 4, "none")
        ),
    )


def _router_seed(family, m, n, k, dtype):
    if family == "vector":
        return _router_config({"BN": 8 if n >= 1024 else 1, "BK": 512})
    if family == "shared":
        return _router_config({"BM": 1, "BN": 4, "BK": 1024})
    if family == "partial":
        return _router_config(
            dict(zip(("BM", "BN", "BK", "SK"), _router_split_seed(m, n)))
        )
    bm, bn, bk, nw, stages, gm, sched = (
        (64, 64, 32, 4, 2, 4, "none")
        if dtype == torch.float32
        else _router_matrix_config(m, n, k)
    )
    return _router_matrix_tile(bm, bn, bk, nw, stages, gm, sched)


def _router_matrix_tile(bm, bn, bk, nw, stages, gm, sched):
    return _router_config(
        {
            "BLOCK_M": bm,
            "BLOCK_N": bn,
            "BLOCK_K": bk,
            "GROUP_M": gm,
            "sched_latency": sched,
        },
        nw,
        stages,
    )


def _router_config_pool(family, expanded=True):
    # The pool defines persistent tuning identity. Shape-aware pruning below
    # keeps the measured candidate set unchanged, rather than searching it all.
    shapes = [(m, n, 4096) for m in (16, 64, 128, 2048) for n in (256, 2048)]
    if family == "matrix":
        shapes += list(_ROUTER_MATRIX_CONFIGS)
    configs = {}
    for m, n, k in shapes:
        for dtype in (torch.bfloat16, torch.float32):
            seed = _router_seed(family, m, n, k, dtype)
            candidates = _router_config_space(family, seed, m) if expanded else [seed]
            for config in candidates:
                configs[str(config)] = config
    return list(configs.values())


_ROUTER_TUNE_OPTIONS = dict(
    strategy="default",
    policy=_RouterTuner,
    flagtune_op_name="router_gemm",
    warmup=2,
    rep=8,
    benchmark_mode="event",
)


@libentry()
@libtuner(
    configs=_router_config_pool("matrix"),
    key=[
        "M",
        "N",
        "K",
        "stride_xm",
        "stride_xk",
        "stride_wk",
        "stride_wn",
        "stride_om",
        "stride_on",
    ],
    **_ROUTER_TUNE_OPTIONS,
)
@triton.heuristics(
    {
        "EVEN_N": lambda a: a["N"] % a["BLOCK_N"] == 0,
        "EVEN_K": lambda a: a["K"] % a["BLOCK_K"] == 0,
    }
)
@triton.jit
def router_gemm_kernel(
    x_ptr,
    weight_ptr,
    out_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_xm: tl.constexpr,
    stride_xk: tl.constexpr,
    stride_wk: tl.constexpr,
    stride_wn: tl.constexpr,
    stride_om: tl.constexpr,
    stride_on: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    pid = ext.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    # Group neighbouring M tiles so a weight tile remains hot in L2.  This is
    # particularly useful for router decode, where M is often below 32.
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    group_size_m = tl.minimum(num_pid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    mask_m = offs_m < M
    mask_n = offs_n < N

    # The modulo form provides a valid aligned hint for the last tile while
    # the original offsets are retained for the store mask.  N is aligned on
    # the model shapes, so avoid carrying a tail mask through every K step.
    load_m = tl.max_contiguous(tl.multiple_of(offs_m % M, BLOCK_M), BLOCK_M)
    if EVEN_N:
        load_n = tl.max_contiguous(tl.multiple_of(offs_n, BLOCK_N), BLOCK_N)
    else:
        load_n = tl.max_contiguous(tl.multiple_of(offs_n % N, BLOCK_N), BLOCK_N)
    x_ptrs = x_ptr + load_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    # Read the [N, K] storage through its logical [K, N] transpose.  The
    # resulting K dimension is contiguous and avoids a register transpose.
    w_ptrs = weight_ptr + offs_k[:, None] * stride_wk + load_n[None, :] * stride_wn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    if EVEN_K:
        loop_end = K
    else:
        loop_end = tl.cdiv(K, BLOCK_K) * BLOCK_K - BLOCK_K
    for _ in range(0, loop_end, BLOCK_K):
        x = tl.load(x_ptrs, mask=mask_m[:, None], other=0.0)
        if EVEN_N:
            weight = tl.load(w_ptrs)
        else:
            weight = tl.load(w_ptrs, mask=mask_n[None, :], other=0.0)
        acc = tl.dot(x, weight, acc, input_precision="ieee")
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    if not EVEN_K:
        rk = loop_end + offs_k
        mask_k = rk < K
        x = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        weight = tl.load(
            w_ptrs,
            mask=mask_k[:, None] & (mask_n[None, :] if not EVEN_N else True),
            other=0.0,
        )
        acc = tl.dot(x, weight, acc, input_precision="ieee")

    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


@libentry()
@libtuner(
    configs=_router_config_pool("vector"),
    key=["M", "N", "K", "XM", "XK", "WN", "WK"],
    **_ROUTER_TUNE_OPTIONS,
)
@triton.jit
def router_gemm_vector_kernel(
    X,
    W,
    OUT,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    XM: tl.constexpr,
    XK: tl.constexpr,
    WN: tl.constexpr,
    WK: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    # Keep K partial sums in registers and reduce once after the K loop.
    p = tl.program_id(0)
    m = p // tl.cdiv(N, BN)
    n = p % tl.cdiv(N, BN) * BN + tl.arange(0, BN)
    k = tl.arange(0, BK)
    acc = tl.zeros((BN, BK), tl.float32)
    for b in range(tl.cdiv(K, BK)):
        kk = b * BK + k
        x = tl.load(X + m * XM + kk * XK, kk < K, 0).to(tl.float32)
        w = tl.load(
            W + n[:, None] * WN + kk[None, :] * WK,
            (n[:, None] < N) & (kk[None, :] < K),
            0,
        ).to(tl.float32)
        acc = tl.fma(w, x[None, :], acc)
    val = tl.sum(acc, 1)
    tl.store(OUT + m * N + n, val, n < N)


@libentry()
@libtuner(
    configs=_router_config_pool("shared"),
    key=["M", "N", "K"],
    **_ROUTER_TUNE_OPTIONS,
)
@triton.jit
def router_gemm_shared_vector_kernel(
    X,
    W,
    OUT,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    p = tl.program_id(0)
    nm = tl.cdiv(M, BM)
    m = p % nm * BM
    ns = p // nm * BN + tl.arange(0, BN)
    ks = tl.arange(0, BK)
    if BM > 0:
        acc0 = tl.zeros((BN, BK), tl.float32)
    if BM > 1:
        acc1 = tl.zeros((BN, BK), tl.float32)
    for block in range(tl.cdiv(K, BK)):
        kk = block * BK + ks
        w = tl.load(
            W + ns[:, None] * K + kk[None, :],
            (ns[:, None] < N) & (kk[None, :] < K),
            0,
        ).to(tl.float32)
        if BM > 0:
            x0 = tl.load(X + (m + 0) * K + kk, (m + 0 < M) & (kk < K), 0).to(tl.float32)
            acc0 = tl.fma(w, x0[None, :], acc0)
        if BM > 1:
            x1 = tl.load(X + (m + 1) * K + kk, (m + 1 < M) & (kk < K), 0).to(tl.float32)
            acc1 = tl.fma(w, x1[None, :], acc1)
    if BM > 0:
        tl.store(OUT + (m + 0) * N + ns, tl.sum(acc0, 1), (m + 0 < M) & (ns < N))
    if BM > 1:
        tl.store(OUT + (m + 1) * N + ns, tl.sum(acc1, 1), (m + 1 < M) & (ns < N))


@triton.jit
def _router_split_k(
    X,
    W,
    P,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    SK: tl.constexpr,
    GROUP_M: tl.constexpr = 1,
    FULL_TILES: tl.constexpr = False,
):
    pid = tl.program_id(0)
    split = tl.program_id(1)
    num_n: tl.constexpr = (N + BN - 1) // BN
    if GROUP_M > 1:
        group = pid // (GROUP_M * num_n)
        group_m = tl.minimum(tl.cdiv(M, BM) - group * GROUP_M, GROUP_M)
        tile = pid % (GROUP_M * num_n)
        pid_m = group * GROUP_M + tile % group_m
        pid_n = tile // group_m
    else:
        pid_m = pid // num_n
        pid_n = pid % num_n
    rows = pid_m * BM + tl.arange(0, BM)
    cols = pid_n * BN + tl.arange(0, BN)
    valid_rows = rows[:, None] < M
    steps: tl.constexpr = (K + SK * BK - 1) // (SK * BK)
    first_k = split * (steps * BK) + tl.arange(0, BK)
    a_ptrs = X + rows[:, None] * K + first_k[None, :]
    b_ptrs = W + cols[None, :] * K + first_k[:, None]
    acc = tl.zeros((BM, BN), tl.float32)
    for block in range(steps):
        if FULL_TILES:
            # The BF16 dispatch uses only full tiles along N and K.
            a = tl.load(a_ptrs, valid_rows, 0)
            b = tl.load(b_ptrs)
        else:
            valid_k = first_k + block * BK < K
            a = tl.load(a_ptrs, valid_rows & valid_k[None, :], 0)
            b = tl.load(b_ptrs, (cols[None, :] < N) & valid_k[:, None], 0)
        acc = tl.dot(a, b, acc, input_precision="ieee")
        a_ptrs += BK
        b_ptrs += BK
    mask = valid_rows if FULL_TILES else valid_rows & (cols[None, :] < N)
    tl.store(P + (split * M + rows[:, None]) * N + cols[None, :], acc, mask)


@libentry()
@libtuner(
    configs=_router_config_pool("partial"),
    key=["M", "N", "K"],
    **_ROUTER_TUNE_OPTIONS,
)
@triton.jit(do_not_specialize=["P"])
def router_gemm_partial_kernel(
    X,
    W,
    P,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    SK: tl.constexpr,
):
    # Retain input specialization for Hygon buffer loads. Scratch stores do
    # not need pointer-range/alignment specialization on the host.
    _router_split_k(X, W, P, M, N, K, BM, BN, BK, SK)


@libentry()
@libtuner(
    configs=_router_config_pool("partial"),
    key=["M", "N", "K"],
    **_ROUTER_TUNE_OPTIONS,
)
@triton.jit(do_not_specialize=["X", "W", "P"])
def router_gemm_partial_generic_kernel(
    X,
    W,
    P,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    ALIGNED: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    SK: tl.constexpr,
):
    # Small output tiles are sensitive to dispatch overhead.
    # Only assert alignment after the caller has checked the actual pointers.
    if ALIGNED:
        X = tl.multiple_of(X, 16)
        W = tl.multiple_of(W, 16)
        P = tl.multiple_of(P, 16)
    _router_split_k(X, W, P, M, N, K, BM, BN, BK, SK)


@libentry()
@triton.jit(do_not_specialize=["P", "OUT", "MN"])
def router_gemm_reduce_kernel(P, OUT, MN, SK: tl.constexpr, B: tl.constexpr):
    # Generic pointers avoid unnecessary alignment/storage-range dispatch work.
    # Runtime MN allows the same reduction program to serve different shapes.
    # Each lane owns an output element. Accumulate slices in a fixed order,
    # without inter-CTA synchronization, atomics, or initialized scratch state.
    offsets = tl.program_id(0) * B + tl.arange(0, B)
    total = tl.zeros((B,), tl.float32)
    for split in tl.static_range(SK):
        total += tl.load(P + split * MN + offsets, offsets < MN, 0)
    tl.store(OUT + offsets, total, offsets < MN)


# Each row sets the upper M bound and (BM, BN, BK, splits, group M,
# warps, matrix instruction width, K pack) for aligned BF16 inputs.
_BF16_SPLIT_CONFIGS = {
    256: (
        (48, (16, 64, 128, 16, 1, 4, 16, 2)),
        (64, (32, 32, 128, 8, 2, 4, 0, 1)),
        (96, (16, 32, 128, 8, 1, 2, 0, 1)),
        (160, (32, 32, 128, 4, 2, 4, 16, 2)),
    ),
    512: (
        (16, (16, 64, 128, 16, 1, 4, 16, 2)),
        (32, (32, 32, 128, 8, 2, 4, 0, 1)),
        (48, (16, 32, 128, 8, 1, 2, 0, 1)),
        (64, (32, 64, 128, 8, 2, 4, 0, 1)),
        (80, (16, 32, 64, 8, 1, 2, 0, 1)),
        (160, (32, 64, 128, 4, 2, 4, 16, 2)),
    ),
    1024: (
        (16, (16, 32, 128, 8, 1, 2, 0, 1)),
        (32, (32, 32, 128, 8, 2, 4, 0, 1)),
        (64, (32, 32, 128, 4, 2, 4, 16, 2)),
    ),
    2048: (
        (16, (16, 16, 128, 8, 1, 2, 16, 2)),
        (64, (32, 32, 128, 4, 2, 4, 16, 2)),
    ),
}


def _bf16_split_config(m, n, k):
    # GEMV still wins for the smallest token counts.
    if m < (5 if n <= 256 else 4):
        return None
    if k == 7168 and n == 256 and m <= 16:
        return (16, 32, 128, 8, 1, 2, 0, 1)
    if k == 4096:
        for max_m, config in _BF16_SPLIT_CONFIGS.get(n, ()):
            if m <= max_m:
                return config
    return None


@libentry()
@triton.jit(do_not_specialize=["m"])
def _bf16_split_kernel(
    X,
    W,
    P,
    m,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    SK: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    _router_split_k(X, W, P, m, N, K, BM, BN, BK, SK, GROUP_M, True)


def _launch_bf16_split(x, weight, out, m, n, k, config):
    bm, bn, bk, sk, group_m, warps, instr, kpack = config
    partials = torch.empty((sk, m, n), device=x.device, dtype=torch.float32)
    _bf16_split_kernel[(triton.cdiv(m, bm) * (n // bn), sk)](
        x,
        weight,
        partials,
        m,
        n,
        k,
        bm,
        bn,
        bk,
        sk,
        group_m,
        num_warps=warps,
        num_stages=2,
        matrix_instr_nonkdim=instr,
        kpack=kpack,
    )
    router_gemm_reduce_kernel[(triton.cdiv(m * n, 256),)](partials, out, m * n, sk, 256)
    return out


def _router_fast(x, weight):
    m, k = x.shape
    n = weight.shape[0]
    out = torch.empty((m, n), device=x.device, dtype=torch.float32)
    contiguous = x.is_contiguous() and weight.is_contiguous()
    if contiguous and x.dtype == torch.bfloat16:
        config = _bf16_split_config(m, n, k)
        if config is not None and x.data_ptr() % 16 == weight.data_ptr() % 16 == 0:
            return _launch_bf16_split(x, weight, out, m, n, k, config)

    # Keep dimensions in LibEntry's dispatch key so every shape is tuned
    # independently. The framework also handles dtype, strides and mode changes.
    if contiguous and (
        (8 < m <= 274) or (4 < m <= 8 and n > 512) or (274 < m <= 2048 and n <= 512)
    ):
        partials = torch.empty((16, m, n), device=x.device, dtype=torch.float32)
        grid = lambda a: (triton.cdiv(m, a["BM"]) * triton.cdiv(n, a["BN"]), a["SK"])
        if (x.dtype != torch.float32 and m <= 128 and m * n <= 65536) or (
            x.dtype == torch.float32 and m <= 64 and n <= 512
        ):
            aligned = all(t.data_ptr() % 16 == 0 for t in (x, weight, partials))
            _, meta = router_gemm_partial_generic_kernel[grid](
                x, weight, partials, m, n, k, aligned
            )
        else:
            _, meta = router_gemm_partial_kernel[grid](x, weight, partials, m, n, k)
        router_gemm_reduce_kernel[(triton.cdiv(m * n, 256),)](
            partials, out, m * n, meta["SK"], 256
        )
    elif contiguous and 4 < m <= 8 and n <= 512:
        router_gemm_shared_vector_kernel[
            lambda a: (triton.cdiv(m, a["BM"]) * triton.cdiv(n, a["BN"]),)
        ](x, weight, out, m, n, k)
    elif m <= 4:
        router_gemm_vector_kernel[lambda a: (m * triton.cdiv(n, a["BN"]),)](
            x, weight, out, m, n, k, *x.stride(), *weight.stride()
        )
    else:
        router_gemm_kernel[
            lambda a: (triton.cdiv(m, a["BLOCK_M"]) * triton.cdiv(n, a["BLOCK_N"]),)
        ](
            x,
            weight,
            out,
            m,
            n,
            k,
            *x.stride(),
            weight.stride(1),
            weight.stride(0),
            *out.stride(),
        )
    return out


def router_gemm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """FP16/BF16/FP32 router projection, with FP32 accumulation and output."""
    if x.ndim != 2 or weight.ndim != 2:
        raise ValueError("router_gemm expects two rank-2 tensors")
    if x.shape[1] != weight.shape[1]:
        raise ValueError("router_gemm expects x.shape[1] == weight.shape[1]")
    if x.device != weight.device:
        raise ValueError("router_gemm inputs must be on the same device")
    if x.dtype != weight.dtype:
        raise ValueError("router_gemm inputs must have the same dtype")
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError("router_gemm supports float16, bfloat16, and float32 inputs")
    m, k = x.shape
    n = weight.shape[0]
    # Empty dimensions require no GEMM; K=0 produces a zero output.
    if m == 0 or n == 0 or k == 0:
        return torch.zeros((m, n), dtype=torch.float32, device=x.device)
    if torch_device_fn.current_device() == x.device.index:
        return _router_fast(x, weight)
    with torch_device_fn.device(x.device):
        return _router_fast(x, weight)
