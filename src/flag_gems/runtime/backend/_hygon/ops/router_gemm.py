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

from pathlib import Path

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, libtuner
from flag_gems.utils import triton_lang_extension as ext
from flag_gems.utils.libentry import LibTuner

__all__ = ["router_gemm"]


_EXPAND_PATH = Path(__file__).resolve().parents[1] / "router_gemm_hygon_expand.yaml"


class _RouterTuner(LibTuner.get("flagtune")):
    def _flagtune_configs_for_mode(self, op_name, mode):
        configs = self._flagtune_default_configs
        if mode is runtime.TuningMode.EXPANDED:
            space = runtime.backend.get_expand_config(
                self._flagtune_expand_op_name, file_path=_EXPAND_PATH
            )
            configs = [*configs]
            for spec in space:
                configs.extend(runtime.config_loader.to_gen_config(spec))
        return list(dict.fromkeys(configs)), self._flagtune_default_strategy


_TUNE_OPTIONS = dict(
    policy=_RouterTuner, flagtune_op_name="router_gemm", warmup=2, rep=8
)


def _matrix_config(bm, bn, bk, warps, stages, group):
    return triton.Config(
        {"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk, "GROUP_M": group},
        num_warps=warps,
        num_stages=stages,
    )


def _split_config(bm, bn, bk, splits, warps=4):
    return triton.Config(
        {"BM": bm, "BN": bn, "BK": bk, "SK": splits},
        num_warps=warps,
        num_stages=2,
    )


def _aligned_config(bm, bn, bk, splits, group, warps, instr=0, pack=1):
    return triton.Config(
        {
            "BM": bm,
            "BN": bn,
            "BK": bk,
            "SK": splits,
            "GROUP_M": group,
            "matrix_instr_nonkdim": instr,
            "kpack": pack,
        },
        num_warps=warps,
        num_stages=2,
    )


_MATRIX_CONFIGS = [
    _matrix_config(16, 16, 256, 4, 2, 1),
    _matrix_config(32, 64, 128, 4, 2, 4),
    _matrix_config(64, 64, 32, 4, 2, 4),
    _matrix_config(128, 256, 64, 8, 1, 8),
]
_SPLIT_CONFIGS = [
    _split_config(32, 32, 128, 16),
    _split_config(16, 64, 128, 16),
    _split_config(32, 64, 128, 8),
    _split_config(64, 64, 64, 4),
]


def _prune_aligned_configs(configs, named_args, **kwargs):
    args = {**named_args, **kwargs}
    m, n, k = args["m"], args["N"], args["K"]
    row_limit = max(16, triton.next_power_of_2(m))
    legal = [
        config
        for config in configs
        if config.kwargs["BM"] <= row_limit
        and n % config.kwargs["BN"] == 0
        and k % (config.kwargs["SK"] * config.kwargs["BK"]) == 0
    ]
    # Once eight FP32 partials would exceed 1 MiB of writes, prefer fewer
    # slices. The producer-only tuner otherwise misses that reduction traffic.
    if m * n * 8 * 4 > 1024**2:
        return [config for config in legal if config.kwargs["SK"] <= 4] or legal
    return legal


@libentry()
@libtuner(
    configs=_MATRIX_CONFIGS,
    flagtune_expand_op_name="matrix",
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
    **_TUNE_OPTIONS,
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
    configs=[
        triton.Config({"BN": bn, "BK": bk}, num_warps=4, num_stages=2)
        for bn, bk in ((1, 512), (2, 2048), (8, 512))
    ],
    flagtune_expand_op_name="vector",
    key=["M", "N", "K", "XM", "XK", "WN", "WK"],
    **_TUNE_OPTIONS,
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
    configs=[
        triton.Config({"BM": bm, "BN": 4, "BK": 1024}, num_warps=4, num_stages=2)
        for bm in (1, 2)
    ],
    flagtune_expand_op_name="shared_vector",
    key=["M", "N", "K"],
    **_TUNE_OPTIONS,
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
    configs=_SPLIT_CONFIGS,
    flagtune_expand_op_name="split",
    key=["M", "N", "K"],
    **_TUNE_OPTIONS,
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
    configs=_SPLIT_CONFIGS,
    flagtune_expand_op_name="split",
    key=["M", "N", "K"],
    **_TUNE_OPTIONS,
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


def _aligned_tuner(configs):
    return libentry()(
        libtuner(
            configs=configs,
            key=["m", "N", "K"],
            flagtune_expand_op_name="aligned_split",
            prune_configs_by={"early_config_prune": _prune_aligned_configs},
            **_TUNE_OPTIONS,
        )(_bf16_split_kernel)
    )


_ALIGNED_NARROW = _aligned_tuner(
    [
        _aligned_config(16, 64, 128, 16, 1, 4, 16, 2),
        _aligned_config(16, 32, 128, 8, 1, 2),
        _aligned_config(32, 32, 128, 8, 2, 4),
        _aligned_config(32, 64, 128, 4, 2, 4, 16, 2),
    ]
)
_ALIGNED_WIDE = _aligned_tuner(
    [
        _aligned_config(16, 16, 128, 8, 1, 2, 16, 2),
        _aligned_config(16, 32, 128, 8, 1, 2),
        _aligned_config(32, 32, 128, 8, 2, 4),
        _aligned_config(32, 32, 128, 4, 2, 4, 16, 2),
    ]
)


def _use_aligned_split(m, n, k):
    # Split K when there are too few output tiles to fill the device. The
    # scratch limit bounds the work added by the second launch.
    return (
        m >= 4
        and (m > 4 or n > 256)
        and n >= 256
        and n % 32 == 0
        and k >= 2048
        and k % 1024 == 0
        and m <= 160
        and (n <= 512 or m <= 64)
        and m * n <= 128 * 1024
    )


def _launch_bf16_split(x, weight, out, m, n, k):
    partials = torch.empty((16, m, n), device=x.device, dtype=torch.float32)
    tuner = _ALIGNED_NARROW if n <= 512 else _ALIGNED_WIDE
    grid = lambda meta: (triton.cdiv(m, meta["BM"]) * (n // meta["BN"]), meta["SK"])
    _, meta = tuner[grid](x, weight, partials, m, n, k)
    router_gemm_reduce_kernel[(triton.cdiv(m * n, 256),)](
        partials, out, m * n, meta["SK"], 256
    )
    return out


def _router_fast(x, weight):
    m, k = x.shape
    n = weight.shape[0]
    out = torch.empty((m, n), device=x.device, dtype=torch.float32)
    contiguous = x.is_contiguous() and weight.is_contiguous()
    if (
        contiguous
        and x.dtype == torch.bfloat16
        and _use_aligned_split(m, n, k)
        and x.data_ptr() % 16 == weight.data_ptr() % 16 == 0
    ):
        return _launch_bf16_split(x, weight, out, m, n, k)

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
