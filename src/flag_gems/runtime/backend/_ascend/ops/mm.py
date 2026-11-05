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

import contextlib
import logging
import os

import torch
import triton
import triton.language as tl
import triton.language.extra.cann.extension as al

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.runtime.backend._ascend import heuristics_config_utils as _hcu
from flag_gems.utils import libentry, libtuner

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def _blockified():
    previous = os.environ.get("TRITON_ALL_BLOCKS_PARALLEL")
    os.environ["TRITON_ALL_BLOCKS_PARALLEL"] = "1"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("TRITON_ALL_BLOCKS_PARALLEL", None)
        else:
            os.environ["TRITON_ALL_BLOCKS_PARALLEL"] = previous


def _prune_mm_configs(configs, named_args, **kwargs):
    if not configs:
        return configs
    args = {**named_args, **kwargs}
    m, n, k = args["M"], args["N"], args["K"]
    if 256 <= m <= 512 and 256 <= n <= 512 and k >= 128:
        preferred = [
            config
            for config in configs
            if (
                config.kwargs["BLOCK_M"],
                config.kwargs["BLOCK_N"],
                config.kwargs["BLOCK_K"],
            )
            == (64, 128, 128)
        ]
        if preferred:
            return preferred
    large_aligned = min(m, n) >= 1024 and m % 128 == 0 and n % 128 == 0
    if not large_aligned:
        configs = [
            config
            for config in configs
            if config.kwargs["BLOCK_M"] <= 128
            and config.kwargs["BLOCK_N"] <= 128
            and config.kwargs["BLOCK_K"] <= 256
        ]
    program_counts = {
        id(config): triton.cdiv(m, config.kwargs["BLOCK_M"])
        * triton.cdiv(n, config.kwargs["BLOCK_N"])
        for config in configs
    }
    min_programs = min(program_counts.values())
    max_programs = max(1, 2 * min_programs)
    candidates = [
        config for config in configs if program_counts[id(config)] <= max_programs
    ]
    if large_aligned:
        if k >= 1024:
            if args["A"].dtype == torch.float32:
                preferred = [
                    config
                    for config in candidates
                    if (
                        config.kwargs["BLOCK_M"],
                        config.kwargs["BLOCK_N"],
                        config.kwargs["BLOCK_K"],
                    )
                    in ((128, 256, 128), (256, 128, 128))
                ]
                if preferred:
                    return preferred
                preferred_block_k = (256,)
            elif args["A"].dtype in (torch.float16, torch.bfloat16):
                preferred = [
                    config
                    for config in candidates
                    if (
                        config.kwargs["BLOCK_M"],
                        config.kwargs["BLOCK_N"],
                        config.kwargs["BLOCK_K"],
                    )
                    in ((128, 256, 256), (256, 128, 256))
                ]
                if preferred:
                    return preferred
                preferred_block_k = (256,)
            else:
                preferred_block_k = (256,)
        else:
            preferred_block_k = (128,)
        preferred = [
            config
            for config in candidates
            if config.kwargs["BLOCK_K"] in preferred_block_k
        ]
        if preferred:
            return preferred
    return candidates


def _mm_heur_divisible_m(args):
    return args["M"] % args["BLOCK_M"] == 0


def _mm_heur_divisible_n(args):
    return args["N"] % args["BLOCK_N"] == 0


_MM_HEURISTICS = {
    **_hcu.HEURISTICS_CONFIGS["mm"],
    "DIVISIBLE_M": _mm_heur_divisible_m,
    "DIVISIBLE_N": _mm_heur_divisible_n,
}


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("mm"),
    key=["M", "N", "K"],
    prune_configs_by={"early_config_prune": _prune_mm_configs},
)
@triton.heuristics(_MM_HEURISTICS)
@triton.jit
def mm_kernel(
    A,
    B,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
    dot_out_dtype: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    SPLIT_K: tl.constexpr,
    EVEN_K: tl.constexpr,
    DIVISIBLE_M: tl.constexpr,
    DIVISIBLE_N: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_z = tl.program_id(1)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)
    ram = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rbn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = pid_z * BLOCK_K + tl.arange(0, BLOCK_K)
    A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
    B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=dot_out_dtype)
    num_k_tiles = tl.cdiv(K, BLOCK_K * SPLIT_K)
    for k in tl.range(0, num_k_tiles):
        if EVEN_K:
            if DIVISIBLE_M:
                a = tl.load(A)
            else:
                a = tl.load(A, mask=(ram < M)[:, None], other=0.0)
            if DIVISIBLE_N:
                b = tl.load(B)
            else:
                b = tl.load(B, mask=(rbn < N)[None, :], other=0.0)
        else:
            k_remaining = K - k * (BLOCK_K * SPLIT_K)
            if DIVISIBLE_M:
                a = tl.load(A, mask=rk[None, :] < k_remaining, other=0.0)
            else:
                a = tl.load(
                    A,
                    mask=(rk[None, :] < k_remaining) & (ram < M)[:, None],
                    other=0.0,
                )
            if DIVISIBLE_N:
                b = tl.load(B, mask=rk[:, None] < k_remaining, other=0.0)
            else:
                b = tl.load(
                    B,
                    mask=(rk[:, None] < k_remaining) & (rbn < N)[None, :],
                    other=0.0,
                )
        al.compile_hint(a, "dot_pad_only_k")
        al.compile_hint(b, "dot_pad_only_k")
        if a.dtype != b.dtype:
            a = a.to(C.dtype.element_ty)
            b = b.to(C.dtype.element_ty)
        acc += tl.dot(a, b, out_dtype=dot_out_dtype, allow_tf32=False)
        A += BLOCK_K * SPLIT_K * stride_ak
        B += BLOCK_K * SPLIT_K * stride_bk
    acc = acc.to(C.dtype.element_ty)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    C = C + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    if DIVISIBLE_M and DIVISIBLE_N:
        mask = None
    elif DIVISIBLE_M:
        mask = (rn < N)[None, :]
    elif DIVISIBLE_N:
        mask = (rm < M)[:, None]
    else:
        mask = (rm < M)[:, None] & (rn < N)[None, :]
    if SPLIT_K == 1:
        tl.store(C, acc, mask=mask)
    else:
        tl.atomic_add(C, acc, mask=mask)


_ordered_datatypes = [torch.float16, torch.bfloat16, torch.float32]


@libentry()
@triton.jit
def _mm_narrow_n_kernel(
    A,
    B,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_OUT: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # A few output columns do not fill a Cube tile. Reduce independent
    # output elements on Vector cores, accumulating products in FP32.
    rows = tl.program_id(0) * BLOCK_OUT + tl.arange(0, BLOCK_OUT)
    rk = tl.arange(0, BLOCK_K)
    row_acc = tl.zeros((BLOCK_OUT, BLOCK_K), tl.float32)
    for start in range(tl.cdiv(K, BLOCK_K)):
        kk = start * BLOCK_K + rk
        a_rows = tl.load(
            A + rows[:, None] * K + kk[None, :],
            mask=(rows < M)[:, None] & (kk < K)[None, :],
            other=0.0,
        ).to(tl.float32)
        b_vec = tl.load(B + kk, mask=kk < K, other=0.0).to(tl.float32)
        row_acc += a_rows * b_vec[None, :]
    tl.store(C + rows, tl.sum(row_acc, axis=1), mask=rows < M)


@libentry()
@triton.jit
def _mm_rectangular_kernel(
    A,
    B,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid // tl.cdiv(N, BLOCK_N) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid % tl.cdiv(N, BLOCK_N) * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for start in tl.range(tl.cdiv(K, BLOCK_K)):
        kk = start * BLOCK_K + rk
        a = tl.load(
            A + rows[:, None] * K + kk[None, :],
            mask=(rows < M)[:, None] & (kk < K)[None, :],
            other=0.0,
        )
        b = tl.load(
            B + kk[:, None] * N + cols[None, :],
            mask=(kk < K)[:, None] & (cols < N)[None, :],
            other=0.0,
        )
        al.compile_hint(a, "dot_pad_only_k")
        al.compile_hint(b, "dot_pad_only_k")
        acc = tl.dot(a, b, acc, allow_tf32=False)
    tl.store(
        C + rows[:, None] * N + cols[None, :],
        acc.to(C.dtype.element_ty),
        mask=(rows < M)[:, None] & (cols < N)[None, :],
    )


@triton.jit
def _mm_pack_narrow4_split_kernel(B, P, K: tl.constexpr, BK: tl.constexpr):
    base = tl.program_id(0) * BK
    idx = tl.arange(0, BK * 4)
    b = tl.load(B + base * 4 + idx).reshape((BK, 2, 2))
    b02, b13 = tl.split(b)
    b0, b2 = tl.split(b02)
    b1, b3 = tl.split(b13)
    k = base + tl.arange(0, BK)
    tl.store(P + k, b0)
    tl.store(P + K + k, b1)
    tl.store(P + 2 * K + k, b2)
    tl.store(P + 3 * K + k, b3)


@triton.jit
def _mm_pack_narrow_b_kernel(
    B, P, N: tl.constexpr, K: tl.constexpr, BK: tl.constexpr, BN: tl.constexpr
):
    k = tl.program_id(0) * BK + tl.arange(0, BK)
    n = tl.arange(0, BN)
    b = tl.load(
        B + k[:, None] * N + n[None, :], (k < K)[:, None] & (n < N)[None, :], other=0
    )
    tl.store(
        P + n[:, None] * K + k[None, :],
        tl.trans(b),
        (n < N)[:, None] & (k < K)[None, :],
    )


@triton.jit
def _mm_pack_narrow4_b_kernel(
    B, P, N: tl.constexpr, K: tl.constexpr, BK: tl.constexpr, BN: tl.constexpr
):
    k = tl.program_id(0) * BK + tl.arange(0, BK)
    n = tl.arange(0, BN)
    b = tl.load(
        B + k[:, None] * N + n[None, :], (k < K)[:, None] & (n < N)[None, :], other=0
    )
    # P is padded to 16 columns so the subsequent Cube load stays aligned.
    tl.store(P + n[:, None] * K + k[None, :], tl.trans(b), (k < K)[None, :])


@triton.jit
def _mm_packed_narrow_kernel(
    A,
    P,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.arange(0, 16)
    rk = tl.arange(0, BK)
    acc = tl.zeros((BM, 16), tl.float32)
    for s in tl.range(tl.cdiv(K, BK)):
        k = s * BK + rk
        a = tl.load(
            A + rows[:, None] * K + k[None, :],
            (rows < M)[:, None] & (k < K)[None, :],
            other=0,
        )
        bt = tl.load(
            P + cols[:, None] * K + k[None, :],
            (cols < N)[:, None] & (k < K)[None, :],
            other=0,
        )
        al.compile_hint(a, "dot_pad_only_k")
        al.compile_hint(bt, "dot_pad_only_k")
        acc = tl.dot(a, tl.trans(bt), acc, allow_tf32=False)
    tl.store(
        C + rows[:, None] * N + cols[None, :],
        acc.to(C.dtype.element_ty),
        (rows < M)[:, None] & (cols < N)[None, :],
    )


@triton.jit
def _mm_packed_narrow4_kernel(
    A,
    P,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.arange(0, 16)
    rk = tl.arange(0, BK)
    acc = tl.zeros((BM, 16), tl.float32)
    for s in tl.range(tl.cdiv(K, BK)):
        k = s * BK + rk
        a = tl.load(
            A + rows[:, None] * K + k[None, :],
            (rows < M)[:, None] & (k < K)[None, :],
            other=0,
        )
        bt = tl.load(
            P + cols[:, None] * K + k[None, :],
            (cols < N)[:, None] & (k < K)[None, :],
            other=0,
        )
        al.compile_hint(a, "dot_pad_only_k")
        al.compile_hint(bt, "dot_pad_only_k")
        acc = tl.dot(a, tl.trans(bt), acc, allow_tf32=False)
    tl.store(
        C + rows[:, None] * N + cols[None, :],
        acc.to(C.dtype.element_ty),
        (rows < M)[:, None] & (cols < N)[None, :],
    )


@triton.jit
def _mm_materialize_broadcast_kernel(
    X,
    Y,
    R: tl.constexpr,
    C: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    BLOCK: tl.constexpr,
):
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    val = tl.load(X + (idx // C) * S0 + (idx % C) * S1, idx < R * C, other=0)
    tl.store(Y + idx, val, idx < R * C)


def _mm_materialize_broadcast(x):
    y = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    with torch_device_fn.device(x.device):
        _mm_materialize_broadcast_kernel[(triton.cdiv(x.numel(), 1024),)](
            x,
            y,
            x.shape[0],
            x.shape[1],
            x.stride(0),
            x.stride(1),
            1024,
            multibuffer=False,
        )
    return y


@triton.jit
def _mm_pack_wide_panels_kernel(
    B, P, N: tl.constexpr, K: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr
):
    pid = tl.program_id(0)
    panel = pid // (K // BK)
    block_k = pid % (K // BK)
    k = block_k * BK + tl.arange(0, BK)
    n = tl.arange(0, BN)
    b = tl.load(B + k[:, None] * N + panel * BN + n[None, :])
    tl.store(P + panel * K * BN + k[:, None] * BN + n[None, :], b)


@triton.jit
def _mm_wide_panel_cube_kernel(
    A,
    P,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    pid = tl.program_id(0)
    panel = pid % (N // BN)
    rows = (pid // (N // BN)) * BM + tl.arange(0, BM)
    cols = tl.arange(0, BN)
    rk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    for step in range(K // BK):
        k = step * BK + rk
        a = tl.load(A + rows[:, None] * K + k[None, :], rows[:, None] < M, other=0)
        b = tl.load(P + panel * K * BN + k[:, None] * BN + cols[None, :])
        al.compile_hint(a, "dot_pad_only_k")
        al.compile_hint(b, "dot_pad_only_k")
        acc = tl.dot(a, b, acc, allow_tf32=False)
    tl.store(
        C + rows[:, None] * N + panel * BN + cols[None, :],
        acc.to(C.dtype.element_ty),
        rows[:, None] < M,
    )


@triton.jit
def _mm_pack_large_panels_kernel(
    B, P, N: tl.constexpr, K: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr
):
    pid = tl.program_id(0)
    panel = pid // (K // BK)
    block_k = pid % (K // BK)
    k = block_k * BK + tl.arange(0, BK)
    n = tl.arange(0, BN)
    b = tl.load(
        B + k[:, None] * N + panel * BN + n[None, :],
        (panel * BN + n[None, :]) < N,
        other=0,
    )
    tl.store(P + panel * K * BN + k[:, None] * BN + n[None, :], b)


@triton.jit
def _mm_large_panel_cube_kernel(
    A,
    P,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    pid = tl.program_id(0)
    panel = pid % tl.cdiv(N, BN)
    rows = (pid // tl.cdiv(N, BN)) * BM + tl.arange(0, BM)
    cols = tl.arange(0, BN)
    rk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    for step in range(K // BK):
        k = step * BK + rk
        a = tl.load(A + rows[:, None] * K + k[None, :], rows[:, None] < M, other=0)
        b = tl.load(P + panel * K * BN + k[:, None] * BN + cols[None, :])
        al.compile_hint(a, "dot_pad_only_k")
        al.compile_hint(b, "dot_pad_only_k")
        acc = tl.dot(a, b, acc, allow_tf32=False)
    tl.store(
        C + rows[:, None] * N + panel * BN + cols[None, :],
        acc.to(C.dtype.element_ty),
        (rows[:, None] < M) & (panel * BN + cols[None, :] < N),
    )


@triton.jit
def _mm_persistent_smallk_kernel(
    A,
    B,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    for step in range(4):
        pid = tl.program_id(0) + step * 16
        rows = pid // tl.cdiv(N, BLOCK_N) * BLOCK_M + tl.arange(0, BLOCK_M)
        cols = pid % tl.cdiv(N, BLOCK_N) * BLOCK_N + tl.arange(0, BLOCK_N)
        rk = tl.arange(0, BLOCK_K)
        acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        for start in tl.range(tl.cdiv(K, BLOCK_K)):
            kk = start * BLOCK_K + rk
            a = tl.load(
                A + rows[:, None] * K + kk[None, :],
                mask=(rows < M)[:, None] & (kk < K)[None, :],
                other=0.0,
            )
            b = tl.load(
                B + kk[:, None] * N + cols[None, :],
                mask=(kk < K)[:, None] & (cols < N)[None, :],
                other=0.0,
            )
            al.compile_hint(a, "dot_pad_only_k")
            al.compile_hint(b, "dot_pad_only_k")
            acc = tl.dot(a, b, acc, allow_tf32=False)
        tl.store(
            C + rows[:, None] * N + cols[None, :],
            acc.to(C.dtype.element_ty),
            mask=(rows < M)[:, None] & (cols < N)[None, :],
        )


def get_higher_dtype(a, b):
    if a is b:
        return a

    assert a in _ordered_datatypes
    assert b in _ordered_datatypes

    for d in _ordered_datatypes:
        if a is d:
            return b
        if b is d:
            return a


def mm(a, b):
    logger.debug("GEMS_ASCEND MM")
    device = a.device
    # Materialize broadcast operands to avoid a zero-stride Cube lowering failure.
    if a.dtype == b.dtype == torch.bfloat16:
        if 0 in a.stride():
            a = _mm_materialize_broadcast(a)
        if 0 in b.stride():
            b = _mm_materialize_broadcast(b)
    # Materialize strided operands before entering the Cube kernels. This
    # avoids backend-specific failures for column-major test inputs.
    if not a.is_contiguous():
        a = a.contiguous()
    if not b.is_contiguous():
        b = b.contiguous()
    # checks constraints
    assert a.shape[1] == b.shape[0], "incompatible dimensions"
    M, K = a.shape
    _, N = b.shape
    # Pack the narrow right operand once so Cube reads contiguous K rows.
    if (
        a.dtype == b.dtype == torch.bfloat16
        and N == 16
        and 2048 <= K <= 4096
        and K % 1024 == 0
        and a.is_contiguous()
        and b.is_contiguous()
    ):
        block_m = 64 if M >= 4096 else 16
        c = torch.empty((M, N), device=device, dtype=a.dtype)
        packed_b = torch.empty((N, K), device=device, dtype=b.dtype)
        with torch_device_fn.device(device), _blockified():
            _mm_pack_narrow_b_kernel[(triton.cdiv(K, 512),)](
                b,
                packed_b,
                N,
                K,
                512,
                16,
                multibuffer=False,
            )
            _mm_packed_narrow_kernel[(triton.cdiv(M, block_m),)](
                a,
                packed_b,
                c,
                M,
                N,
                K,
                block_m,
                1024,
                multibuffer=True,
                unit_flag=True,
                sync_solver=False,
            )
        return c
    # Deinterleave four BF16 columns into contiguous K vectors for Cube.
    if (
        a.dtype == b.dtype == torch.bfloat16
        and N == 4
        and 8192 < K <= 16384
        and K % 512 == 0
        and a.is_contiguous()
        and b.is_contiguous()
    ):
        c = torch.empty((M, N), device=device, dtype=a.dtype)
        packed_b = torch.empty((4, K), device=device, dtype=b.dtype)
        with torch_device_fn.device(device), _blockified():
            _mm_pack_narrow4_split_kernel[(K // 512,)](
                b,
                packed_b,
                K,
                512,
                multibuffer=False,
            )
            _mm_packed_narrow_kernel[(triton.cdiv(M, 16),)](
                a,
                packed_b,
                c,
                M,
                N,
                K,
                16,
                512,
                multibuffer=True,
                unit_flag=True,
                sync_solver=False,
            )
        return c
    # For the very wide BF16 layout, pack B into contiguous N panels so that
    # Cube reads short contiguous rows instead of the original K-strided rows.
    if (
        a.dtype == b.dtype == torch.bfloat16
        and N >= 131072
        and N % 256 == 0
        and 1024 < K <= 2048
        and K % 128 == 0
        and 0 < M <= 128
        and a.is_contiguous()
        and b.is_contiguous()
    ):
        block_m = 16 if M <= 16 else 128
        block_n = 256
        c = torch.empty((M, N), device=device, dtype=a.dtype)
        packed_b = torch.empty((N // block_n, K, block_n), device=device, dtype=b.dtype)
        with torch_device_fn.device(device), _blockified():
            _mm_pack_wide_panels_kernel[(N // block_n * (K // 64),)](
                b, packed_b, N, K, block_n, 64, multibuffer=False
            )
            _mm_wide_panel_cube_kernel[(triton.cdiv(M, block_m) * (N // block_n),)](
                a,
                packed_b,
                c,
                M,
                N,
                K,
                block_m,
                block_n,
                128,
                multibuffer=True,
                unit_flag=True,
                sync_solver=False,
            )
        return c
    # Panel families: tall/narrow outputs and wide outputs with a large
    # reduction. K alignment is required by the unmasked panel kernels.
    if (
        a.dtype == b.dtype == torch.bfloat16
        and K % 256 == 0
        and (
            (15360 < M <= 16384 and 2048 < N <= 2304 and 4096 < K <= 8192)
            or (1024 < M <= 2048 and N >= 131072 and 1024 < K <= 2048)
            or (
                1024 < M <= 2048
                and 8192 < N <= 32768
                and N % 256 != 0
                and 2048 < K <= 8192
            )
        )
        and a.is_contiguous()
        and b.is_contiguous()
    ):
        c = torch.empty((M, N), device=device, dtype=a.dtype)
        packed_b = torch.empty(
            (triton.cdiv(N, 256), K, 256), device=device, dtype=b.dtype
        )
        with torch_device_fn.device(device), _blockified():
            _mm_pack_large_panels_kernel[(triton.cdiv(N, 256) * (K // 64),)](
                b,
                packed_b,
                N,
                K,
                256,
                64,
                multibuffer=False,
            )
            _mm_large_panel_cube_kernel[(triton.cdiv(M, 128) * triton.cdiv(N, 256),)](
                a,
                packed_b,
                c,
                M,
                N,
                K,
                128,
                256,
                256,
                multibuffer=True,
                unit_flag=True,
                sync_solver=False,
            )
        return c
    # Compatibility exceptions: nearby shapes use the generic path in the
    # established workload. Keep their dispatch until wider ranges are measured.
    # Four M tiles per program keep this 64-tile workload balanced over 16 cores.
    if (
        a.dtype == b.dtype == torch.bfloat16
        and (M, N, K) == (480, 4096, 128)
        and a.is_contiguous()
        and b.is_contiguous()
    ):
        c = torch.empty((M, N), device=device, dtype=a.dtype)
        with torch_device_fn.device(device), _blockified():
            _mm_persistent_smallk_kernel[(16,)](
                a,
                b,
                c,
                M,
                N,
                K,
                128,
                256,
                128,
                multibuffer=True,
                unit_flag=True,
                sync_solver=False,
            )
        return c
    # Wider N tiles reduce program count and K-loop overhead on these layouts.
    if (
        a.dtype == b.dtype == torch.bfloat16
        and a.is_contiguous()
        and b.is_contiguous()
        and (
            (6144 < N <= 7168 and K == 256 and M > 128)
            or (M, N, K)
            in (
                (200, 7168, 2304),
                (13422, 256, 2048),
                (480, 4608, 7168),
            )
        )
    ):
        c = torch.empty((M, N), device=device, dtype=a.dtype)
        with torch_device_fn.device(device), _blockified():
            _mm_rectangular_kernel[(triton.cdiv(M, 128) * triton.cdiv(N, 256),)](
                a,
                b,
                c,
                M,
                N,
                K,
                128,
                256,
                min(256, K),
                multibuffer=True,
                unit_flag=True,
                sync_solver=False,
            )
        return c
    # Explicit tiles for narrow outputs and the remaining slow rectangular layouts.
    if (
        a.dtype == b.dtype == torch.bfloat16
        and a.is_contiguous()
        and b.is_contiguous()
        and (M, N, K)
        in (
            (1, 64, 4096),
            (6, 64, 4096),
            (16, 64, 2048),
            (4434, 2048, 1024),
            (1352, 3072, 2048),
            (80, 12288, 2048),
            (368, 3072, 2048),
        )
    ):
        bm = 16 if M <= 16 else 128
        bn = 64 if N == 64 else 256
        c = torch.empty((M, N), device=device, dtype=a.dtype)
        with torch_device_fn.device(device), _blockified():
            _mm_rectangular_kernel[(triton.cdiv(M, bm) * triton.cdiv(N, bn),)](
                a,
                b,
                c,
                M,
                N,
                K,
                bm,
                bn,
                256,
                multibuffer=True,
                unit_flag=True,
                sync_solver=False,
            )
        return c
    # allocates output
    c_dtype = get_higher_dtype(a.dtype, b.dtype)
    c = torch.empty((M, N), device=device, dtype=c_dtype)
    dot_out_dtype = tl.float32

    # Keep tiny reductions off the autotuned Cube path. Some Ascend
    # toolchains select an invalid tile for these partial shapes.
    if M <= 16 and N <= 16 and K < 128:
        if not a.is_contiguous():
            a = a.contiguous()
        if not b.is_contiguous():
            b = b.contiguous()
        block_k = min(128, triton.next_power_of_2(K))
        with torch_device_fn.device(device), _blockified():
            _mm_rectangular_kernel[(1,)](
                a,
                b,
                c,
                M,
                N,
                K,
                16,
                16,
                block_k,
                multibuffer=True,
                unit_flag=True,
                sync_solver=False,
            )
        return c

    # Keep all short reductions off the autotuned Cube path. Some Ascend
    # toolchains select an invalid tile for partial K tiles.
    if K < 128:
        block_m = 16 if M <= 16 else 128
        block_n = 32 if N <= 32 else (128 if N <= 128 else 256)
        with torch_device_fn.device(device), _blockified():
            _mm_rectangular_kernel[
                (triton.cdiv(M, block_m) * triton.cdiv(N, block_n),)
            ](
                a,
                b,
                c,
                M,
                N,
                K,
                block_m,
                block_n,
                triton.next_power_of_2(K),
                multibuffer=True,
                unit_flag=True,
                sync_solver=False,
            )
        return c

    if (
        a.dtype == b.dtype
        and a.dtype in (torch.float16, torch.bfloat16)
        and a.is_contiguous()
        and b.is_contiguous()
        and K >= 128
    ):
        if N == 1:
            with torch_device_fn.device(device):
                _mm_narrow_n_kernel[(triton.cdiv(M, 4),)](
                    a,
                    b,
                    c,
                    M,
                    N,
                    K,
                    BLOCK_OUT=4,
                    BLOCK_K=min(2048, triton.next_power_of_2(K)),
                )
            return c
        if M <= 32 and N >= 16384:
            block_m = 16 if M <= 16 else 32
            block_n = 256
            with torch_device_fn.device(device), _blockified():
                _mm_rectangular_kernel[
                    (triton.cdiv(M, block_m) * triton.cdiv(N, block_n),)
                ](
                    a,
                    b,
                    c,
                    M,
                    N,
                    K,
                    BLOCK_M=block_m,
                    BLOCK_N=block_n,
                    BLOCK_K=256,
                    multibuffer=True,
                    unit_flag=True,
                    sync_solver=False,
                )
            return c

    # launch kernel
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
        META.get("SPLIT_K", 1),
    )
    with torch_device_fn.device(a.device), _blockified():
        mm_kernel[grid](
            a,
            b,
            c,
            M,
            N,
            K,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            c.stride(0),
            c.stride(1),
            dot_out_dtype=dot_out_dtype,
            GROUP_M=1,
            multibuffer=True,
            unit_flag=True,
            sync_solver=False,
        )
    return c


def mm_out(a, b, *, out):
    logger.debug("GEMS_ASCEND MM_OUT")
    if not a.is_contiguous():
        a = a.contiguous()
    if not b.is_contiguous():
        b = b.contiguous()
    assert a.shape[1] == b.shape[0], "incompatible dimensions"
    M, K = a.shape
    _, N = b.shape
    dot_out_dtype = tl.float32
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
        META.get("SPLIT_K", 1),
    )
    with torch_device_fn.device(a.device), _blockified():
        mm_kernel[grid](
            a,
            b,
            out,
            M,
            N,
            K,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            out.stride(0),
            out.stride(1),
            dot_out_dtype=dot_out_dtype,
            GROUP_M=8,
        )
    return out
