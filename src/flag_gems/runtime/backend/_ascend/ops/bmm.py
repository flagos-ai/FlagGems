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

import logging

import torch
import triton
import triton.language as tl
import triton.language.extra.cann.extension as extension

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.runtime.backend._ascend import heuristics_config_utils as _hcu
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

from .mm import _blockified

logger = logging.getLogger(__name__)


_BMM_CONFIGS = runtime.get_tuned_config("bmm")


def _prune_bmm_configs(configs, named_args, **kwargs):
    args = {**named_args, **kwargs}
    A = args["A"]
    M, N, K = args["M"], args["N"], args["K"]
    batch = args["BATCH"]
    # Keep one balanced tile for small aligned squares with enough programs
    # to occupy the cores; timing noise otherwise selects padded large tiles.
    square_programs = batch * (M // 128) * (N // 128)
    small_square = (
        A.dtype in (torch.float16, torch.float32, torch.bfloat16)
        and M == N == K
        and 256 <= M <= 512
        and M % 128 == 0
        and 16 <= square_programs <= 64
    )
    short_k_fp16 = (
        A.dtype == torch.float16
        and batch == 1
        and 256 <= M <= 512
        and N >= 4096
        and K == 256
    )
    target = None
    if small_square:
        target = (128, 128, 128)
    elif short_k_fp16:
        target = (128, 256, 256)
    elif A.dtype == torch.bfloat16 and batch == 1 and M >= 256 and K >= 256:
        if N >= 4096:
            target = (128, 256, 256)
        elif N >= 1024 and K >= 4096:
            target = (256, 128, 256)
    if target is None:
        return configs

    selected = [
        config
        for config in configs
        if (
            config.kwargs["TILE_M"],
            config.kwargs["TILE_N"],
            config.kwargs["TILE_K"],
        )
        == target
    ]
    return selected or configs


# avoid
@libentry()
@triton.autotune(
    configs=_BMM_CONFIGS,
    key=["M", "N", "K", "BATCH", "DOT_PAD_ONLY_K"],
    prune_configs_by={"early_config_prune": _prune_bmm_configs},
)
@triton.heuristics(_hcu.HEURISTICS_CONFIGS["bmm"])
@triton.jit
def bmm_kernel(
    A,
    B,
    O,
    M,
    N,
    K,
    BATCH: tl.constexpr,
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
    TILE_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    DOT_PAD_ONLY_K: tl.constexpr,
    DIVISIBLE_M: tl.constexpr,
    DIVISIBLE_N: tl.constexpr,
    DIVISIBLE_K: tl.constexpr,
):
    # batch offsets
    pid_b = ext.program_id(2)
    A += pid_b * M * K
    B += pid_b * K * N
    O += pid_b * M * N

    pidx = ext.program_id(0)
    pidy = ext.program_id(1)
    if GROUP_M == 1:
        pid_m, pid_n = pidx, pidy
    else:
        # reorder CTAs
        gridx = ext.num_programs(0)
        gridy = ext.num_programs(1)
        pid = pidx + pidy * gridx

        num_CTA_per_group = gridy * GROUP_M

        group_id = pid // num_CTA_per_group
        inner_group_id = pid % num_CTA_per_group
        GROUP_SIZE = tl.where(
            (group_id * GROUP_M + GROUP_M) > gridx, gridx % GROUP_M, GROUP_M
        )
        pid_m = group_id * GROUP_M + inner_group_id % GROUP_SIZE
        pid_n = inner_group_id // GROUP_SIZE

    offs_m = pid_m * TILE_M + tl.arange(0, TILE_M)
    offs_n = pid_n * TILE_N + tl.arange(0, TILE_N)
    offs_k = tl.arange(0, TILE_K)

    a_ptrs = A + offs_m[:, None] * K + offs_k[None, :]
    b_ptrs = B + offs_k[:, None] * N + offs_n[None, :]
    o_ptrs = O + offs_m[:, None] * N + offs_n[None, :]

    if not DIVISIBLE_M:
        mask_m = offs_m < M
    if not DIVISIBLE_N:
        mask_n = offs_n < N

    num_iters = tl.cdiv(K, TILE_K)
    o = tl.zeros((TILE_M, TILE_N), dtype=tl.float32)
    for i in range(num_iters):
        if DIVISIBLE_K:
            mask_a = None if DIVISIBLE_M else mask_m[:, None]
            mask_b = None if DIVISIBLE_N else mask_n[None, :]
        else:
            mask_k = offs_k < K - i * TILE_K
            mask_a = (
                mask_k[None, :] if DIVISIBLE_M else mask_m[:, None] & mask_k[None, :]
            )
            mask_b = (
                mask_k[:, None] if DIVISIBLE_N else mask_k[:, None] & mask_n[None, :]
            )
        if DIVISIBLE_K and DIVISIBLE_M:
            a = tl.load(a_ptrs)
        else:
            a = tl.load(a_ptrs, mask=mask_a, other=0.0)
        if DIVISIBLE_K and DIVISIBLE_N:
            b = tl.load(b_ptrs)
        else:
            b = tl.load(b_ptrs, mask=mask_b, other=0.0)

        a_ptrs += TILE_K
        b_ptrs += TILE_K * N

        if DOT_PAD_ONLY_K:
            extension.compile_hint(a, "dot_pad_only_k")
            extension.compile_hint(b, "dot_pad_only_k")
        o += tl.dot(a, b, allow_tf32=False)

    if DIVISIBLE_M and DIVISIBLE_N:
        mask_c = None
    elif DIVISIBLE_M:
        mask_c = mask_n[None, :]
    elif DIVISIBLE_N:
        mask_c = mask_m[:, None]
    else:
        mask_c = mask_m[:, None] & mask_n[None, :]
    if DIVISIBLE_M and DIVISIBLE_N:
        tl.store(o_ptrs, o)
    else:
        tl.store(o_ptrs, o, mask_c)


def bmm(A, B):
    logger.debug("GEMS_ASCEND BMM")
    batch, M, K = A.shape
    _, _, N = B.shape
    A = A.contiguous()
    out = torch.empty((batch, M, N), dtype=A.dtype, device=A.device)
    if _use_nt_gemm(A, B):
        return _launch_nt_gemm(A, B, out)
    if _use_narrow_gemv(A, B):
        return _launch_narrow_gemv(A, B, out)
    if _use_narrow_pack(A, B):
        return _launch_narrow_pack(A, B, out)
    tile = _select_rectangular_tile(A, B)
    if tile is not None:
        return _launch_tiled_gemm(A, B, out, tile)
    B = B.contiguous()
    dot_pad_only_k = (
        A.dtype == torch.bfloat16
        and batch == 1
        and M >= 256
        and N >= 128
        and N % 16 == 0
        and K >= 128
    )

    grid_fn = lambda meta: (
        triton.cdiv(meta["M"], meta["TILE_M"]),
        triton.cdiv(meta["N"], meta["TILE_N"]),
        batch,
    )

    with torch_device_fn.device(A.device), _blockified():
        bmm_kernel[grid_fn](
            A,
            B,
            out,
            M,
            N,
            K,
            BATCH=batch,
            DOT_PAD_ONLY_K=dot_pad_only_k,
            enable_auto_blockify=True,
        )
    return out


# Shared Ascend matmul fast paths for addmm, bmm, and baddbmm.
def _use_nt_gemm(a, b):
    m, k = a.shape[-2:]
    n = b.shape[-1]
    return (
        a.dtype == b.dtype
        and a.dtype in (torch.float16, torch.bfloat16)
        and a.is_contiguous()
        and not b.is_contiguous()
        and b.transpose(-2, -1).is_contiguous()
        and a.shape[:-2] == b.shape[:-2]
        and (a.ndim == 2 or a.shape[0] > 0)
        and b.shape[-2] == k
        and 0 < m <= 512
        and 4 <= n <= 8192
        and n % 4 == 0
        and k >= 256
        and k % 256 == 0
    )


def _select_rectangular_tile(a, b, *, rectangular_nn=True):
    m, k = a.shape[-2:]
    n = b.shape[-1]
    nt = not b.is_contiguous()
    if (
        a.dtype != b.dtype
        or a.dtype not in (torch.float16, torch.bfloat16)
        or not a.is_contiguous()
        or (nt and not b.transpose(-2, -1).is_contiguous())
        or a.shape[:-2] != b.shape[:-2]
        or (a.ndim == 3 and a.shape[0] == 0)
        or b.shape[-2] != k
    ):
        return None
    if nt and k == 64 and 64 <= m <= 256 and 64 <= n <= 256:
        return 64, 64, 64, 1, False
    if 32 <= m <= 128 and 64 <= n <= 512 and 1024 <= k <= 4096 and (k % 16 or n % 16):
        # Larger batches already supply enough work to use a taller NN tile.
        bm = (
            64
            if rectangular_nn and not nt and a.ndim == 3 and a.shape[0] >= 8 and m > 64
            else 32
        )
        return bm, 64, 256, 1, False
    if nt:
        return None
    if 0 < m <= 128 and 32 <= n <= 128 and n % 32 == 0 and k in (2048, 4096):
        return 16, 64, 256, 4, True
    if rectangular_nn and 16 <= m <= 128 and 128 < n <= 512 and n % 16 == 0:
        if 2048 <= k <= 8192 and k % 512 == 0:
            return 32, 64, 512, 1, True
    if rectangular_nn and 64 <= m <= 128 and 8192 <= n <= 32768 and n % 256 == 0:
        if 4096 <= k <= 8192 and k % 128 == 0:
            return 64, 256, 128, 1, True
    return None


@triton.jit
def _tiled_gemm_kernel(
    A,
    B,
    Bias,
    Out,
    Scratch,
    BATCH: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    TRANSPOSED: tl.constexpr,
    WITH_BIAS: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    SPLIT: tl.constexpr,
    HINT: tl.constexpr,
):
    pid = tl.program_id(0)
    split = pid % SPLIT
    tile = pid // SPLIT
    batch = tl.program_id(1)
    m = (tile // tl.cdiv(N, BN)) * BM + tl.arange(0, BM)
    n = (tile % tl.cdiv(N, BN)) * BN + tl.arange(0, BN)
    k0 = tl.arange(0, BK)
    a_base = A + batch * M * K
    b_base = B + batch * K * N
    acc = tl.zeros((BM, BN), tl.float32)
    for start in range(tl.cdiv(K, BK * SPLIT)):
        k = (start * SPLIT + split) * BK + k0
        valid_k = k < K if K % (BK * SPLIT) else tl.full((BK,), True, tl.int1)
        a = tl.load(
            a_base + m[:, None] * K + k[None, :], (m[:, None] < M) & valid_k[None, :], 0
        )
        if TRANSPOSED:
            # The transposed view stores consecutive K elements for each column.
            b = tl.load(
                b_base + n[:, None] * K + k[None, :],
                (n[:, None] < N) & valid_k[None, :],
                0,
            ).T
        else:
            b = tl.load(
                b_base + k[:, None] * N + n[None, :],
                valid_k[:, None] & (n[None, :] < N),
                0,
            )
        if HINT:
            extension.compile_hint(a, "dot_pad_only_k")
            extension.compile_hint(b, "dot_pad_only_k")
        acc = tl.dot(a, b, acc, allow_tf32=False)
    offset = batch * M * N + m[:, None] * N + n[None, :]
    valid = (m[:, None] < M) & (n[None, :] < N)
    if SPLIT > 1:
        tl.store(Scratch + split * BATCH * M * N + offset, acc, valid)
    else:
        if WITH_BIAS:
            acc += tl.load(Bias + offset, valid, 0).to(tl.float32)
        tl.store(Out + offset, acc, valid)


@triton.jit
def _finish_split_gemm(
    Scratch,
    Bias,
    Out,
    TOTAL: tl.constexpr,
    SPLIT: tl.constexpr,
    BLOCK: tl.constexpr,
    WITH_BIAS: tl.constexpr,
):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    split = tl.arange(0, SPLIT)
    parts = tl.load(
        Scratch + split[:, None] * TOTAL + offset[None, :],
        offset[None, :] < TOTAL,
        0,
    )
    value = tl.sum(parts, axis=0)
    if WITH_BIAS:
        value += tl.load(Bias + offset, offset < TOTAL, 0).to(tl.float32)
    tl.store(Out + offset, value, offset < TOTAL)


def _launch_tiled_gemm(a, b, out, tile, bias=None):
    bm, bn, bk, split, hint = tile
    m, k = a.shape[-2:]
    n = b.shape[-1]
    batch = a.shape[0] if a.ndim == 3 else 1
    total = batch * m * n
    bias_arg = a if bias is None else bias
    partial = (
        torch.empty((split, total), device=a.device, dtype=torch.float32)
        if split > 1
        else out
    )
    grid = (triton.cdiv(m, bm) * triton.cdiv(n, bn) * split, batch)
    with torch_device_fn.device(a.device):
        with _blockified():
            _tiled_gemm_kernel[grid](
                a,
                b,
                bias_arg,
                out,
                partial,
                batch,
                m,
                n,
                k,
                not b.is_contiguous(),
                bias is not None,
                bm,
                bn,
                bk,
                split,
                hint,
                multibuffer=True,
                unit_flag=True,
                sync_solver=False,
                enable_auto_blockify=True,
            )
        if split > 1:
            _finish_split_gemm[(triton.cdiv(total, 256),)](
                partial, bias_arg, out, total, split, 256, bias is not None
            )
    return out


def _launch_nt_gemm(a, b, out, bias=None, block_k=256):
    m = a.shape[-2]
    n = b.shape[-1]
    if m >= 256 and n >= 128 and m % 128 == 0 and n % 128 == 0:
        bm = bn = 128
    else:
        bm = 16 if m <= 64 else 64
        bn = 16 if n <= 16 else 64
    return _launch_tiled_gemm(a, b, out, (bm, bn, block_k, 1, False), bias)


def _use_narrow_gemv(a, b):
    m, k = a.shape[-2:]
    return (
        a.dtype == b.dtype
        and a.dtype in (torch.float16, torch.bfloat16)
        and b.shape[-2:] == (k, 1)
        and a.shape[:-2] == b.shape[:-2]
        and (a.ndim == 2 or a.shape[0] > 0)
        and a.is_contiguous()
        and b.is_contiguous()
        and m >= 256
        and m % 8 == 0
        and k >= 1024
        and k % 1024 == 0
    )


@triton.jit
def _narrow_gemv_kernel(
    A,
    B,
    Bias,
    Out,
    M: tl.constexpr,
    K: tl.constexpr,
    WITH_BIAS: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    batch = tl.program_id(1)
    rk = tl.arange(0, BK)
    a_base = A + (batch * M + rows[:, None]) * K
    b_base = B + batch * K
    acc = tl.zeros((BM, BK), tl.float32)
    for start in range(K // BK):
        k = start * BK + rk
        a = tl.load(a_base + k[None, :]).to(tl.float32)
        b = tl.load(b_base + k).to(tl.float32)
        acc += a * b[None, :]
    result = tl.sum(acc, axis=1)
    offset = batch * M + rows
    if WITH_BIAS:
        result += tl.load(Bias + offset).to(tl.float32)
    tl.store(Out + offset, result)


def _launch_narrow_gemv(a, b, out, bias=None):
    m, k = a.shape[-2:]
    batch = a.shape[0] if a.ndim == 3 else 1
    with torch_device_fn.device(a.device):
        _narrow_gemv_kernel[(m // 8, batch)](
            a, b, a if bias is None else bias, out, m, k, bias is not None, 8, 1024
        )
    return out


def _use_narrow_pack(a, b):
    m, k = a.shape[-2:]
    n = b.shape[-1]
    return (
        a.dtype == b.dtype
        and a.dtype in (torch.float16, torch.bfloat16)
        and a.is_contiguous()
        and b.is_contiguous()
        and a.shape[:-2] == b.shape[:-2]
        and (a.ndim == 2 or a.shape[0] > 0)
        and b.shape[-2] == k
        and 0 < m <= 64
        and k % 512 == 0
        and ((n == 4 and 8192 <= k <= 16384) or (n == 16 and 2048 <= k <= 4096))
    )


@triton.jit
def _pack_narrow_rhs(B, Packed, N: tl.constexpr, K: tl.constexpr, BK: tl.constexpr):
    start = tl.program_id(0) * BK
    batch = tl.program_id(1)
    B += batch * N * K
    Packed += batch * N * K
    k = start + tl.arange(0, BK)
    if N == 4:
        # Deinterleave the four columns after one contiguous load.
        b = tl.load(B + start * 4 + tl.arange(0, BK * 4)).reshape((BK, 2, 2))
        b02, b13 = tl.split(b)
        b0, b2 = tl.split(b02)
        b1, b3 = tl.split(b13)
        tl.store(Packed + k, b0)
        tl.store(Packed + K + k, b1)
        tl.store(Packed + 2 * K + k, b2)
        tl.store(Packed + 3 * K + k, b3)
    else:
        n = tl.arange(0, N)
        b = tl.load(B + k[:, None] * N + n[None, :])
        tl.store(Packed + n[:, None] * K + k[None, :], b.T)


def _launch_narrow_pack(a, b, out, bias=None):
    k, n = b.shape[-2:]
    batch = b.shape[0] if b.ndim == 3 else 1
    packed = torch.empty((*b.shape[:-2], n, k), device=b.device, dtype=b.dtype)
    with torch_device_fn.device(b.device):
        _pack_narrow_rhs[(k // 512, batch)](b, packed, n, k, 512, multibuffer=False)
    return _launch_nt_gemm(a, packed.transpose(-2, -1), out, bias, block_k=512)
