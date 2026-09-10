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
import os
from collections import OrderedDict

import torch
import triton
import triton.language as tl

from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)

MM_GENERIC_CONFIG_TABLE = (
    # Decode-like long vocab projection prefers narrower N tiles.
    {"m_max": 1, "n_min": 65536, "k_min": 0, "config": (4, 16, 8)},
    # Batched decode/prefill small-M cases with bf16-direct inputs.
    # BM=4, BN=8 is optimal for M=2-4 (1.08-1.19x vs native bf16 on ARM).
    {"m_max": 4, "n_min": 2048, "k_min": 0, "config": (4, 8, 8)},
    # Prefill with large K: use larger BLOCK_K to reduce loop iterations.
    {"m_max": 8, "n_min": 0, "k_min": 2048, "config": (8, 8, 32)},
    {"m_max": 8, "n_min": 2048, "k_min": 0, "config": (8, 8, 8)},
    {"m_max": 8, "n_min": 0, "k_min": 0, "config": (8, 8, 8)},
    # Prefill M>8: (64,32,32) benchmarked as best on CIX P1 (2026-03-07).
    # Triton BF16 prefill is still ~3x slower than ATen BFMMLA — fundamental
    # limit of Triton not emitting BFMMLA for tl.dot(bf16,bf16). Larger tiles
    # reduce overhead vs (8,8,8) default but cannot close the BFMMLA gap.
    {"m_max": None, "n_min": 0, "k_min": 0, "config": (64, 32, 32)},
)

MM_M1_CONFIG_TABLE = (
    # Keep very large vocab projection on the generic kernel.
    {"n_min": 65536, "k_min": 0, "config": None},
    # Qwen3-4B gate/up (N=9728, K=2560): BN=64 BK=16 is 9% faster.
    # K≥2560 threshold avoids regressing 1.7B (K=2048) shapes.
    {"n_min": 4096, "k_min": 2560, "config": (64, 16)},
    {"n_min": 2048, "k_min": 0, "config": (32, 8)},
    # Small N (e.g. k/v_proj N=128): use smaller BLOCK_N for better efficiency.
    {"n_min": 256, "k_min": 3072, "config": (128, 8)},
    {"n_min": 256, "k_min": 2048, "config": (32, 16)},
    {"n_min": 256, "k_min": 0, "config": (64, 8)},
    # N < 256: skip M1 fastpath, fall through to generic kernel.
)

MM_M1_TRANSPOSED_CONFIG_TABLE = (
    # Large vocab projection (lm_head N~=152k): BN=2 for fine-grained OMP
    # load balancing; BK=64 fills a full 64-byte cache line per K-step.
    # Tuned on CIX P1 aarch64 (2026-03-04): 30ms vs ATen 65ms (2.17x faster).
    {"n_min": 65536, "k_min": 0, "k_max": 1536, "config": (2, 64)},
    {"n_min": 2048, "k_min": 0, "k_max": 1536, "config": (4, 64)},
    {"n_min": 0, "k_min": 2048, "config": (4, 64)},
    {"n_min": 0, "k_min": 0, "config": (4, 64)},
)

_MM_PREPACK_CACHE = OrderedDict()
_MM_PREPACK_CACHE_BYTES = 0
_MM_FP32_CAST_CACHE = OrderedDict()
_MM_FP32_CAST_CACHE_BYTES = 0


@triton.jit
def mm_kernel(
    A,
    B,
    C,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    dot_out_dtype: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    SPLIT_K: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    # matrix multiplication
    pid = tle.program_id(0)
    pid_z = tle.program_id(1)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)
    # do matrix multiplication
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    rk = pid_z * BLOCK_K + tl.arange(0, BLOCK_K)
    # pointers
    A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
    B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=dot_out_dtype)
    for k in range(0, tl.cdiv(K, BLOCK_K * SPLIT_K)):
        if EVEN_K:
            a = tl.load(A)
            b = tl.load(B)
        else:
            k_remaining = K - k * (BLOCK_K * SPLIT_K)
            _0 = tl.zeros((1, 1), dtype=C.dtype.element_ty)
            a = tl.load(A, mask=rk[None, :] < k_remaining, other=_0)
            b = tl.load(B, mask=rk[:, None] < k_remaining, other=_0)
        if a.dtype != b.dtype:
            a = a.to(C.dtype.element_ty)
            b = b.to(C.dtype.element_ty)
        acc += tl.dot(a, b, out_dtype=dot_out_dtype, allow_tf32=False)
        A += BLOCK_K * SPLIT_K * stride_ak
        B += BLOCK_K * SPLIT_K * stride_bk
    acc = acc.to(C.dtype.element_ty)
    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    C = C + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    mask = (rm < M)[:, None] & (rn < N)[None, :]
    # handles write-back with reduction-splitting
    if SPLIT_K == 1:
        tl.store(C, acc, mask=mask)
    else:
        tl.atomic_add(C, acc, mask=mask)


@triton.jit
def mm_m1_kernel(
    A,
    B,
    C,
    N,
    K,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cn,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    pid_n = tle.program_id(0)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)

    a_ptr = A + rk * stride_ak
    b_ptr = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        if EVEN_K:
            a = tl.load(a_ptr)
            b = tl.load(b_ptr)
        else:
            k_remaining = K - k * BLOCK_K
            a = tl.load(a_ptr, mask=rk < k_remaining, other=0.0)
            b = tl.load(
                b_ptr,
                mask=(rk[:, None] < k_remaining) & (rn[None, :] < N),
                other=0.0,
            )

        if a.dtype != b.dtype:
            a = a.to(C.dtype.element_ty)
            b = b.to(C.dtype.element_ty)

        acc += tl.sum(b * a[:, None], axis=0)
        a_ptr += BLOCK_K * stride_ak
        b_ptr += BLOCK_K * stride_bk

    c_ptr = C + rn * stride_cn
    tl.store(c_ptr, acc.to(C.dtype.element_ty), mask=rn < N)


@triton.jit
def mm_m1_transposed_rhs_kernel(
    A,
    B,
    C,
    N,
    K,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cn,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    pid_n = tle.program_id(0)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)

    a_ptr = A + rk * stride_ak
    # For transposed RHS views (stride_bk == 1), load [BLOCK_N, BLOCK_K]
    # so the K dimension is contiguous in memory.
    bt_ptr = B + rn[:, None] * stride_bn + rk[None, :] * stride_bk
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        if EVEN_K:
            a = tl.load(a_ptr)
            bt = tl.load(bt_ptr, mask=rn[:, None] < N, other=0.0)
        else:
            k_remaining = K - k * BLOCK_K
            a = tl.load(a_ptr, mask=rk < k_remaining, other=0.0)
            bt = tl.load(
                bt_ptr,
                mask=(rn[:, None] < N) & (rk[None, :] < k_remaining),
                other=0.0,
            )

        a_fp = a.to(tl.float32)
        bt_fp = bt.to(tl.float32)
        acc += tl.sum(bt_fp * a_fp[None, :], axis=1)
        a_ptr += BLOCK_K * stride_ak
        bt_ptr += BLOCK_K * stride_bk

    c_ptr = C + rn * stride_cn
    tl.store(c_ptr, acc.to(C.dtype.element_ty), mask=rn < N)


_ordered_datatypes = [torch.float16, torch.bfloat16, torch.float32]


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


def _match_mnk_rule(M, N, K, rule):
    m_max = rule.get("m_max")
    n_min = rule.get("n_min", 0)
    k_min = rule.get("k_min", 0)
    if m_max is not None and M > m_max:
        return False
    if N < n_min:
        return False
    if K < k_min:
        return False
    return True


def _select_mm_config(M, N, K):
    for rule in MM_GENERIC_CONFIG_TABLE:
        if _match_mnk_rule(M, N, K, rule):
            return rule["config"]
    return 8, 8, 8


def _select_mm_m1_config(N, K):
    for rule in MM_M1_CONFIG_TABLE:
        if N >= rule.get("n_min", 0) and K >= rule.get("k_min", 0):
            return rule["config"]
    # No matching rule (e.g. N < 256): skip M1 fastpath
    return None


def _select_mm_m1_transposed_config(N, K):
    for rule in MM_M1_TRANSPOSED_CONFIG_TABLE:
        k_max = rule.get("k_max")
        if (
            N >= rule.get("n_min", 0)
            and K >= rule.get("k_min", 0)
            and (k_max is None or K <= k_max)
        ):
            return rule["config"]
    return 64, 8


def _m1_fastpath_enabled():
    return os.getenv("FLAGGEMS_ARM_M1_FASTPATH", "1").lower() in ("1", "true", "on")


def _m1_transposed_fastpath_enabled():
    return os.getenv("FLAGGEMS_ARM_M1_TRANSPOSED_FASTPATH", "1").lower() in (
        "1",
        "true",
        "on",
    )


def _use_m1_transposed_fastpath_shape(N, K):
    # Tiny matrices can hit unstable LLVM lowering on ARM cpu backend for this
    # specialized kernel; keep generic path for those shapes.
    return N >= 256 and K >= 256


def _mm_prepack_enabled():
    return os.getenv("FLAGGEMS_ARM_MM_PREPACK", "0").lower() in ("1", "true", "on")


def _get_env_int(name, default):
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _tensor_nbytes(t):
    return int(t.numel()) * int(t.element_size())


def _is_rhs_transposed_layout(rhs):
    if rhs.ndim != 2:
        return False
    # Typical weight.t() view: stride(0) == 1, stride(1) == K.
    return rhs.stride(0) == 1 and rhs.stride(1) >= rhs.shape[0]


def _prepack_key(rhs):
    return (
        int(rhs.data_ptr()),
        tuple(rhs.shape),
        tuple(rhs.stride()),
        str(rhs.dtype),
        str(rhs.device),
    )


def _maybe_get_prepacked_rhs(rhs):
    global _MM_PREPACK_CACHE_BYTES
    if not _mm_prepack_enabled():
        return None

    max_bytes = max(_get_env_int("FLAGGEMS_ARM_MM_PREPACK_MAX_BYTES", 0), 0)
    if max_bytes <= 0:
        return None

    max_tensor_bytes = max(
        _get_env_int("FLAGGEMS_ARM_MM_PREPACK_MAX_TENSOR_BYTES", 8 * 1024 * 1024), 0
    )
    rhs_bytes = _tensor_nbytes(rhs)
    if max_tensor_bytes > 0 and rhs_bytes > max_tensor_bytes:
        return None
    if rhs_bytes > max_bytes:
        return None

    key = _prepack_key(rhs)
    packed = _MM_PREPACK_CACHE.get(key)
    if packed is not None:
        _MM_PREPACK_CACHE.move_to_end(key)
        return packed

    packed = rhs.contiguous()
    packed_bytes = _tensor_nbytes(packed)
    max_entries = max(_get_env_int("FLAGGEMS_ARM_MM_PREPACK_MAX_ENTRIES", 32), 1)
    while _MM_PREPACK_CACHE and (
        _MM_PREPACK_CACHE_BYTES + packed_bytes > max_bytes
        or len(_MM_PREPACK_CACHE) >= max_entries
    ):
        _, evicted = _MM_PREPACK_CACHE.popitem(last=False)
        _MM_PREPACK_CACHE_BYTES -= _tensor_nbytes(evicted)

    if packed_bytes > max_bytes:
        return None

    _MM_PREPACK_CACHE[key] = packed
    _MM_PREPACK_CACHE_BYTES += packed_bytes
    return packed


def _mm_fp32_cast_cache_enabled():
    return os.getenv("FLAGGEMS_ARM_MM_FP32_CAST_CACHE", "1").lower() in (
        "1",
        "true",
        "on",
    )


def _fp32_cast_key(t):
    return (
        int(t.data_ptr()),
        tuple(t.shape),
        tuple(t.stride()),
        int(getattr(t, "_version", 0)),
        str(t.dtype),
        str(t.device),
    )


def _maybe_get_cached_fp32(t):
    global _MM_FP32_CAST_CACHE_BYTES
    if not _mm_fp32_cast_cache_enabled():
        return t.to(torch.float32)
    if t.dtype is not torch.bfloat16:
        return t.to(torch.float32)
    if t.requires_grad:
        return t.to(torch.float32)

    min_numel = max(_get_env_int("FLAGGEMS_ARM_MM_FP32_CAST_MIN_NUMEL", 4096), 0)
    if t.numel() < min_numel:
        return t.to(torch.float32)

    max_bytes = max(_get_env_int("FLAGGEMS_ARM_MM_FP32_CAST_MAX_BYTES", 2**31), 0)
    if max_bytes <= 0:
        return t.to(torch.float32)

    key = _fp32_cast_key(t)
    cached = _MM_FP32_CAST_CACHE.get(key)
    if cached is not None:
        _MM_FP32_CAST_CACHE.move_to_end(key)
        return cached

    fp32_t = t.to(torch.float32)
    fp32_bytes = _tensor_nbytes(fp32_t)
    max_tensor_bytes = max(
        _get_env_int("FLAGGEMS_ARM_MM_FP32_CAST_MAX_TENSOR_BYTES", 2**30), 0
    )
    if (
        max_tensor_bytes > 0 and fp32_bytes > max_tensor_bytes
    ) or fp32_bytes > max_bytes:
        return fp32_t

    max_entries = max(_get_env_int("FLAGGEMS_ARM_MM_FP32_CAST_MAX_ENTRIES", 64), 1)
    while _MM_FP32_CAST_CACHE and (
        _MM_FP32_CAST_CACHE_BYTES + fp32_bytes > max_bytes
        or len(_MM_FP32_CAST_CACHE) >= max_entries
    ):
        _, evicted = _MM_FP32_CAST_CACHE.popitem(last=False)
        _MM_FP32_CAST_CACHE_BYTES -= _tensor_nbytes(evicted)

    if fp32_bytes > max_bytes:
        return fp32_t

    _MM_FP32_CAST_CACHE[key] = fp32_t
    _MM_FP32_CAST_CACHE_BYTES += fp32_bytes
    return fp32_t


def _launch_mm_m1_kernel(a, b, c, N, K):
    m1_cfg = _select_mm_m1_config(N, K)
    if m1_cfg is None:
        return False
    BLOCK_N, BLOCK_K = m1_cfg
    EVEN_K = K % BLOCK_K == 0
    grid = lambda META: (triton.cdiv(N, BLOCK_N),)
    mm_m1_kernel[grid](
        a,
        b,
        c,
        N,
        K,
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(1),
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        EVEN_K=EVEN_K,
    )
    return True


def _launch_mm_m1_transposed_rhs_kernel(a, b, c, N, K):
    cfg = _select_mm_m1_transposed_config(N, K)
    if cfg is None:
        return False
    BLOCK_N, BLOCK_K = cfg
    EVEN_K = K % BLOCK_K == 0
    grid = lambda META: (triton.cdiv(N, BLOCK_N),)
    mm_m1_transposed_rhs_kernel[grid](
        a,
        b,
        c,
        N,
        K,
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(1),
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        EVEN_K=EVEN_K,
    )
    return True


def mm(a, b):
    logger.debug("GEMS_ARM MM")
    device = a.device
    # handle non-contiguous inputs if necessary
    if a.stride(0) > 1 and a.stride(1) > 1:
        a = a.contiguous()
    if b.stride(0) > 1 and b.stride(1) > 1:
        b = b.contiguous()
    # checks constraints
    assert a.shape[1] == b.shape[0], "incompatible dimensions"
    M, K = a.shape
    _, N = b.shape
    # Small-shape fallback: use numpy BLAS for shapes where Triton has excessive
    # overhead (e.g., k/v_proj decode M=1, N=128, K=896).
    if N < 256 and M <= 8 and a.dtype in (torch.float32, torch.float64):
        import numpy as np

        return torch.from_numpy(np.dot(a.detach().numpy(), b.detach().numpy()))
    # allocates output
    c_dtype = get_higher_dtype(a.dtype, b.dtype)
    use_fp32_kernel = a.dtype is torch.bfloat16 or b.dtype is torch.bfloat16
    if M == 1:
        # Keep decode-path tensors in native dtype to avoid expensive full-tensor
        # bf16<->fp32 copies; kernels accumulate in fp32 internally.
        a_kernel = a
        b_kernel = b
        m1_out_fp32 = use_fp32_kernel
        c_kernel = torch.empty(
            (M, N),
            device=device,
            dtype=(torch.float32 if m1_out_fp32 else c_dtype),
        )
        if (
            _m1_transposed_fastpath_enabled()
            and _use_m1_transposed_fastpath_shape(N, K)
            and _is_rhs_transposed_layout(b_kernel)
        ):
            packed_rhs = _maybe_get_prepacked_rhs(b_kernel)
            if packed_rhs is not None and _launch_mm_m1_kernel(
                a_kernel, packed_rhs, c_kernel, N, K
            ):
                return c_kernel.to(c_dtype) if m1_out_fp32 else c_kernel
            if _launch_mm_m1_transposed_rhs_kernel(a_kernel, b_kernel, c_kernel, N, K):
                return c_kernel.to(c_dtype) if m1_out_fp32 else c_kernel
        if _m1_fastpath_enabled() and _launch_mm_m1_kernel(
            a_kernel, b_kernel, c_kernel, N, K
        ):
            return c_kernel.to(c_dtype) if m1_out_fp32 else c_kernel

    # M>1 BF16: fallback to ATen native mm (ARM BFMMLA, 3-5x faster than Triton).
    # Cannot call torch.mm() here (infinite recursion via torch.library override).
    # torch.addmm(beta=0) bypasses aten::mm dispatch and uses ATen BFMMLA directly.
    if M > 1 and use_fp32_kernel:
        return torch.addmm(
            torch.empty(N, device=device, dtype=c_dtype), a, b, beta=0, alpha=1
        )

    # Generic path: for M>1 bf16, pass bf16 inputs directly to the Triton kernel
    # instead of casting to fp32 first. The kernel uses tl.dot(out_dtype=tl.float32)
    # for fp32 accumulation, so bf16 inputs are handled natively. This avoids the
    # expensive full-tensor bf16->fp32 conversion that was 2-4x slower than native.
    if use_fp32_kernel and M > 1:
        a_kernel = a
        b_kernel = b
    else:
        a_kernel = a.to(torch.float32) if use_fp32_kernel else a
        b_kernel = _maybe_get_cached_fp32(b) if use_fp32_kernel else b
    c_kernel = torch.empty(
        (M, N),
        device=device,
        dtype=(torch.float32 if use_fp32_kernel else c_dtype),
    )

    BLOCK_M, BLOCK_N, BLOCK_K = _select_mm_config(M, N, K)
    EVEN_K = K % BLOCK_K == 0
    # launch kernel
    grid = lambda META: (
        triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),
        1,
    )
    mm_kernel[grid](
        a_kernel,
        b_kernel,
        c_kernel,
        M,
        N,
        K,
        a_kernel.stride(0),
        a_kernel.stride(1),
        b_kernel.stride(0),
        b_kernel.stride(1),
        c_kernel.stride(0),
        c_kernel.stride(1),
        dot_out_dtype=tl.float32,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        GROUP_M=8,
        SPLIT_K=1,
        EVEN_K=EVEN_K,
    )
    return c_kernel.to(c_dtype) if use_fp32_kernel else c_kernel


def mm_out(a, b, *, out):
    logger.debug("GEMS_ARM MM_OUT")
    if a.stride(0) > 1 and a.stride(1) > 1:
        a = a.contiguous()
    if b.stride(0) > 1 and b.stride(1) > 1:
        b = b.contiguous()

    assert a.shape[1] == b.shape[0], "incompatible dimensions"
    M, K = a.shape
    _, N = b.shape
    assert out is not None, "out tensor is required"
    assert out.shape == (M, N), "incompatible out shape"
    use_fp32_kernel = a.dtype is torch.bfloat16 or b.dtype is torch.bfloat16
    if M == 1:
        a_kernel = a
        b_kernel = b
        m1_out_fp32 = use_fp32_kernel
        out_kernel = (
            torch.empty((M, N), device=out.device, dtype=torch.float32)
            if m1_out_fp32
            else out
        )
        if (
            _m1_transposed_fastpath_enabled()
            and _use_m1_transposed_fastpath_shape(N, K)
            and _is_rhs_transposed_layout(b_kernel)
        ):
            packed_rhs = _maybe_get_prepacked_rhs(b_kernel)
            if packed_rhs is not None and _launch_mm_m1_kernel(
                a_kernel, packed_rhs, out_kernel, N, K
            ):
                if m1_out_fp32:
                    out.copy_(out_kernel.to(out.dtype))
                return out
            if _launch_mm_m1_transposed_rhs_kernel(
                a_kernel, b_kernel, out_kernel, N, K
            ):
                if m1_out_fp32:
                    out.copy_(out_kernel.to(out.dtype))
                return out
        if _m1_fastpath_enabled() and _launch_mm_m1_kernel(
            a_kernel, b_kernel, out_kernel, N, K
        ):
            if m1_out_fp32:
                out.copy_(out_kernel.to(out.dtype))
            return out

    # M>1 BF16: fallback to ATen native mm (see mm() for rationale).
    if M > 1 and use_fp32_kernel:
        torch.addmm(
            torch.empty(N, device=out.device, dtype=out.dtype),
            a,
            b,
            beta=0,
            alpha=1,
            out=out,
        )
        return out

    # For M>1 bf16, pass bf16 inputs directly to Triton kernel (see mm() comment).
    if use_fp32_kernel and M > 1:
        a_kernel = a
        b_kernel = b
    else:
        a_kernel = a.to(torch.float32) if use_fp32_kernel else a
        b_kernel = _maybe_get_cached_fp32(b) if use_fp32_kernel else b
    out_kernel = (
        torch.empty((M, N), device=out.device, dtype=torch.float32)
        if use_fp32_kernel
        else out
    )

    BLOCK_M, BLOCK_N, BLOCK_K = _select_mm_config(M, N, K)
    EVEN_K = K % BLOCK_K == 0

    grid = lambda META: (
        triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),
        1,
    )
    mm_kernel[grid](
        a_kernel,
        b_kernel,
        out_kernel,
        M,
        N,
        K,
        a_kernel.stride(0),
        a_kernel.stride(1),
        b_kernel.stride(0),
        b_kernel.stride(1),
        out_kernel.stride(0),
        out_kernel.stride(1),
        dot_out_dtype=tl.float32,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        GROUP_M=8,
        SPLIT_K=1,
        EVEN_K=EVEN_K,
    )
    if use_fp32_kernel:
        out.copy_(out_kernel.to(out.dtype))
    return out
