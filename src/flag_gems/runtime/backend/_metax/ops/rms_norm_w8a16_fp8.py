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

"""MetaX W8A16 RMSNorm.

Activation is 16-bit (FP16/BF16). Weight is grouped FP8 E4M3FN
(``torch.float8_e4m3fn``) plus a per-group scale (default group size 128),
matching the public FlagGems entry and PPU PR #5957.

Eager execution caches a BF16/FP16 dequantized weight while the source
tensor identities and version counters stay unchanged. CUDA Graph capture
and tensors without version counters use the fused dynamic-weight path so
graph replay keeps reading the live weight and scale buffers.

MetaX C550 limits a CTA to 8 warps of 64 threads (512 threads). Native
``float8_e4m3fn`` load/cast is used instead of the PPU E4M3B15 conversion.
"""

import logging
import math
import weakref

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)

_FP8_DTYPE = getattr(torch, "float8_e4m3fn", None)
_MAX_WARPS = 8
_DEQUANT_WEIGHT_CACHE = {}
_LAST_DEQUANT_WEIGHT = None


@triton.jit
def prev_multiple_of(a, b):
    return tl.cdiv(a, b) * b - b


@libentry()
@triton.jit(do_not_specialize=["eps"])
def rms_norm_cached_kernel(
    out_ptr,
    in_ptr,
    weight_ptr,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
    NUM_WARPS: tl.constexpr,
):
    pid = ext.program_id(0)
    tl.static_assert(NUM_WARPS > 0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    x = tl.load(in_ptr + pid * N + cols, mask=mask, other=0.0).to(tl.float32)
    rrms = tl.math.rsqrt(tl.sum(x * x, axis=0) / N + eps)
    weight = tl.load(weight_ptr + cols, mask=mask, other=0.0)
    y = (x * rrms).to(in_ptr.dtype.element_ty) * weight
    tl.store(out_ptr + pid * N + cols, y, mask=mask)


@libentry()
@triton.jit(do_not_specialize=["eps"])
def rms_norm_cached_grouped_kernel(
    out_ptr,
    in_ptr,
    weight_ptr,
    N: tl.constexpr,
    eps,
    GROUP_SIZE: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    NUM_WARPS: tl.constexpr,
):
    pid = ext.program_id(0)
    tl.static_assert(NUM_WARPS > 0)
    groups = tl.arange(0, NUM_GROUPS)
    cols = tl.arange(0, GROUP_SIZE)
    offsets = groups[:, None] * GROUP_SIZE + cols[None, :]
    x = tl.load(in_ptr + pid * N + offsets).to(tl.float32)
    rrms = tl.math.rsqrt(tl.sum(x * x) / N + eps)
    weight = tl.load(weight_ptr + offsets)
    y = (x * rrms).to(in_ptr.dtype.element_ty) * weight
    tl.store(out_ptr + pid * N + offsets, y)


@libentry()
@triton.jit(do_not_specialize=["eps"])
def rms_norm_cached_loop_kernel(
    out_ptr,
    in_ptr,
    weight_ptr,
    N,
    eps,
    TILE_N: tl.constexpr,
    NUM_WARPS: tl.constexpr,
):
    pid = ext.program_id(0)
    tl.static_assert(NUM_WARPS > 0)
    acc = tl.zeros((TILE_N,), dtype=tl.float32)
    num_steps = tl.cdiv(N, TILE_N)
    for step in range(0, num_steps - 1):
        n_offsets = step * TILE_N + tl.arange(0, TILE_N)
        x = tl.load(in_ptr + pid * N + n_offsets).to(tl.float32)
        acc += x * x
    n_offsets = (num_steps - 1) * TILE_N + tl.arange(0, TILE_N)
    mask = n_offsets < N
    x = tl.load(in_ptr + pid * N + n_offsets, mask=mask, other=0.0).to(tl.float32)
    acc += x * x
    rrms = tl.math.rsqrt(tl.sum(acc) / N + eps)

    prev_multiple = prev_multiple_of(N, TILE_N)
    for start_n in range(0, TILE_N, TILE_N):
        n_offsets = (prev_multiple - start_n) + tl.arange(0, TILE_N)
        mask = n_offsets < N
        x_tile = tl.load(
            in_ptr + pid * N + n_offsets,
            mask=mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        weight = tl.load(weight_ptr + n_offsets, mask=mask, other=0.0)
        y = (x_tile * rrms).to(in_ptr.dtype.element_ty) * weight
        tl.store(out_ptr + pid * N + n_offsets, y, mask=mask)
    for start_n in range(TILE_N, N, TILE_N):
        n_offsets = (prev_multiple - start_n) + tl.arange(0, TILE_N)
        x_tile = tl.load(
            in_ptr + pid * N + n_offsets,
            eviction_policy="evict_first",
        ).to(tl.float32)
        weight = tl.load(weight_ptr + n_offsets)
        y = (x_tile * rrms).to(in_ptr.dtype.element_ty) * weight
        tl.store(out_ptr + pid * N + n_offsets, y)


@libentry()
@triton.jit(do_not_specialize=["eps"])
def rms_norm_fp8_kernel(
    out_ptr,
    in_ptr,
    w_ptr,
    scale_ptr,
    N,
    eps,
    GROUP_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUM_WARPS: tl.constexpr,
):
    pid = ext.program_id(0)
    tl.static_assert(NUM_WARPS > 0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    x = tl.load(in_ptr + pid * N + cols, mask=mask, other=0.0).to(tl.float32)
    rrms = tl.math.rsqrt(tl.sum(x * x, axis=0) / N + eps)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    scale = tl.load(scale_ptr + cols // GROUP_SIZE, mask=mask, other=0.0).to(tl.float32)
    weight = (w * scale).to(in_ptr.dtype.element_ty)
    y = (x * rrms).to(in_ptr.dtype.element_ty) * weight
    tl.store(out_ptr + pid * N + cols, y, mask=mask)


@libentry()
@triton.jit(do_not_specialize=["eps"])
def rms_norm_fp8_grouped_kernel(
    out_ptr,
    in_ptr,
    w_ptr,
    scale_ptr,
    N: tl.constexpr,
    eps,
    GROUP_SIZE: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    NUM_WARPS: tl.constexpr,
):
    pid = ext.program_id(0)
    tl.static_assert(NUM_WARPS > 0)
    groups = tl.arange(0, NUM_GROUPS)
    cols = tl.arange(0, GROUP_SIZE)
    offsets = groups[:, None] * GROUP_SIZE + cols[None, :]
    x = tl.load(in_ptr + pid * N + offsets).to(tl.float32)
    rrms = tl.math.rsqrt(tl.sum(x * x) / N + eps)
    w = tl.load(w_ptr + offsets).to(tl.float32)
    scale = tl.load(scale_ptr + groups).to(tl.float32)[:, None]
    weight = (w * scale).to(in_ptr.dtype.element_ty)
    y = (x * rrms).to(in_ptr.dtype.element_ty) * weight
    tl.store(out_ptr + pid * N + offsets, y)


@libentry()
@triton.jit(do_not_specialize=["eps"])
def rms_norm_fp8_loop_kernel(
    out_ptr,
    in_ptr,
    w_ptr,
    scale_ptr,
    N,
    eps,
    GROUP_SIZE: tl.constexpr,
    TILE_N: tl.constexpr,
    NUM_WARPS: tl.constexpr,
):
    pid = ext.program_id(0)
    tl.static_assert(NUM_WARPS > 0)
    acc = tl.zeros((TILE_N,), dtype=tl.float32)
    num_steps = tl.cdiv(N, TILE_N)
    for step in range(0, num_steps - 1):
        n_offsets = step * TILE_N + tl.arange(0, TILE_N)
        x = tl.load(in_ptr + pid * N + n_offsets).to(tl.float32)
        acc += x * x
    n_offsets = (num_steps - 1) * TILE_N + tl.arange(0, TILE_N)
    mask = n_offsets < N
    x = tl.load(in_ptr + pid * N + n_offsets, mask=mask, other=0.0).to(tl.float32)
    acc += x * x
    rrms = tl.math.rsqrt(tl.sum(acc) / N + eps)

    prev_multiple = prev_multiple_of(N, TILE_N)
    for start_n in range(0, TILE_N, TILE_N):
        n_offsets = (prev_multiple - start_n) + tl.arange(0, TILE_N)
        mask = n_offsets < N
        x_tile = tl.load(
            in_ptr + pid * N + n_offsets,
            mask=mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        w = tl.load(w_ptr + n_offsets, mask=mask, other=0.0).to(tl.float32)
        scale = tl.load(scale_ptr + n_offsets // GROUP_SIZE, mask=mask, other=0.0).to(
            tl.float32
        )
        weight = (w * scale).to(in_ptr.dtype.element_ty)
        y = (x_tile * rrms).to(in_ptr.dtype.element_ty) * weight
        tl.store(out_ptr + pid * N + n_offsets, y, mask=mask)
    for start_n in range(TILE_N, N, TILE_N):
        n_offsets = (prev_multiple - start_n) + tl.arange(0, TILE_N)
        x_tile = tl.load(
            in_ptr + pid * N + n_offsets,
            eviction_policy="evict_first",
        ).to(tl.float32)
        w = tl.load(w_ptr + n_offsets).to(tl.float32)
        scale = tl.load(scale_ptr + n_offsets // GROUP_SIZE).to(tl.float32)
        weight = (w * scale).to(in_ptr.dtype.element_ty)
        y = (x_tile * rrms).to(in_ptr.dtype.element_ty) * weight
        tl.store(out_ptr + pid * N + n_offsets, y)


def _clamp_warps(num_warps):
    return max(1, min(int(num_warps), _MAX_WARPS))


def _is_pow2(n):
    return n > 0 and (n & (n - 1)) == 0


def _tile_n(N):
    if N <= 2048:
        return triton.next_power_of_2(N)
    if N <= 8192:
        return 2048
    return 2048


def _simple_warps(M, N):
    if N >= 8192:
        return _clamp_warps(8)
    if M >= 256:
        return _clamp_warps(4)
    return _clamp_warps(8)


def _use_grouped(N, group_size):
    return (
        group_size == 128 and 4096 <= N <= 8192 and N % group_size == 0 and _is_pow2(N)
    )


def _is_capturing():
    try:
        return bool(torch.cuda.is_current_stream_capturing())
    except Exception:
        return False


def _tensor_version(tensor):
    try:
        return tensor._version
    except RuntimeError:
        return None


def _remove_dequant_weight(key, dead_ref):
    global _LAST_DEQUANT_WEIGHT
    entry = _DEQUANT_WEIGHT_CACHE.get(key)
    if entry is not None and (entry[0] is dead_ref or entry[1] is dead_ref):
        _DEQUANT_WEIGHT_CACHE.pop(key, None)
        if _LAST_DEQUANT_WEIGHT is entry:
            _LAST_DEQUANT_WEIGHT = None


def _get_dequant_weight(x, weight_q, weight_scale, group_size):
    global _LAST_DEQUANT_WEIGHT
    weight_version = _tensor_version(weight_q)
    scale_version = _tensor_version(weight_scale)
    if weight_version is None or scale_version is None:
        return None

    entry = _LAST_DEQUANT_WEIGHT
    if (
        entry is not None
        and entry[0]() is weight_q
        and entry[1]() is weight_scale
        and entry[2] == weight_version
        and entry[3] == scale_version
        and entry[4].dtype == x.dtype
        and entry[5] == group_size
    ):
        return entry[4]

    key = (id(weight_q), id(weight_scale), x.dtype, group_size)
    entry = _DEQUANT_WEIGHT_CACHE.get(key)
    if (
        entry is not None
        and entry[0]() is weight_q
        and entry[1]() is weight_scale
        and entry[2] == weight_version
        and entry[3] == scale_version
    ):
        _LAST_DEQUANT_WEIGHT = entry
        return entry[4]

    # Pad a trailing partial group so the broadcast dequant still applies the
    # last scale to the remaining elements, then trim it back off.
    flat_weight = weight_q.float().reshape(-1)
    pad = (-flat_weight.numel()) % group_size
    if pad:
        flat_weight = torch.nn.functional.pad(flat_weight, (0, pad))
    dequant_weight = (
        (flat_weight.reshape(-1, group_size) * weight_scale.float().reshape(-1, 1))
        .reshape(-1)[: weight_q.numel()]
        .to(x.dtype)
    )
    weight_ref = weakref.ref(
        weight_q, lambda dead_ref: _remove_dequant_weight(key, dead_ref)
    )
    scale_ref = weakref.ref(
        weight_scale, lambda dead_ref: _remove_dequant_weight(key, dead_ref)
    )
    entry = (
        weight_ref,
        scale_ref,
        weight_version,
        scale_version,
        dequant_weight,
        group_size,
    )
    _DEQUANT_WEIGHT_CACHE[key] = entry
    _LAST_DEQUANT_WEIGHT = entry
    return dequant_weight


def _peek_cached_weight(x, weight_q, weight_scale, group_size):
    entry = _LAST_DEQUANT_WEIGHT
    if entry is None:
        return None
    try:
        weight_version = weight_q._version
        scale_version = weight_scale._version
    except RuntimeError:
        return None
    if (
        entry[0]() is weight_q
        and entry[1]() is weight_scale
        and entry[2] == weight_version
        and entry[3] == scale_version
        and entry[4].dtype == x.dtype
        and entry[4].device == x.device
        and entry[5] == group_size
    ):
        return entry[4]
    return None


def _launch_cached(y, x, weight, M, N, eps):
    num_warps = _simple_warps(M, N)
    if _use_grouped(N, 128):
        rms_norm_cached_grouped_kernel[M,](
            y,
            x,
            weight,
            N,
            eps,
            128,
            N // 128,
            num_warps,
            num_warps=num_warps,
        )
        return
    if N <= 4096:
        rms_norm_cached_kernel[M,](
            y,
            x,
            weight,
            N,
            eps,
            triton.next_power_of_2(N),
            num_warps,
            num_warps=num_warps,
        )
        return
    tile_n = _tile_n(N)
    rms_norm_cached_loop_kernel[M,](
        y,
        x,
        weight,
        N,
        eps,
        tile_n,
        num_warps,
        num_warps=num_warps,
    )


def _launch_fp8(y, x, weight_q, weight_scale, M, N, eps, group_size):
    num_warps = _simple_warps(M, N)
    if _use_grouped(N, group_size):
        rms_norm_fp8_grouped_kernel[M,](
            y,
            x,
            weight_q,
            weight_scale,
            N,
            eps,
            group_size,
            N // group_size,
            num_warps,
            num_warps=num_warps,
        )
        return
    if N <= 4096:
        rms_norm_fp8_kernel[M,](
            y,
            x,
            weight_q,
            weight_scale,
            N,
            eps,
            group_size,
            triton.next_power_of_2(N),
            num_warps,
            num_warps=num_warps,
        )
        return
    rms_norm_fp8_loop_kernel[M,](
        y,
        x,
        weight_q,
        weight_scale,
        N,
        eps,
        group_size,
        _tile_n(N),
        num_warps,
        num_warps=num_warps,
    )


def rms_norm_w8a16_fp8(
    x, normalized_shape, weight_q, weight_scale, eps=1e-5, group_size=128
):
    if x.is_contiguous():
        cached = _peek_cached_weight(x, weight_q, weight_scale, group_size)
        if cached is not None:
            N = math.prod(normalized_shape)
            if cached.numel() == N and not _is_capturing():
                with torch_device_fn.device(x.device):
                    M = x.numel() // N
                    y = torch.empty_like(x)
                    _launch_cached(y, x, cached, M, N, eps)
                    return y

    logger.debug("GEMS_METAX RMS_NORM W8A16 FORWARD")
    dim = x.ndim - len(normalized_shape)
    M = math.prod(x.shape[:dim])
    N = math.prod(normalized_shape)
    num_groups = -(-N // group_size)
    if _FP8_DTYPE is None or weight_q.dtype != _FP8_DTYPE:
        raise TypeError(
            f"MetaX W8A16 RMSNorm expects float8_e4m3fn weight, got {weight_q.dtype}"
        )
    if weight_q.numel() != N:
        raise ValueError(f"weight_q numel {weight_q.numel()} != {N} elements")
    if weight_scale.numel() != num_groups:
        raise ValueError(
            f"weight_scale numel {weight_scale.numel()} != {num_groups} groups"
        )
    if not x.is_contiguous():
        x = x.contiguous()
    y = torch.empty(x.shape, device=x.device, dtype=x.dtype)
    with torch_device_fn.device(x.device):
        if not _is_capturing():
            dequant_weight = _get_dequant_weight(x, weight_q, weight_scale, group_size)
            if dequant_weight is not None:
                _launch_cached(y, x, dequant_weight, M, N, eps)
                return y
        w = weight_q if weight_q.is_contiguous() else weight_q.contiguous()
        scale = (
            weight_scale if weight_scale.is_contiguous() else weight_scale.contiguous()
        )
        _launch_fp8(y, x, w, scale, M, N, eps, group_size)
    return y
