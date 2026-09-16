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

"""THead / PPU W8A16 RMSNorm.

Activation is 16-bit (FP16/BF16). Weight is grouped FP8 E4M3
(``torch.float8_e4m3fn``) plus per-group scale (group_size=128), matching
PR #4437.

Eager execution caches a BF16/FP16 dequantized weight while the source
tensors' version counters are unchanged. CUDA Graph capture and tensors
without version counters use the fused dynamic-weight kernel so every replay
continues to read the current weight and scale buffers.
"""

import logging
import math
import weakref

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)

_FP8_DTYPE = getattr(torch, "float8_e4m3fn", None)
# This cache contains only the fixed 256-value E4M3 decoding table. It never
# contains values derived from an input weight or scale tensor.
_FP8_LUT_CACHE = {}
# Entries hold weak references to both source tensors so cached device memory
# is released with the model. Source version counters invalidate eager cache
# entries after an in-place update.
_DEQUANT_WEIGHT_CACHE = {}
_LAST_DEQUANT_WEIGHT = None


@triton.jit
def prev_multiple_of(a, b):
    return tl.cdiv(a, b) * b - b


@triton.jit
def _decode_e4m3fn(bits):
    # E4M3B15 has the same layout as E4M3FN with an exponent bias larger by
    # eight. PPU has a packed software conversion for E4M3B15, so rescale its
    # FP16 result by 2**8 and restore E4M3FN's two NaN encodings.
    bits = bits.to(tl.uint8)
    magnitude = bits & 0x7F
    e4m3b15 = bits.to(tl.float8e4b15, bitcast=True)
    value = e4m3b15.to(tl.float16) * 256.0
    return tl.where(magnitude == 0x7F, float("nan"), value)


@libentry()
@triton.jit(do_not_specialize=["eps"])
def rms_norm_simple_kernel(
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
    rrms = 1 / tl.sqrt(tl.sum(x * x, axis=0) / N + eps)
    bits = tl.load(w_ptr + cols, mask=mask, other=0)
    scale = tl.load(scale_ptr + cols // GROUP_SIZE, mask=mask, other=0)
    w = _decode_e4m3fn(bits).to(in_ptr.dtype.element_ty) * scale
    y = (x * rrms).to(in_ptr.dtype.element_ty) * w
    tl.store(out_ptr + pid * N + cols, y, mask=mask)


@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("rms_norm_loop"),
    key=["N"],
)
@triton.jit(do_not_specialize=["eps"])
def rms_norm_simple_loop_kernel(
    out_ptr,
    in_ptr,
    w_ptr,
    scale_ptr,
    N,
    eps,
    GROUP_SIZE: tl.constexpr,
    TILE_N: tl.constexpr,
):
    if tl.constexpr(in_ptr.dtype.element_ty == tl.float16) or tl.constexpr(
        in_ptr.dtype.element_ty == tl.bfloat16
    ):
        cdtype = tl.float32
    else:
        cdtype = in_ptr.dtype.element_ty

    pid = ext.program_id(0)
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
    rrms = 1 / tl.sqrt(tl.sum(acc) / N + eps)

    prev_multiple = prev_multiple_of(N, TILE_N)
    for start_n in range(0, TILE_N, TILE_N):
        n_offsets = (prev_multiple - start_n) + tl.arange(0, TILE_N)
        mask = n_offsets < N
        x = tl.load(
            in_ptr + pid * N + n_offsets,
            mask=mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(cdtype)
        bits = tl.load(w_ptr + n_offsets, mask=mask, other=0)
        scale = tl.load(scale_ptr + n_offsets // GROUP_SIZE, mask=mask, other=0)
        w = _decode_e4m3fn(bits).to(in_ptr.dtype.element_ty) * scale
        y = (x * rrms).to(in_ptr.dtype.element_ty) * w
        tl.store(out_ptr + pid * N + n_offsets, y, mask=mask)
    for start_n in range(TILE_N, N, TILE_N):
        n_offsets = (prev_multiple - start_n) + tl.arange(0, TILE_N)
        x = tl.load(
            in_ptr + pid * N + n_offsets,
            eviction_policy="evict_first",
        ).to(cdtype)
        bits = tl.load(w_ptr + n_offsets)
        scale = tl.load(scale_ptr + n_offsets // GROUP_SIZE)
        w = _decode_e4m3fn(bits).to(in_ptr.dtype.element_ty) * scale
        y = (x * rrms).to(in_ptr.dtype.element_ty) * w
        tl.store(out_ptr + pid * N + n_offsets, y)


@libentry()
@triton.jit(do_not_specialize=["eps"])
def rms_norm_grouped_kernel(
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
    rrms = 1 / tl.sqrt(tl.sum(x * x) / N + eps)
    bits = tl.load(w_ptr + offsets)
    scale = tl.load(scale_ptr + groups)
    w = _decode_e4m3fn(bits).to(in_ptr.dtype.element_ty) * scale[:, None]
    y = (x * rrms).to(in_ptr.dtype.element_ty) * w
    tl.store(out_ptr + pid * N + offsets, y)


@libentry()
@triton.jit(do_not_specialize=["eps"])
def rms_norm_batched_kernel(
    out_ptr,
    in_ptr,
    w_ptr,
    scale_ptr,
    fp8_lut_ptr,
    M,
    N: tl.constexpr,
    eps,
    GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_WARPS: tl.constexpr,
):
    tl.static_assert(NUM_WARPS > 0)
    rows = ext.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)
    offsets = rows[:, None] * N + cols[None, :]
    mask = (rows[:, None] < M) & (cols[None, :] < N)
    x = tl.load(in_ptr + offsets, mask=mask, other=0).to(tl.float32)
    rrms = 1 / tl.sqrt(tl.sum(x * x, axis=1) / N + eps)
    bits = tl.load(w_ptr + cols, mask=cols < N, other=0)
    decoded = tl.load(fp8_lut_ptr + bits, mask=cols < N, other=0)
    scale = tl.load(scale_ptr + cols // GROUP_SIZE, mask=cols < N, other=0)
    w = decoded * scale
    y = (x * rrms[:, None]).to(in_ptr.dtype.element_ty) * w[None, :]
    tl.store(out_ptr + offsets, y, mask=mask)


@libentry()
@triton.jit(do_not_specialize=["eps"])
def rms_norm_tiled_batched_kernel(
    out_ptr,
    in_ptr,
    w_ptr,
    scale_ptr,
    fp8_lut_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    eps,
    GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    TILE_N: tl.constexpr,
    NUM_WARPS: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    tl.static_assert(NUM_WARPS > 0)
    tl.static_assert(NUM_STAGES > 0)
    rows = ext.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, TILE_N)
    acc = tl.zeros((BLOCK_M, TILE_N), tl.float32)
    for step in tl.static_range(0, N // TILE_N):
        current_cols = step * TILE_N + cols
        offsets = rows[:, None] * N + current_cols[None, :]
        x = tl.load(in_ptr + offsets).to(tl.float32)
        acc += x * x
    rrms = 1 / tl.sqrt(tl.sum(acc, axis=1) / N + eps)
    for step in tl.static_range(0, N // TILE_N):
        current_cols = step * TILE_N + cols
        offsets = rows[:, None] * N + current_cols[None, :]
        x = tl.load(in_ptr + offsets)
        bits = tl.load(w_ptr + current_cols)
        decoded = tl.load(fp8_lut_ptr + bits)
        scale = tl.load(scale_ptr + current_cols // GROUP_SIZE)
        w = decoded * scale
        y = (x * rrms[:, None]).to(in_ptr.dtype.element_ty) * w[None, :]
        tl.store(out_ptr + offsets, y)


@libentry()
@triton.jit(do_not_specialize=["eps"])
def rms_norm_cached_weight_kernel(
    out_ptr,
    in_ptr,
    weight_ptr,
    N: tl.constexpr,
    eps,
    BLOCK_SIZE: tl.constexpr,
    NUM_WARPS: tl.constexpr,
):
    pid = ext.program_id(0)
    tl.static_assert(NUM_WARPS > 0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    x = tl.load(in_ptr + pid * N + cols, mask=mask, other=0).to(tl.float32)
    rrms = tl.rsqrt(tl.sum(x * x, axis=0) / N + eps)
    weight = tl.load(weight_ptr + cols, mask=mask, other=0)
    y = (x * rrms).to(in_ptr.dtype.element_ty) * weight
    tl.store(out_ptr + pid * N + cols, y, mask=mask)


def _num_warps_for(M, N):
    if M == 1 and N >= 16384:
        return 32
    if M <= 64:
        return 16
    if N >= 32768:
        return 8
    return 4


def _get_fp8_lut(x):
    key = (x.device, x.dtype)
    lut = _FP8_LUT_CACHE.get(key)
    if lut is None:
        with torch_device_fn.device(x.device):
            lut = (
                torch.arange(256, device=x.device, dtype=torch.uint8)
                .view(_FP8_DTYPE)
                .to(x.dtype)
            )
        _FP8_LUT_CACHE[key] = lut
    return lut


def _remove_dequant_weight(key, dead_ref):
    global _LAST_DEQUANT_WEIGHT
    entry = _DEQUANT_WEIGHT_CACHE.get(key)
    if entry is not None and (entry[0] is dead_ref or entry[1] is dead_ref):
        _DEQUANT_WEIGHT_CACHE.pop(key, None)
        if _LAST_DEQUANT_WEIGHT is entry:
            _LAST_DEQUANT_WEIGHT = None


def _tensor_version(tensor):
    try:
        return tensor._version
    except RuntimeError:
        # Tensors created in inference_mode do not have version counters, so
        # their in-place updates cannot safely invalidate an eager cache.
        return None


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

    dequant_weight = (
        (weight_q.float().reshape(-1, group_size) * weight_scale.float().reshape(-1, 1))
        .reshape(-1)
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


def _cached_weight_num_warps(M, N):
    if N <= 4096:
        if M <= 16 or M >= 256:
            return 4
        return 8
    if N <= 8192:
        if M == 1:
            return 8
        if M <= 64:
            return 16
        return 8
    return 16


def _launch_rms_norm(y, x, w, scale, M, N, eps, group_size):
    if N == 4096 and M >= 256:
        if M == 256:
            block_m, tile_n, num_warps, num_stages = 2, 2048, 8, 3
        elif M % 4 == 0:
            block_m, tile_n, num_warps, num_stages = 4, 1024, 4, 2
        else:
            block_m = 0
        if block_m:
            fp8_lut = _get_fp8_lut(x)
            rms_norm_tiled_batched_kernel[M // block_m,](
                y,
                x,
                w,
                scale,
                fp8_lut,
                M,
                N,
                eps,
                group_size,
                block_m,
                tile_n,
                num_warps,
                num_stages,
                num_warps=num_warps,
                num_stages=num_stages,
            )
            return
    # Reuse decoded weights across rows when there are enough rows to keep
    # the device occupied. Larger row tiles otherwise increase register use.
    if M >= 256 and 4096 <= N <= 8192 and N & (N - 1) == 0:
        block_m, num_warps = (4, 4) if M >= 512 and N <= 4096 else (2, 8)
        fp8_lut = _get_fp8_lut(x)
        rms_norm_batched_kernel[triton.cdiv(M, block_m),](
            y,
            x,
            w,
            scale,
            fp8_lut,
            M,
            N,
            eps,
            group_size,
            block_m,
            N,
            num_warps,
            num_warps=num_warps,
        )
        return
    num_warps = _num_warps_for(M, N)
    # NUM_WARPS is a constexpr so libentry caches each warp count separately
    # (launch kwargs are not part of the entry key).
    # PPU can hold a full row up to 32k; Gems switches to a 2-pass loop at 4k.
    if N <= 16384 and N % 128 == 0 and N & (N - 1) == 0 and group_size == 128:
        rms_norm_grouped_kernel[M,](
            y, x, w, scale, N, eps, 128, N // 128, num_warps, num_warps=num_warps
        )
        return
    if N <= 32768:
        rms_norm_simple_kernel[M,](
            y,
            x,
            w,
            scale,
            N,
            eps,
            group_size,
            triton.next_power_of_2(N),
            num_warps,
            num_warps=num_warps,
        )
        return
    rms_norm_simple_loop_kernel[M,](y, x, w, scale, N, eps, group_size)


def rms_norm_w8a16_fp8(
    x, normalized_shape, weight_q, weight_scale, eps=1e-5, group_size=128
):
    entry = _LAST_DEQUANT_WEIGHT
    if entry is not None and x.is_contiguous():
        try:
            weight_version = weight_q._version
            scale_version = weight_scale._version
        except RuntimeError:
            pass
        else:
            if (
                entry[0]() is weight_q
                and entry[1]() is weight_scale
                and entry[2] == weight_version
                and entry[3] == scale_version
                and entry[4].dtype == x.dtype
                and entry[4].device == x.device
                and entry[5] == group_size
            ):
                N = math.prod(normalized_shape)
                if entry[4].numel() == N and N <= 16384:
                    with torch_device_fn.device(x.device):
                        if not torch.cuda.is_current_stream_capturing():
                            M = x.numel() // N
                            y = torch.empty_like(x)
                            num_warps = _cached_weight_num_warps(M, N)
                            rms_norm_cached_weight_kernel[M,](
                                y,
                                x,
                                entry[4],
                                N,
                                eps,
                                triton.next_power_of_2(N),
                                num_warps,
                                num_warps=num_warps,
                            )
                            return y
    logger.debug("GEMS_THEAD RMS_NORM W8A16 FORWARD")
    dim = x.ndim - len(normalized_shape)
    M = math.prod(x.shape[:dim])
    N = math.prod(normalized_shape)
    if N % group_size != 0:
        raise ValueError(
            f"normalized_shape product {N} must be divisible by group_size={group_size}"
        )
    if _FP8_DTYPE is None or weight_q.dtype != _FP8_DTYPE:
        raise TypeError(
            f"PPU W8A16 RMSNorm expects float8_e4m3fn weight, got {weight_q.dtype}"
        )
    if weight_q.numel() != N:
        raise ValueError(f"weight_q numel {weight_q.numel()} != {N} elements")
    if weight_scale.numel() != N // group_size:
        raise ValueError(
            f"weight_scale numel {weight_scale.numel()} != {N // group_size} groups"
        )
    if not x.is_contiguous():
        x = x.contiguous()
    y = torch.empty(x.shape, device=x.device, dtype=x.dtype)
    with torch_device_fn.device(x.device):
        if not torch.cuda.is_current_stream_capturing() and N <= 16384:
            dequant_weight = _get_dequant_weight(x, weight_q, weight_scale, group_size)
            if dequant_weight is not None:
                num_warps = _cached_weight_num_warps(M, N)
                rms_norm_cached_weight_kernel[M,](
                    y,
                    x,
                    dequant_weight,
                    N,
                    eps,
                    triton.next_power_of_2(N),
                    num_warps,
                    num_warps=num_warps,
                )
                return y
        # A dtype view only changes metadata; it does not launch a conversion.
        w = weight_q.contiguous().view(torch.uint8)
        scale = weight_scale.contiguous()
        _launch_rms_norm(y, x, w, scale, M, N, eps, group_size)
    return y
