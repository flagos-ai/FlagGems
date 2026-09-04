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

Weights are typically static, so they are dequantized once per unique
storage and reused. The hot path is then a Gems-like RMSNorm that does not
write ``inv_rms``. CUDA Graph capture after warmup therefore records only
the RMSNorm launch.
"""

import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)

_FP8_DTYPE = getattr(torch, "float8_e4m3fn", None)
_DEQUANT_CACHE = {}
_DEQUANT_CACHE_MAX = 16
_LAST_DEQUANT_KEY = None
_LAST_DEQUANT_W = None


@triton.jit
def prev_multiple_of(a, b):
    return tl.cdiv(a, b) * b - b


@libentry()
@triton.jit(do_not_specialize=["eps"])
def rms_norm_simple_kernel(
    out_ptr,
    in_ptr,
    w_ptr,
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
    rrms = 1 / tl.sqrt(tl.sum(x * x, axis=0) / N + eps)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0)
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
    N,
    eps,
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
        w = tl.load(w_ptr + n_offsets, mask=mask, other=0.0)
        y = (x * rrms).to(in_ptr.dtype.element_ty) * w
        tl.store(out_ptr + pid * N + n_offsets, y, mask=mask)
    for start_n in range(TILE_N, N, TILE_N):
        n_offsets = (prev_multiple - start_n) + tl.arange(0, TILE_N)
        x = tl.load(
            in_ptr + pid * N + n_offsets,
            eviction_policy="evict_first",
        ).to(cdtype)
        w = tl.load(w_ptr + n_offsets)
        y = (x * rrms).to(in_ptr.dtype.element_ty) * w
        tl.store(out_ptr + pid * N + n_offsets, y)


@libentry()
@triton.jit(do_not_specialize=["eps"])
def rms_norm_grouped_kernel(
    out_ptr,
    in_ptr,
    w_ptr,
    N,
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
    w = tl.load(w_ptr + offsets)
    y = (x * rrms).to(in_ptr.dtype.element_ty) * w
    tl.store(out_ptr + pid * N + offsets, y)


def _num_warps_for(M, N):
    if M <= 64:
        return 16
    if N >= 32768:
        return 8
    return 4


def _launch_rms_norm(y, x, w, M, N, eps):
    num_warps = _num_warps_for(M, N)
    # NUM_WARPS is a constexpr so libentry caches each warp count separately
    # (launch kwargs are not part of the entry key).
    # PPU can hold a full row up to 32k; Gems switches to a 2-pass loop at 4k.
    if N <= 16384 and N % 128 == 0:
        rms_norm_grouped_kernel[M,](
            y, x, w, N, eps, 128, N // 128, num_warps, num_warps=num_warps
        )
        return
    if N <= 32768:
        rms_norm_simple_kernel[M,](
            y,
            x,
            w,
            N,
            eps,
            triton.next_power_of_2(N),
            num_warps,
            num_warps=num_warps,
        )
        return
    rms_norm_simple_loop_kernel[M,](y, x, w, N, eps)


def _dequant_weight(weight_q, weight_scale, group_size, out_dtype):
    global _LAST_DEQUANT_KEY, _LAST_DEQUANT_W
    n = weight_q.numel()
    key = (weight_q.data_ptr(), weight_scale.data_ptr(), n, out_dtype, group_size)
    if key == _LAST_DEQUANT_KEY and _LAST_DEQUANT_W is not None:
        return _LAST_DEQUANT_W
    cached = _DEQUANT_CACHE.get(key)
    if cached is not None:
        _LAST_DEQUANT_KEY = key
        _LAST_DEQUANT_W = cached
        return cached
    # Cast in PyTorch so PPU does not need Triton fp8e4nv loads.
    w = (
        weight_q.to(torch.float32)
        * weight_scale.to(torch.float32).repeat_interleave(group_size)
    ).to(out_dtype)
    if len(_DEQUANT_CACHE) >= _DEQUANT_CACHE_MAX:
        _DEQUANT_CACHE.pop(next(iter(_DEQUANT_CACHE)))
    _DEQUANT_CACHE[key] = w
    _LAST_DEQUANT_KEY = key
    _LAST_DEQUANT_W = w
    return w


def rms_norm_w8a16_fp8(
    x, normalized_shape, weight_q, weight_scale, eps=1e-5, group_size=128
):
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
    if weight_scale.numel() != N // group_size:
        raise ValueError(
            f"weight_scale numel {weight_scale.numel()} != {N // group_size} groups"
        )
    if not x.is_contiguous():
        x = x.contiguous()
    y = torch.empty(x.shape, device=x.device, dtype=x.dtype)
    with torch_device_fn.device(x.device):
        w = _dequant_weight(weight_q, weight_scale, group_size, x.dtype)
        _launch_rms_norm(y, x, w, M, N, eps)
    return y
