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

"""Antialiased Lanczos3 2-D resampling for Kunlunxin XPU.

Vendor override of the generic implementation, which does not compile on the
XPU compiler (``int64`` index tensors compared against ``int32`` loop scalars
produce ``arith.cmpi`` type mismatches) and, once it does compile, relies on
per-element gathers that run two orders of magnitude slower than the ATen
composition.

This implementation keeps the Lanczos weights in a dense (input, output)
matrix -- the same representation the ATen reference uses -- and evaluates the
two separable passes as fused block matrix products (``tl.dot``), so both
passes stream contiguous data.  The weight matrices are sparse by construction
(``MAX_TAPS`` non-zeros per column) and are zero-initialised, so only the taps
inside the support window are written.
"""

import logging
import math
from collections import OrderedDict
from typing import Optional, Sequence, Tuple

import torch
import triton
import triton.language as tl

from flag_gems.runtime import device, torch_device_fn

logger = logging.getLogger(__name__)
device = device.name

_BLOCK_M = 64
_BLOCK_N = 64
_BLOCK_K = 64
_BLOCK_WEIGHT_O = 64

# Resampling weights depend only on (in, out, scale, dtype, device, layout) and
# are reused across calls with the same geometry (the ATen reference caches its
# weight matrices the same way). The cache is bounded; eviction is LRU.
_WEIGHT_CACHE = OrderedDict()
_WEIGHT_CACHE_MAX = 16


@triton.jit
def _lanczos3(x):
    abs_x = tl.abs(x)
    pix = x * 3.141592653589793
    pix_over_three = pix / 3.0
    sinc_x = tl.where(abs_x == 0.0, 1.0, tl.sin(pix) / pix)
    sinc_x_over_three = tl.where(
        abs_x == 0.0, 1.0, tl.sin(pix_over_three) / pix_over_three
    )
    return tl.where(abs_x < 3.0, sinc_x * sinc_x_over_three, 0.0)


# Stores the unnormalised taps of every output index into a zeroed dense
# (input, output) matrix and the per-column normaliser into `colsum`; the dot
# kernels apply the normaliser, keeping the sin evaluation to a single pass.
# (Kept outside the kernel body: comments inside it trip the flagtree 0.7.1a1
# "pipeline event id" pass limit.)
@triton.jit
def _lanczos_weights_kernel(
    weights,
    colsum,
    input_size,
    output_size,
    SCALE: tl.constexpr,
    SUPPORT: tl.constexpr,
    INVSCALE: tl.constexpr,
    MAX_TAPS: tl.constexpr,
    BLOCK_T: tl.constexpr,
    TRANSPOSED: tl.constexpr,
    ROW_STRIDE: tl.constexpr,
    BLOCK_O: tl.constexpr,
):
    # Writes the unnormalised taps of every output index into a zeroed dense
    # matrix and the per-column normaliser into `colsum`; the dot kernels apply
    # the normaliser, so sin() is evaluated only once per weight. The two
    # layouts keep the innermost axis of the store contiguous, because XPU
    # miscompiles the equivalent strided stores.
    o = tl.program_id(0) * BLOCK_O + tl.arange(0, BLOCK_O)
    omask = o < output_size
    center = SCALE * (o.to(tl.float32) + 0.5)
    index_min = tl.maximum((center - SUPPORT + 0.5).to(tl.int32), 0)
    index_max = tl.minimum((center + SUPPORT + 0.5).to(tl.int32), input_size)
    index_size = tl.minimum(tl.maximum(index_max - index_min, 0), MAX_TAPS)

    if TRANSPOSED:
        tap = tl.arange(0, BLOCK_T)
        distance = (
            tap[None, :] + index_min[:, None] - center[:, None] + 0.5
        ) * INVSCALE
        mask = omask[:, None] & (tap[None, :] < index_size[:, None])
        weight = tl.where(mask, _lanczos3(distance), 0.0)
        tl.store(
            weights + o[:, None] * ROW_STRIDE + (index_min[:, None] + tap[None, :]),
            weight,
            mask=mask,
        )
        tl.store(colsum + o, tl.sum(weight, axis=1), mask=omask)
    else:
        total = tl.zeros((BLOCK_O,), dtype=tl.float32)
        for tap in tl.static_range(MAX_TAPS):
            tap_mask = omask & (tap < index_size)
            weight = tl.where(
                tap_mask, _lanczos3((tap + index_min - center + 0.5) * INVSCALE), 0.0
            )
            tl.store(
                weights + (index_min + tap) * output_size + o, weight, mask=tap_mask
            )
            total += weight
        tl.store(colsum + o, total, mask=omask)


@triton.jit
def _lanczos_horizontal_kernel(
    x_ptr,
    w_ptr,
    colsum_ptr,
    out_ptr,
    rows,
    input_w,
    output_w,
    IS_UINT8: tl.constexpr,
    F32_DOT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # out[rows, output_w] = x[rows, input_w] @ weights[input_w, output_w]
    rm = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    rmask = rm < rows
    nmask = rn < output_w

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, input_w, BLOCK_K):
        rk = k0 + tl.arange(0, BLOCK_K)
        kmask = rk < input_w
        if F32_DOT:
            a = tl.load(
                x_ptr + rm[:, None] * input_w + rk[None, :],
                mask=rmask[:, None] & kmask[None, :],
                other=0.0,
            ).to(tl.float32)
        else:
            a = tl.load(
                x_ptr + rm[:, None] * input_w + rk[None, :],
                mask=rmask[:, None] & kmask[None, :],
                other=0.0,
            )
        b = tl.load(
            w_ptr + rk[:, None] * output_w + rn[None, :],
            mask=kmask[:, None] & nmask[None, :],
            other=0.0,
        )
        acc += tl.dot(a, b, input_precision="ieee")

    # XPU miscompiles a broadcast divide, so scale by the reciprocal.
    normaliser = tl.load(colsum_ptr + rn, mask=nmask, other=1.0)
    acc *= (1.0 / tl.where(normaliser != 0.0, normaliser, 1.0))[None, :]
    if IS_UINT8:
        acc = tl.floor(tl.minimum(tl.maximum(acc, 0.0), 255.0) + 0.5)
    tl.store(
        out_ptr + rm[:, None] * output_w + rn[None, :],
        acc,
        mask=rmask[:, None] & nmask[None, :],
    )


@triton.jit
def _lanczos_vertical_kernel(
    x_ptr,
    w_ptr,
    colsum_ptr,
    out_ptr,
    w_stride,
    input_h,
    output_h,
    output_w,
    IS_UINT8: tl.constexpr,
    F32_DOT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # out[output_h, output_w] = weights[output_h, input_h] @ x[input_h, output_w]
    batch = tl.program_id(0)
    rm = tl.program_id(1) * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = tl.program_id(2) * BLOCK_N + tl.arange(0, BLOCK_N)
    mmask = rm < output_h
    nmask = rn < output_w

    x_base = x_ptr + batch * input_h * output_w
    out_base = out_ptr + batch * output_h * output_w

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, input_h, BLOCK_K):
        rk = k0 + tl.arange(0, BLOCK_K)
        kmask = rk < input_h
        # weights are stored as (output_h, padded input_h)
        a = tl.load(
            w_ptr + rm[:, None] * w_stride + rk[None, :],
            mask=mmask[:, None] & kmask[None, :],
            other=0.0,
        )
        if F32_DOT:
            b = tl.load(
                x_base + rk[:, None] * output_w + rn[None, :],
                mask=kmask[:, None] & nmask[None, :],
                other=0.0,
            ).to(tl.float32)
        else:
            b = tl.load(
                x_base + rk[:, None] * output_w + rn[None, :],
                mask=kmask[:, None] & nmask[None, :],
                other=0.0,
            )
        acc += tl.dot(a, b, input_precision="ieee")

    # XPU miscompiles a broadcast divide, so scale by the reciprocal.
    normaliser = tl.load(colsum_ptr + rm, mask=mmask, other=1.0)
    acc *= (1.0 / tl.where(normaliser != 0.0, normaliser, 1.0))[:, None]
    if IS_UINT8:
        acc = tl.floor(tl.minimum(tl.maximum(acc, 0.0), 255.0) + 0.5)
    tl.store(
        out_base + rm[:, None] * output_w + rn[None, :],
        acc,
        mask=mmask[:, None] & nmask[None, :],
    )


def _reciprocal_scale(
    input_size: int,
    output_size: int,
    align_corners: bool,
    scale: Optional[float],
) -> float:
    if align_corners:
        return (input_size - 1) / (output_size - 1) if output_size > 1 else 0.0
    if scale is not None and scale > 0.0:
        return 1.0 / scale
    return input_size / output_size


def _support_and_taps(scale: float) -> Tuple[float, int]:
    support = 3.0 * scale if scale >= 1.0 else 3.0
    return support, math.ceil(support) * 2 + 1


def _validate(input: torch.Tensor, output_size: Sequence[int]) -> Tuple[int, int]:
    if input.device.type != device:
        raise RuntimeError(f"Expected input on {device}, but got {input.device.type}")
    if input.ndim != 4:
        raise RuntimeError(
            f"It is expected input_size equals to 4, but got size {input.ndim}"
        )
    if len(output_size) != 2:
        raise RuntimeError(
            f"It is expected output_size equals to 2, but got size {len(output_size)}"
        )
    if not input.is_floating_point() and input.dtype != torch.uint8:
        raise RuntimeError(
            f'"compute_index_ranges_weights" not implemented for {input.dtype}'
        )

    output_h, output_w = int(output_size[0]), int(output_size[1])
    if min(input.shape[-2], input.shape[-1], output_h, output_w) <= 0:
        raise RuntimeError("Input and output sizes should be greater than 0")
    if input.shape[1] == 0:
        raise RuntimeError("Non-empty 4D data tensor expected")
    return output_h, output_w


def _weight_row_stride(input_size: int) -> int:
    # XPU miscompiles stores with an odd row stride (e.g. 3); round up to 4.
    return (input_size + 3) // 4 * 4


def _weight_dtype(input: torch.Tensor) -> torch.dtype:
    # 16-bit inputs feed the matmul units directly; everything else is f32.
    if input.dtype in (torch.float16, torch.bfloat16):
        return input.dtype
    return torch.float32


def _make_lanczos_weights(
    input_size: int,
    output_size: int,
    scale: float,
    device: torch.device,
    dtype: torch.dtype,
    transposed: bool = False,
):
    key = (input_size, output_size, scale, device, dtype, transposed)
    cached = _WEIGHT_CACHE.get(key)
    if cached is not None:
        _WEIGHT_CACHE.move_to_end(key)
        return cached

    support, taps = _support_and_taps(scale)
    row_stride = _weight_row_stride(input_size)
    if transposed:
        weights = torch.zeros((output_size, row_stride), device=device, dtype=dtype)
    else:
        weights = torch.zeros((input_size, output_size), device=device, dtype=dtype)
    colsum = torch.empty(output_size, device=device, dtype=torch.float32)
    _lanczos_weights_kernel[(triton.cdiv(output_size, _BLOCK_WEIGHT_O),)](
        weights,
        colsum,
        input_size,
        output_size,
        SCALE=scale,
        SUPPORT=support,
        INVSCALE=1.0 / scale if scale >= 1.0 else 1.0,
        MAX_TAPS=taps,
        BLOCK_T=max(1, triton.next_power_of_2(taps)),
        TRANSPOSED=transposed,
        ROW_STRIDE=row_stride,
        BLOCK_O=_BLOCK_WEIGHT_O,
    )
    _WEIGHT_CACHE[key] = (weights, colsum)
    if len(_WEIGHT_CACHE) > _WEIGHT_CACHE_MAX:
        _WEIGHT_CACHE.popitem(last=False)
    return weights, colsum


def _upsample_lanczos2d_aa_contiguous(
    input: torch.Tensor,
    output_size: Tuple[int, int],
    align_corners: bool,
    scales_h: Optional[float],
    scales_w: Optional[float],
) -> torch.Tensor:
    n, c, input_h, input_w = input.shape
    output_h, output_w = output_size
    if input_h == output_h and input_w == output_w:
        return input.clone()

    scale_w = _reciprocal_scale(input_w, output_w, align_corners, scales_w)
    scale_h = _reciprocal_scale(input_h, output_h, align_corners, scales_h)
    is_uint8 = input.dtype == torch.uint8
    weight_dtype = _weight_dtype(input)
    f32_dot = weight_dtype == torch.float32

    with torch_device_fn.device(input.device):
        if input_w != output_w:
            weights_w, colsum_w = _make_lanczos_weights(
                input_w, output_w, scale_w, input.device, weight_dtype
            )
            horizontal = torch.empty(
                (n, c, input_h, output_w), device=input.device, dtype=input.dtype
            )
            rows = n * c * input_h
            _lanczos_horizontal_kernel[
                (
                    triton.cdiv(rows, _BLOCK_M),
                    triton.cdiv(output_w, _BLOCK_N),
                )
            ](
                input,
                weights_w,
                colsum_w,
                horizontal,
                rows,
                input_w,
                output_w,
                IS_UINT8=is_uint8,
                F32_DOT=f32_dot,
                BLOCK_M=_BLOCK_M,
                BLOCK_N=_BLOCK_N,
                BLOCK_K=_BLOCK_K,
            )
        else:
            horizontal = input

        if input_h != output_h:
            weights_h, colsum_h = _make_lanczos_weights(
                input_h, output_h, scale_h, input.device, weight_dtype, transposed=True
            )
            output = torch.empty(
                (n, c, output_h, output_w), device=input.device, dtype=input.dtype
            )
            _lanczos_vertical_kernel[
                (
                    n * c,
                    triton.cdiv(output_h, _BLOCK_M),
                    triton.cdiv(output_w, _BLOCK_N),
                )
            ](
                horizontal,
                weights_h,
                colsum_h,
                output,
                _weight_row_stride(input_h),
                input_h,
                output_h,
                output_w,
                IS_UINT8=is_uint8,
                F32_DOT=f32_dot,
                BLOCK_M=_BLOCK_M,
                BLOCK_N=_BLOCK_N,
                BLOCK_K=_BLOCK_K,
            )
        else:
            output = horizontal
    return output


def _upsample_lanczos2d_aa(
    input: torch.Tensor,
    output_size: Tuple[int, int],
    align_corners: bool = False,
    scales_h: Optional[float] = None,
    scales_w: Optional[float] = None,
) -> torch.Tensor:
    logger.debug("GEMS_KUNLUNXIN UPSAMPLE_LANCZOS2D_AA")
    output_size = _validate(input, output_size)
    channels_last = input.is_contiguous(memory_format=torch.channels_last)
    if input.shape[0] == 0:
        output = torch.empty(
            (*input.shape[:2], *output_size), device=input.device, dtype=input.dtype
        )
    else:
        output = _upsample_lanczos2d_aa_contiguous(
            input.contiguous(),
            output_size,
            align_corners,
            scales_h,
            scales_w,
        )
    if channels_last:
        output = output.contiguous(memory_format=torch.channels_last)
    return output


def _upsample_lanczos2d_aa_out(
    input: torch.Tensor,
    output_size: Tuple[int, int],
    align_corners: bool = False,
    scales_h: Optional[float] = None,
    scales_w: Optional[float] = None,
    *,
    out: torch.Tensor,
) -> torch.Tensor:
    if out.device != input.device:
        raise RuntimeError(
            f"Expected out tensor on {input.device}, but got {out.device}"
        )
    if out.dtype != input.dtype:
        raise RuntimeError(
            f"Expected out tensor to have dtype {input.dtype}, but got {out.dtype}"
        )
    result = _upsample_lanczos2d_aa(
        input, output_size, align_corners, scales_h, scales_w
    )
    out.resize_(result.shape)
    out.copy_(result)
    return out


def _upsample_lanczos2d_aa_vec(
    input: torch.Tensor,
    output_size: Optional[Sequence[int]],
    align_corners: bool,
    scale_factors: Optional[Sequence[float]],
) -> torch.Tensor:
    if (output_size is None) == (scale_factors is None):
        raise RuntimeError("Must specify exactly one of output_size and scale_factors")
    if output_size is not None:
        return _upsample_lanczos2d_aa(input, tuple(output_size), align_corners)
    if len(scale_factors) != 2:
        raise RuntimeError("scale_factors must have two elements")
    output_size = (
        int(input.shape[-2] * scale_factors[0]),
        int(input.shape[-1] * scale_factors[1]),
    )
    return _upsample_lanczos2d_aa(
        input,
        output_size,
        align_corners,
        float(scale_factors[0]),
        float(scale_factors[1]),
    )
