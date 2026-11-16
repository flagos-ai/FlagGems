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
import math
from typing import Optional, Tuple

import numpy as np
import torch
import triton
import triton.language as tl

from flag_gems.runtime import device, torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

_BLOCK = 1024
# The general candidate scan holds cand_d*cand_h*cand_w live loads per lane.
# Keep the widest tile that fits the buffer tuner: XPU3 is program-dispatch
# bound, so fewer, fatter programs win whenever register pressure allows.
_BLOCK_SCAN = 256
_EXTRA_CANDIDATES = 1


@libentry()
@triton.jit
def _upsample_nearest_exact3d_backward_kernel(
    grad_output,
    grad_input,
    total,
    C,
    ID,
    IH,
    IW,
    OD,
    OH,
    OW,
    go_stride_n,
    go_stride_c,
    go_stride_d,
    go_stride_h,
    go_stride_w,
    gi_stride_n,
    gi_stride_c,
    gi_stride_d,
    gi_stride_h,
    gi_stride_w,
    rec_d: tl.constexpr,
    rec_h: tl.constexpr,
    rec_w: tl.constexpr,
    scale_d: tl.constexpr,
    scale_h: tl.constexpr,
    scale_w: tl.constexpr,
    cand_d: tl.constexpr,
    cand_h: tl.constexpr,
    cand_w: tl.constexpr,
    IS_FP64: tl.constexpr,
    IS_UINT8: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Gather backward for ``_upsample_nearest_exact3d_backward``.

    One lane per ``grad_input`` voxel.  A candidate output voxel is accepted
    only when ATen's official *forward* exact formula

        src = min(int(floor((o + 0.5) * reciprocal_scale)), in_size - 1)

    maps it back to the current input voxel.  This reproduces the native
    scatter+atomic result (including the topmost-input clamp tail) without
    atomics.  A downsampled input voxel may receive no contribution at all.
    """
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < total
    # Store masks are not honoured on this backend; tail lanes decode element 0
    # and recompute lane 0's value so their write is harmless.
    rem = tl.where(mask, offsets, 0)

    iw = rem % IW
    tmp = rem // IW
    ih = tmp % IH
    tmp2 = tmp // IH
    id_ = tmp2 % ID
    nc = tmp2 // ID
    c = nc % C
    n = nc // C

    # Candidate window starts; slide the topmost input of each dimension so its
    # window ends at the output extent and absorbs every clamped position.
    d_lo = tl.ceil(id_.to(tl.float32) * scale_d - 0.5).to(tl.int32) - 1
    h_lo = tl.ceil(ih.to(tl.float32) * scale_h - 0.5).to(tl.int32) - 1
    w_lo = tl.ceil(iw.to(tl.float32) * scale_w - 0.5).to(tl.int32) - 1
    d_lo = tl.where(id_ == ID - 1, tl.maximum(d_lo, OD - cand_d), d_lo)
    h_lo = tl.where(ih == IH - 1, tl.maximum(h_lo, OH - cand_h), h_lo)
    w_lo = tl.where(iw == IW - 1, tl.maximum(w_lo, OW - cand_w), w_lo)

    go_base = n * go_stride_n + c * go_stride_c
    gi_base = (
        n * gi_stride_n
        + c * gi_stride_c
        + id_ * gi_stride_d
        + ih * gi_stride_h
        + iw * gi_stride_w
    )

    if IS_UINT8:
        acc = tl.zeros((BLOCK,), dtype=tl.int32)
    elif IS_FP64:
        acc = tl.zeros((BLOCK,), dtype=tl.float64)
    else:
        acc = tl.zeros((BLOCK,), dtype=tl.float32)

    for kd in tl.static_range(cand_d):
        od = d_lo + kd
        src_d = tl.minimum(((od.to(tl.float32) + 0.5) * rec_d).to(tl.int32), ID - 1)
        ok_d = (od >= 0) & (od < OD) & (src_d == id_)
        for kh in tl.static_range(cand_h):
            oh = h_lo + kh
            src_h = tl.minimum(((oh.to(tl.float32) + 0.5) * rec_h).to(tl.int32), IH - 1)
            ok_h = ok_d & (oh >= 0) & (oh < OH) & (src_h == ih)
            for kw in tl.static_range(cand_w):
                ow = w_lo + kw
                src_w = tl.minimum(
                    ((ow.to(tl.float32) + 0.5) * rec_w).to(tl.int32), IW - 1
                )
                ok = ok_h & (ow >= 0) & (ow < OW) & (src_w == iw)
                # Clamp every address; rejected lanes are neutralized below.
                go_off = (
                    go_base
                    + tl.where(ok_d, od, 0) * go_stride_d
                    + tl.where(ok_h, oh, 0) * go_stride_h
                    + tl.where(ok, ow, 0) * go_stride_w
                )
                value = tl.load(grad_output + go_off, mask=ok, other=0)
                if IS_UINT8:
                    acc += tl.where(ok, value.to(tl.int32), 0)
                elif IS_FP64:
                    acc += tl.where(ok, value.to(tl.float64), 0.0)
                else:
                    acc += tl.where(ok, value.to(tl.float32), 0.0)

    tl.store(
        grad_input + gi_base,
        acc.to(grad_input.dtype.element_ty),
        mask=mask,
    )


@libentry()
@triton.jit
def _upsample_nearest_exact3d_backward_2x_kernel(
    grad_output,
    grad_input,
    total,
    C,
    ID,
    IH,
    IW,
    OD,
    OH,
    OW,
    go_stride_n,
    go_stride_c,
    go_stride_d,
    go_stride_h,
    go_stride_w,
    gi_stride_n,
    gi_stride_c,
    gi_stride_d,
    gi_stride_h,
    gi_stride_w,
    IS_FP64: tl.constexpr,
    IS_UINT8: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Exact-2x fast path.

    For ``OD == 2*ID`` (and likewise H/W) the ``exact`` forward map is
    ``src = floor((dst + 0.5) * 0.5) == dst // 2`` exactly, so every input voxel
    gathers the fixed 2x2x2 output block around it -- no candidate scan.
    """
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < total
    rem = tl.where(mask, offsets, 0)

    iw = rem % IW
    tmp = rem // IW
    ih = tmp % IH
    tmp2 = tmp // IH
    id_ = tmp2 % ID
    nc = tmp2 // ID
    c = nc % C
    n = nc // C

    go_base = n * go_stride_n + c * go_stride_c
    gi_base = (
        n * gi_stride_n
        + c * gi_stride_c
        + id_ * gi_stride_d
        + ih * gi_stride_h
        + iw * gi_stride_w
    )
    od0 = id_ * 2
    oh0 = ih * 2
    ow0 = iw * 2

    if IS_UINT8:
        acc = tl.zeros((BLOCK,), dtype=tl.int32)
    elif IS_FP64:
        acc = tl.zeros((BLOCK,), dtype=tl.float64)
    else:
        acc = tl.zeros((BLOCK,), dtype=tl.float32)

    for dd in tl.static_range(2):
        for hh in tl.static_range(2):
            for ww in tl.static_range(2):
                go_off = (
                    go_base
                    + (od0 + dd) * go_stride_d
                    + (oh0 + hh) * go_stride_h
                    + (ow0 + ww) * go_stride_w
                )
                value = tl.load(grad_output + go_off, mask=mask, other=0)
                if IS_UINT8:
                    acc += value.to(tl.int32)
                elif IS_FP64:
                    acc += value.to(tl.float64)
                else:
                    acc += value.to(tl.float32)

    tl.store(
        grad_input + gi_base,
        acc.to(grad_input.dtype.element_ty),
        mask=mask,
    )


def _scale_value(scale: Optional[float], output_size: int, input_size: int) -> float:
    # ATen ``compute_scales_value_backwards<float>``: an explicit positive
    # scale wins, otherwise the backward scale is output / input.
    if scale is not None and scale > 0:
        return float(scale)
    return output_size / input_size


def _reciprocal_scale(scale: Optional[float], in_size: int, out_size: int) -> float:
    """Match ATen ``compute_scales_value<float>`` (forward reciprocal, fp32)."""
    if scale is not None and scale > 0:
        return float(np.float32(1.0 / float(scale)))
    return float(np.float32(in_size) / np.float32(out_size))


def _upsample_nearest_exact3d_backward_impl(
    grad_output: torch.Tensor,
    output_size: Tuple[int, int, int],
    input_size: Tuple[int, int, int, int, int],
    scales_d: Optional[float],
    scales_h: Optional[float],
    scales_w: Optional[float],
    grad_input: Optional[torch.Tensor],
) -> torch.Tensor:
    if grad_output.device.type != device.name:
        raise RuntimeError(
            f"expected grad_output to be on {device.name}, got {grad_output.device}"
        )
    if grad_output.ndim != 5:
        raise RuntimeError("grad_output must be a 5D tensor")
    if len(output_size) != 3:
        raise RuntimeError("output_size must contain three values")
    if len(input_size) != 5:
        raise RuntimeError("input_size must contain five values")

    od, oh, ow = map(int, output_size)
    n, c, id_, ih, iw = map(int, input_size)
    expected_shape = (n, c, od, oh, ow)
    if tuple(grad_output.shape) != expected_shape:
        raise RuntimeError(
            f"grad_output shape {tuple(grad_output.shape)} does not match "
            f"expected shape {expected_shape}"
        )

    if grad_input is None:
        grad_input = torch.empty(
            (n, c, id_, ih, iw),
            dtype=grad_output.dtype,
            device=grad_output.device,
        )
    else:
        if grad_input.device != grad_output.device:
            raise RuntimeError("grad_input and grad_output must be on the same device")
        if grad_input.dtype != grad_output.dtype:
            raise RuntimeError(
                f"Expected out tensor to have dtype {grad_output.dtype}, "
                f"but got {grad_input.dtype} instead"
            )
        if tuple(grad_input.shape) != (n, c, id_, ih, iw):
            grad_input.resize_((n, c, id_, ih, iw))

    total = grad_input.numel()
    if total == 0:
        return grad_input

    scale_d = _scale_value(scales_d, od, id_)
    scale_h = _scale_value(scales_h, oh, ih)
    scale_w = _scale_value(scales_w, ow, iw)
    rec_d = _reciprocal_scale(scales_d, id_, od)
    rec_h = _reciprocal_scale(scales_h, ih, oh)
    rec_w = _reciprocal_scale(scales_w, iw, ow)
    scale_d = float(np.float32(scale_d))
    scale_h = float(np.float32(scale_h))
    scale_w = float(np.float32(scale_w))

    def _candidates(scale, in_size, out_size):
        cand = max(1, math.ceil(scale)) + _EXTRA_CANDIDATES
        # The topmost input also absorbs every clamped output position.
        top_start = math.ceil((in_size - 1) * scale - 0.5)
        return min(out_size, max(cand, out_size - top_start))

    cand_d = _candidates(scale_d, id_, od)
    cand_h = _candidates(scale_h, ih, oh)
    cand_w = _candidates(scale_w, iw, ow)

    exact_2x = (
        od == 2 * id_ and oh == 2 * ih and ow == 2 * iw and scale_d == 2.0
        and scale_h == 2.0 and scale_w == 2.0
    )

    grid = (triton.cdiv(total, _BLOCK_SCAN),)
    with torch_device_fn.device(grad_output.device):
        if exact_2x:
            grid = (triton.cdiv(total, _BLOCK),)
            _upsample_nearest_exact3d_backward_2x_kernel[grid](
                grad_output,
                grad_input,
                total,
                c,
                id_,
                ih,
                iw,
                od,
                oh,
                ow,
                *grad_output.stride(),
                *grad_input.stride(),
                IS_FP64=grad_output.dtype == torch.float64,
                IS_UINT8=grad_output.dtype == torch.uint8,
                BLOCK=_BLOCK,
                num_warps=1,
                buffer_size_limit=2048,
            )
        else:
            scan_block = _BLOCK_SCAN
            grid = (triton.cdiv(total, scan_block),)
            _upsample_nearest_exact3d_backward_kernel[grid](
                grad_output,
                grad_input,
                total,
                c,
                id_,
                ih,
                iw,
                od,
                oh,
                ow,
                *grad_output.stride(),
                *grad_input.stride(),
                rec_d=rec_d,
                rec_h=rec_h,
                rec_w=rec_w,
                scale_d=scale_d,
                scale_h=scale_h,
                scale_w=scale_w,
                cand_d=cand_d,
                cand_h=cand_h,
                cand_w=cand_w,
                IS_FP64=grad_output.dtype == torch.float64,
                IS_UINT8=grad_output.dtype == torch.uint8,
                BLOCK=scan_block,
                num_warps=1,
                buffer_size_limit=2048,
            )
    return grad_input


def _upsample_nearest_exact3d_backward(
    grad_output: torch.Tensor,
    output_size: Tuple[int, int, int],
    input_size: Tuple[int, int, int, int, int],
    scales_d: Optional[float] = None,
    scales_h: Optional[float] = None,
    scales_w: Optional[float] = None,
) -> torch.Tensor:
    logger.debug("GEMS_KUNLUNXIN _UPSAMPLE_NEAREST_EXACT3D_BACKWARD")
    return _upsample_nearest_exact3d_backward_impl(
        grad_output, output_size, input_size, scales_d, scales_h, scales_w, None
    )


def _upsample_nearest_exact3d_backward_grad_input(
    grad_output: torch.Tensor,
    output_size: Tuple[int, int, int],
    input_size: Tuple[int, int, int, int, int],
    scales_d: Optional[float] = None,
    scales_h: Optional[float] = None,
    scales_w: Optional[float] = None,
    *,
    grad_input: torch.Tensor,
) -> torch.Tensor:
    logger.debug("GEMS_KUNLUNXIN _UPSAMPLE_NEAREST_EXACT3D_BACKWARD_GRAD_INPUT")
    return _upsample_nearest_exact3d_backward_impl(
        grad_output,
        output_size,
        input_size,
        scales_d,
        scales_h,
        scales_w,
        grad_input,
    )
