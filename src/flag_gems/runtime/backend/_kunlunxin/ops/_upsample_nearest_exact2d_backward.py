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
# How many extra output positions are scanned around the analytically estimated
# window.  The window has to absorb the (at most one-step) difference between
# ATen's ``floor((dst + 0.5) * reciprocal_scale)`` and the analytic inverse
# ``ceil(src * scale - 0.5)``: both are only exact in real arithmetic and the
# fp32 rounding of the reciprocal can move a boundary by one position.
# ``cand = ceil(scale) + 1`` is what an exhaustive sweep of every
# (in_size, out_size) in [1, 64] x [1, 256] together with 4000 random larger
# pairs actually needs; +1 more is kept as margin.
_EXTRA_CANDIDATES = 2


@libentry()
@triton.jit
def _upsample_nearest_exact2d_backward_kernel(
    grad_output,
    grad_input,
    total,
    in_h,
    in_w,
    out_h,
    out_w,
    rec_h: tl.constexpr,
    rec_w: tl.constexpr,
    scale_h: tl.constexpr,
    scale_w: tl.constexpr,
    cand_h: tl.constexpr,
    cand_w: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Gather backward: one lane per grad_input element.

    For every input position (ih, iw) the kernel scans the candidate output
    positions ``oh in [ceil(ih * scale_h - 0.5) - 1, +cand_h)`` and accepts a
    candidate only if ATen's *forward* source-index formula maps it back to
    (ih, iw).  Using the official formula as the acceptance test makes the
    result bit-compatible with ATen at the fp32 boundaries where the purely
    analytic ``ceil(ih * scale - 0.5)`` range differs by one position.
    """
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < total

    iw = offsets % in_w
    ih = (offsets // in_w) % in_h
    nc = offsets // (in_h * in_w)
    # Lanes past ``total`` only exist in the padding of the over-allocated
    # output; keep their addresses in bounds (their result is discarded).
    base = tl.where(mask, nc, 0) * (out_h * out_w)

    h_lo = tl.ceil(ih.to(tl.float32) * scale_h - 0.5).to(tl.int32) - 1
    w_lo = tl.ceil(iw.to(tl.float32) * scale_w - 0.5).to(tl.int32) - 1
    # The topmost input row also absorbs every output row past the clamp of
    # ATen's ``min(..., in_h - 1)``; slide its window so that it ends at
    # ``out_h`` instead of starting at the analytic estimate.
    h_lo = tl.where(ih == in_h - 1, tl.maximum(h_lo, out_h - cand_h), h_lo)
    w_lo = tl.where(iw == in_w - 1, tl.maximum(w_lo, out_w - cand_w), w_lo)

    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for dh in tl.static_range(cand_h):
        oh = h_lo + dh
        # ATen: min(int(floor((dst + 0.5f) * scale)), src_size - 1)
        src_h = tl.minimum(((oh.to(tl.float32) + 0.5) * rec_h).to(tl.int32), in_h - 1)
        ok_h = mask & (oh >= 0) & (oh < out_h) & (src_h == ih)
        for dw in tl.static_range(cand_w):
            ow = w_lo + dw
            src_w = tl.minimum(
                ((ow.to(tl.float32) + 0.5) * rec_w).to(tl.int32), in_w - 1
            )
            ok = ok_h & (ow >= 0) & (ow < out_w) & (src_w == iw)
            # Keep the masked-load form used before this fix: an unmasked
            # gather from a clamped address was observed to fault the device
            # (KL_XID_KERNEL_EXCEPTION on XPU 5) as soon as a block contained
            # idle lanes, which all collapse onto the same fallback address.
            value = tl.load(
                grad_output
                + base
                + tl.where(ok_h, oh, 0) * out_w
                + tl.where(ok, ow, 0),
                mask=ok,
                other=0.0,
            )
            acc += tl.where(ok, value.to(tl.float32), 0.0)

    # ``grad_input`` is over-allocated to a whole tile so a store mask that is
    # not honoured still stays inside the allocation.
    tl.store(grad_input + offsets, acc.to(grad_input.dtype.element_ty), mask=mask)


def _f32(value):
    """Round a Python float to the nearest float32 (bit-exact fp32 constant)."""
    return float(np.float32(value))


def _reciprocal_scale(in_size, out_size, scale):
    """Match ATen ``compute_scales_value<float>`` exactly."""
    if scale is not None and scale > 0:
        # static_cast<float>(1.0 / scale)
        return _f32(1.0 / scale)
    # static_cast<float>(in) / out  (a genuine fp32 division)
    return float(np.float32(in_size) / np.float32(out_size))


def _shape_args(grad_output, output_size, input_size):
    N, C, IH, IW = (int(s) for s in input_size)
    OH, OW = int(output_size[-2]), int(output_size[-1])
    if grad_output.shape != (N, C, OH, OW):
        raise RuntimeError(
            f"Expected grad_output to have the same shape as output; output.size(2) "
            f"= {OH} and output.size(3) = {OW} but got grad_output.size(2) = "
            f"{grad_output.shape[-2]} and grad_output.size(3) = "
            f"{grad_output.shape[-1]}"
        )
    if OH < 0 or OW < 0:
        raise ValueError("Output size must be non-negative.")
    if OH * OW > 0 and (IH == 0 or IW == 0):
        raise RuntimeError(
            "Input and output sizes should be greater than 0, but got input "
            f"(H: {IH}, W: {IW}) output (H: {OH}, W: {OW})"
        )
    return N, C, IH, IW, OH, OW


# Highest rank the staging gather can decode.  ``grad_output`` is always 4-D
# here (the public entry asserts it before calling ``_stage_contiguous``); the
# helper pads shorter ranks with size-1 / zero-stride dimensions, so 8 is pure
# head-room and the guard below is a dead branch in practice.
_STAGE_MAX_NDIM = 8
# Bounded tile for the staging gather.  The destination is contiguous, so its
# store is a stride-1 block DMA; 2048 lanes keeps that block wide while staying
# clear of the 1024-lane tile associated with the registered bf16 pointwise
# defect (and far below the 32768 that is known to wedge this backend).
_STAGE_BLOCK = 2048


@libentry()
@triton.jit
def _stage_contiguous_kernel(
    src_ptr,
    dst_ptr,
    total,
    d0,
    d1,
    d2,
    d3,
    d4,
    d5,
    d6,
    d7,
    s0,
    s1,
    s2,
    s3,
    s4,
    s5,
    s6,
    s7,
    NDIM: tl.constexpr,
    NEED_MASK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Gather ``src`` (arbitrary strides) into a contiguous ``dst``.

    One lane per element.  ``dst`` is addressed stride-1, so its store lowers
    to a block DMA on this backend, while ``src`` is addressed through its
    *real* strides.  Dimensions are passed innermost-first, so the flat index
    is decoded with a divmod chain and the source offset is the dot product
    ``sum_k idx_k * stride_k`` -- which makes ``stride == 0`` (broadcast) and
    negative strides fall out for free.
    """
    dims = (d0, d1, d2, d3, d4, d5, d6, d7)
    strides = (s0, s1, s2, s3, s4, s5, s6, s7)

    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < total
    # A masked store is not honoured on this backend (it degrades to an address
    # select and every lane writes), so the tail lanes must still produce an
    # in-bounds *source* address: decoding ``0`` for them keeps the gather
    # inside ``src`` while their result lands in the over-allocated tail of
    # ``dst`` and is discarded.
    rem = tl.where(mask, offsets, 0)
    src_off = tl.zeros((BLOCK,), dtype=tl.int32)
    for k in tl.static_range(NDIM):
        idx = rem % dims[k]
        rem = rem // dims[k]
        src_off += idx * strides[k]

    if NEED_MASK:
        value = tl.load(src_ptr + src_off, mask=mask, other=0.0)
        tl.store(dst_ptr + offsets, value.to(dst_ptr.dtype.element_ty), mask=mask)
    else:
        value = tl.load(src_ptr + src_off)
        tl.store(dst_ptr + offsets, value.to(dst_ptr.dtype.element_ty))


def _stage_contiguous(grad_output):
    """Return ``grad_output`` itself, or a contiguous staging copy of it.

    ``Tensor.contiguous`` / the vendor ``copy_`` are unsafe on strided 2-byte
    sources on this backend (``_kunlunxin/ops/copy.py::copy_slice`` wedges the
    device on a transposed fp16 ``(512, 512)``), so the layout is normalised
    with this file's own Triton gather (``_stage_contiguous_kernel``) instead
    of any ``torch`` copy primitive.  Contiguous inputs return early and never
    enter this path, so the official matrices pay nothing for it.
    """
    if grad_output.is_contiguous():
        return grad_output

    ndim = grad_output.dim()
    if ndim > _STAGE_MAX_NDIM:
        raise NotImplementedError(
            f"_stage_contiguous supports grad_output up to {_STAGE_MAX_NDIM} "
            f"dimensions, got {ndim}"
        )

    # Innermost-first, padded with size-1 / zero-stride dimensions: the padding
    # contributes `0 * 0` to every source offset, i.e. it is a no-op.
    shape = grad_output.shape
    stride = grad_output.stride()
    dims = list(shape[::-1]) + [1] * (_STAGE_MAX_NDIM - ndim)
    strides = list(stride[::-1]) + [0] * (_STAGE_MAX_NDIM - ndim)
    total = grad_output.numel()

    # Over-allocate to a whole tile so the (unhonoured) store mask still lands
    # inside the allocation; the returned view carries the requested shape and
    # is contiguous, exactly like the staging buffer it replaces.
    buf = torch.empty(
        triton.cdiv(total, _STAGE_BLOCK) * _STAGE_BLOCK,
        device=grad_output.device,
        dtype=grad_output.dtype,
    )
    staged = buf[:total].view(shape)

    grid = (triton.cdiv(total, _STAGE_BLOCK),)
    with torch_device_fn.device(grad_output.device):
        _stage_contiguous_kernel[grid](
            grad_output,
            staged,
            total,
            *dims,
            *strides,
            NDIM=ndim,
            NEED_MASK=(total % _STAGE_BLOCK != 0),
            BLOCK=_STAGE_BLOCK,
        )
    return staged


def _upsample_nearest_exact2d_backward(
    grad_output: torch.Tensor,
    output_size: Tuple[int, int],
    input_size: Tuple[int, int, int, int],
    scales_h: Optional[float] = None,
    scales_w: Optional[float] = None,
) -> torch.Tensor:
    logger.debug("GEMS_KUNLUNXIN UPSAMPLE NEAREST EXACT2D BACKWARD")

    assert grad_output.device.type == device.name
    assert grad_output.ndim == 4, "The ndim of grad_output must be 4"
    assert len(output_size) == 2, "The len of output_size must be 2"
    assert len(input_size) == 4, "The len of input_size must be 4"

    N, C, IH, IW, OH, OW = _shape_args(grad_output, output_size, input_size)
    total = N * C * IH * IW
    if total == 0:
        return torch.empty(
            tuple(input_size), device=grad_output.device, dtype=grad_output.dtype
        )

    grad_output = _stage_contiguous(grad_output)

    rec_h = _reciprocal_scale(IH, OH, scales_h)
    rec_w = _reciprocal_scale(IW, OW, scales_w)
    scale_h = float(scales_h) if scales_h is not None else OH / IH
    scale_w = float(scales_w) if scales_w is not None else OW / IW
    cand_h = max(math.ceil(scale_h), 1) + _EXTRA_CANDIDATES
    cand_w = max(math.ceil(scale_w), 1) + _EXTRA_CANDIDATES

    # Over-allocate to a whole tile (a store mask that is not honoured on this
    # backend then still lands inside the allocation), then hand back a view
    # with the requested shape (strides/contiguity unchanged).
    buf = torch.empty(
        triton.cdiv(total, _BLOCK) * _BLOCK,
        device=grad_output.device,
        dtype=grad_output.dtype,
    )
    grad_input = buf[:total].view(tuple(input_size))

    grid = (triton.cdiv(total, _BLOCK),)
    with torch_device_fn.device(grad_output.device):
        _upsample_nearest_exact2d_backward_kernel[grid](
            grad_output,
            grad_input,
            total,
            IH,
            IW,
            OH,
            OW,
            rec_h=rec_h,
            rec_w=rec_w,
            scale_h=scale_h,
            scale_w=scale_w,
            cand_h=cand_h,
            cand_w=cand_w,
            BLOCK=_BLOCK,
            num_warps=1,
            buffer_size_limit=2048,
            isCloseVectorization=True,
        )
    return grad_input
