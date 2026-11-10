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
# The candidate window has to absorb the (at most one-step) difference between
# ATen's ``floor(dst * reciprocal_scale)`` and the analytic inverse
# ``ceil(src * scale)``: both are only exact in real arithmetic and the fp32
# rounding of the reciprocal can move a boundary by one output position.  One
# extra candidate (plus the ``- 1`` low margin in the kernel) is sufficient for
# every (in_size, out_size) in the official test matrix.
_EXTRA_CANDIDATES = 1


@libentry()
@triton.jit
def _upsample_nearest3d_backward_kernel(
    grad_output,
    grad_input,
    total,
    ID,
    IH,
    IW,
    OD,
    OH,
    OW,
    rec_d: tl.constexpr,
    rec_h: tl.constexpr,
    rec_w: tl.constexpr,
    scale_d: tl.constexpr,
    scale_h: tl.constexpr,
    scale_w: tl.constexpr,
    cand_d: tl.constexpr,
    cand_h: tl.constexpr,
    cand_w: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Gather backward: one lane per ``grad_input`` element.

    For every input voxel (id, ih, iw) the kernel scans the candidate output
    positions and accepts a candidate only if ATen's *forward* source-index
    formula maps it back to (id, ih, iw).  Using the official formula as the
    acceptance test makes the result bit-compatible with ATen at the fp32
    boundaries where the purely analytic inverse range differs by one position.
    """
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < total

    # Decode the flat input index into (nc, id, ih, iw).
    iw = offsets % IW
    tmp = offsets // IW
    ih = tmp % IH
    tmp2 = tmp // IH
    id_ = tmp2 % ID
    nc = tmp2 // ID

    # Lanes past ``total`` only exist in the padding of the over-allocated
    # output; keep their base address in bounds (their result is discarded).
    base = tl.where(mask, nc, 0) * (OD * OH * OW)

    d_lo = tl.ceil(id_.to(tl.float32) * scale_d).to(tl.int32) - 1
    h_lo = tl.ceil(ih.to(tl.float32) * scale_h).to(tl.int32) - 1
    w_lo = tl.ceil(iw.to(tl.float32) * scale_w).to(tl.int32) - 1
    # The topmost input coordinate also absorbs every output position past the
    # clamp of ATen's ``min(..., in_size - 1)``; slide its window so that it
    # ends at ``out_size`` instead of starting at the analytic estimate.
    d_lo = tl.where(id_ == ID - 1, tl.maximum(d_lo, OD - cand_d), d_lo)
    h_lo = tl.where(ih == IH - 1, tl.maximum(h_lo, OH - cand_h), h_lo)
    w_lo = tl.where(iw == IW - 1, tl.maximum(w_lo, OW - cand_w), w_lo)

    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for dd in tl.static_range(cand_d):
        od = d_lo + dd
        # ATen forward: min(int(floor(dst * rec)), src_size - 1)
        src_d = tl.minimum((od.to(tl.float32) * rec_d).to(tl.int32), ID - 1)
        ok_d = mask & (od >= 0) & (od < OD) & (src_d == id_)
        for hh in tl.static_range(cand_h):
            oh = h_lo + hh
            src_h = tl.minimum((oh.to(tl.float32) * rec_h).to(tl.int32), IH - 1)
            ok_h = ok_d & (oh >= 0) & (oh < OH) & (src_h == ih)
            for ww in tl.static_range(cand_w):
                ow = w_lo + ww
                src_w = tl.minimum((ow.to(tl.float32) * rec_w).to(tl.int32), IW - 1)
                ok = ok_h & (ow >= 0) & (ow < OW) & (src_w == iw)
                # The masked load is not honoured on this backend (the mask is
                # dropped and every lane gathers), so the address is clamped to
                # stay in bounds for rejected lanes and the value is neutralized
                # in registers via tl.where.  An unmasked gather from a clamped
                # address is safe; an unmasked gather from an unclamped address
                # faults the device when a block contains idle lanes.
                value = tl.load(
                    grad_output
                    + base
                    + tl.where(ok_d, od, 0) * (OH * OW)
                    + tl.where(ok_h, oh, 0) * OW
                    + tl.where(ok, ow, 0),
                    mask=ok,
                    other=0.0,
                )
                acc += tl.where(ok, value.to(tl.float32), 0.0)

    tl.store(grad_input + offsets, acc.to(grad_input.dtype.element_ty), mask=mask)


@libentry()
@triton.jit
def _upsample_nearest3d_backward_exact2x_kernel(
    grad_output,
    grad_input,
    total,
    ID,
    IH,
    IW,
    OD,
    OH,
    OW,
    BLOCK: tl.constexpr,
):
    """Fast path for exact 2x upscaling.

    When ``OD == 2*ID`` (and likewise H/W) the forward map is exactly
    ``src = dst // 2``, so every input voxel (id, ih, iw) gathers from the fixed
    2x2x2 output block ``[2*id .. 2*id+1] x [2*ih .. 2*ih+1] x [2*iw .. 2*iw+1]``
    with no candidate scanning and no fp32 boundary checks.  The eight source
    offsets are affine in the decoded input index, which lets the compiler emit
    a far tighter gather than the general candidate-scan kernel.
    """
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < total

    iw = offsets % IW
    tmp = offsets // IW
    ih = tmp % IH
    tmp2 = tmp // IH
    id_ = tmp2 % ID
    nc = tmp2 // ID

    # Idle lanes (past ``total``) decode to garbage; clamp every decoded index
    # so their gather address stays in bounds (their result is discarded by the
    # store mask / over-allocation).
    nc = tl.where(mask, nc, 0)
    id_ = tl.where(mask, id_, 0)
    ih = tl.where(mask, ih, 0)
    iw = tl.where(mask, iw, 0)

    base = nc * (OD * OH * OW)
    od0 = id_ * 2
    oh0 = ih * 2
    ow0 = iw * 2

    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for dd in tl.static_range(2):
        for hh in tl.static_range(2):
            for ww in tl.static_range(2):
                off = base + (od0 + dd) * (OH * OW) + (oh0 + hh) * OW + (ow0 + ww)
                value = tl.load(grad_output + off, mask=mask, other=0.0)
                acc += value.to(tl.float32)

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


def _candidate_count(in_size, out_size, scale):
    """Number of output positions per input position (~ the up-scaling factor)."""
    if scale is not None and scale > 0:
        s = float(scale)
    else:
        s = out_size / in_size
    return max(math.ceil(s), 1) + _EXTRA_CANDIDATES


def _upsample_nearest3d_backward_impl(
    grad_output: torch.Tensor,
    output_size: Tuple[int, int, int],
    input_size: Tuple[int, int, int, int, int],
    scales_d: Optional[float] = None,
    scales_h: Optional[float] = None,
    scales_w: Optional[float] = None,
    grad_input: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    assert grad_output.device.type == device.name
    assert grad_output.ndim == 5, "The ndim of grad_output must be 5"
    assert len(output_size) == 3, "The len of output_size must be 3"
    assert len(input_size) == 5, "The len of input_size must be 5"

    OD, OH, OW = output_size
    N, C, ID, IH, IW = input_size

    assert grad_output.shape == (N, C, OD, OH, OW), (
        f"grad_output shape {grad_output.shape} does not match "
        f"expected shape (N={N}, C={C}, OD={OD}, OH={OH}, OW={OW})"
    )

    grad_output = grad_output.contiguous()

    rec_d = _reciprocal_scale(ID, OD, scales_d)
    rec_h = _reciprocal_scale(IH, OH, scales_h)
    rec_w = _reciprocal_scale(IW, OW, scales_w)
    scale_d = float(scales_d) if scales_d is not None else OD / ID
    scale_h = float(scales_h) if scales_h is not None else OH / IH
    scale_w = float(scales_w) if scales_w is not None else OW / IW

    cand_d = _candidate_count(ID, OD, scales_d)
    cand_h = _candidate_count(IH, OH, scales_h)
    cand_w = _candidate_count(IW, OW, scales_w)

    total = N * C * ID * IH * IW
    if total == 0:
        if grad_input is None:
            return torch.empty(
                (N, C, ID, IH, IW),
                device=grad_output.device,
                dtype=grad_output.dtype,
            )
        return grad_input

    if grad_input is None:
        # Over-allocate to a whole tile so a store mask that is not honoured on
        # this backend still lands inside the allocation, then hand back a view
        # with the requested shape.
        buf = torch.empty(
            triton.cdiv(total, _BLOCK) * _BLOCK,
            device=grad_output.device,
            dtype=grad_output.dtype,
        )
        grad_input = buf[:total].view((N, C, ID, IH, IW))

    grid = (triton.cdiv(total, _BLOCK),)
    exact_2x = (
        scale_d == 2.0
        and scale_h == 2.0
        and scale_w == 2.0
        and OD == 2 * ID
        and OH == 2 * IH
        and OW == 2 * IW
    )
    with torch_device_fn.device(grad_output.device):
        if exact_2x:
            _upsample_nearest3d_backward_exact2x_kernel[grid](
                grad_output,
                grad_input,
                total,
                ID,
                IH,
                IW,
                OD,
                OH,
                OW,
                BLOCK=_BLOCK,
                num_warps=1,
                buffer_size_limit=2048,
            )
        else:
            _upsample_nearest3d_backward_kernel[grid](
                grad_output,
                grad_input,
                total,
                ID,
                IH,
                IW,
                OD,
                OH,
                OW,
                rec_d=rec_d,
                rec_h=rec_h,
                rec_w=rec_w,
                scale_d=scale_d,
                scale_h=scale_h,
                scale_w=scale_w,
                cand_d=cand_d,
                cand_h=cand_h,
                cand_w=cand_w,
                BLOCK=_BLOCK,
                num_warps=1,
                buffer_size_limit=2048,
            )
    return grad_input


def upsample_nearest3d_backward(
    grad_output: torch.Tensor,
    output_size: Tuple[int, int, int],
    input_size: Tuple[int, int, int, int, int],
    scales_d: Optional[float] = None,
    scales_h: Optional[float] = None,
    scales_w: Optional[float] = None,
) -> torch.Tensor:
    logger.debug("GEMS_KUNLUNXIN UPSAMPLE_NEAREST3D_BACKWARD")
    return _upsample_nearest3d_backward_impl(
        grad_output, output_size, input_size, scales_d, scales_h, scales_w
    )


def upsample_nearest3d_backward_grad_input(
    grad_output: torch.Tensor,
    output_size: Tuple[int, int, int],
    input_size: Tuple[int, int, int, int, int],
    scales_d: Optional[float] = None,
    scales_h: Optional[float] = None,
    scales_w: Optional[float] = None,
    grad_input: torch.Tensor = None,
) -> torch.Tensor:
    logger.debug("GEMS_KUNLUNXIN UPSAMPLE_NEAREST3D_BACKWARD_GRAD_INPUT")
    return _upsample_nearest3d_backward_impl(
        grad_output,
        output_size,
        input_size,
        scales_d,
        scales_h,
        scales_w,
        grad_input,
    )
