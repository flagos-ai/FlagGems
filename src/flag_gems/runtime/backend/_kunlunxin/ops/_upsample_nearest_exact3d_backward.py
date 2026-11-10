import math
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from flag_gems.runtime import device, torch_device_fn
from flag_gems.utils import libentry

logger = __import__("logging").getLogger(__name__)

_BLOCK = 1024


@libentry()
@triton.jit
def _upsample_nearest_exact3d_backward_kernel(
    grad_output,
    grad_input,
    total,
    C,
    OD,
    OH,
    OW,
    ID,
    IH,
    IW,
    scale_d: tl.constexpr,
    scale_h: tl.constexpr,
    scale_w: tl.constexpr,
    cand_d: tl.constexpr,
    cand_h: tl.constexpr,
    cand_w: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Gather backward over the *spatial* slice (N is folded into the runtime
    # loop below to keep the grid small); each input voxel sums the output
    # positions whose exact forward map lands on it.  One lane per input voxel.
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < total

    iw = offsets % IW
    tmp = offsets // IW
    ih = tmp % IH
    tmp2 = tmp // IH
    id_ = tmp2 % ID
    c = tmp2 // ID

    base = tl.where(mask, c, 0) * (OD * OH * OW)

    # exact interval: [ceil(i*scale - 0.5), ceil((i+1)*scale - 0.5))
    d_lo = tl.ceil(id_.to(tl.float32) * scale_d - 0.5).to(tl.int32)
    h_lo = tl.ceil(ih.to(tl.float32) * scale_h - 0.5).to(tl.int32)
    w_lo = tl.ceil(iw.to(tl.float32) * scale_w - 0.5).to(tl.int32)
    d_hi = tl.ceil((id_ + 1).to(tl.float32) * scale_d - 0.5).to(tl.int32)
    h_hi = tl.ceil((ih + 1).to(tl.float32) * scale_h - 0.5).to(tl.int32)
    w_hi = tl.ceil((iw + 1).to(tl.float32) * scale_w - 0.5).to(tl.int32)

    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for dd in tl.static_range(cand_d):
        od = d_lo + dd
        ok_d = mask & (od < d_hi)
        for hh in tl.static_range(cand_h):
            oh = h_lo + hh
            ok_h = ok_d & (oh < h_hi)
            for ww in tl.static_range(cand_w):
                ow = w_lo + ww
                ok = ok_h & (ow < w_hi)
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
def _upsample_nearest_exact3d_backward_exact2x_kernel(
    grad_output,
    grad_input,
    total,
    C,
    OD,
    OH,
    OW,
    ID,
    IH,
    IW,
    BLOCK: tl.constexpr,
):
    # exact 2x: src == dst // 2, so each input voxel gathers the fixed 2x2x2
    # output block with no candidate scan / fp32 boundary checks.
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < total

    iw = offsets % IW
    tmp = offsets // IW
    ih = tmp % IH
    tmp2 = tmp // IH
    id_ = tmp2 % ID
    c = tmp2 // ID

    c = tl.where(mask, c, 0)
    id_ = tl.where(mask, id_, 0)
    ih = tl.where(mask, ih, 0)
    iw = tl.where(mask, iw, 0)

    base = c * (OD * OH * OW)
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


@libentry()
@triton.jit
def _upsample_nearest_exact3d_backward_strided_kernel(
    grad_output,
    grad_input,
    numel,
    C,
    OD,
    OH,
    OW,
    ID,
    IH,
    IW,
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
    scale_d: tl.constexpr,
    scale_h: tl.constexpr,
    scale_w: tl.constexpr,
    cand_d: tl.constexpr,
    cand_h: tl.constexpr,
    cand_w: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < numel

    iw = offsets % IW
    rest = offsets // IW
    ih = rest % IH
    rest = rest // IH
    id_ = rest % ID
    rest = rest // ID
    c = rest % C
    n = rest // C

    d_lo = tl.ceil(id_.to(tl.float32) * scale_d - 0.5).to(tl.int32)
    h_lo = tl.ceil(ih.to(tl.float32) * scale_h - 0.5).to(tl.int32)
    w_lo = tl.ceil(iw.to(tl.float32) * scale_w - 0.5).to(tl.int32)
    d_hi = tl.ceil((id_ + 1).to(tl.float32) * scale_d - 0.5).to(tl.int32)
    h_hi = tl.ceil((ih + 1).to(tl.float32) * scale_h - 0.5).to(tl.int32)
    w_hi = tl.ceil((iw + 1).to(tl.float32) * scale_w - 0.5).to(tl.int32)

    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for dd in tl.static_range(cand_d):
        od = d_lo + dd
        ok_d = mask & (od < d_hi)
        for hh in tl.static_range(cand_h):
            oh = h_lo + hh
            ok_h = ok_d & (oh < h_hi)
            for ww in tl.static_range(cand_w):
                ow = w_lo + ww
                ok = ok_h & (ow < w_hi)
                go_offset = (
                    n * go_stride_n
                    + c * go_stride_c
                    + tl.where(ok_d, od, 0) * go_stride_d
                    + tl.where(ok_h, oh, 0) * go_stride_h
                    + tl.where(ok, ow, 0) * go_stride_w
                )
                value = tl.load(grad_output + go_offset, mask=ok, other=0.0)
                acc += tl.where(ok, value.to(tl.float32), 0.0)

    gi_offset = (
        n * gi_stride_n
        + c * gi_stride_c
        + id_ * gi_stride_d
        + ih * gi_stride_h
        + iw * gi_stride_w
    )
    tl.store(grad_input + gi_offset, acc, mask=mask)


def _scale_value(scale, output_size, input_size):
    if scale is not None and scale > 0:
        return float(scale)
    return output_size / input_size


def _candidate_count(scale):
    return max(1, math.ceil(scale)) + 1


def _impl(grad_output, output_size, input_size, scales_d, scales_h, scales_w, grad_input):
    if grad_output.device.type != device.name:
        raise RuntimeError(f"expected grad_output to be on {device.name}")
    if grad_output.ndim != 5 or len(output_size) != 3 or len(input_size) != 5:
        raise RuntimeError("bad ndim/len")
    od, oh, ow = map(int, output_size)
    n, c, id_, ih, iw = map(int, input_size)
    if tuple(grad_output.shape) != (n, c, od, oh, ow):
        raise RuntimeError("grad_output shape mismatch")

    grad_output = grad_output.contiguous()

    if grad_input is None:
        grad_input = torch.empty((n, c, id_, ih, iw), dtype=grad_output.dtype, device=grad_output.device)
    else:
        if tuple(grad_input.shape) != (n, c, id_, ih, iw):
            grad_input.resize_((n, c, id_, ih, iw))

    total = n * c * id_ * ih * iw
    if total == 0:
        return grad_input

    scale_d = _scale_value(scales_d, od, id_)
    scale_h = _scale_value(scales_h, oh, ih)
    scale_w = _scale_value(scales_w, ow, iw)
    cand_d = _candidate_count(scale_d)
    cand_h = _candidate_count(scale_h)
    cand_w = _candidate_count(scale_w)

    exact_2x = (
        scale_d == 2.0 and scale_h == 2.0 and scale_w == 2.0
        and od == 2 * id_ and oh == 2 * ih and ow == 2 * iw
    )
    with torch_device_fn.device(grad_output.device):
        if grad_input.is_contiguous():
            grid = (triton.cdiv(total, _BLOCK),)
            if exact_2x:
                _upsample_nearest_exact3d_backward_exact2x_kernel[grid](
                    grad_output, grad_input, total, c, od, oh, ow, id_, ih, iw,
                    BLOCK=_BLOCK, num_warps=1, buffer_size_limit=2048,
                )
            else:
                _upsample_nearest_exact3d_backward_kernel[grid](
                    grad_output, grad_input, total, c, od, oh, ow, id_, ih, iw,
                    scale_d=scale_d, scale_h=scale_h, scale_w=scale_w,
                    cand_d=cand_d, cand_h=cand_h, cand_w=cand_w,
                    BLOCK=_BLOCK, num_warps=1, buffer_size_limit=2048,
                )
        else:
            grid = (triton.cdiv(grad_input.numel(), _BLOCK),)
            _upsample_nearest_exact3d_backward_strided_kernel[grid](
                grad_output, grad_input, grad_input.numel(), c, od, oh, ow, id_, ih, iw,
                *grad_output.stride(), *grad_input.stride(),
                scale_d=scale_d, scale_h=scale_h, scale_w=scale_w,
                cand_d=cand_d, cand_h=cand_h, cand_w=cand_w,
                BLOCK=_BLOCK, num_warps=1, buffer_size_limit=2048,
            )
    return grad_input


def _upsample_nearest_exact3d_backward(grad_output, output_size, input_size,
                                       scales_d=None, scales_h=None, scales_w=None):
    return _impl(grad_output, output_size, input_size, scales_d, scales_h, scales_w, None)


def _upsample_nearest_exact3d_backward_grad_input(grad_output, output_size, input_size,
                                                 scales_d=None, scales_h=None, scales_w=None,
                                                 *, grad_input=None):
    return _impl(grad_output, output_size, input_size, scales_d, scales_h, scales_w, grad_input)
