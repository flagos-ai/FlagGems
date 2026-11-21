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
import struct
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from flag_gems.runtime import device, torch_device_fn

logger = logging.getLogger(__name__)

# One lane per grad_input element; the grid is (spatial tile, batch*channel) so
# the batch and channel axes stay parallel instead of being looped per program.
_BLOCK = 256
_FP64_BLOCK = 128


@triton.jit
def _upsample_nearest_exact3d_backward_kernel(
    go_ptr,
    gi_ptr,
    IN,  # ID * IH * IW
    IW,
    IH,
    ID,
    OD,
    OH,
    OW,
    MAX_OFF,
    scale_d,
    scale_h,
    scale_w,
    MAX_D: tl.constexpr,
    MAX_H: tl.constexpr,
    MAX_W: tl.constexpr,
    IS_FP64: tl.constexpr,
    IS_BYTE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    nc = tl.program_id(1)

    sp = pid * BLOCK + tl.arange(0, BLOCK)
    valid = sp < IN

    iw = sp % IW
    rest = sp // IW
    ih = rest % IH
    id_ = rest // IH

    # PyTorch's exact-nearest backward maps an input coordinate to the half-open
    # output interval [ceil(i * scale - 0.5), ceil((i + 1) * scale - 0.5)),
    # clamped at the output size; the last input coordinate absorbs every
    # remaining output position because the forward source index is clamped to
    # input_size - 1.
    start_d = tl.minimum(tl.ceil(id_.to(tl.float32) * scale_d - 0.5), OD).to(tl.int32)
    end_d = tl.minimum(tl.ceil((id_ + 1).to(tl.float32) * scale_d - 0.5), OD).to(
        tl.int32
    )
    end_d = tl.maximum(end_d, (id_ >= ID - 1).to(tl.int32) * OD)
    start_h = tl.minimum(tl.ceil(ih.to(tl.float32) * scale_h - 0.5), OH).to(tl.int32)
    end_h = tl.minimum(tl.ceil((ih + 1).to(tl.float32) * scale_h - 0.5), OH).to(
        tl.int32
    )
    end_h = tl.maximum(end_h, (ih >= IH - 1).to(tl.int32) * OH)
    start_w = tl.minimum(tl.ceil(iw.to(tl.float32) * scale_w - 0.5), OW).to(tl.int32)
    end_w = tl.minimum(tl.ceil((iw + 1).to(tl.float32) * scale_w - 0.5), OW).to(
        tl.int32
    )
    end_w = tl.maximum(end_w, (iw >= IW - 1).to(tl.int32) * OW)

    if IS_FP64:
        acc = tl.zeros((BLOCK,), dtype=tl.float64)
    elif IS_BYTE:
        acc = tl.zeros((BLOCK,), dtype=tl.int32)
    else:
        acc = tl.zeros((BLOCK,), dtype=tl.float32)

    out_nc = nc * (OD * OH * OW)
    ohw = OH * OW
    for kd in tl.static_range(MAX_D):
        od = start_d + kd
        mask_d = od < end_d
        base_d = out_nc + od * ohw
        for kh in tl.static_range(MAX_H):
            oh = start_h + kh
            mask_h = mask_d & (oh < end_h)
            base = base_d + oh * OW + start_w
            for kw in tl.static_range(MAX_W):
                m = mask_h & (start_w + kw < end_w)
                # Clamped flat offset keeps a switched-off lane inside the
                # allocation; its value is neutralised in registers.
                off = tl.minimum(base + kw, MAX_OFF)
                v = tl.load(go_ptr + off)
                if IS_BYTE:
                    acc += tl.where(m, v.to(tl.int32), 0)
                else:
                    acc += tl.where(m, v, 0.0)

    tl.store(gi_ptr + (nc * IN + sp), acc, mask=valid)


@triton.jit
def _upsample_nearest_exact3d_backward_strided_kernel(
    go_ptr,
    gi_ptr,
    numel,
    N,
    C,
    ID,
    IH,
    IW,
    OD,
    OH,
    OW,
    gs_n,
    gs_c,
    gs_d,
    gs_h,
    gs_w,
    is_n,
    is_c,
    is_d,
    is_h,
    is_w,
    scale_d,
    scale_h,
    scale_w,
    MAX_D: tl.constexpr,
    MAX_H: tl.constexpr,
    MAX_W: tl.constexpr,
    IS_FP64: tl.constexpr,
    IS_BYTE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = offsets < numel

    iw = offsets % IW
    rest = offsets // IW
    ih = rest % IH
    rest = rest // IH
    id_ = rest % ID
    rest = rest // ID
    c = rest % C
    n = tl.minimum(rest // C, N - 1)

    start_d = tl.minimum(tl.ceil(id_.to(tl.float32) * scale_d - 0.5), OD).to(tl.int64)
    end_d = tl.minimum(tl.ceil((id_ + 1).to(tl.float32) * scale_d - 0.5), OD).to(
        tl.int64
    )
    end_d = tl.maximum(end_d, (id_ >= ID - 1).to(tl.int64) * OD)
    start_h = tl.minimum(tl.ceil(ih.to(tl.float32) * scale_h - 0.5), OH).to(tl.int64)
    end_h = tl.minimum(tl.ceil((ih + 1).to(tl.float32) * scale_h - 0.5), OH).to(
        tl.int64
    )
    end_h = tl.maximum(end_h, (ih >= IH - 1).to(tl.int64) * OH)
    start_w = tl.minimum(tl.ceil(iw.to(tl.float32) * scale_w - 0.5), OW).to(tl.int64)
    end_w = tl.minimum(tl.ceil((iw + 1).to(tl.float32) * scale_w - 0.5), OW).to(
        tl.int64
    )
    end_w = tl.maximum(end_w, (iw >= IW - 1).to(tl.int64) * OW)

    if IS_FP64:
        acc = tl.zeros((BLOCK,), dtype=tl.float64)
    elif IS_BYTE:
        acc = tl.zeros((BLOCK,), dtype=tl.int32)
    else:
        acc = tl.zeros((BLOCK,), dtype=tl.float32)

    go_base = n * gs_n + c * gs_c
    for kd in tl.static_range(MAX_D):
        od = start_d + kd
        mask_d = od < end_d
        od_c = tl.minimum(od, OD - 1)
        for kh in tl.static_range(MAX_H):
            oh = start_h + kh
            mask_h = mask_d & (oh < end_h)
            oh_c = tl.minimum(oh, OH - 1)
            for kw in tl.static_range(MAX_W):
                ow = start_w + kw
                m = mask_h & (ow < end_w)
                off = (
                    go_base + od_c * gs_d + oh_c * gs_h + tl.minimum(ow, OW - 1) * gs_w
                )
                v = tl.load(go_ptr + off)
                if IS_BYTE:
                    acc += tl.where(m, v.to(tl.int32), 0)
                else:
                    acc += tl.where(m, v, 0.0)

    gi_off = n * is_n + c * is_c + id_ * is_d + ih * is_h + iw * is_w
    tl.store(gi_ptr + gi_off, acc, mask=valid)


def _scale_value(scale: Optional[float], output_size: int, input_size: int) -> float:
    # compute_scales_value_backwards<float> in PyTorch uses the supplied
    # forward scale directly and falls back for absent/non-positive scales.
    if scale is not None and scale > 0:
        return float(scale)
    return output_size / input_size


def _float32(value: float) -> float:
    return struct.unpack("f", struct.pack("f", value))[0]


def _max_contributors(scale: float, input_size: int, output_size: int) -> int:
    """Per-dimension contributor bound, including the clamped last index.

    ``ceil((i + 1) * scale - 0.5) - ceil(i * scale - 0.5)`` is at most
    ``ceil(scale)``; the only entry that can exceed that is the last source
    index, whose interval end is forced to ``output_size``.
    """
    last_start = math.ceil(
        _float32(_float32(_float32(input_size - 1) * _float32(scale)) - 0.5)
    )
    last_start = min(max(last_start, 0), output_size)
    return max(1, math.ceil(scale), output_size - last_start)


def _upsample_nearest_exact3d_backward_impl(
    grad_output: torch.Tensor,
    output_size: Tuple[int, int, int],
    input_size: Tuple[int, int, int, int, int],
    scales_d: Optional[float],
    scales_h: Optional[float],
    scales_w: Optional[float],
    grad_input: Optional[torch.Tensor] = None,
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

    numel = grad_input.numel()
    if numel == 0 or grad_output.numel() == 0:
        return grad_input

    scale_d = _scale_value(scales_d, od, id_)
    scale_h = _scale_value(scales_h, oh, ih)
    scale_w = _scale_value(scales_w, ow, iw)

    max_d = _max_contributors(scale_d, id_, od)
    max_h = _max_contributors(scale_h, ih, oh)
    max_w = _max_contributors(scale_w, iw, ow)
    block = _FP64_BLOCK if grad_output.dtype == torch.float64 else _BLOCK
    block = min(block, triton.next_power_of_2(max(1, numel)))

    with torch_device_fn.device(grad_output.device):
        if grad_output.is_contiguous() and grad_input.is_contiguous():
            inner = id_ * ih * iw
            grid = (triton.cdiv(inner, block), n * c)
            _upsample_nearest_exact3d_backward_kernel[grid](
                grad_output,
                grad_input,
                inner,
                iw,
                ih,
                id_,
                od,
                oh,
                ow,
                grad_output.numel() - 1,
                scale_d,
                scale_h,
                scale_w,
                max_d,
                max_h,
                max_w,
                grad_output.dtype == torch.float64,
                grad_output.dtype == torch.uint8,
                block,
            )
        else:
            grid = (triton.cdiv(numel, block),)
            _upsample_nearest_exact3d_backward_strided_kernel[grid](
                grad_output,
                grad_input,
                numel,
                n,
                c,
                id_,
                ih,
                iw,
                od,
                oh,
                ow,
                *grad_output.stride(),
                *grad_input.stride(),
                scale_d,
                scale_h,
                scale_w,
                max_d,
                max_h,
                max_w,
                grad_output.dtype == torch.float64,
                grad_output.dtype == torch.uint8,
                block,
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
        grad_output,
        output_size,
        input_size,
        scales_d,
        scales_h,
        scales_w,
        None,
    )


# The shared module also exposes this entry point; keep both namespaces on the
# same implementation.
import sys as _sys  # noqa: E402

_generic_module = _sys.modules.get("flag_gems.ops._upsample_nearest_exact3d_backward")
if _generic_module is not None:
    setattr(
        _generic_module,
        "_upsample_nearest_exact3d_backward",
        _upsample_nearest_exact3d_backward,
    )
