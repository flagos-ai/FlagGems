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
import warnings

import numpy as np
import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)

_MAX_STATIC_SPAN = 32


def _src_map(input_length, output_length, scale):
    if input_length == output_length:
        return np.arange(output_length, dtype=np.int64)
    if output_length == 2 * input_length:
        return np.arange(output_length, dtype=np.int64) >> 1
    ratio = (
        np.float32(input_length / output_length)
        if scale is None
        else np.float32(1.0 / scale)
    )
    idx = np.arange(output_length, dtype=np.float32) * ratio
    return np.minimum(idx.astype(np.int64), input_length - 1)


def _windows(input_length, output_length, scale):
    src = _src_map(input_length, output_length, scale)
    lo = np.zeros(input_length, dtype=np.int32)
    length = np.zeros(input_length, dtype=np.int32)
    for out_index, in_index in enumerate(src.tolist()):
        if length[in_index] == 0:
            lo[in_index] = out_index
        length[in_index] += 1
    return lo, length, int(length.max()) if input_length else 0


@triton.jit
def _vec_kernel(
    GO,
    GI,
    HLO,
    HLEN,
    WLO,
    WLEN,
    TOTAL,
    C,
    IH,
    IW,
    OH,
    OW,
    GO_N,
    GO_C,
    GO_H,
    GO_W,
    GI_N,
    GI_C,
    GI_H,
    GI_W,
    IS_INT: tl.constexpr,
    MAX_H: tl.constexpr,
    MAX_W: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    index = pid * BLOCK + tl.arange(0, BLOCK)
    valid = index < TOTAL
    idx = tl.where(valid, index, 0)
    iw_i = idx % IW
    tmp = idx // IW
    ih_i = tmp % IH
    tmp2 = tmp // IH
    c_i = tmp2 % C
    n_i = tmp2 // C

    out_off = n_i * GI_N + c_i * GI_C + ih_i * GI_H + iw_i * GI_W
    base = n_i * GO_N + c_i * GO_C

    hlo = tl.load(HLO + ih_i)
    hlen = tl.load(HLEN + ih_i)
    wlo = tl.load(WLO + iw_i)
    wlen = tl.load(WLEN + iw_i)

    if IS_INT:
        acc = tl.zeros((BLOCK,), dtype=tl.int32)
    else:
        acc = tl.zeros((BLOCK,), dtype=tl.float32)

    for dy in tl.static_range(MAX_H):
        oy = tl.minimum(hlo + dy, OH - 1)
        for dx in tl.static_range(MAX_W):
            ox = tl.minimum(wlo + dx, OW - 1)
            v = tl.load(GO + base + oy * GO_H + ox * GO_W)
            keep = (dy < hlen) & (dx < wlen)
            acc += tl.where(keep, v.to(acc.dtype), tl.zeros_like(acc))

    tl.store(GI + out_off, acc.to(GI.dtype.element_ty), mask=valid)


@triton.jit
def _scalar_kernel(
    GO,
    GI,
    HLO,
    HLEN,
    WLO,
    WLEN,
    TOTAL,
    C,
    IH,
    IW,
    OH,
    OW,
    GO_N,
    GO_C,
    GO_H,
    GO_W,
    GI_N,
    GI_C,
    GI_H,
    GI_W,
    IS_INT: tl.constexpr,
):
    start = tl.program_id(0).to(tl.int64)
    step = tl.num_programs(0).to(tl.int64)
    for index in range(start, TOTAL, step):
        iw_i = index % IW
        tmp = index // IW
        ih_i = tmp % IH
        tmp2 = tmp // IH
        c_i = tmp2 % C
        n_i = tmp2 // C

        out_off = n_i * GI_N + c_i * GI_C + ih_i * GI_H + iw_i * GI_W
        base = n_i * GO_N + c_i * GO_C

        hlo = tl.load(HLO + ih_i)
        hlen = tl.load(HLEN + ih_i)
        wlo = tl.load(WLO + iw_i)
        wlen = tl.load(WLEN + iw_i)

        if IS_INT:
            acc = tl.full((), 0, tl.int32)
        else:
            acc = tl.full((), 0, tl.float32)
        for dy in range(hlen):
            oy = (hlo + dy).to(tl.int64)
            row = base + oy * GO_H
            for dx in range(wlen):
                ox = (wlo + dx).to(tl.int64)
                acc += tl.load(GO + row + ox * GO_W).to(acc.dtype)
        tl.store(GI + out_off, acc.to(GI.dtype.element_ty))


def _dispatch(grad_output, output_size, input_size, scales_h, scales_w, grad_input):
    if len(output_size) != 2 or len(input_size) != 4:
        raise RuntimeError("output_size must have length 2 and input_size length 4")
    n, c, ih, iw = (int(x) for x in input_size)
    oh, ow = (int(x) for x in output_size)
    if min(ih, iw, oh, ow) <= 0 or n < 0 or c < 0:
        raise RuntimeError("Input and output spatial sizes must be greater than 0")
    if grad_output.ndim != 4:
        raise RuntimeError("Expected grad_output to be a tensor of dimension 4")
    if tuple(grad_output.shape) != (n, c, oh, ow):
        raise RuntimeError("Expected grad_output to have the same shape as output")
    for scale in (scales_h, scales_w):
        if scale is not None and not scale > 0:
            raise RuntimeError("Explicit scales must be greater than 0")
    if grad_output.dtype not in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
        torch.uint8,
    ):
        raise RuntimeError("upsample_nearest2d_backward received an unsupported dtype")
    if grad_input is None:
        if grad_output.is_contiguous(memory_format=torch.channels_last):
            grad_input = torch.empty_strided(
                (n, c, ih, iw),
                (c * ih * iw, 1, c * iw, c),
                dtype=grad_output.dtype,
                device=grad_output.device,
            )
        else:
            grad_input = torch.empty(
                (n, c, ih, iw), dtype=grad_output.dtype, device=grad_output.device
            )
    else:
        if grad_input.dtype != grad_output.dtype:
            raise RuntimeError(
                f"Expected out tensor to have dtype {grad_output.dtype}, "
                f"but got {grad_input.dtype} instead"
            )
        if grad_input.device != grad_output.device:
            raise RuntimeError(
                f"Expected out tensor to have device {grad_output.device}, "
                f"but got {grad_input.device} instead"
            )
        if tuple(grad_input.shape) != (n, c, ih, iw):
            if grad_input.numel():
                warnings.warn(
                    "An output with one or more elements was resized since it "
                    "had a different shape from the required output shape.",
                    UserWarning,
                    stacklevel=3,
                )
            grad_input.resize_((n, c, ih, iw))

    total = n * c * ih * iw
    if total == 0:
        return grad_input

    hlo_np, hlen_np, max_h = _windows(ih, oh, scales_h)
    wlo_np, wlen_np, max_w = _windows(iw, ow, scales_w)
    dev = grad_output.device
    hlo = torch.from_numpy(hlo_np).to(dev)
    hlen = torch.from_numpy(hlen_np).to(dev)
    wlo = torch.from_numpy(wlo_np).to(dev)
    wlen = torch.from_numpy(wlen_np).to(dev)

    is_int = grad_output.dtype == torch.uint8

    int64_index = any(
        sum((size - 1) * stride for size, stride in zip(t.shape, t.stride()))
        > torch.iinfo(torch.int32).max
        for t in (grad_output, grad_input)
    )

    go_stride = grad_output.stride()
    gi_stride = grad_input.stride()

    with torch_device_fn.device(dev):
        if not int64_index and max_h * max_w <= _MAX_STATIC_SPAN:
            block = 512
            grid = (triton.cdiv(total, block),)
            _vec_kernel[grid](
                grad_output,
                grad_input,
                hlo,
                hlen,
                wlo,
                wlen,
                total,
                c,
                ih,
                iw,
                oh,
                ow,
                *go_stride,
                *gi_stride,
                IS_INT=is_int,
                MAX_H=max(max_h, 1),
                MAX_W=max(max_w, 1),
                BLOCK=block,
            )
        else:
            grid = (min(total, 65535),)
            _scalar_kernel[grid](
                grad_output,
                grad_input,
                hlo,
                hlen,
                wlo,
                wlen,
                total,
                c,
                ih,
                iw,
                oh,
                ow,
                *go_stride,
                *gi_stride,
                IS_INT=is_int,
            )
    return grad_input


def upsample_nearest2d_backward(
    grad_output, output_size, input_size, scales_h=None, scales_w=None
):
    logger.debug("GEMS_KUNLUNXIN UPSAMPLE_NEAREST2D_BACKWARD")
    return _dispatch(grad_output, output_size, input_size, scales_h, scales_w, None)


def upsample_nearest2d_backward_grad_input(
    grad_output,
    output_size,
    input_size,
    scales_h=None,
    scales_w=None,
    *,
    grad_input,
):
    logger.debug("GEMS_KUNLUNXIN UPSAMPLE_NEAREST2D_BACKWARD.GRAD_INPUT")
    return _dispatch(
        grad_output, output_size, input_size, scales_h, scales_w, grad_input
    )


def _install_into_flag_gems_ops():
    try:
        import flag_gems.ops as _fg_ops
    except Exception:  # pragma: no cover - defensive; import ordering
        return
    _fg_ops.upsample_nearest2d_backward = upsample_nearest2d_backward
    _fg_ops.upsample_nearest2d_backward_grad_input = (
        upsample_nearest2d_backward_grad_input
    )


_install_into_flag_gems_ops()
