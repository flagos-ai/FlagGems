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
import warnings
from functools import lru_cache

import torch
import triton
import triton.language as tl

from flag_gems.runtime import device as runtime_device
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

# These native backends use ceil intervals. CPU instead transposes its legacy
# forward floor map, which differs when explicit scales disagree with size.
_CEIL_BACKWARD = runtime_device.vendor_name in (
    "nvidia",
    "ascend",
    "mthreads",
    "hygon",
    "iluvatar",
)


# Cache the installed CANN version once. Direct aclnn calls in 8.5.0 ignore
# explicit scales; 9.0.0 honors them. This is independent of torch-npu version.
_ASCEND_SIZE_ONLY = False
if runtime_device.vendor_name == "ascend":
    from torch_npu.npu.utils import get_cann_version

    _ASCEND_SIZE_ONLY = get_cann_version() == "8.5.0"


@lru_cache(maxsize=256)
def _ceil_span_bound(scale, input_length, output_length):
    scale = struct.unpack("f", struct.pack("f", scale))[0]
    maximum = min(input_length * scale, output_length)
    if scale.is_integer() and maximum <= 2**24:
        return min(int(scale), output_length)
    # Input coordinates below 2^20 are exactly representable in FP32. The
    # difference of the two rounded products adds at most one product ULP.
    product_ulp = math.ldexp(1.0, math.frexp(maximum)[1] - 24)
    return min(math.ceil(scale + product_ulp), output_length)


@lru_cache(maxsize=16)
def _ascend_vector_cores(device_index):
    properties = triton.runtime.driver.active.utils.get_device_properties(device_index)
    return properties["num_vectorcore"]


@libentry()
@triton.jit
def _nearest2d_backward_ascend_copy(GO, GI, TOTAL: tl.constexpr, BLOCK: tl.constexpr):
    tiles: tl.constexpr = triton.cdiv(TOTAL, BLOCK)
    count = tl.cdiv(tiles, tl.num_programs(0))
    first = tl.program_id(0) * count
    for tile in range(first, tl.minimum(first + count, tiles)):
        index = tile * BLOCK + tl.arange(0, BLOCK)
        mask = index.to(tl.float32) < TOTAL
        value = tl.load(GO + index, mask, other=0)
        tl.store(GI + index, value, mask)


@libentry()
@triton.jit
def _nearest2d_backward_ascend_rows(
    GO,
    GI,
    PLANES: tl.constexpr,
    IH: tl.constexpr,
    IW: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    SCALE_H: tl.constexpr,
    SCALE_W: tl.constexpr,
    H_STEPS: tl.constexpr,
    W_STEPS: tl.constexpr,
    ROWS: tl.constexpr,
    BLOCK: tl.constexpr,
    PATCH: tl.constexpr,
):
    # Load a contiguous group of output rows once, then gather the short
    # inverse windows in local memory. Each core owns consecutive row tiles.
    tiles: tl.constexpr = triton.cdiv(IH, ROWS)
    tasks: tl.constexpr = PLANES * tiles
    count = tl.cdiv(tasks, tl.num_programs(0))
    first = tl.program_id(0) * count
    index = tl.arange(0, BLOCK)
    index_f = index.to(tl.float32)
    row = tl.floor(tl.div_rn(index_f, IW))
    column = index_f - row * IW
    patch_index = tl.arange(0, PATCH)
    for task in range(first, tl.minimum(first + count, tasks)):
        plane = task // tiles
        first_row = (task - plane * tiles) * ROWS
        ih = first_row + row
        # An affine prefix lets the compiler shorten the physical store.
        # A row-coordinate mask instead reads and rewrites neighboring tiles.
        valid = index < tl.minimum(ROWS, IH - first_row) * IW
        h0 = tl.minimum(tl.ceil(ih * SCALE_H), OH)
        h1 = tl.minimum(tl.ceil((ih.to(tl.int32) + 1).to(tl.float32) * SCALE_H), OH)
        w0 = tl.minimum(tl.ceil(column * SCALE_W), OW)
        w1 = tl.minimum(tl.ceil((column.to(tl.int32) + 1).to(tl.float32) * SCALE_W), OW)
        patch_row = tl.minimum(tl.ceil(first_row.to(tl.float32) * SCALE_H), OH)
        if PATCH == 1:
            values = tl.load(GO + plane * OH * OW).to(tl.float32)
        else:
            values = tl.load(
                GO + plane * OH * OW + patch_row.to(tl.int32) * OW + patch_index,
                patch_index < (OH - patch_row.to(tl.int32)) * OW,
                other=0,
            ).to(tl.float32)
        acc = tl.full((BLOCK,), 0, tl.float32)
        for dy in tl.static_range(H_STEPS):
            for dx in tl.static_range(W_STEPS):
                oy = h0 + dy
                ox = w0 + dx
                local = ((oy - patch_row) * OW + ox).to(tl.int32)
                local = tl.minimum(tl.maximum(local, 0), PATCH - 1)
                if PATCH == 1:
                    value = values
                else:
                    value = tl.gather(values, local, 0)
                acc += tl.where(valid & (oy < h1) & (ox < w1), value, 0.0)
        tl.store(GI + plane * IH * IW + first_row * IW + index, acc, valid)


@libentry()
@triton.jit
def _nearest2d_backward_ascend_reduce(
    GO,
    GI,
    PLANES: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    SCALE_H: tl.constexpr,
    SCALE_W: tl.constexpr,
    BLOCK: tl.constexpr,
):
    count = tl.cdiv(PLANES, tl.num_programs(0))
    first = tl.program_id(0) * count
    index = tl.arange(0, BLOCK)
    index_f = index.to(tl.float32)
    row = tl.floor(tl.div_rn(index_f, OW))
    column = index_f - row * OW
    mask = (row < tl.ceil(SCALE_H)) & (column < tl.ceil(SCALE_W)) & (index_f < OH * OW)
    for plane in range(first, tl.minimum(first + count, PLANES)):
        value = tl.load(GO + plane * OH * OW + index, mask, other=0).to(tl.float32)
        tl.store(GI + plane, tl.sum(value, 0))


@triton.jit
def _nearest2d_backward_bound(
    index,
    INPUT: tl.constexpr,
    OUTPUT: tl.constexpr,
    SCALE: tl.constexpr,
    CEIL_BACKWARD: tl.constexpr,
):
    if CEIL_BACKWARD:
        bound = tl.minimum(tl.ceil(index.to(tl.float32) * SCALE), OUTPUT).to(
            index.dtype
        )
    elif INPUT == OUTPUT:
        bound = index.to(index.dtype)
    elif OUTPUT == 2 * INPUT:
        bound = index.to(index.dtype) * 2
    elif SCALE == 0:
        bound = tl.where(index == 0, 0, OUTPUT).to(index.dtype)
    else:
        # Invert the actual float32 forward map, including rounding at integer
        # boundaries, rather than assuming that reciprocal scales are exact.
        bound = tl.minimum(tl.ceil(tl.div_rn(index.to(tl.float32), SCALE)), OUTPUT).to(
            index.dtype
        )
        left = (bound > 0) & ((bound - 1).to(tl.float32) * SCALE >= index)
        while tl.max(left.to(tl.int32), 0) != 0:
            bound -= left.to(index.dtype)
            left = (bound > 0) & ((bound - 1).to(tl.float32) * SCALE >= index)
        right = (bound < OUTPUT) & (bound.to(tl.float32) * SCALE < index)
        while tl.max(right.to(tl.int32), 0) != 0:
            bound += right.to(index.dtype)
            right = (bound < OUTPUT) & (bound.to(tl.float32) * SCALE < index)
        # Forward clamps every remaining output pixel to the last input pixel.
        bound = tl.where(index == INPUT, OUTPUT, bound)
    return bound


@libentry()
@triton.jit
def _upsample_nearest2d_backward_kernel(
    GO,
    GI,
    TOTAL: tl.constexpr,
    C: tl.constexpr,
    IH: tl.constexpr,
    IW: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    GO_N: tl.constexpr,
    GO_C: tl.constexpr,
    GO_H: tl.constexpr,
    GO_W: tl.constexpr,
    GI_N: tl.constexpr,
    GI_C: tl.constexpr,
    GI_H: tl.constexpr,
    GI_W: tl.constexpr,
    SCALE_H: tl.constexpr,
    SCALE_W: tl.constexpr,
    CEIL_BACKWARD: tl.constexpr,
    CHANNELS_LAST: tl.constexpr,
    IDENTITY: tl.constexpr,
    INT64_INDEX: tl.constexpr,
    H_STEPS: tl.constexpr,
    W_STEPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    if INT64_INDEX:
        pid = pid.to(tl.int64)
    index = pid * BLOCK + tl.arange(0, BLOCK)
    valid = index < TOTAL
    if CHANNELS_LAST:
        channel = index % C
        iw = index // C % IW
        ih = index // (C * IW) % IH
        batch = index // (C * IW * IH)
    else:
        iw = index % IW
        ih = index // IW % IH
        channel = index // (IW * IH) % C
        batch = index // (C * IW * IH)
    output_offset = batch * GI_N + channel * GI_C + ih * GI_H + iw * GI_W
    input_base = batch * GO_N + channel * GO_C
    if IDENTITY:
        value = tl.load(GO + input_base + ih * GO_H + iw * GO_W, valid, other=0)
        tl.store(GI + output_offset, value, valid)
    else:
        h0 = _nearest2d_backward_bound(ih, IH, OH, SCALE_H, CEIL_BACKWARD)
        h1 = _nearest2d_backward_bound(ih + 1, IH, OH, SCALE_H, CEIL_BACKWARD)
        w0 = _nearest2d_backward_bound(iw, IW, OW, SCALE_W, CEIL_BACKWARD)
        w1 = _nearest2d_backward_bound(iw + 1, IW, OW, SCALE_W, CEIL_BACKWARD)
        if GI.dtype.element_ty == tl.float64:
            acc = tl.full((BLOCK,), 0, tl.float64)
        elif GI.dtype.element_ty == tl.uint8:
            acc = tl.full((BLOCK,), 0, tl.int32)
        else:
            acc = tl.full((BLOCK,), 0, tl.float32)
        # Each input pixel has a single writer. The traversal is deterministic
        # and accumulates low precision inputs in FP32 before the final cast.
        if H_STEPS > 0:
            for step in tl.static_range(H_STEPS * W_STEPS):
                oy = h0 + step // W_STEPS
                ox = w0 + step % W_STEPS
                value = tl.load(
                    GO + input_base + oy * GO_H + ox * GO_W,
                    valid & (oy < h1) & (ox < w1),
                    other=0,
                )
                acc += value.to(acc.dtype)
        else:
            h_count = tl.max(tl.where(valid, h1 - h0, 0), 0)
            w_count = tl.max(tl.where(valid, w1 - w0, 0), 0)
            for dy in range(h_count):
                oy = h0 + dy
                for dx in range(w_count):
                    ox = w0 + dx
                    value = tl.load(
                        GO + input_base + oy * GO_H + ox * GO_W,
                        valid & (oy < h1) & (ox < w1),
                        other=0,
                    )
                    acc += value.to(acc.dtype)
        tl.store(GI + output_offset, acc, valid)


@libentry()
@triton.jit
def _upsample_nearest2d_backward_scalar_kernel(
    GO,
    GI,
    TOTAL: tl.constexpr,
    C: tl.constexpr,
    IH: tl.constexpr,
    IW: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    GO_N: tl.constexpr,
    GO_C: tl.constexpr,
    GO_H: tl.constexpr,
    GO_W: tl.constexpr,
    GI_N: tl.constexpr,
    GI_C: tl.constexpr,
    GI_H: tl.constexpr,
    GI_W: tl.constexpr,
    SCALE_H: tl.constexpr,
    SCALE_W: tl.constexpr,
    IDENTITY: tl.constexpr,
):
    # Keep address arithmetic scalar where the backend vector gather
    # fails on the validated sparse strides above 32 bits.
    for index in range(tl.program_id(0).to(tl.int64), TOTAL, tl.num_programs(0)):
        iw = index % IW
        ih = index // IW % IH
        channel = index // (IW * IH) % C
        batch = index // (C * IW * IH)
        input_base = GO + batch * GO_N + channel * GO_C
        output = GI + batch * GI_N + channel * GI_C + ih * GI_H + iw * GI_W
        if IDENTITY:
            value = tl.load(input_base + ih * GO_H + iw * GO_W)
        else:
            h0 = _nearest2d_backward_bound(ih, IH, OH, SCALE_H, True)
            h1 = _nearest2d_backward_bound(ih + 1, IH, OH, SCALE_H, True)
            w0 = _nearest2d_backward_bound(iw, IW, OW, SCALE_W, True)
            w1 = _nearest2d_backward_bound(iw + 1, IW, OW, SCALE_W, True)
            if GI.dtype.element_ty == tl.float64:
                value = tl.full((), 0, tl.float64)
            elif GI.dtype.element_ty == tl.uint8:
                value = tl.full((), 0, tl.int32)
            else:
                value = tl.full((), 0, tl.float32)
            for oy in range(h0, h1):
                for ox in range(w0, w1):
                    value += tl.load(input_base + oy * GO_H + ox * GO_W).to(value.dtype)
        tl.store(output, value)


def _upsample_nearest2d_backward(
    grad_output, output_size, input_size, scales_h, scales_w, grad_input
):
    if len(output_size) != 2 or len(input_size) != 4:
        raise RuntimeError("output_size must have length 2 and input_size length 4")
    n, c, ih, iw = input_size
    oh, ow = output_size
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
    if runtime_device.vendor_name == "iluvatar" and grad_output.dtype == torch.float64:
        raise RuntimeError("Iluvatar does not support FP64 backward arithmetic")
    if grad_output.device.type != runtime_device.name:
        raise RuntimeError("grad_output must be on the active accelerator backend")
    if grad_input is None:
        if grad_output.is_contiguous(memory_format=torch.channels_last):
            # Explicit strides also work on backends whose tensor factories
            # reject the channels_last memory_format keyword.
            grad_input = torch.empty_strided(
                input_size,
                (c * ih * iw, 1, c * iw, c),
                dtype=grad_output.dtype,
                device=grad_output.device,
            )
        else:
            grad_input = torch.empty(
                input_size, dtype=grad_output.dtype, device=grad_output.device
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
        if tuple(grad_input.shape) != tuple(input_size):
            if grad_input.numel():
                warnings.warn(
                    "An output with one or more elements was resized since it "
                    "had a different shape from the required output shape.",
                    UserWarning,
                    stacklevel=3,
                )
            grad_input.resize_(input_size)
    total = n * c * ih * iw
    if total == 0:
        return grad_input
    if _CEIL_BACKWARD:
        scale_h = oh / ih if scales_h is None or _ASCEND_SIZE_ONLY else scales_h
        scale_w = ow / iw if scales_w is None or _ASCEND_SIZE_ONLY else scales_w
    else:
        scale_h = 1.0 / scales_h if scales_h is not None else ih / oh
        scale_w = 1.0 / scales_w if scales_w is not None else iw / ow
    # Infinity is a positive scale: bound every nonzero source coordinate to
    # the output extent without passing infinity through an integer cast.
    if _CEIL_BACKWARD:
        scale_h = min(scale_h, float(oh))
        scale_w = min(scale_w, float(ow))
    int64_index = any(
        sum((size - 1) * stride for size, stride in zip(t.shape, t.stride()))
        > torch.iinfo(torch.int32).max
        for t in (grad_output, grad_input)
    )
    identity = (ih, iw) == (oh, ow)
    if (
        runtime_device.vendor_name == "ascend"
        and not _ASCEND_SIZE_ONLY
        and not int64_index
        and total <= 2**24
        and max(ih, iw, oh, ow) <= 4096
        and grad_output.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and grad_output.is_contiguous()
        and grad_input.is_contiguous()
    ):
        cores = _ascend_vector_cores(grad_output.device.index)
        with torch_device_fn.device(grad_output.device):
            if identity:
                block = 4096
                _nearest2d_backward_ascend_copy[
                    (min(cores, triton.cdiv(total, block)),)
                ](grad_output, grad_input, total, block)
                return grad_input
            if ih == 1 and iw == 1 and oh * ow <= 16384:
                _nearest2d_backward_ascend_reduce[(min(cores, n * c),)](
                    grad_output,
                    grad_input,
                    n * c,
                    oh,
                    ow,
                    scale_h,
                    scale_w,
                    triton.next_power_of_2(oh * ow),
                )
                return grad_input
            h_steps = _ceil_span_bound(scale_h, ih, oh)
            w_steps = _ceil_span_bound(scale_w, iw, ow)
            patch_limit = 4096 if grad_output.dtype == torch.float32 else 8192
            rows = min(16, triton.next_power_of_2(ih))
            patch = triton.next_power_of_2(min(oh, rows * h_steps) * ow)
            block = triton.next_power_of_2(rows * iw)
            # The installed compiler enables multiple buffers automatically.
            # These measured bounds keep the live row and patch tiles in UB.
            while rows > 1 and (block > 2048 or patch > patch_limit):
                rows //= 2
                block = triton.next_power_of_2(rows * iw)
                patch = triton.next_power_of_2(min(oh, rows * h_steps) * ow)
            if h_steps * w_steps <= 16 and block <= 2048 and patch <= patch_limit:
                _nearest2d_backward_ascend_rows[
                    (min(cores, n * c * triton.cdiv(ih, rows)),)
                ](
                    grad_output,
                    grad_input,
                    n * c,
                    ih,
                    iw,
                    oh,
                    ow,
                    scale_h,
                    scale_w,
                    h_steps,
                    w_steps,
                    rows,
                    block,
                    patch,
                )
                return grad_input
    if int64_index and (_ASCEND_SIZE_ONLY or runtime_device.vendor_name == "hygon"):
        with torch_device_fn.device(grad_output.device):
            _upsample_nearest2d_backward_scalar_kernel[(min(total, 65535),)](
                grad_output,
                grad_input,
                total,
                c,
                ih,
                iw,
                oh,
                ow,
                *grad_output.stride(),
                *grad_input.stride(),
                scale_h,
                scale_w,
                identity,
            )
        return grad_input
    h_steps, w_steps = 0, 0
    if runtime_device.vendor_name == "mthreads" and max(ih, iw, oh, ow) <= 2**20:
        h_steps = _ceil_span_bound(scale_h, ih, oh)
        w_steps = _ceil_span_bound(scale_w, iw, ow)
        if h_steps * w_steps > 16:
            h_steps, w_steps = 0, 0
    block = 1024 if identity else 256
    with torch_device_fn.device(grad_output.device):
        _upsample_nearest2d_backward_kernel[(triton.cdiv(total, block),)](
            grad_output,
            grad_input,
            total,
            c,
            ih,
            iw,
            oh,
            ow,
            *grad_output.stride(),
            *grad_input.stride(),
            scale_h,
            scale_w,
            _CEIL_BACKWARD,
            grad_input.stride(1) == 1,
            identity,
            int64_index,
            h_steps,
            w_steps,
            block,
        )
    return grad_input


def upsample_nearest2d_backward(
    grad_output, output_size, input_size, scales_h=None, scales_w=None
):
    logger.debug("GEMS UPSAMPLE_NEAREST2D_BACKWARD")
    return _upsample_nearest2d_backward(
        grad_output, output_size, input_size, scales_h, scales_w, None
    )


def upsample_nearest2d_backward_grad_input(
    grad_output,
    output_size,
    input_size,
    scales_h=None,
    scales_w=None,
    *,
    grad_input,
):
    logger.debug("GEMS UPSAMPLE_NEAREST2D_BACKWARD.GRAD_INPUT")
    return _upsample_nearest2d_backward(
        grad_output, output_size, input_size, scales_h, scales_w, grad_input
    )
