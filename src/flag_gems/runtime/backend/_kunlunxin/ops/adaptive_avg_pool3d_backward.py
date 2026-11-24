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

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def _adaptive_avg_pool3d_backward_exact_kernel(
    grad_output_ptr,
    grad_input_ptr,
    IN_D: tl.constexpr,
    IN_H: tl.constexpr,
    IN_W: tl.constexpr,
    OUT_D: tl.constexpr,
    OUT_H: tl.constexpr,
    OUT_W: tl.constexpr,
    KD: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    AREA: tl.constexpr,
    n_elems: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Fast path: exact integer ratio (in % out == 0 on every dim).
    # Each input element belongs to exactly one output's pooling region, so
    # grad_input[i] = grad_output[i // K] / (KD*KH*KW).  One load, one store.
    #
    # The five shapes are constexpr on purpose.  With them as runtime
    # arguments every index below costs a hardware integer division, and on
    # this backend the resulting scalar (non-vectorised) address stream runs
    # the kernel at ~1/3 of its constexpr speed (measured: 1.07 ms vs 0.51 ms
    # for 4x8x64^3 bf16).  Specialising per shape trades a recompilation for
    # that factor, which is the right trade for a pooling backward pass.
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elems

    d = (offsets // (IN_H * IN_W)) % IN_D
    h = (offsets // IN_W) % IN_H
    w = offsets % IN_W
    nc = offsets // (IN_D * IN_H * IN_W)

    # All loads below are provably in-bounds (indices are exact), so only the
    # flat tail mask is needed.
    o_off = (
        nc * (OUT_D * OUT_H * OUT_W)
        + (d // KD) * (OUT_H * OUT_W)
        + (h // KH) * OUT_W
        + (w // KW)
    )
    val = tl.load(grad_output_ptr + o_off, mask=mask).to(tl.float32)
    tl.store(
        grad_input_ptr + offsets,
        (val / AREA).to(grad_input_ptr.dtype.element_ty),
        mask=mask,
    )


@libentry()
@triton.jit
def _adaptive_avg_pool3d_backward_general_kernel(
    grad_output_ptr,
    grad_input_ptr,
    out_last,
    n_elems,
    in_d,
    in_h,
    in_w,
    out_d,
    out_h,
    out_w,
    MAX_D: tl.constexpr,
    MAX_H: tl.constexpr,
    MAX_W: tl.constexpr,
    BLOCK: tl.constexpr,
    USE_STATIC: tl.constexpr,
):
    # General path (non-integer ratios, handles upsampling).
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    lane_ok = offsets < n_elems
    safe = tl.where(lane_ok, offsets, 0)

    w = safe % in_w
    rem = safe // in_w
    h = rem % in_h
    rem = rem // in_h
    d = rem % in_d
    nc = rem // in_d

    d_min = (d * out_d) // in_d
    d_max = tl.minimum(((d + 1) * out_d + in_d - 1) // in_d, out_d)
    h_min = (h * out_h) // in_h
    h_max = tl.minimum(((h + 1) * out_h + in_h - 1) // in_h, out_h)
    w_min = (w * out_w) // in_w
    w_max = tl.minimum(((w + 1) * out_w + in_w - 1) // in_w, out_w)

    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    if USE_STATIC:
        for od in tl.static_range(0, MAX_D):
            o_d = d_min + od
            c_d = tl.minimum(o_d, out_d - 1)
            ds = (c_d * in_d) // out_d
            de = tl.minimum(((c_d + 1) * in_d + out_d - 1) // out_d, in_d)
            d_ok = (o_d < d_max) & (d >= ds) & (d < de)
            sz_d = de - ds
            for oh in tl.static_range(0, MAX_H):
                o_h = h_min + oh
                c_h = tl.minimum(o_h, out_h - 1)
                hs = (c_h * in_h) // out_h
                he = tl.minimum(((c_h + 1) * in_h + out_h - 1) // out_h, in_h)
                h_ok = (o_h < h_max) & (h >= hs) & (h < he)
                sz_dh = sz_d * (he - hs)
                row_ok = d_ok & h_ok
                for ow in tl.static_range(0, MAX_W):
                    o_w = w_min + ow
                    c_w = tl.minimum(o_w, out_w - 1)
                    ws = (c_w * in_w) // out_w
                    we = tl.minimum(((c_w + 1) * in_w + out_w - 1) // out_w, in_w)
                    active = row_ok & (o_w < w_max) & (w >= ws) & (w < we)
                    area = sz_dh * (we - ws)
                    o_off = ((nc * out_d + c_d) * out_h + c_h) * out_w + c_w
                    o_off = tl.minimum(tl.maximum(o_off, 0), out_last)
                    val = tl.load(grad_output_ptr + o_off).to(tl.float32)
                    acc += tl.where(active, val / tl.cast(area, tl.float32), 0.0)
    else:
        for od in range(0, MAX_D):
            o_d = d_min + od
            c_d = tl.minimum(o_d, out_d - 1)
            ds = (c_d * in_d) // out_d
            de = tl.minimum(((c_d + 1) * in_d + out_d - 1) // out_d, in_d)
            d_ok = (o_d < d_max) & (d >= ds) & (d < de)
            sz_d = de - ds
            for oh in range(0, MAX_H):
                o_h = h_min + oh
                c_h = tl.minimum(o_h, out_h - 1)
                hs = (c_h * in_h) // out_h
                he = tl.minimum(((c_h + 1) * in_h + out_h - 1) // out_h, in_h)
                h_ok = (o_h < h_max) & (h >= hs) & (h < he)
                sz_dh = sz_d * (he - hs)
                row_ok = d_ok & h_ok
                for ow in range(0, MAX_W):
                    o_w = w_min + ow
                    c_w = tl.minimum(o_w, out_w - 1)
                    ws = (c_w * in_w) // out_w
                    we = tl.minimum(((c_w + 1) * in_w + out_w - 1) // out_w, in_w)
                    active = row_ok & (o_w < w_max) & (w >= ws) & (w < we)
                    area = sz_dh * (we - ws)
                    o_off = ((nc * out_d + c_d) * out_h + c_h) * out_w + c_w
                    o_off = tl.minimum(tl.maximum(o_off, 0), out_last)
                    val = tl.load(grad_output_ptr + o_off).to(tl.float32)
                    acc += tl.where(active, val / tl.cast(area, tl.float32), 0.0)

    tl.store(
        grad_input_ptr + offsets, acc.to(grad_input_ptr.dtype.element_ty), mask=lane_ok
    )


def _fill_grad_input(grad_output, input, grad_input):
    """Launch the backward kernel, writing the result into ``grad_input``.

    ``grad_input`` must be a contiguous buffer of ``input``'s shape.  Both
    kernels accumulate in float32 and the store converts to the buffer's
    element type, so bf16/fp16 outputs are written straight into
    ``grad_input`` -- no fp32 scratch and no host-side copy are needed.
    """
    if not grad_output.is_contiguous():
        # Stage a contiguous copy with a gems Triton kernel (never a torch
        # data-movement fallback).  The kernels below index grad_output with a
        # flat contiguous offset, so a strided grad_output must be regularised
        # first.
        from flag_gems.ops.copy import copy_ as _gems_copy_

        staged = torch.empty(
            grad_output.shape, dtype=grad_output.dtype, device=grad_output.device
        )
        _gems_copy_(staged, grad_output)
        grad_output = staged
    in_n, in_c, in_d, in_h, in_w = input.shape
    out_d, out_h, out_w = grad_output.shape[-3:]

    with torch_device_fn.device(input.device):
        if in_d % out_d == 0 and in_h % out_h == 0 and in_w % out_w == 0:
            kd, kh, kw = in_d // out_d, in_h // out_h, in_w // out_w
            n_elems = in_n * in_c * in_d * in_h * in_w
            # 4096 lanes per program measured fastest here; the kernel is a
            # streaming gather, so it wants few, wide programs.
            block = 4096
            grid = (triton.cdiv(n_elems, block),)
            _adaptive_avg_pool3d_backward_exact_kernel[grid](
                grad_output,
                grad_input,
                IN_D=in_d,
                IN_H=in_h,
                IN_W=in_w,
                OUT_D=out_d,
                OUT_H=out_h,
                OUT_W=out_w,
                KD=kd,
                KH=kh,
                KW=kw,
                AREA=kd * kh * kw,
                n_elems=n_elems,
                BLOCK=block,
                num_warps=4,
            )
        else:
            n_elems = in_n * in_c * in_d * in_h * in_w
            max_d = (out_d + in_d - 1) // in_d + 1
            max_h = (out_h + in_h - 1) // in_h + 1
            max_w = (out_w + in_w - 1) // in_w + 1
            use_static = max_d * max_h * max_w <= 27
            block = 1024
            if not use_static:
                block = 64
            while block > 64 and block > n_elems:
                block //= 2
            n_tiles = triton.cdiv(n_elems, block)
            out_total = grad_output.numel()
            _adaptive_avg_pool3d_backward_general_kernel[(n_tiles,)](
                grad_output,
                grad_input,
                out_total - 1,
                n_elems,
                in_d,
                in_h,
                in_w,
                out_d,
                out_h,
                out_w,
                MAX_D=max_d,
                MAX_H=max_h,
                MAX_W=max_w,
                BLOCK=block,
                USE_STATIC=use_static,
                num_warps=1,
            )


def _adaptive_avg_pool3d_backward(grad_output, input):
    """Gradient of adaptive_avg_pool3d (Kunlunxin/XPU implementation)."""
    logger.debug("GEMS_KUNLUNXIN _ADAPTIVE_AVG_POOL3D_BACKWARD")

    # Allocate a fresh contiguous buffer (torch.empty on an explicit shape is a
    # pure allocation and always contiguous, so no .contiguous() call on
    # ``input`` is required -- the kernels only read grad_output).
    grad_input = torch.empty(input.shape, dtype=input.dtype, device=input.device)
    if grad_output.numel() == 0 or input.numel() == 0:
        return grad_input.zero_()

    _fill_grad_input(grad_output, input, grad_input)
    return grad_input


def adaptive_avg_pool3d_backward_grad_input(grad_output, input, *, grad_input):
    """Out-variant of adaptive_avg_pool3d_backward (Kunlunxin/XPU).

    Corresponds to ``aten::adaptive_avg_pool3d_backward.grad_input``: the
    result is written into the caller-provided ``grad_input`` buffer, which is
    also returned.
    """
    logger.debug("GEMS_KUNLUNXIN ADAPTIVE_AVG_POOL3D_BACKWARD_GRAD_INPUT")

    if grad_output.numel() == 0 or input.numel() == 0:
        return grad_input.zero_()

    if grad_input.is_contiguous():
        _fill_grad_input(grad_output, input, grad_input)
        return grad_input

    # Non-contiguous destination: compute into a contiguous scratch buffer,
    # then scatter into the strided grad_input with a gems Triton copy kernel
    # (pointwise_dynamic honours the destination strides).  No torch
    # data-movement fallback is used.
    from flag_gems.ops.copy import copy_ as _gems_copy_

    scratch = torch.empty(input.shape, dtype=input.dtype, device=input.device)
    _fill_grad_input(grad_output, input, scratch)
    _gems_copy_(grad_input, scratch)
    return grad_input
