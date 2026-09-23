# Copyright 2026, The FlagOS Contributors.
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

# Lanes per program over the `w` axis; `in_w` is split across programs when it
# exceeds this.
MAX_BLOCK_W = 128

# Largest unrolled window count for the uneven-division flat map; beyond this
# the row-per-program kernel wins because the unrolled body grows with
# MAX_OH * MAX_OW.
_FLAT_MAX_CANDIDATES = 16


@libentry()
@triton.jit
def _adaptive_avg_pool2d_backward_kernel(
    grad_output_ptr,
    grad_input_ptr,
    # Input/Output shapes
    in_c,
    in_h,
    in_w,
    out_h,
    out_w,
    # Strides for grad_output
    out_stride_n,
    out_stride_c,
    out_stride_h,
    out_stride_w,
    # Strides for grad_input
    in_stride_n,
    in_stride_c,
    in_stride_h,
    in_stride_w,
    # Meta-parameters
    BLOCK_W: tl.constexpr,
    MAX_OH: tl.constexpr,
    MAX_OW: tl.constexpr,
):
    """
    Backward kernel for adaptive average pooling 2D (Kunlunxin/XPU).

    Semantics match the generic implementation: for each input position
    (ih, iw), accumulate grad_output[oh, ow] / window_size over every output
    position (oh, ow) whose pooling window covers (ih, iw):

        grad_input[ih, iw] = sum_{(oh, ow) : (ih, iw) in window(oh, ow)}
                                 grad_output[oh, ow] / window_size(oh, ow)

    The adaptive pooling window boundaries are:
        start_h(oh) = (oh * in_h) // out_h
        end_h(oh) = ((oh + 1) * in_h + out_h - 1) // out_h

    Two XPU-specific deviations from the generic kernel:

    1. Every mask stays one-dimensional, so no value of shape (N, 1) ever meets
       a value of shape (1, M).  The generic kernel builds
       `in_window = h_in_window & w_in_window`, an `arith.andi` between a
       (BLOCK_H, 1) and a (1, BLOCK_W) mask; TritonXPULegalize rewrites that
       into an op whose operands keep their pre-broadcast shapes and rejects it
       with "'arith.andi' op requires the same type for all operands and
       results" -- the generic kernel does not compile on XPU at all.  A program
       here owns one input row `h` (a scalar, so the height test is a scalar)
       times a run of `w` lanes (a vector), so only scalars ever meet vectors.
       This is the data flow of `_adaptive_avg_pool3d_backward_general_kernel`.

    2. Instead of sweeping all out_h * out_w output positions for every input
       position, each position visits only the output positions it can belong
       to.  Input index `i` belongs to output `o` iff

           (i * out) // in <= o < ((i + 1) * out + in - 1) // in

       Both bounds are monotone in `i`, so `[o_min, o_max)` is exactly the set
       of outputs whose window covers `i` -- which is why the mask does not
       need a `w >= start_w` comparison, only `ow < ow_max`.  Only MAX_OH /
       MAX_OW (the worst case over all rows/lanes, computed host-side by
       `_max_candidates`) are unrolled; the rest is masked off.  At the 14x14
       output sizes this removes ~8x of the work of the generic sweep.
    """
    pid_row = tl.program_id(0)
    w_block = tl.program_id(1)

    h = pid_row % in_h
    nc = pid_row // in_h
    c_idx = nc % in_c
    n_idx = nc // in_c

    w = w_block * BLOCK_W + tl.arange(0, BLOCK_W)
    valid = w < in_w

    grad_output_base = grad_output_ptr + n_idx * out_stride_n + c_idx * out_stride_c
    grad_input_base = grad_input_ptr + n_idx * in_stride_n + c_idx * in_stride_c

    # Candidate output positions of this row (scalar) and of every lane (vector).
    oh_min = (h * out_h) // in_h
    oh_max = ((h + 1) * out_h + in_h - 1) // in_h
    ow_min = (w * out_w) // in_w
    ow_max = tl.minimum(((w + 1) * out_w + in_w - 1) // in_w, out_w)

    grad_acc = tl.zeros((BLOCK_W,), dtype=tl.float32)

    for jh in tl.static_range(MAX_OH):
        oh = oh_min + jh
        oh_ok = oh < oh_max
        # Clamped so the (masked-off) tail lanes cannot address out of range.
        safe_oh = tl.minimum(oh, out_h - 1)
        start_h = (safe_oh * in_h) // out_h
        end_h = ((safe_oh + 1) * in_h + out_h - 1) // out_h
        window_h = end_h - start_h
        grad_out_row_ptr = grad_output_base + safe_oh * out_stride_h

        for jw in tl.static_range(MAX_OW):
            ow = ow_min + jw
            active = valid & oh_ok & (ow < ow_max)

            safe_ow = tl.minimum(ow, out_w - 1)
            start_w = (safe_ow * in_w) // out_w
            end_w = ((safe_ow + 1) * in_w + out_w - 1) // out_w

            window_size = (window_h * (end_w - start_w)).to(tl.float32)
            grad_out_val = tl.load(grad_out_row_ptr + safe_ow * out_stride_w)

            grad_acc += tl.where(active, grad_out_val.to(tl.float32) / window_size, 0.0)

    grad_input_store_ptr = grad_input_base + h * in_stride_h + w * in_stride_w
    tl.store(
        grad_input_store_ptr,
        grad_acc.to(grad_input_ptr.type.element_ty),
        mask=valid,
    )


@triton.jit
def _adaptive_avg_pool2d_backward_map_kernel(
    grad_output_ptr,
    grad_input_ptr,
    total,
    IN_HW: tl.constexpr,
    IN_W: tl.constexpr,
    OUT_W: tl.constexpr,
    RH: tl.constexpr,
    RW: tl.constexpr,
    PLANE_GO: tl.constexpr,
    BLOCK: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    """Even-division fast path: pure elementwise map.

    Used only when `in_h % out_h == 0 and in_w % out_w == 0`, where every input
    element belongs to exactly one output window, so
        grad_input[nc, h, w] = grad_output[nc, h // rh, w // rw] / (rh * rw)
    with no window loop and no accumulation.

    Every shape constant is a `tl.constexpr`: the per-element index chain is
    dominated by the `offs // IN_HW` / `% IN_HW` pair, and a *runtime* divisor
    costs ~30 extra instructions per element on this backend (measured
    965us -> 247us on the [4,128,112,112 -> 14,14] fp32 core shape when IN_HW
    was made constexpr; evidence/adaptive-avg-pool2d-bwd-recon-20260918 §7).
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    nc = offs // IN_HW
    rem = offs % IN_HW
    h = rem // IN_W
    w = rem % IN_W
    oh = h // RH
    ow = w // RW
    if NEED_MASK:
        mask = offs < total
        v = tl.load(
            grad_output_ptr + nc * PLANE_GO + oh * OUT_W + ow, mask=mask, other=0.0
        )
        tl.store(grad_input_ptr + offs, v / (RH * RW), mask=mask)
    else:
        v = tl.load(grad_output_ptr + nc * PLANE_GO + oh * OUT_W + ow)
        tl.store(grad_input_ptr + offs, v / (RH * RW))


@triton.jit
def _adaptive_avg_pool2d_backward_flat_kernel(
    grad_output_ptr,
    grad_input_ptr,
    total,
    IN_H: tl.constexpr,
    IN_W: tl.constexpr,
    IN_HW: tl.constexpr,
    OUT_H: tl.constexpr,
    OUT_W: tl.constexpr,
    PLANE_GO: tl.constexpr,
    MAX_OH: tl.constexpr,
    MAX_OW: tl.constexpr,
    BLOCK: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    """Uneven-division path: flat elementwise gather with unrolled candidates.

    With `in % out != 0` an input position can belong to several pooling
    windows, but never to more than MAX_OH x MAX_OW of them, so the candidate
    sweep becomes a fully unrolled per-lane accumulation:

        grad_input[nc, h, w] = sum_{jh, jw} grad_output[nc, oh, ow] / area

    Same shape as the even-division map kernel -- flat 1-D lanes with every
    shape constant a `tl.constexpr` -- which is what makes it fast.  The
    row-per-program kernel it replaces ran one program per input row at 32
    lanes (under the reliable 64-lane width) and kept a runtime divisor in
    every window bound; at (4, 3, 32, 32 -> 7, 7) it is ~3x slower at fp32
    (evidence/adaptive-avg-pool2d-bwd-recon-20260918 §7.5).
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)

    nc = offs // IN_HW
    rem = offs % IN_HW
    h = rem // IN_W
    w = rem % IN_W

    # Candidate output positions of this lane: [o_min, o_max) both monotone.
    oh_min = (h * OUT_H) // IN_H
    oh_max = ((h + 1) * OUT_H + IN_H - 1) // IN_H
    ow_min = (w * OUT_W) // IN_W
    ow_max = ((w + 1) * OUT_W + IN_W - 1) // IN_W

    grad_output_base = grad_output_ptr + nc * PLANE_GO
    grad_acc = tl.zeros((BLOCK,), dtype=tl.float32)

    for jh in tl.static_range(MAX_OH):
        oh = oh_min + jh
        oh_active = oh < oh_max
        # Clamped so the (masked-off) lanes cannot address out of range.
        safe_oh = tl.minimum(oh, OUT_H - 1)
        start_h = (safe_oh * IN_H) // OUT_H
        end_h = ((safe_oh + 1) * IN_H + OUT_H - 1) // OUT_H
        window_h = (end_h - start_h).to(tl.float32)

        for jw in tl.static_range(MAX_OW):
            ow = ow_min + jw
            active = oh_active & (ow < ow_max)

            safe_ow = tl.minimum(ow, OUT_W - 1)
            start_w = (safe_ow * IN_W) // OUT_W
            end_w = ((safe_ow + 1) * IN_W + OUT_W - 1) // OUT_W

            window_size = window_h * (end_w - start_w).to(tl.float32)
            grad_out_val = tl.load(grad_output_base + safe_oh * OUT_W + safe_ow)

            grad_acc += tl.where(active, grad_out_val.to(tl.float32) / window_size, 0.0)

    if NEED_MASK:
        mask = offs < total
        tl.store(
            grad_input_ptr + offs,
            grad_acc.to(grad_input_ptr.type.element_ty),
            mask=mask,
        )
    else:
        tl.store(grad_input_ptr + offs, grad_acc.to(grad_input_ptr.type.element_ty))


def _max_candidates(in_size, out_size):
    """
    Largest number of output positions a single input index along one
    dimension can belong to, i.e. the worst-case size of [o_min, o_max).
    """
    best = 0
    for i in range(in_size):
        lo = (i * out_size) // in_size
        hi = min(((i + 1) * out_size + in_size - 1) // in_size, out_size)
        best = max(best, hi - lo)
    return best


def _adaptive_avg_pool2d_backward(
    grad_output: torch.Tensor,
    self: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the backward pass for adaptive average pooling 2D (Kunlunxin/XPU).

    Args:
        grad_output: Gradient with respect to the output of adaptive_avg_pool2d.
                    Shape: (N, C, out_H, out_W) or (C, out_H, out_W)
        self: The input tensor from the forward pass.
              Shape: (N, C, in_H, in_W) or (C, in_H, in_W)

    Returns:
        grad_input: Gradient with respect to the input.
                   Shape: same as self
    """
    logger.debug("GEMS_KUNLUNXIN _ADAPTIVE_AVG_POOL2D_BACKWARD")

    # Handle 3D input (C, H, W) by adding batch dimension
    input_is_3d = self.dim() == 3
    if input_is_3d:
        grad_output = grad_output.unsqueeze(0)
        self = self.unsqueeze(0)

    grad_output = grad_output.contiguous()

    in_n, in_c, in_h, in_w = self.shape
    out_h, out_w = grad_output.shape[2], grad_output.shape[3]

    # Create output tensor (same shape as input)
    grad_input = torch.empty_like(self, dtype=self.dtype)

    if grad_output.numel() == 0 or self.numel() == 0:
        if input_is_3d:
            return grad_input.squeeze(0)
        return grad_input

    # Both flat paths rewrite grad_input as a flat 1-D array, so they need a
    # contiguous destination; anything else falls through to the strided kernel.
    flat_ok = self.is_contiguous()

    # Even-division fast path: no window overlap -> elementwise map.
    if in_h % out_h == 0 and in_w % out_w == 0 and flat_ok:
        rh = in_h // out_h
        rw = in_w // out_w
        total = grad_input.numel()
        BLOCK = 2048
        map_grid = (triton.cdiv(total, BLOCK),)
        with torch_device_fn.device(grad_input.device):
            _adaptive_avg_pool2d_backward_map_kernel[map_grid](
                grad_output,
                grad_input,
                total,
                IN_HW=in_h * in_w,
                IN_W=in_w,
                OUT_W=out_w,
                RH=rh,
                RW=rw,
                PLANE_GO=out_h * out_w,
                BLOCK=BLOCK,
                NEED_MASK=total % BLOCK != 0,
                num_warps=8,
            )
        if input_is_3d:
            return grad_input.squeeze(0)
        return grad_input

    max_oh = _max_candidates(in_h, out_h)
    max_ow = _max_candidates(in_w, out_w)

    # Uneven division: every input position belongs to at most
    # MAX_OH x MAX_OW pooling windows, so the gather is unrolled per lane.
    if flat_ok and max_oh * max_ow <= _FLAT_MAX_CANDIDATES:
        total = grad_input.numel()
        BLOCK = 1024
        flat_grid = (triton.cdiv(total, BLOCK),)
        with torch_device_fn.device(grad_input.device):
            _adaptive_avg_pool2d_backward_flat_kernel[flat_grid](
                grad_output,
                grad_input,
                total,
                IN_H=in_h,
                IN_W=in_w,
                IN_HW=in_h * in_w,
                OUT_H=out_h,
                OUT_W=out_w,
                PLANE_GO=out_h * out_w,
                MAX_OH=max_oh,
                MAX_OW=max_ow,
                BLOCK=BLOCK,
                NEED_MASK=total % BLOCK != 0,
                num_warps=4,
            )
        if input_is_3d:
            return grad_input.squeeze(0)
        return grad_input

    block_w = min(triton.next_power_of_2(in_w), MAX_BLOCK_W)
    grid = (
        in_n * in_c * in_h,
        triton.cdiv(in_w, block_w),
    )

    with torch_device_fn.device(grad_input.device):
        _adaptive_avg_pool2d_backward_kernel[grid](
            grad_output,
            grad_input,
            in_c,
            in_h,
            in_w,
            out_h,
            out_w,
            grad_output.stride(0),
            grad_output.stride(1),
            grad_output.stride(2),
            grad_output.stride(3),
            grad_input.stride(0),
            grad_input.stride(1),
            grad_input.stride(2),
            grad_input.stride(3),
            BLOCK_W=block_w,
            MAX_OH=max_oh,
            MAX_OW=max_ow,
            num_warps=4,
        )

    if input_is_3d:
        return grad_input.squeeze(0)

    return grad_input
