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

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

from ..utils.tle_copy import tle_copy

logger = logging.getLogger(__name__)


@triton.jit
def reflection_pad2d_kernel(
    in_ptr,
    out_ptr,
    H_in,
    W_in,
    pad_left,
    pad_top,
    W_out,
    HW_out,
    HW_in,
    total_out,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    o = pid * BLOCK + tl.arange(0, BLOCK)
    mask = o < total_out

    b = o // HW_out
    rem = o % HW_out
    h_idx = rem // W_out
    w_idx = rem % W_out

    # Reflected height index. pad_top < H_in is validated on the host, so a single
    # period (abs + where) is exact — no `% (2*(H_in-1))` needed.
    y = h_idx.to(tl.int32) - pad_top
    pH = 2 * (H_in - 1)
    t_h = tl.abs(y)
    ih = tl.where(t_h < H_in, t_h, pH - t_h)

    # Reflected width index (same reasoning; pad_left < W_in validated).
    x = w_idx.to(tl.int32) - pad_left
    pW = 2 * (W_in - 1)
    t_w = tl.abs(x)
    iw = tl.where(t_w < W_in, t_w, pW - t_w)

    in_offs = b * HW_in + ih * W_in + iw
    vals = tl.load(in_ptr + in_offs, mask=mask)
    tl.store(out_ptr + o, vals, mask=mask)


@triton.jit
def copy_tensor_kernel(in_ptr, out_ptr, total, BLOCK: tl.constexpr):
    # Flat contiguous copy (no padding path). Mask-based, contiguous offsets ->
    # block DMA, same as the padded kernel's store side.
    pid = tl.program_id(axis=0)
    o = pid * BLOCK + tl.arange(0, BLOCK)
    mask = o < total
    vals = tl.load(in_ptr + o, mask=mask)
    tl.store(out_ptr + o, vals, mask=mask)


@triton.jit
def pad2d_hside_kernel(
    in_ptr,
    out_ptr,
    H_in: tl.constexpr,
    W_in: tl.constexpr,
    pad_left: tl.constexpr,
    pad_top: tl.constexpr,
    pad_bottom: tl.constexpr,
    W_out: tl.constexpr,
    HW_out: tl.constexpr,
    HW_in: tl.constexpr,
    total_h,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    o = pid * BLOCK + tl.arange(0, BLOCK)
    m = o < total_h
    oc = tl.minimum(o, total_h - 1)

    R = pad_top + pad_bottom
    RW = R * W_out
    # All divisors below are constexpr -> compile-time magic-number division
    # (a runtime divisor on XPU costs ~30 instructions: the flat kernel's
    # measured soft-div is the dominant cost of the gather path).
    b = oc // RW
    rem = oc - b * RW
    r = rem // W_out
    w = rem - r * W_out

    # Output row: top region rows [0, pad_top), bottom region rows
    # [H_in+pad_top, H_out).
    h_out = tl.where(r < pad_top, r, H_in + r)

    # Reflected height index (single-reflection exact: host validates
    # pad_top/pad_bottom < H_in, so |h_out - pad_top| <= 2*(H_in-1)).
    y = h_out - pad_top
    t_h = tl.abs(y)
    pH = 2 * (H_in - 1)
    ih = tl.where(t_h < H_in, t_h, pH - t_h)

    # Reflected width index (same single-reflection argument for pad_left/right).
    x = w - pad_left
    t_w = tl.abs(x)
    pW = 2 * (W_in - 1)
    iw = tl.where(t_w < W_in, t_w, pW - t_w)

    vals = tl.load(in_ptr + b * HW_in + ih * W_in + iw)
    tl.store(out_ptr + b * HW_out + h_out * W_out + w, vals, mask=m)


@triton.jit
def pad2d_wside_kernel(
    in_ptr,
    out_ptr,
    H_in: tl.constexpr,
    W_in: tl.constexpr,
    pad_left: tl.constexpr,
    pad_right: tl.constexpr,
    pad_top: tl.constexpr,
    W_out: tl.constexpr,
    HW_out: tl.constexpr,
    HW_in: tl.constexpr,
    total_w,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    o = pid * BLOCK + tl.arange(0, BLOCK)
    m = o < total_w
    oc = tl.minimum(o, total_w - 1)

    P = pad_left + pad_right
    row = oc // P
    j = oc - row * P
    b = row // H_in
    y = row - b * H_in

    # output column (left segment [0, pad_left), right segment
    # [pad_left+W_in, W_out)); source column reversed, exact when
    # pad_left/pad_right < W_in (host-validated).
    w_out = tl.where(j < pad_left, j, pad_left + W_in + (j - pad_left))
    src = tl.where(j < pad_left, pad_left - j, W_in - 2 - (j - pad_left))

    vals = tl.load(in_ptr + b * HW_in + y * W_in + src)
    tl.store(out_ptr + b * HW_out + (pad_top + y) * W_out + w_out, vals, mask=m)


@triton.jit
def interior_copy_kernel(
    in_ptr,
    out_ptr,
    HW_out,
    interior_off,
    H_in: tl.constexpr,
    W_in: tl.constexpr,
    W_out: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Interior block, one program per (row, column-block): the interior is
    # H_in*W_in elements per batch laid out as H_in rows of W_in at row stride
    # W_out, NOT one contiguous run. Grid axis 0 = b*H_in + y and axis 1 =
    # column blocks, so every address is affine in the program ids -- no
    # runtime div/mod (H_in constexpr -> compile-time magic-number division;
    # a runtime divisor on XPU costs ~30 instructions, see
    # pad2d_hside_kernel). The tail is masked exactly like copy_tensor_kernel
    # (affine indices + mask, verified maxdiff=0).
    r = tl.program_id(axis=0)
    b = r // H_in
    y = r - b * H_in
    o = tl.program_id(axis=1) * BLOCK + tl.arange(0, BLOCK)
    mask = o < W_in
    vals = tl.load(in_ptr + b * (H_in * W_in) + y * W_in + o, mask=mask)
    tl.store(out_ptr + b * HW_out + y * W_out + interior_off + o, vals, mask=mask)


def _launch_reflection_pad2d_split(
    x, out, pad_left, pad_right, pad_top, pad_bottom, H_in, W_in, H_out, W_out, B
):
    """Big-shape split: copy-family recipe (tle SDNN row transfer, Triton
    fallback) for the contiguous interior + two small Triton kernels for the
    H-side (top/bottom rows) and W-side (interior left/right columns) borders.
    No `torch.ops.aten.slice` / `_copy_from` here: slice is intercepted by gems
    and `_copy_from` lands on the XPU fallback, so both are replaced by the
    same copy-family path as alias_copy / lift_out (tle first, pointwise
    fallback)."""
    HW_out = H_out * W_out
    HW_in = H_in * W_in
    interior_off = pad_top * W_out + pad_left
    with torch_device_fn.device(x.device):
        # 1. Interior block: B*H_in rows of W_in elements at row stride W_out.
        interior = torch.as_strided(
            out,
            size=(B, H_in, W_in),
            stride=(HW_out, W_out, 1),
            storage_offset=interior_off,
        )
        if not tle_copy(x, interior):
            grid = (B * H_in, triton.cdiv(W_in, 4096))
            interior_copy_kernel[grid](
                x,
                out,
                HW_out,
                interior_off,
                H_in,
                W_in,
                W_out,
                BLOCK=4096,
            )
        # 2. Top/bottom rows.
        if pad_top > 0 or pad_bottom > 0:
            total_h = B * (pad_top + pad_bottom) * W_out
            pad2d_hside_kernel[(triton.cdiv(total_h, 4096),)](
                x,
                out,
                H_in,
                W_in,
                pad_left,
                pad_top,
                pad_bottom,
                W_out,
                HW_out,
                HW_in,
                total_h,
                BLOCK=4096,
            )
        # 3. Interior rows' left/right columns.
        if pad_left > 0 or pad_right > 0:
            total_w = B * H_in * (pad_left + pad_right)
            pad2d_wside_kernel[(triton.cdiv(total_w, 4096),)](
                x,
                out,
                H_in,
                W_in,
                pad_left,
                pad_right,
                pad_top,
                W_out,
                HW_out,
                HW_in,
                total_w,
                BLOCK=4096,
            )
    return out


def launch_reflection_pad2d(input: torch.Tensor, padding, out: torch.Tensor = None):
    # Validate padding format
    if not isinstance(padding, (list, tuple)):
        raise ValueError("padding must be a sequence")
    if len(padding) != 4:
        raise ValueError(
            "padding must be a sequence of length 4: (pad_left, pad_right, pad_top, pad_bottom)"
        )
    pad_left, pad_right, pad_top, pad_bottom = [int(p) for p in padding]

    # Validate padding values
    if pad_left < 0 or pad_right < 0 or pad_top < 0 or pad_bottom < 0:
        raise ValueError("padding values must be >= 0")

    # Validate input
    if input.dim() < 3:
        raise ValueError("input must have at least 3 dimensions")

    x = input.contiguous()
    H_in = int(x.shape[-2])
    W_in = int(x.shape[-1])
    # Validate reflection padding constraints
    if H_in < 2 or W_in < 2:
        raise ValueError(
            "input spatial dimensions must be at least 2 for reflection padding when padding > 0"
        )
    if H_in <= 0 or W_in <= 0:
        raise ValueError("spatial dimensions must be > 0")
    if pad_left >= W_in or pad_right >= W_in or pad_top >= H_in or pad_bottom >= H_in:
        raise ValueError(
            "padding values must be less than the input spatial dimensions for reflection padding"
        )

    H_out = H_in + pad_top + pad_bottom
    W_out = W_in + pad_left + pad_right

    leading_shape = x.shape[:-2]
    B = int(math.prod(leading_shape)) if len(leading_shape) > 0 else 1

    # Handle output tensor
    if out is None:
        out = torch.empty(
            (*leading_shape, H_out, W_out), device=x.device, dtype=x.dtype
        )
    else:
        expected_shape = (*leading_shape, H_out, W_out)
        if tuple(out.shape) != expected_shape:
            raise ValueError(
                f"out tensor has shape {tuple(out.shape)}, expected {expected_shape}"
            )
        if out.dtype != x.dtype:
            raise ValueError(
                f"out dtype {out.dtype} does not match input dtype {x.dtype}"
            )
        if out.device != x.device:
            raise ValueError("out must be on the same device as input")
        out = out.contiguous()

    # No padding: just copy
    if pad_left == 0 and pad_right == 0 and pad_top == 0 and pad_bottom == 0:
        BLOCK = 1024
        total = B * H_in * W_in
        grid = (triton.cdiv(total, BLOCK),)
        with torch_device_fn.device(x.device):
            copy_tensor_kernel[grid](x, out, total, BLOCK=BLOCK)
        return out

    HW_out = H_out * W_out
    HW_in = H_in * W_in
    total_out = B * HW_out

    if total_out >= 1048576:
        return _launch_reflection_pad2d_split(
            x,
            out,
            pad_left,
            pad_right,
            pad_top,
            pad_bottom,
            H_in,
            W_in,
            H_out,
            W_out,
            B,
        )

    # BLOCK=1024 is the best all-round tile on XPU: small shapes avoid the
    # per-program waste of a huge block, while medium/large shapes still get
    # enough work per program to stay off the launch floor (measured sweep).
    BLOCK = 1024
    grid = (triton.cdiv(total_out, BLOCK),)
    with torch_device_fn.device(x.device):
        reflection_pad2d_kernel[grid](
            x,
            out,
            H_in,
            W_in,
            pad_left,
            pad_top,
            W_out,
            HW_out,
            HW_in,
            total_out,
            BLOCK=BLOCK,
        )
    return out


def reflection_pad2d(input: torch.Tensor, padding):
    logger.debug("GEMS_KUNLUNXIN REFLECTION_PAD2D")
    return launch_reflection_pad2d(input, padding, out=None)


def reflection_pad2d_out(input: torch.Tensor, padding, out: torch.Tensor):
    logger.debug("GEMS_KUNLUNXIN REFLECTION_PAD2D_OUT")
    return launch_reflection_pad2d(input, padding, out=out)
