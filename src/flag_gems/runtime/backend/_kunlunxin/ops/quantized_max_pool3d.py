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
#
# kunlunxin (XPU) re-implementation of quantized_max_pool3d / .out.
#
# Two platform defects in the generic implementation
# (flag_gems/ops/quantized_max_pool3d.py), both measured on XPU3 with
# `tests/test_quantized_max_pool3d.py --ref cpu` (baseline 38 failed / 68 passed):
#
#  1. The kernel expresses padding with `tl.load(..., mask=in_mask,
#     other=min_val)`. This backend silently ignores `other=` (wiki
#     xpu3-masked-load-other-inbounds-leak, KERNEL_OPT_EXPERIENCE #4.3-6), so a
#     padding lane returns a neighbouring real value and pollutes the max ->
#     integer mismatches for every window that touches padding.
#  2. The `.out` variant finishes with `out.copy_(result)`. QuantizedCUDA has no
#     `copy_kernel` on this stack (`[NOT IMPLEMENTED] ... copy_kernel.cpp`), and
#     on XPU even a working copy would not adopt the input's scale/zero_point as
#     ATen's generated `.out` wrapper does.
#
# Fix: keep the generic parameter parsing / shape / error-text helpers (they are
# pure host logic and already carry the full ATen contract), replace the kernel
# load with a clamp-then-`tl.where` (always in-bounds, no masked-load `other`),
# and make `.out` launch the kernel straight into the user buffer and adopt the
# input quantization parameters in place.
import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops.quantized_max_pool3d import (
    _aten_dtype_name,
    _check_quantized_input,
    _int_view,
    _parse_pool3d_params,
    _quantized_max_pool3d_output_shape,
    _resize_quantized_out,
)
from flag_gems.utils.limits import get_dtype_min

logger = logging.getLogger(__name__)

__all__ = ["quantized_max_pool3d", "quantized_max_pool3d_out"]


@triton.jit
def _qmp3d_forward_kernel(
    input_ptr,
    output_ptr,
    in_stride_n,
    in_stride_c,
    in_stride_d,
    in_stride_h,
    in_stride_w,
    out_stride_n,
    out_stride_c,
    out_stride_d,
    out_stride_h,
    out_stride_w,
    in_c,
    in_d,
    in_h,
    in_w,
    out_d,
    out_h,
    out_w,
    kernel_d: tl.constexpr,
    kernel_h: tl.constexpr,
    kernel_w: tl.constexpr,
    stride_d: tl.constexpr,
    stride_h: tl.constexpr,
    stride_w: tl.constexpr,
    padding_d: tl.constexpr,
    padding_h: tl.constexpr,
    padding_w: tl.constexpr,
    dilation_d: tl.constexpr,
    dilation_h: tl.constexpr,
    dilation_w: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    pid_nc = tl.program_id(0)
    pid_dhw = tl.program_id(1)

    num_h_blocks = tl.cdiv(out_h, BLOCK_H)
    num_w_blocks = tl.cdiv(out_w, BLOCK_W)

    d_block_idx = pid_dhw // (num_h_blocks * num_w_blocks)
    hw_remainder = pid_dhw % (num_h_blocks * num_w_blocks)
    h_block_idx = hw_remainder // num_w_blocks
    w_block_idx = hw_remainder % num_w_blocks

    n_idx = pid_nc // in_c
    c_idx = pid_nc % in_c

    d_out = d_block_idx

    h_out_offsets = h_block_idx * BLOCK_H + tl.arange(0, BLOCK_H)
    w_out_offsets = w_block_idx * BLOCK_W + tl.arange(0, BLOCK_W)

    dtype = input_ptr.type.element_ty
    min_val = get_dtype_min(dtype)
    max_val_acc = tl.full((BLOCK_H, BLOCK_W), min_val, dtype=dtype)

    input_base_ptr = input_ptr + n_idx * in_stride_n + c_idx * in_stride_c

    for kd in tl.static_range(0, kernel_d):
        d_in = d_out * stride_d - padding_d + kd * dilation_d
        d_valid = (d_in >= 0) & (d_in < in_d)
        # Clamp the index so the (unmasked) load is always in-bounds; the value
        # is discarded by the `tl.where` below whenever the tap is a padding
        # lane. This avoids the masked-load `other=` path, which is unreliable
        # here.
        d_safe = tl.minimum(tl.maximum(d_in, 0), in_d - 1)
        for kh in tl.static_range(0, kernel_h):
            h_in = h_out_offsets[:, None] * stride_h - padding_h + kh * dilation_h
            h_valid = (h_in >= 0) & (h_in < in_h)
            h_safe = tl.minimum(tl.maximum(h_in, 0), in_h - 1)
            for kw in tl.static_range(0, kernel_w):
                w_in = w_out_offsets[None, :] * stride_w - padding_w + kw * dilation_w
                w_valid = (w_in >= 0) & (w_in < in_w)
                w_safe = tl.minimum(tl.maximum(w_in, 0), in_w - 1)
                valid = d_valid & h_valid & w_valid
                input_offset = (
                    d_safe * in_stride_d
                    + h_safe * in_stride_h
                    + w_safe * in_stride_w
                )
                current_val = tl.load(input_base_ptr + input_offset)
                current_val = tl.where(valid, current_val, min_val)
                max_val_acc = tl.maximum(max_val_acc, current_val)

    out_base_ptr = (
        output_ptr + n_idx * out_stride_n + c_idx * out_stride_c + d_out * out_stride_d
    )
    output_block_ptr = (
        out_base_ptr
        + h_out_offsets[:, None] * out_stride_h
        + w_out_offsets[None, :] * out_stride_w
    )

    out_mask = (h_out_offsets[:, None] < out_h) & (w_out_offsets[None, :] < out_w)
    tl.store(output_block_ptr, max_val_acc, mask=out_mask)


def _np2_cap(v, cap):
    p = 1
    while p * 2 <= v and p * 2 <= cap:
        p *= 2
    return max(1, min(p, cap))


def _launch(input, params, out_shape, out_int):
    kd, kh, kw, sd, sh, sw, pd, ph, pw, dd, dh, dw = params
    in_n, in_c, in_d, in_h, in_w = input.shape
    _, _, out_d, out_h, out_w = out_shape

    int_repr = _int_view(input)

    # One program per (n, c); the tile only has to cover one output d/h/w plane.
    # Size it to the actual output plane so a tiny plane does not launch mostly
    # masked lanes (the default 16x16 wastes up to 16x lanes for a 4x4 plane).
    BLOCK_H = _np2_cap(out_h, 16)
    BLOCK_W = _np2_cap(out_w, 16)
    grid = (
        in_n * in_c,
        out_d * triton.cdiv(out_h, BLOCK_H) * triton.cdiv(out_w, BLOCK_W),
    )
    _qmp3d_forward_kernel[grid](
        int_repr,
        out_int,
        *int_repr.stride(),
        *out_int.stride(),
        in_c,
        in_d,
        in_h,
        in_w,
        out_d,
        out_h,
        out_w,
        kd,
        kh,
        kw,
        sd,
        sh,
        sw,
        pd,
        ph,
        pw,
        dd,
        dh,
        dw,
        BLOCK_H=BLOCK_H,
        BLOCK_W=BLOCK_W,
    )
    return out_int


def _memory_format(input):
    # ATen keeps a channels_last_3d input in that layout; every other layout
    # yields an NCDHW-contiguous result.
    return (
        torch.channels_last_3d
        if input.is_contiguous(memory_format=torch.channels_last_3d)
        and not input.is_contiguous()
        else torch.contiguous_format
    )


def _adopt_qparams(out, scale, zero_point):
    """Rewrite ``out``'s scale/zero_point in place, keeping its strides.

    ATen's generated ``quantized_max_pool3d.out`` wrapper leaves ``out``
    carrying the *input's* quantization parameters. Rebuild a quantized view
    over ``out``'s own storage (so strides + offset are preserved) with the new
    parameters and install it as ``out``'s data.
    """
    q = torch._empty_affine_quantized(
        tuple(out.shape),
        scale=scale,
        zero_point=zero_point,
        dtype=out.dtype,
        device=out.device,
    )
    q.set_(
        out.untyped_storage(),
        out.storage_offset(),
        tuple(out.shape),
        out.stride(),
    )
    out.data = q


def quantized_max_pool3d(
    input: torch.Tensor,
    kernel_size,
    stride=(),
    padding=0,
    dilation=1,
    ceil_mode=False,
):
    logger.debug("GEMS_KUNLUNXIN QUANTIZED_MAX_POOL3D")

    params = _parse_pool3d_params(kernel_size, stride, padding, dilation)
    _check_quantized_input(input)
    out_shape = _quantized_max_pool3d_output_shape(input, params, ceil_mode)

    scale = float(input.q_scale())
    zero_point = int(input.q_zero_point())

    out = torch._empty_affine_quantized(
        out_shape,
        scale=scale,
        zero_point=zero_point,
        dtype=input.dtype,
        device=input.device,
        memory_format=_memory_format(input),
    )

    if out.numel() != 0:
        _launch(input, params, out_shape, _int_view(out))

    return out


def quantized_max_pool3d_out(
    input: torch.Tensor,
    kernel_size,
    stride=(),
    padding=0,
    dilation=1,
    ceil_mode=False,
    *,
    out: torch.Tensor,
):
    logger.debug("GEMS_KUNLUNXIN QUANTIZED_MAX_POOL3D_OUT")

    params = _parse_pool3d_params(kernel_size, stride, padding, dilation)
    _check_quantized_input(input)
    out_shape = _quantized_max_pool3d_output_shape(input, params, ceil_mode)

    # ATen's generated ``resize_out`` checks dtype before device.
    if out.dtype != input.dtype:
        raise RuntimeError(
            f"Expected out tensor to have dtype {_aten_dtype_name(input.dtype)}, "
            f"but got {_aten_dtype_name(out.dtype)} instead"
        )
    if out.device != input.device:
        raise RuntimeError(
            f"Expected out tensor to have device {input.device}, "
            f"but got {out.device} instead"
        )

    if tuple(out.shape) != tuple(out_shape):
        _resize_quantized_out(out, out_shape)

    if out.numel() != 0:
        _launch(input, params, out_shape, _int_view(out))

    # ATen's out variant leaves the input's scale/zero_point on ``out``.
    _adopt_qparams(out, float(input.q_scale()), int(input.q_zero_point()))
    return out
