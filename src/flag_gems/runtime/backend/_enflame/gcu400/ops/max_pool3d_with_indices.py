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

from flag_gems.utils import libentry
from flag_gems.utils.limits import get_dtype_min

logger = logging.getLogger(__name__)

# One program handles a flat tile of output elements and strides over the whole
# output, so the grid never grows with the tensor. Measured on gcu400: a CTA
# costs ~120ns to launch, and the previous (n*c, depth*height*width-block) grid
# reached 8192 CTAs on the benchmark core shapes, which dominated the runtime of
# the small output volumes (32x256x2x7x7 -> 9.5ms with 8192 CTAs vs 0.25ms for
# 4x3x16x56x56 with 384 CTAs).
_MAX_GRID_X = 65535
_MAX_PROGRAMS = 96


def pool3d_output_size(
    in_size: int,
    kernel_size: int,
    stride: int,
    padding: int,
    dilation: int,
    ceil_mode: bool = False,
) -> int:
    """Compute one spatial dimension of the 3-D max-pool output."""
    effective_kernel_size = (kernel_size - 1) * dilation + 1
    numerator = in_size + 2 * padding - effective_kernel_size
    if ceil_mode:
        output_size = (numerator + stride - 1) // stride + 1
        # PyTorch-compatible adjustment for ceil_mode
        if (output_size - 1) * stride >= in_size + padding:
            output_size -= 1
    else:
        output_size = numerator // stride + 1
    return output_size


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK": 1024}, num_stages=2, num_warps=8),
        triton.Config({"BLOCK": 2048}, num_stages=2, num_warps=8),
        triton.Config({"BLOCK": 512}, num_stages=2, num_warps=4),
    ],
    key=[
        "in_d",
        "in_h",
        "in_w",
        "out_d",
        "out_h",
        "out_w",
        "kernel_d",
        "kernel_h",
        "kernel_w",
        "stride_d",
        "stride_h",
        "stride_w",
    ],
)
@triton.jit
def max_pool3d_forward_kernel(
    input_ptr,
    output_ptr,
    indices_ptr,
    # Input tensor strides
    in_stride_n,
    in_stride_c,
    in_stride_d,
    in_stride_h,
    in_stride_w,
    # Input/Output shapes
    in_c,
    in_d,
    in_h,
    in_w,
    out_d,
    out_h,
    out_w,
    # Total number of output elements (N * C * out_d * out_h * out_w)
    total,
    # Pooling parameters
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
    # Tiling of the flat output index space
    BLOCK: tl.constexpr,
):
    """Flat max-pooling kernel.

    A tile is a contiguous range of the flat output index, so the output and the
    index tensor are written with plain contiguous stores and the tile shape
    does not have to match out_h x out_w (the previous 2-D tile wasted most lanes
    on shapes like 4x4 or 7x7). The flat index decomposes into
    (plane, od, oh, ow) with (out_d * out_h * out_w) as the spatial extent of one
    (n, c) plane.
    """
    dtype = input_ptr.type.element_ty
    min_val = get_dtype_min(dtype)

    n_tiles = tl.cdiv(total, BLOCK)
    n_prog = tl.num_programs(0)
    out_spatial = out_d * out_h * out_w
    for tile in tl.range(tl.program_id(0), n_tiles, n_prog):
        offset = tile * BLOCK + tl.arange(0, BLOCK)
        tile_mask = offset < total

        # Decompose the flat output index; every component stays in range for a
        # masked lane because the modulo of the quotient is taken first.
        spatial = offset % out_spatial
        plane = offset // out_spatial
        ow = spatial % out_w
        spatial_h = spatial // out_w
        oh = spatial_h % out_h
        od = spatial_h // out_h

        plane_ptr = (
            input_ptr + (plane // in_c) * in_stride_n + (plane % in_c) * in_stride_c
        )

        max_val_acc = tl.full((BLOCK,), min_val, dtype=dtype)
        # The flat spatial index always fits in int32 for a poolable volume, and
        # int32 keeps the 27 compare/select updates off the slow 64-bit path.
        max_idx_acc = tl.full((BLOCK,), -1, dtype=tl.int32)

        for kd in tl.static_range(0, kernel_d):
            d_in = od * stride_d - padding_d + kd * dilation_d
            d_valid = (d_in >= 0) & (d_in < in_d)
            for kh in tl.static_range(0, kernel_h):
                h_in = oh * stride_h - padding_h + kh * dilation_h
                dh_valid = d_valid & (h_in >= 0) & (h_in < in_h)
                for kw in tl.static_range(0, kernel_w):
                    w_in = ow * stride_w - padding_w + kw * dilation_w
                    in_mask = tile_mask & dh_valid & (w_in >= 0) & (w_in < in_w)
                    input_offset = (
                        d_in * in_stride_d + h_in * in_stride_h + w_in * in_stride_w
                    )
                    current_val = tl.load(
                        plane_ptr + input_offset, mask=in_mask, other=min_val
                    )
                    current_idx = d_in * in_h * in_w + h_in * in_w + w_in
                    # A masked lane loads `min_val`, so a strict `>` can never
                    # select it and the index update needs no extra mask term.
                    is_new_max = current_val > max_val_acc
                    max_val_acc = tl.where(is_new_max, current_val, max_val_acc)
                    max_idx_acc = tl.where(is_new_max, current_idx, max_idx_acc)

        tl.store(output_ptr + offset, max_val_acc, mask=tile_mask)
        tl.store(indices_ptr + offset, max_idx_acc.to(tl.int64), mask=tile_mask)


def _parse_pool3d_params(kernel_size, stride, padding, dilation):
    """Parse and validate 3-D pooling parameters.

    Each parameter can be an int (applied to all 3 spatial dims) or a
    3-element tuple/list (D, H, W).
    """

    def _parse_param(param, name, default=None):
        if param is None:
            return default
        if isinstance(param, int):
            return param, param, param
        if isinstance(param, (list, tuple)) and len(param) == 3:
            return tuple(param)
        raise ValueError(f"Invalid {name}: {param}")

    kd, kh, kw = _parse_param(kernel_size, "kernel_size")
    sd, sh, sw = _parse_param(stride, "stride", default=(kd, kh, kw))
    pd, ph, pw = _parse_param(padding, "padding", default=(0, 0, 0))
    dd, dh, dw = _parse_param(dilation, "dilation", default=(1, 1, 1))

    if sd <= 0 or sh <= 0 or sw <= 0:
        raise ValueError(f"stride must be positive, but got stride=({sd}, {sh}, {sw})")
    if pd < 0 or ph < 0 or pw < 0:
        raise ValueError(
            f"padding must be non-negative, but got padding=({pd}, {ph}, {pw})"
        )
    if dd <= 0 or dh <= 0 or dw <= 0:
        raise ValueError(
            f"dilation must be positive, but got dilation=({dd}, {dh}, {dw})"
        )

    return kd, kh, kw, sd, sh, sw, pd, ph, pw, dd, dh, dw


def max_pool3d_with_indices(
    input: torch.Tensor,
    kernel_size,
    stride=None,
    padding=0,
    dilation=1,
    ceil_mode=False,
):
    """Compute 3-D max pooling, returning (output, indices).

    Indices are flat offsets into the (D, H, W) spatial volume of the input.
    """
    logger.debug("GEMS_ENFLAME MAX_POOL3D_WITH_INDICES")
    input = input.contiguous()

    params = _parse_pool3d_params(kernel_size, stride, padding, dilation)
    kd, kh, kw, sd, sh, sw, pd, ph, pw, dd, dh, dw = params

    in_n, in_c, in_d, in_h, in_w = input.shape
    out_d = pool3d_output_size(in_d, kd, sd, pd, dd, ceil_mode)
    out_h = pool3d_output_size(in_h, kh, sh, ph, dh, ceil_mode)
    out_w = pool3d_output_size(in_w, kw, sw, pw, dw, ceil_mode)

    output = torch.empty(
        (in_n, in_c, out_d, out_h, out_w), device=input.device, dtype=input.dtype
    )
    indices = torch.empty(
        (in_n, in_c, out_d, out_h, out_w), device=input.device, dtype=torch.int64
    )

    if output.numel() == 0:
        return output, indices

    total = output.numel()

    # Fixed grid: a CTA costs ~120ns to launch on this backend, so the number of
    # programs is capped and each one strides over the flat output space.
    grid = lambda meta: (
        min(triton.cdiv(total, meta["BLOCK"]), _MAX_PROGRAMS, _MAX_GRID_X),
    )

    max_pool3d_forward_kernel[grid](
        input,
        output,
        indices,
        input.stride(0),
        input.stride(1),
        input.stride(2),
        input.stride(3),
        input.stride(4),
        in_c,
        in_d,
        in_h,
        in_w,
        out_d,
        out_h,
        out_w,
        total,
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
    )

    return output, indices
