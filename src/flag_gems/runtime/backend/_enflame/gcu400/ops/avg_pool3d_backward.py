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

from .avg_pool3d import _parse_pool3d_params

logger = logging.getLogger(__name__)

# Fixed Grid width of the GCU400 device (SP count); the flat stride loop below
# keeps every program inside one wave regardless of problem size.
_NUM_SPC = 48


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_R": 64, "BLOCK_W": 8}, num_stages=3, num_warps=1),
        triton.Config({"BLOCK_R": 128, "BLOCK_W": 8}, num_stages=3, num_warps=1),
        triton.Config({"BLOCK_R": 256, "BLOCK_W": 8}, num_stages=3, num_warps=1),
        triton.Config({"BLOCK_R": 64, "BLOCK_W": 16}, num_stages=2, num_warps=1),
        triton.Config({"BLOCK_R": 128, "BLOCK_W": 16}, num_stages=2, num_warps=1),
        triton.Config({"BLOCK_R": 256, "BLOCK_W": 16}, num_stages=2, num_warps=1),
        triton.Config({"BLOCK_R": 64, "BLOCK_W": 32}, num_stages=2, num_warps=1),
        triton.Config({"BLOCK_R": 128, "BLOCK_W": 32}, num_stages=2, num_warps=1),
        triton.Config({"BLOCK_R": 256, "BLOCK_W": 32}, num_stages=2, num_warps=1),
        triton.Config({"BLOCK_R": 64, "BLOCK_W": 64}, num_stages=2, num_warps=1),
        triton.Config({"BLOCK_R": 128, "BLOCK_W": 64}, num_stages=2, num_warps=1),
        triton.Config({"BLOCK_R": 512, "BLOCK_W": 8}, num_stages=3, num_warps=1),
        triton.Config({"BLOCK_R": 512, "BLOCK_W": 16}, num_stages=2, num_warps=1),
        triton.Config({"BLOCK_R": 512, "BLOCK_W": 32}, num_stages=2, num_warps=1),
    ],
    key=[
        "IN_H",
        "IN_W",
        "kernel_d",
        "kernel_h",
        "kernel_w",
        "stride_d",
        "stride_h",
        "stride_w",
    ],
)
@triton.jit
def avg_pool3d_backward_kernel(
    grad_output_ptr,
    grad_input_ptr,
    # Strides for grad_input
    in_stride_n,
    in_stride_c,
    in_stride_d,
    in_stride_h,
    in_stride_w,
    # Strides for grad_output
    out_stride_n,
    out_stride_c,
    out_stride_d,
    out_stride_h,
    out_stride_w,
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
    # AvgPool specific parameters
    COUNT_INCLUDE_PAD: tl.constexpr,
    UNIFORM_DIVISOR: tl.constexpr,
    UNIFORM_DIVISOR_VALUE: tl.constexpr,
    # Shapes as constexpr: every tile-index division below divides by a
    # compile-time constant, so the flat stride loop carries no software
    # integer division.
    IN_C: tl.constexpr,
    IN_D: tl.constexpr,
    IN_H: tl.constexpr,
    IN_W: tl.constexpr,
    OUT_D: tl.constexpr,
    OUT_H: tl.constexpr,
    OUT_W: tl.constexpr,
    # Number of contributing output windows per input position along each dim:
    # output o covers input i iff o*s-p <= i <= o*s-p+k-1, i.e.
    # o in [ceil((i+p-k+1)/s), floor((i+p)/s)] -- at most (k-1)//s + 1 values.
    WINDOWS_D: tl.constexpr,
    WINDOWS_H: tl.constexpr,
    WINDOWS_W: tl.constexpr,
    total_nc,
    NUM_SPC: tl.constexpr,
    # Tiling meta-parameters: rows bundle (n*c, d, h_in), cols are w_in.
    BLOCK_R: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    # Input-centric backward: iterate over input positions, gather from output.
    # Rows of one 2D tile flatten (n*c, d, h) and columns are w, so each tile
    # carries BLOCK_R*BLOCK_W lanes behind a fixed per-tile instruction count;
    # GCU prices vector instructions ~independently of tile lanes, so fat tiles
    # amortize the loop/load issue cost (small-spatial large-batch shapes used
    # to be dominated by per-tile fixed cost). The flat 1D tile space is walked
    # by a Fixed-Grid stride loop: the program count stays at the SP count
    # instead of scaling with N*C (tens of thousands of tiny CTAs serialize
    # badly on GCU400).
    pid = tl.program_id(0)

    ROWS_PER_NC: tl.constexpr = IN_D * IN_H
    num_col_blocks = (IN_W + BLOCK_W - 1) // BLOCK_W
    total_rows = total_nc * ROWS_PER_NC
    num_row_blocks = (total_rows + BLOCK_R - 1) // BLOCK_R
    total_tiles = num_row_blocks * num_col_blocks

    w_in_offsets = tl.arange(0, BLOCK_W)

    for tile_id in tl.range(pid, total_tiles, NUM_SPC):
        row_block = tile_id // num_col_blocks
        col_block = tile_id - row_block * num_col_blocks

        rows = row_block * BLOCK_R + tl.arange(0, BLOCK_R)
        row_valid = rows < total_rows
        nc_i = rows // ROWS_PER_NC
        rem = rows - nc_i * ROWS_PER_NC
        d_vec = rem // IN_H
        h_vec = rem - d_vec * IN_H
        n_idx = nc_i // IN_C
        c_idx = nc_i - n_idx * IN_C

        w0 = col_block * BLOCK_W
        w_in = w0 + w_in_offsets
        w_lane_valid = w_in < IN_W
        ow_max_vec = (w_in + padding_w) // stride_w

        in_row_base = (
            grad_input_ptr
            + n_idx * in_stride_n
            + c_idx * in_stride_c
            + d_vec * in_stride_d
            + h_vec * in_stride_h
        )
        out_row_base = grad_output_ptr + n_idx * out_stride_n + c_idx * out_stride_c

        od_max_vec = (d_vec + padding_d) // stride_d
        oh_max_vec = (h_vec + padding_h) // stride_h

        grad_acc = tl.zeros((BLOCK_R, BLOCK_W), dtype=tl.float32)

        for wd in range(WINDOWS_D):
            od_vec = od_max_vec - wd
            d_valid_vec = (
                (od_vec >= 0)
                & (od_vec < OUT_D)
                & (od_vec * stride_d >= d_vec + padding_d - kernel_d + 1)
            )
            if UNIFORM_DIVISOR:
                fd_vec = tl.full((BLOCK_R,), 1, dtype=tl.int32)
            elif COUNT_INCLUDE_PAD:
                d_start = od_vec * stride_d - padding_d
                fd_raw = tl.minimum(d_start + kernel_d, IN_D + padding_d) - tl.maximum(
                    d_start, -padding_d
                )
                fd_vec = tl.maximum(fd_raw, 1)
            else:
                d_start = od_vec * stride_d - padding_d
                fd_raw = tl.minimum(d_start + kernel_d, IN_D) - tl.maximum(d_start, 0)
                fd_vec = tl.maximum(fd_raw, 1)

            for wh in range(WINDOWS_H):
                oh_vec = oh_max_vec - wh
                h_valid_vec = (
                    (oh_vec >= 0)
                    & (oh_vec < OUT_H)
                    & (oh_vec * stride_h >= h_vec + padding_h - kernel_h + 1)
                )
                if UNIFORM_DIVISOR:
                    fh_vec = tl.full((BLOCK_R,), 1, dtype=tl.int32)
                elif COUNT_INCLUDE_PAD:
                    h_start = oh_vec * stride_h - padding_h
                    fh_raw = tl.minimum(
                        h_start + kernel_h, IN_H + padding_h
                    ) - tl.maximum(h_start, -padding_h)
                    fh_vec = tl.maximum(fh_raw, 1)
                else:
                    h_start = oh_vec * stride_h - padding_h
                    fh_raw = tl.minimum(h_start + kernel_h, IN_H) - tl.maximum(
                        h_start, 0
                    )
                    fh_vec = tl.maximum(fh_raw, 1)

                rd_valid = row_valid & d_valid_vec & h_valid_vec
                row_out_base = (
                    out_row_base + od_vec * out_stride_d + oh_vec * out_stride_h
                )

                fdh = fd_vec * fh_vec

                for ww in range(WINDOWS_W):
                    ow_vec = ow_max_vec - ww
                    w_valid = (
                        (ow_vec >= 0)
                        & (ow_vec < OUT_W)
                        & (ow_vec * stride_w >= w_in + padding_w - kernel_w + 1)
                    )
                    out_mask = rd_valid[:, None] & w_valid[None, :]

                    if UNIFORM_DIVISOR:
                        divisor = UNIFORM_DIVISOR_VALUE
                    else:
                        if COUNT_INCLUDE_PAD:
                            w_start = ow_vec * stride_w - padding_w
                            fw_raw = tl.minimum(
                                w_start + kernel_w, IN_W + padding_w
                            ) - tl.maximum(w_start, -padding_w)
                            fw = tl.maximum(fw_raw, 1)
                        else:
                            w_start = ow_vec * stride_w - padding_w
                            fw_raw = tl.minimum(w_start + kernel_w, IN_W) - tl.maximum(
                                w_start, 0
                            )
                            fw = tl.maximum(fw_raw, 1)
                        divisor = (fdh[:, None] * fw[None, :]).to(tl.float32)

                    grad_out_ptr = (
                        row_out_base[:, None] + ow_vec[None, :] * out_stride_w
                    )
                    # No `other=`: masked lanes stay undefined and are filtered
                    # by the explicit where below (masked 2D affine loads with
                    # other= miscompile on large tiles).
                    grad_out_val = tl.load(grad_out_ptr, mask=out_mask)
                    grad_acc += tl.where(out_mask, grad_out_val / divisor, 0.0)

        store_ptr = in_row_base[:, None] + w_in[None, :] * in_stride_w
        tl.store(
            store_ptr,
            grad_acc.to(grad_input_ptr.type.element_ty),
            mask=row_valid[:, None] & w_lane_valid[None, :],
        )


def avg_pool3d_backward(
    grad_output: torch.Tensor,
    input: torch.Tensor,
    kernel_size,
    stride,
    padding,
    ceil_mode,
    count_include_pad,
    divisor_override,
):
    """Compute the gradient of avg_pool3d.

    Args:
        grad_output: Gradient of the output tensor.
        input: Original input tensor (used for shape information).
        kernel_size: Size of the pooling window.
        stride: Stride of the pooling window.
        padding: Implicit zero padding.
        ceil_mode: Whether ceil was used for output shape.
        count_include_pad: Whether padding was included in averaging.
        divisor_override: Custom divisor override.

    Returns:
        Gradient with respect to the input tensor.
    """
    logger.debug("GEMS_ENFLAME AVG_POOL3D_BACKWARD")

    if divisor_override is not None and divisor_override == 0:
        raise ValueError("divisor_override cannot be zero")

    grad_output = grad_output.contiguous()

    (
        kernel_d,
        kernel_h,
        kernel_w,
        stride_d,
        stride_h,
        stride_w,
        padding_d,
        padding_h,
        padding_w,
    ) = _parse_pool3d_params(kernel_size, stride, padding)

    in_n, in_c, in_d, in_h, in_w = input.shape
    out_d, out_h, out_w = (
        grad_output.shape[2],
        grad_output.shape[3],
        grad_output.shape[4],
    )

    grad_input = torch.empty_like(input)

    if grad_output.numel() == 0:
        return grad_input.zero_()

    # At most (k-1)//s + 1 output windows cover any input position per dim;
    # enumerating only those windows cuts the gather loop from k^3 to
    # windows_d * windows_h * windows_w iterations (3x3x3/stride2: 8 vs 27).
    windows_d = (kernel_d - 1) // stride_d + 1
    windows_h = (kernel_h - 1) // stride_h + 1
    windows_w = (kernel_w - 1) // stride_w + 1

    # With zero padding in floor mode every window is fully inside the input,
    # so the forward divisor is the constant kernel volume (divisor_override
    # is constant by definition); skip the separable divisor machinery.
    if divisor_override is not None:
        uniform_divisor = True
        uniform_divisor_value = float(divisor_override)
    elif padding_d == 0 and padding_h == 0 and padding_w == 0 and not ceil_mode:
        uniform_divisor = True
        uniform_divisor_value = float(kernel_d * kernel_h * kernel_w)
    else:
        uniform_divisor = False
        uniform_divisor_value = 1.0

    # Fixed Grid: a flat 1D tile space over (row-block, col-block), where rows
    # bundle (n*c, d, h), walked by NUM_SPC programs, keeping the CTA count at
    # the SP count instead of scaling with N*C (large-batch shapes launched
    # tens of thousands of tiny CTAs and serialized badly on GCU400).
    grid = (_NUM_SPC,)

    avg_pool3d_backward_kernel[grid](
        grad_output,
        grad_input,
        grad_input.stride(0),
        grad_input.stride(1),
        grad_input.stride(2),
        grad_input.stride(3),
        grad_input.stride(4),
        grad_output.stride(0),
        grad_output.stride(1),
        grad_output.stride(2),
        grad_output.stride(3),
        grad_output.stride(4),
        kernel_d,
        kernel_h,
        kernel_w,
        stride_d,
        stride_h,
        stride_w,
        padding_d,
        padding_h,
        padding_w,
        COUNT_INCLUDE_PAD=count_include_pad,
        UNIFORM_DIVISOR=uniform_divisor,
        UNIFORM_DIVISOR_VALUE=uniform_divisor_value,
        IN_C=in_c,
        IN_D=in_d,
        IN_H=in_h,
        IN_W=in_w,
        OUT_D=out_d,
        OUT_H=out_h,
        OUT_W=out_w,
        WINDOWS_D=windows_d,
        WINDOWS_H=windows_h,
        WINDOWS_W=windows_w,
        total_nc=in_n * in_c,
        NUM_SPC=_NUM_SPC,
    )

    return grad_input
