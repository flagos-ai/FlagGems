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

"""GCU400 specialization of ``cudnn_grid_sampler``.

The generic kernel folds the batch and channel axes into one launch dimension
(``grid.y = N * C``). GCU400 caps every grid dimension at 255, so the widest benchmark
plane ``(8, 32, 256, 256) -> (64, 64)`` -- ``N * C = 256`` -- dies with
``OutOfResources: grid.y, Required: 256, Hardware limit: 255`` before the kernel runs.

This version launches ``(spatial_blocks, N, C / CHANNELS_PER_PROGRAM)`` instead: the
same one-program-per-spatial-block decomposition, but no launch dimension ever scales
with ``N * C``, so the wide planes work.  Every plane whose output grid is at least
``_WIDE_PLANE_SPATIAL`` pixels also takes this kernel, because there the coordinate
math is a large enough share of the work that sharing it across 4 output channels per
program measured 1.08x-1.17x (float16 and float32); the narrow planes stay on the
generic kernel, where halving the program count costs more than the reused math saves.
Everything else -- denormalization, the float bounds tests, the mask and the NaN
propagation -- is carried over unchanged, and the autotuner is replaced by a
deterministic block choice.
"""

import logging
import warnings

import torch
import triton
import triton.language as tl

from flag_gems.ops.cudnn_grid_sampler import (
    cudnn_grid_sampler_out as _generic_grid_sampler_out,
)
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

# Largest ``N * C`` that the generic kernel can fold into ``grid.y``; GCU400 caps every
# grid dimension at 255.
_MAX_FOLDED_PLANES = 255
# Planes at least this wide keep the custom kernel even when the generic one could
# launch: the coordinate math is then a large enough share of the work for the
# channel grouping to win (measured on 128x128 and 64x64 output planes).
_WIDE_PLANE_SPATIAL = 8192
# Output channels that share one program and one set of bilinear weights.
_CHANNELS_PER_PROGRAM = 4

logger = logging.getLogger(__name__)

_OUT_RESIZED_WARNING = (
    "An output with one or more elements was resized since it had shape "
    "{old_shape}, which does not match the required output shape {new_shape}. "
    "This behavior is deprecated, and in a future PyTorch release outputs will "
    "not be resized unless they have zero elements. You can explicitly reuse an "
    "out tensor t by resizing it, inplace, to zero elements with t.resize_(0)."
)


def _dtype_name(dtype):
    return str(dtype).removeprefix("torch.")


@libentry()
@triton.jit
def cudnn_grid_sampler_multi_c_kernel(
    out_ptr,
    inp_ptr,
    grid_ptr,
    C,
    H_in,
    W_in,
    num_spatial,
    CHANNELS_PER_PROGRAM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Bilinear 2D grid sampler (zeros padding, align_corners=True), N channels per program.

    Coordinates and bilinear weights are computed once per spatial block and reused for
    ``CHANNELS_PER_PROGRAM`` consecutive output channels.
    """
    pid_spatial = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_c = tl.program_id(2)

    idx = pid_spatial * BLOCK + tl.arange(0, BLOCK)
    mask = idx < num_spatial

    # Grid has shape (N, H_out, W_out, 2) and is contiguous.
    grid_offsets = pid_n * num_spatial * 2 + idx * 2
    grid_x = tl.load(grid_ptr + grid_offsets, mask=mask, other=0.0).to(tl.float32)
    grid_y = tl.load(grid_ptr + grid_offsets + 1, mask=mask, other=0.0).to(tl.float32)

    # Non-finite grid coordinates produce NaN (matches the cuDNN behaviour).
    non_finite = (grid_x != grid_x) | (grid_y != grid_y)
    non_finite = non_finite | (tl.abs(grid_x) == float("inf"))
    non_finite = non_finite | (tl.abs(grid_y) == float("inf"))

    # align_corners=True: pixel centers are located at -1 and 1.
    x = (grid_x + 1.0) * (W_in - 1) * 0.5
    y = (grid_y + 1.0) * (H_in - 1) * 0.5

    x0f = tl.floor(x)
    y0f = tl.floor(y)
    wx = x - x0f
    wy = y - y0f

    # Bounds are tested in floating point, before casting, so that non-finite or huge
    # coordinates never produce an out-of-range integer index.
    x0_in = (x0f >= 0.0) & (x0f < W_in)
    x1_in = (x0f + 1.0 >= 0.0) & (x0f + 1.0 < W_in)
    y0_in = (y0f >= 0.0) & (y0f < H_in)
    y1_in = (y0f + 1.0 >= 0.0) & (y0f + 1.0 < H_in)

    x0 = tl.cast(x0f, tl.int32)
    y0 = tl.cast(y0f, tl.int32)
    x1 = x0 + 1
    y1 = y0 + 1

    valid = mask & ~non_finite
    m00 = valid & x0_in & y0_in
    m01 = valid & x1_in & y0_in
    m10 = valid & x0_in & y1_in
    m11 = valid & x1_in & y1_in

    one_minus_wx = 1.0 - wx
    one_minus_wy = 1.0 - wy
    w00 = one_minus_wx * one_minus_wy
    w01 = wx * one_minus_wy
    w10 = one_minus_wx * wy
    w11 = wx * wy

    row0 = y0 * W_in
    row1 = y1 * W_in
    plane_size = H_in * W_in
    # The whole (n, c) plane set of this program shares the spatial block; only the
    # plane base and the output base advance with the channel.
    base_n = pid_n * C
    for c_index in tl.static_range(CHANNELS_PER_PROGRAM):
        c = pid_c * CHANNELS_PER_PROGRAM + c_index
        if c < C:
            plane_in = (base_n + c) * plane_size
            v00 = tl.load(inp_ptr + plane_in + row0 + x0, mask=m00, other=0.0).to(
                tl.float32
            )
            v01 = tl.load(inp_ptr + plane_in + row0 + x1, mask=m01, other=0.0).to(
                tl.float32
            )
            v10 = tl.load(inp_ptr + plane_in + row1 + x0, mask=m10, other=0.0).to(
                tl.float32
            )
            v11 = tl.load(inp_ptr + plane_in + row1 + x1, mask=m11, other=0.0).to(
                tl.float32
            )
            value = v00 * w00 + v01 * w01 + v10 * w10 + v11 * w11
            # Propagate NaN for non-finite grid samples.
            value = tl.where(mask & non_finite, float("nan"), value)
            tl.store(out_ptr + (base_n + c) * num_spatial + idx, value, mask=mask)


def _select_launch(num_spatial, channel_count):
    """Deterministic spatial block for the launch.

    BLOCK caps at 1024, which is what the generic autotuner settles on for these
    planes.  Sharing the coordinate math and the four bilinear gathers indices across
    a few channels per program pays off on the heavy planes: 4 channels per program
    measured 1.08x-1.17x on ``(2,64,64,64,128,128)`` and ``(8,32,256,256,64,64)`` for
    both float16 and float32, and 8 or 16 channels per program were already declining
    again.  On the small planes the channel group only shrinks the program count, so
    those stay on the generic kernel (see ``_cudnn_grid_sampler_launch``).
    """
    block = min(triton.next_power_of_2(max(num_spatial, 1)), 1024)
    return block, max(1, min(_CHANNELS_PER_PROGRAM, channel_count))


def _cudnn_grid_sampler_launch(inp, grid, out):
    N, C, H_in, W_in = inp.shape
    H_out, W_out = grid.shape[1], grid.shape[2]
    num_spatial = H_out * W_out

    if num_spatial == 0 or C == 0 or N == 0:
        return out

    # The generic kernel is at parity or slightly ahead on the narrow planes, so only
    # the planes that would overflow the folded grid cap -- or that are wide enough for
    # the channel grouping to pay off -- use this kernel.
    if N * C <= _MAX_FOLDED_PLANES and num_spatial < _WIDE_PLANE_SPATIAL:
        return _generic_grid_sampler_out(inp, grid, out=out)

    block, channels_per_program = _select_launch(num_spatial, C)
    grid_size = (
        triton.cdiv(num_spatial, block),
        N,
        triton.cdiv(C, channels_per_program),
    )
    cudnn_grid_sampler_multi_c_kernel[grid_size](
        out,
        inp,
        grid,
        C,
        H_in,
        W_in,
        num_spatial,
        CHANNELS_PER_PROGRAM=channels_per_program,
        BLOCK=block,
    )
    return out


def _validate_cudnn_grid_sampler_args(input, grid):
    """Mirror the argument checks of ``aten::cudnn_grid_sampler``."""
    if input.dim() != 4:
        raise ValueError(
            f"cudnn_grid_sampler only supports 4D input, but got {input.dim()}D input"
        )
    if grid.dim() != 4:
        raise ValueError(
            f"cudnn_grid_sampler only supports 4D grid, but got {grid.dim()}D grid"
        )
    if grid.shape[0] != input.shape[0]:
        raise ValueError(
            "cudnn_grid_sampler expects grid and input to have the same batch size, "
            f"but got input with shape {list(input.shape)} and grid with shape "
            f"{list(grid.shape)}"
        )
    if grid.shape[-1] != 2:
        raise ValueError(
            "cudnn_grid_sampler expects grid to have size 2 in the last dimension, "
            f"but got grid with shape {list(grid.shape)}"
        )
    if input.dtype != grid.dtype:
        raise ValueError(
            "cudnn_grid_sampler expects input and grid to have the same dtype, "
            f"but got {input.dtype} and {grid.dtype}"
        )
    if grid.device != input.device:
        raise ValueError(
            "cudnn_grid_sampler expects input and grid to be on the same device, "
            f"but got {input.device} and {grid.device}"
        )


def _prepare_out(out, expected_shape, dtype, device):
    """Resize ``out`` the way ATen's ``resize_output`` does, then check it."""
    if tuple(out.shape) != expected_shape:
        if out.numel() != 0:
            warnings.warn(
                _OUT_RESIZED_WARNING.format(
                    old_shape=list(out.shape), new_shape=list(expected_shape)
                ),
                UserWarning,
                stacklevel=3,
            )
        out.resize_(expected_shape)
    if out.dtype != dtype:
        raise ValueError(
            f"Expected out tensor to have dtype {_dtype_name(dtype)}, "
            f"but got {_dtype_name(out.dtype)} instead"
        )
    if out.device != device:
        raise ValueError(
            f"Expected out tensor to have device {device}, but got {out.device} instead"
        )
    return out


def cudnn_grid_sampler(input, grid):
    logger.debug("GEMS_ENFLAME CUDNN_GRID_SAMPLER")
    _validate_cudnn_grid_sampler_args(input, grid)

    inp = input.contiguous()
    grd = grid.contiguous()
    N, C, _, _ = inp.shape
    H_out, W_out = grd.shape[1], grd.shape[2]
    out = torch.empty((N, C, H_out, W_out), dtype=inp.dtype, device=inp.device)
    with torch_device_fn.device(inp.device):
        return _cudnn_grid_sampler_launch(inp, grd, out)


def cudnn_grid_sampler_out(input, grid, *, out):
    logger.debug("GEMS_ENFLAME CUDNN_GRID_SAMPLER_OUT")
    _validate_cudnn_grid_sampler_args(input, grid)

    inp = input.contiguous()
    grd = grid.contiguous()
    N, C, _, _ = inp.shape
    expected_shape = (N, C, grd.shape[1], grd.shape[2])
    out = _prepare_out(out, expected_shape, inp.dtype, inp.device)
    with torch_device_fn.device(inp.device):
        if out.is_contiguous():
            return _cudnn_grid_sampler_launch(inp, grd, out)
        # A strided ``out`` cannot be addressed with the kernel's flat offsets, so
        # sample into a contiguous temporary and let the strided copy place the values.
        tmp = torch.empty(expected_shape, dtype=inp.dtype, device=inp.device)
        _cudnn_grid_sampler_launch(inp, grd, tmp)
        out.copy_(tmp)
        return out
