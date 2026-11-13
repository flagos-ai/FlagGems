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

"""Triton implementation of ``cudnn_grid_sampler_backward`` (enflame GCU400).

``aten::cudnn_grid_sampler_backward(self, grid, grad_output) -> (grad_self,
grad_grid)`` is the backward pass of cuDNN's grid sampler.  cuDNN only exposes
one configuration: bilinear interpolation with zero padding and
``align_corners=True``, therefore the operator takes no mode arguments and this
implementation hardcodes those semantics.

For every output location ``(n, h_out, w_out)`` the normalized grid
coordinates ``(gx, gy)`` are denormalized to pixel space with

    px = (gx + 1) * (W_in - 1) / 2,   py = (gy + 1) * (H_in - 1) / 2

The four neighbouring pixels ``(floor(px) + dx, floor(py) + dy)`` are combined
with bilinear weights ``wx, wy``; corners outside of the input count as zeros.
The two gradients are then

    grad_self[n, c, y, x] += grad_output[n, c, h_out, w_out] * w_x * w_y

    grad_grid[n, h_out, w_out, 0] =
        sum_c grad_output * d(value)/d(gx) * (W_in - 1) / 2
    grad_grid[n, h_out, w_out, 1] =
        sum_c grad_output * d(value)/d(gy) * (H_in - 1) / 2

Both gradients are produced by a single kernel so the input pixels needed by
``grad_grid`` are read only once.  ``grad_self`` is accumulated with atomics
into a zero-filled float32 (or float64) buffer, which is exact enough for a
scatter of a handful of contributions per input pixel; when the destination
already has that dtype the caller's buffer is used directly instead of
allocating a scratch and copying afterwards.

NaN handling follows cuDNN: a non-finite grid coordinate propagates into the
``grad_grid`` component whose derivative is multiplied by the corresponding
bilinear weight, and a non-finite ``grad_output`` only contributes where the
sample point actually touches the input (a sample taken entirely outside of
the image contributes nothing, so ``grad_grid`` stays finite there).

Enflame GCU400 deviations from the generic kernel
--------------------------------------------------
1. 1D position tiles: the generic kernel walks the channels with a
   ``[BLOCK_C, BLOCK_P]`` 2D tile whose four corner loads/atomics overlap by
   one element in both tile dimensions.  That "d == 1 overlapping 2D tile"
   construct fails to compile on this backend
   (``Pipeline run failed: PassManager execution failed``, no pass
   diagnostics; see knowledge/compiler_traps.md) whenever both tile
   dimensions are >= 2.  Tiles with a unit dimension compile fine, and the
   1D d=1-overlapping loads/atomics are proven by ``grid_sampler_2d_backward``
   running green on this device, so this variant keeps the same launch grid
   and math but processes one channel per loop iteration on 1D position
   tiles.
2. No fp64 arithmetic: the device reports ``support_fp64 == False`` and the
   Triton compiler rejects f64 ops (compiler_problems P04).  Native aten
   already computes Double as FP32 here ("GCU not support Double use Float
   replace"), so the wrapper mirrors that: fp64 requests are computed
   value-level in FP32 and the outputs are widened, like
   ``upsample_bicubic2d`` / ``gradient``.
"""

import logging
import warnings

import torch
import triton
import triton.language as tl

from flag_gems.runtime import device as runtime_device
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

# Launch configuration of the fused kernel: (BLOCK_P, num_warps).
# BLOCK_P tiles the output positions; the channel dimension is walked by a
# scalar loop inside the kernel (see the module docstring for why the 2D
# [BLOCK_C, BLOCK_P] tile of the generic kernel cannot compile here).
KERNEL_CONFIG = (64, 4)
KERNEL_CONFIG_FP64 = (32, 2)

# Flat element offsets are computed in int32 unless one of the tensors is
# larger than this.  2**31 is the first value an int32 offset cannot hold.
_INT32_MAX = 2**31

_OUT_RESIZED_WARNING = (
    "An output with one or more elements was resized since it had shape "
    "{old_shape}, which does not match the required output shape {new_shape}. "
    "This behavior is deprecated, and in a future PyTorch release outputs will "
    "not be resized unless they have zero elements. You can explicitly reuse an "
    "out tensor t by resizing it, inplace, to zero elements with t.resize_(0)."
)


def _dtype_name(dtype):
    return str(dtype).removeprefix("torch.")


@triton.jit
def _cast_scalar(value, dtype: tl.constexpr):
    """Cast a scalar kernel argument.

    Triton specializes integer arguments to Python constants, which have no
    ``.to()`` method, so ``tl.cast`` is used instead.
    """
    return tl.cast(value, dtype)


@triton.jit
def _grid_lookup(
    grid_ptr,
    n,
    pos,
    p_mask,
    num_out,
    W_in,
    H_in,
    CALC_DTYPE: tl.constexpr,
    USE_INT64_INDEX: tl.constexpr,
):
    """Load grid coordinates and turn them into bilinear corner descriptions."""
    index_dtype = tl.int64 if USE_INT64_INDEX else tl.int32
    grid_base = (n * num_out + pos) * 2
    gx = tl.load(grid_ptr + grid_base, mask=p_mask, other=0.0).to(CALC_DTYPE)
    gy = tl.load(grid_ptr + grid_base + 1, mask=p_mask, other=0.0).to(CALC_DTYPE)

    # align_corners=True denormalization.
    px = (gx + 1.0) * (_cast_scalar(W_in, CALC_DTYPE) - 1.0) * 0.5
    py = (gy + 1.0) * (_cast_scalar(H_in, CALC_DTYPE) - 1.0) * 0.5

    x0f = tl.floor(px)
    y0f = tl.floor(py)
    wx = px - x0f
    wy = py - y0f

    # Bounds are tested in floating point, before casting, so NaN (and far
    # out-of-range) coordinates never reach an undefined float -> int conversion.
    # ``W_in``/``H_in`` are loop-invariant integers, so the comparisons promote
    # them to the coordinate dtype.
    x0_ok = (x0f >= 0) & (x0f < W_in)
    x1_ok = (x0f + 1.0 >= 0) & (x0f + 1.0 < W_in)
    y0_ok = (y0f >= 0) & (y0f < H_in)
    y1_ok = (y0f + 1.0 >= 0) & (y0f + 1.0 < H_in)

    x0 = tl.cast(tl.where(x0_ok, x0f, 0.0), index_dtype)
    x1 = tl.cast(tl.where(x1_ok, x0f + 1.0, 0.0), index_dtype)
    y0 = tl.cast(tl.where(y0_ok, y0f, 0.0), index_dtype)
    y1 = tl.cast(tl.where(y1_ok, y0f + 1.0, 0.0), index_dtype)

    return x0, x1, y0, y1, wx, wy, x0_ok, x1_ok, y0_ok, y1_ok


@libentry()
@triton.jit
def cudnn_grid_sampler_backward_kernel(
    grad_input_ptr,
    grad_grid_ptr,
    grad_output_ptr,
    input_ptr,
    grid_ptr,
    C,
    num_out,
    H_in,
    W_in,
    num_p_blocks,
    BLOCK_P: tl.constexpr,
    CALC_DTYPE: tl.constexpr,
    USE_INT64_INDEX: tl.constexpr,
):
    """Scatter ``grad_input`` and reduce the channels for ``grad_grid``.

    Enflame lowering: one channel per loop iteration on a 1D position tile
    (the generic [BLOCK_C, BLOCK_P] 2D tile does not compile on this
    backend - see the module docstring).  The 2D tile's ``tl.sum(axis=0)``
    channel reduction degenerates into an elementwise accumulate.
    """
    index_dtype = tl.int64 if USE_INT64_INDEX else tl.int32

    pid = tl.program_id(0)
    # Flattened launch: the position blocks of every batch entry in one
    # dimension, so neither the batch size nor ``H_out * W_out`` is limited by
    # the 65535 cap of the y grid dimension.
    n = (pid // num_p_blocks).to(index_dtype)
    pid_p = pid % num_p_blocks

    pos = pid_p.to(index_dtype) * BLOCK_P + tl.arange(0, BLOCK_P).to(index_dtype)
    p_mask = pos < num_out

    x0, x1, y0, y1, wx, wy, x0_ok, x1_ok, y0_ok, y1_ok = _grid_lookup(
        grid_ptr,
        n,
        pos,
        p_mask,
        num_out,
        W_in,
        H_in,
        CALC_DTYPE,
        USE_INT64_INDEX,
    )

    one_minus_wx = 1.0 - wx
    one_minus_wy = 1.0 - wy

    corner_mask_00 = x0_ok & y0_ok
    corner_mask_01 = x1_ok & y0_ok
    corner_mask_10 = x0_ok & y1_ok
    corner_mask_11 = x1_ok & y1_ok

    acc_x = tl.zeros([BLOCK_P], dtype=CALC_DTYPE)
    acc_y = tl.zeros([BLOCK_P], dtype=CALC_DTYPE)

    hw = _cast_scalar(H_in, index_dtype) * _cast_scalar(W_in, index_dtype)
    c_total = _cast_scalar(C, index_dtype)
    input_base = n * c_total * hw
    grad_output_base = n * c_total * num_out

    for c0 in range(0, C):
        c = _cast_scalar(c0, index_dtype)

        input_offset = input_base + c * hw
        grad_output_offset = grad_output_base + c * num_out + pos

        p00 = tl.load(
            input_ptr + input_offset + y0 * W_in + x0,
            mask=p_mask & corner_mask_00,
            other=0.0,
        ).to(CALC_DTYPE)
        p01 = tl.load(
            input_ptr + input_offset + y0 * W_in + x1,
            mask=p_mask & corner_mask_01,
            other=0.0,
        ).to(CALC_DTYPE)
        p10 = tl.load(
            input_ptr + input_offset + y1 * W_in + x0,
            mask=p_mask & corner_mask_10,
            other=0.0,
        ).to(CALC_DTYPE)
        p11 = tl.load(
            input_ptr + input_offset + y1 * W_in + x1,
            mask=p_mask & corner_mask_11,
            other=0.0,
        ).to(CALC_DTYPE)

        go = tl.load(
            grad_output_ptr + grad_output_offset,
            mask=p_mask,
            other=0.0,
        ).to(CALC_DTYPE)

        # grad_input: scatter the bilinear weights of the four corners.
        grad_input_base = grad_input_ptr + (n * c_total + c) * hw
        tl.atomic_add(
            grad_input_base + y0 * W_in + x0,
            go * one_minus_wx * one_minus_wy,
            mask=p_mask & corner_mask_00,
            sem="relaxed",
        )
        tl.atomic_add(
            grad_input_base + y0 * W_in + x1,
            go * wx * one_minus_wy,
            mask=p_mask & corner_mask_01,
            sem="relaxed",
        )
        tl.atomic_add(
            grad_input_base + y1 * W_in + x0,
            go * one_minus_wx * wy,
            mask=p_mask & corner_mask_10,
            sem="relaxed",
        )
        tl.atomic_add(
            grad_input_base + y1 * W_in + x1,
            go * wx * wy,
            mask=p_mask & corner_mask_11,
            sem="relaxed",
        )

        # cuDNN drops sample points that miss the image entirely, so a NaN
        # ``grad_output`` there must not leak into ``grad_grid``.
        touched = corner_mask_00 | corner_mask_01 | corner_mask_10 | corner_mask_11
        go_grid = tl.where(touched, go, 0.0)

        # grad_grid: per-channel elementwise contribution; the channel sum
        # falls out of the loop-carried accumulators.
        acc_x += ((p01 - p00) * one_minus_wy + (p11 - p10) * wy) * go_grid
        acc_y += ((p10 - p00) * one_minus_wx + (p11 - p01) * wx) * go_grid

    grid_base = (n * num_out + pos) * 2
    scale_x = (_cast_scalar(W_in, CALC_DTYPE) - 1.0) * 0.5
    scale_y = (_cast_scalar(H_in, CALC_DTYPE) - 1.0) * 0.5
    tl.store(grad_grid_ptr + grid_base, acc_x * scale_x, mask=p_mask)
    tl.store(grad_grid_ptr + grid_base + 1, acc_y * scale_y, mask=p_mask)


def _check_inputs(input, grid, grad_output):
    if input.dim() != 4:
        raise ValueError(
            "cudnn_grid_sampler_backward: expected 4D input, "
            f"but got {input.dim()}D input"
        )
    if grid.dim() != 4 or grid.size(-1) != 2:
        raise ValueError(
            "cudnn_grid_sampler_backward: expected grid of shape "
            f"(N, H_out, W_out, 2), but got {tuple(grid.shape)}"
        )
    if input.size(0) != grid.size(0):
        raise ValueError(
            "cudnn_grid_sampler_backward: input and grid must have the same "
            f"batch size, but got {input.size(0)} and {grid.size(0)}"
        )
    expected = (input.size(0), input.size(1), grid.size(1), grid.size(2))
    if tuple(grad_output.shape) != expected:
        raise ValueError(
            "cudnn_grid_sampler_backward: expected grad_output of shape "
            f"{expected}, but got {tuple(grad_output.shape)}"
        )
    if grad_output.dtype != input.dtype:
        raise ValueError(
            "cudnn_grid_sampler_backward: expected grad_output to have the same "
            f"dtype as input, but got {_dtype_name(grad_output.dtype)} and "
            f"{_dtype_name(input.dtype)}"
        )
    for name, tensor in (("grid", grid), ("grad_output", grad_output)):
        if tensor.device != input.device:
            raise ValueError(
                f"cudnn_grid_sampler_backward: expected {name} to be on "
                f"{input.device}, but got {tensor.device}"
            )


def _prepare_out(out, expected_shape, dtype, device, name):
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
            f"Expected {name} to have dtype {_dtype_name(dtype)}, "
            f"but got {_dtype_name(out.dtype)} instead"
        )
    if out.device != device:
        raise ValueError(
            f"Expected {name} to have device {device}, but got {out.device}"
        )
    return out


def _select_config(is_fp64, total_out):
    """Pick a launch configuration and clamp the tile to the problem size."""
    block_p, num_warps = KERNEL_CONFIG_FP64 if is_fp64 else KERNEL_CONFIG
    block_p = min(block_p, triton.next_power_of_2(total_out))
    return block_p, num_warps


def _needs_int64_index(N, C, H_in, W_in, total_out):
    """True when any flat element offset of the kernel can exceed int32."""
    largest = max(
        N * C * H_in * W_in,  # input / grad_input
        N * C * total_out,  # grad_output
        N * total_out * 2,  # grid / grad_grid
    )
    return largest >= _INT32_MAX


def _run(input, grid, grad_output, out0=None, out1=None):
    """Shared driver for the base and ``.out`` variants."""
    _check_inputs(input, grid, grad_output)

    # The caller-visible dtypes: outputs are always produced in them.
    self_dtype = input.dtype
    grid_dtype = grid.dtype

    inp = input.contiguous()
    grd = grid.contiguous()
    go = grad_output.contiguous()

    N, C, H_in, W_in = inp.shape
    H_out, W_out = grd.shape[1], grd.shape[2]
    total_out = H_out * W_out

    input_shape = (N, C, H_in, W_in)
    grid_shape = (N, H_out, W_out, 2)
    device = inp.device

    if out0 is not None:
        out0 = _prepare_out(out0, input_shape, inp.dtype, device, "out0")
    if out1 is not None:
        out1 = _prepare_out(out1, grid_shape, grd.dtype, device, "out1")

    if N == 0 or C == 0 or H_in == 0 or W_in == 0 or total_out == 0:
        # No sample point can touch the input: every gradient is zero.
        grad_input = (
            out0
            if out0 is not None
            else torch.empty(input_shape, dtype=inp.dtype, device=device)
        )
        grad_input.zero_()
        grad_grid = (
            out1
            if out1 is not None
            else torch.empty(grid_shape, dtype=grd.dtype, device=device)
        )
        grad_grid.zero_()
        return grad_input, grad_grid

    # Enflame GCU400 lowering: the device has no fp64 arithmetic (the Triton
    # compiler rejects f64 ops, compiler_problems P04) while fp64 tensors
    # themselves allocate and copy fine.  Native aten already computes Double
    # as FP32 here, so mirror that: compute value-level in FP32 and widen the
    # outputs, like upsample_bicubic2d / gradient.
    downcast_fp64 = not runtime_device.support_fp64 and (
        inp.dtype == torch.float64 or grd.dtype == torch.float64
    )
    if downcast_fp64:
        if inp.dtype == torch.float64:
            inp = inp.to(torch.float32)
            go = go.to(torch.float32)
        if grd.dtype == torch.float64:
            grd = grd.to(torch.float32)

    # grad_grid is produced by a plain reduction, so the kernel stores straight
    # into the destination and lets the store perform the dtype conversion.
    grad_grid = (
        out1
        if out1 is not None
        else torch.empty(grid_shape, dtype=grid_dtype, device=device)
    )
    if grad_grid.is_contiguous() and not (
        grad_grid.dtype == torch.float64 and not runtime_device.support_fp64
    ):
        kernel_grid = grad_grid
    else:
        # A strided or FP64 destination cannot be addressed by the kernel
        # (flat offsets / no f64 store lanes), so it writes a contiguous
        # compute-dtype staging buffer that is copied afterwards.
        kernel_grid = torch.empty(grid_shape, dtype=grd.dtype, device=device)

    # grad_input is accumulated with atomics, which needs a zero-filled buffer
    # in the accumulation dtype.  When the destination already has that dtype
    # (float32/float64 input) the buffer is used directly instead of allocating
    # a scratch and converting afterwards, which would add a full-size pass.
    # After an fp64 value-level downcast the destination dtype never matches,
    # so the scratch + widening copy path is taken.
    acc_dtype = torch.float64 if inp.dtype == torch.float64 else torch.float32
    if out0 is not None and out0.dtype == acc_dtype and out0.is_contiguous():
        grad_input = out0
        kernel_input = grad_input
        kernel_input.zero_()
    elif out0 is None and inp.dtype == acc_dtype and not downcast_fp64:
        grad_input = torch.zeros(input_shape, dtype=acc_dtype, device=device)
        kernel_input = grad_input
    else:
        kernel_input = torch.zeros(input_shape, dtype=acc_dtype, device=device)
        grad_input = None

    block_p, num_warps = _select_config(acc_dtype == torch.float64, total_out)
    num_p_blocks = triton.cdiv(total_out, block_p)

    with torch_device_fn.device(device):
        cudnn_grid_sampler_backward_kernel[(num_p_blocks * N,)](
            kernel_input,
            kernel_grid,
            go,
            inp,
            grd,
            C,
            total_out,
            H_in,
            W_in,
            num_p_blocks,
            BLOCK_P=block_p,
            CALC_DTYPE=tl.float64 if acc_dtype == torch.float64 else tl.float32,
            USE_INT64_INDEX=_needs_int64_index(N, C, H_in, W_in, total_out),
            num_warps=num_warps,
        )

    if grad_input is None:
        grad_input = (
            out0
            if out0 is not None
            else torch.empty(input_shape, dtype=self_dtype, device=device)
        )
        grad_input.copy_(kernel_input)
    if kernel_grid is not grad_grid:
        grad_grid.copy_(kernel_grid)
    return grad_input, grad_grid


def cudnn_grid_sampler_backward(input, grid, grad_output):
    logger.debug("GEMS CUDNN_GRID_SAMPLER_BACKWARD")
    return _run(input, grid, grad_output)


def cudnn_grid_sampler_backward_out(input, grid, grad_output, out0, out1):
    logger.debug("GEMS CUDNN_GRID_SAMPLER_BACKWARD_OUT")
    return _run(input, grid, grad_output, out0=out0, out1=out1)
