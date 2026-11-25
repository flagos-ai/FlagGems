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
from triton import language as tl

from flag_gems.ops.index_select import index_select
from flag_gems.ops.index_select_backward import index_select_backward
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext
from flag_gems.utils.pointwise_dynamic import pointwise_dynamic
from flag_gems.utils.shape_utils import c_contiguous_stride
from flag_gems.utils.tensor_wrapper import StridedBuffer

logger = logging.getLogger(__name__)


@pointwise_dynamic(num_inputs=1, promotion_methods=[(0, "DEFAULT")])
@triton.jit
def copy_func(x):
    return x


def repeat_interleave_self_int(inp, repeats, dim=None, *, output_size=None):
    logger.debug("GEMS_HYGON REPEAT_INTERLEAVE_SELF_INT")
    if dim is None:
        inp = inp.flatten()
        dim = 0
    else:
        if (dim < -inp.ndim) or (dim >= inp.ndim):
            raise IndexError(
                "Dimension out of range (expected to be in range of [{}, {}], but got {})".format(
                    -inp.ndim, inp.ndim - 1, dim
                )
            )
    inp_shape = list(inp.shape)
    inp_stride = list(inp.stride())
    output_shape = list(inp.shape)

    if dim < 0:
        dim = dim + len(inp_shape)

    output_shape[dim] *= repeats

    if output_size is not None and output_size != output_shape[dim]:
        raise RuntimeError(
            "repeat_interleave: Invalid output_size, expected {} but got {}".format(
                output_shape[dim], output_size
            )
        )

    output = torch.empty(output_shape, dtype=inp.dtype, device=inp.device)

    if repeats == 0:
        return output

    in_view_stride = inp_stride[: dim + 1] + [0] + inp_stride[dim + 1 :]
    out_view_shape = inp_shape[: dim + 1] + [repeats] + inp_shape[dim + 1 :]
    out_view_stride = c_contiguous_stride(out_view_shape)

    in_view = StridedBuffer(inp, out_view_shape, in_view_stride)
    out_view = StridedBuffer(output, out_view_shape, out_view_stride)
    ndim = len(out_view_shape)
    copy_func.instantiate(ndim)(in_view, out0=out_view)
    return output


@triton.jit
def repeat_interleave_tensor_kernel(
    repeats_ptr, cumsum_ptr, out_ptr, size, BLOCK_SIZE: tl.constexpr
):
    pid = ext.program_id(0)
    mask = pid < size
    cumsum = tl.load(cumsum_ptr + pid, mask, other=0)
    repeats = tl.load(repeats_ptr + pid, mask, other=0)
    out_offset = cumsum - repeats

    tl.device_assert(repeats >= 0, "repeats can not be negative")

    out_ptr += out_offset
    for start_k in range(0, repeats, BLOCK_SIZE):
        offsets_k = start_k + tl.arange(0, BLOCK_SIZE)
        mask_k = offsets_k < repeats
        tl.store(out_ptr + offsets_k, pid, mask=mask_k)


def repeat_interleave_tensor(repeats, *, output_size=None):
    logger.debug("GEMS_HYGON REPEAT_INTERLEAVE_TENSOR")

    assert repeats.ndim == 1, "repeat_interleave only accept 1D vector as repeat"
    repeats = repeats.contiguous()
    cumsum = repeats.cumsum(axis=0)
    result_size = cumsum[-1].item()

    assert result_size >= 0, "repeats can not be negative"

    out = torch.empty((result_size,), dtype=repeats.dtype, device=repeats.device)
    size = repeats.size(0)

    grid = (size,)
    BLOCK_SIZE = 32
    repeat_interleave_tensor_kernel[grid](
        repeats,
        cumsum,
        out,
        size,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=1,
    )
    return out


# Copy complete contiguous slices directly instead of materializing indices and
# sending every output element through the generic index_select kernel.
_REPEAT_COPY_DTYPES = (
    torch.bool,
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
)


@libentry()
@triton.jit
def _repeat_interleave_prefix_kernel(
    repeats_ptr,
    prefix_ptr,
    N: tl.constexpr,
    STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK)
    repeats = tl.load(repeats_ptr + offsets * STRIDE, offsets < N, other=0).to(tl.int64)
    prefix = tl.cumsum(repeats, axis=0)
    tl.store(prefix_ptr + offsets, prefix, offsets < N)
    # Check every count, including negative counts whose sum is nonnegative.
    invalid = tl.min(repeats, axis=0) < 0
    total = tl.where(invalid, -1, tl.sum(repeats, axis=0))
    tl.store(prefix_ptr + N, total)


@libentry()
@triton.jit
def _repeat_interleave_scatter_kernel(
    inp_ptr,
    prefix_ptr,
    out_ptr,
    INNER: tl.constexpr,
    BLOCK: tl.constexpr,
):
    source_row = tl.program_id(1)
    first = tl.load(prefix_ptr + source_row - 1, source_row > 0, other=0)
    last = tl.load(prefix_ptr + source_row)
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    source = source_row.to(tl.int64) * INNER + offsets
    values = tl.load(inp_ptr + source, offsets < INNER, other=0)
    # Reuse the input slice in registers for small repeat vectors.
    for row in range(first, last):
        tl.store(out_ptr + row * INNER + offsets, values, offsets < INNER)


@libentry()
@triton.jit
def _repeat_interleave_gather_kernel(
    inp_ptr,
    prefix_ptr,
    out_ptr,
    total,
    AXIS: tl.constexpr,
    INNER: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    output_row = pid // tl.cdiv(INNER, BLOCK)
    offsets = (pid % tl.cdiv(INNER, BLOCK)) * BLOCK + tl.arange(0, BLOCK)
    outer = output_row // total
    repeated_row = output_row % total

    # The upper bound also skips zero-count entries. One lookup serves a whole
    # contiguous output tile, keeping both reads and writes vectorized.
    left = tl.full((), 0, tl.int32)
    right = tl.full((), AXIS, tl.int32)
    while left < right:
        middle = (left + right) // 2
        boundary = tl.load(prefix_ptr + middle)
        move_right = boundary <= repeated_row
        left = tl.where(move_right, middle + 1, left)
        right = tl.where(move_right, right, middle)

    source_row = outer.to(tl.int64) * AXIS + left
    values = tl.load(inp_ptr + source_row * INNER + offsets, offsets < INNER, other=0)
    destination = output_row.to(tl.int64) * INNER + offsets
    tl.store(out_ptr + destination, values, offsets < INNER)


def _repeat_interleave_contiguous(inp, repeats, dim, inner, output_size):
    axis = inp.shape[dim]
    prefix = torch.empty((axis + 1,), dtype=torch.int64, device=inp.device)
    with torch_device_fn.device(inp.device):
        _repeat_interleave_prefix_kernel[(1,)](
            repeats,
            prefix,
            N=axis,
            STRIDE=repeats.stride(0),
            BLOCK=triton.next_power_of_2(axis),
            num_warps=4,
        )
        total = prefix[-1].item()
        if total < 0:
            raise RuntimeError("repeats can not be negative")
        if output_size is not None and output_size != total:
            raise RuntimeError(
                f"repeat_interleave: Invalid output_size, expected {total} "
                f"but got {output_size}"
            )

        shape = list(inp.shape)
        shape[dim] = total
        out = torch.empty(shape, dtype=inp.dtype, device=inp.device)
        if out.numel() == 0:
            return out

        if dim == 0 and axis <= 32 and inner >= 4096:
            _repeat_interleave_scatter_kernel[(triton.cdiv(inner, 4096), axis)](
                inp, prefix, out, INNER=inner, BLOCK=4096, num_warps=4
            )
        else:
            block, warps = (4096, 8) if inner >= 131072 else (2048, 4)
            rows = out.numel() // inner
            _repeat_interleave_gather_kernel[(rows * triton.cdiv(inner, block),)](
                inp,
                prefix,
                out,
                total,
                AXIS=axis,
                INNER=inner,
                BLOCK=block,
                num_warps=warps,
            )
    return out


def _complex_lanes(inp):
    return inp.contiguous().unsqueeze(-1).view(inp.real.dtype)


class _RepeatInterleaveGather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp, dim, indices):
        ctx.save_for_backward(indices)
        ctx.input_shape = inp.shape
        ctx.dim = dim

        if inp.numel() == 0 or indices.numel() == 0:
            shape = list(inp.shape)
            shape[dim] = indices.numel()
            return inp.new_empty(shape)

        inp = inp.resolve_conj().resolve_neg()
        with torch_device_fn.device(inp.device):
            if inp.is_complex():
                out = index_select(_complex_lanes(inp), dim, indices)
                return out.view(inp.dtype).squeeze(-1)
            return index_select(inp, dim, indices)

    @staticmethod
    def backward(ctx, grad):
        (indices,) = ctx.saved_tensors
        grad_inp = _RepeatInterleaveGatherBackward.apply(
            grad, ctx.input_shape, ctx.dim, indices
        )
        return grad_inp, None, None


class _RepeatInterleaveGatherBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, grad, input_shape, dim, indices):
        ctx.save_for_backward(indices)
        ctx.dim = dim

        if grad.numel() == 0:
            return grad.new_zeros(input_shape)

        grad = grad.resolve_conj().resolve_neg()
        with torch_device_fn.device(grad.device):
            if grad.is_complex():
                grad_inp = index_select_backward(
                    _complex_lanes(grad), (*input_shape, 2), dim, indices
                )
                return grad_inp.view(grad.dtype).squeeze(-1)
            return index_select_backward(grad, input_shape, dim, indices)

    @staticmethod
    def backward(ctx, grad_inp):
        (indices,) = ctx.saved_tensors
        return (
            _RepeatInterleaveGather.apply(grad_inp, ctx.dim, indices),
            None,
            None,
            None,
        )


def repeat_interleave_self_tensor(inp, repeats, dim=None, *, output_size=None):
    logger.debug("GEMS_HYGON REPEAT_INTERLEAVE_SELF_TENSOR")

    if repeats.numel() == 0:
        return inp.clone()

    if dim is None:
        inp = inp.flatten()
        dim = 0
    else:
        if (dim < -inp.ndim) or (dim >= inp.ndim):
            raise IndexError(
                "Dimension out of range (expected to be in range of [{}, {}], but got {})".format(
                    -inp.ndim, inp.ndim - 1, dim
                )
            )

    if repeats.ndim == 0 or (repeats.ndim == 1 and repeats.size(0) == 1):
        return repeat_interleave_self_int(
            inp, repeats.item(), dim=dim, output_size=output_size
        )
    elif repeats.ndim > 1:
        raise RuntimeError("repeats must be 0-dim or 1-dim tensor")

    inp_shape = list(inp.shape)
    if dim < 0:
        dim = dim + len(inp_shape)

    if repeats.size(0) != inp_shape[dim]:
        raise RuntimeError(
            "repeats must have the same size as input along dim, but got \
                repeats.size(0) = {} and input.size({}) = {}".format(
                repeats.size(0), dim, inp_shape[dim]
            )
        )

    if repeats.device != inp.device:
        raise RuntimeError("repeats and input must be on the same device")
    if repeats.dtype not in (torch.int32, torch.int64):
        raise RuntimeError("repeats must have int32 or int64 dtype")

    inner = math.prod(inp_shape[dim + 1 :])
    if (
        inp.is_contiguous()
        and inp.dtype in _REPEAT_COPY_DTYPES
        and 1 < inp_shape[dim] <= 4096
        and inner >= 128
    ):
        return _repeat_interleave_contiguous(inp, repeats, dim, inner, output_size)

    with torch_device_fn.device(inp.device):
        indices = repeat_interleave_tensor(repeats)
    if output_size is not None and output_size != indices.numel():
        raise RuntimeError(
            f"repeat_interleave: Invalid output_size, expected {indices.numel()} "
            f"but got {output_size}"
        )
    return _RepeatInterleaveGather.apply(inp, dim, indices)
