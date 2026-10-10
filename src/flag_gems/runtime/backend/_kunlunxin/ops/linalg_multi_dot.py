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
import warnings

import torch

from flag_gems.ops.copy import copy_ as _gems_copy_

from .mm import mm as _vendor_mm

logger = logging.getLogger(__name__)


def _to_f32_contig(t):
    """Return a contiguous float32 copy of ``t`` using only gems kernels.

    ``torch.empty`` is a pure allocation (never a compute/movement fallback) and
    the generic gems Triton copy (``flag_gems.ops.copy.copy_``) casts dtype AND
    lands contiguous in a single pointwise pass -- it replaces the previous
    ``t.float().contiguous()`` which relied on torch's own cast + materialize
    fallbacks. For fp16/bf16/fp32 real strided sources into a contiguous fp32
    destination the copy stays on the Triton branch of ``copy.py`` (not complex,
    not float8, not zerotensor, not an alias, numel < 2**31, numel != 0).
    """
    if t.dtype == torch.float32 and t.is_contiguous():
        return t
    dst = torch.empty(t.shape, dtype=torch.float32, device=t.device)
    _gems_copy_(dst, t)
    return dst


def _matrix_multiply(left, right, out=None):
    """Pairwise matmul in fp32 so the chain matches ATen multi_dot precision.

    fp32 inputs make the vendor ``mm`` accurate to ~1e-5; rounding the product
    back to the working dtype reproduces hardware bf16/fp16 matmul semantics
    (fp32 accumulate, round-to-dtype). No native/ATen matmul fallback is used.
    """
    result = _vendor_mm(_to_f32_contig(left), _to_f32_contig(right))
    if out is not None:
        # Write back through the gems Triton copy kernel (flag_gems.ops.copy),
        # not torch's ``.copy_``/``.to`` fallback. The pointwise copy casts the
        # fp32 product to ``out``'s working dtype AND honours a non-contiguous
        # view of the user's tensor (strided stores) in a single pass.
        _gems_copy_(out, result)
        return out
    if result.dtype == left.dtype:
        return result
    # Round the fp32 product back to the working dtype via the gems copy kernel
    # (matches ATen bf16/fp16 matmul: fp32 accumulate, round-to-dtype).
    typed = torch.empty(result.shape, dtype=left.dtype, device=result.device)
    _gems_copy_(typed, result)
    return typed


def _validate_and_prepare(tensors):
    num_tensors = len(tensors)
    if num_tensors < 2:
        raise RuntimeError(
            f"multi_dot(): expected at least 2 tensors but got {num_tensors}"
        )

    arrays = list(tensors)
    first = arrays[0]
    last = arrays[-1]
    if first.ndim not in (1, 2):
        raise RuntimeError(
            f"multi_dot(): the first tensor must be 1D or 2D but got {first.ndim}D"
        )
    if last.ndim not in (1, 2):
        raise RuntimeError(
            f"multi_dot(): the last tensor must be 1D or 2D but got {last.ndim}D"
        )

    for index, tensor in enumerate(arrays[1:-1], 1):
        if tensor.ndim != 2:
            raise RuntimeError(
                f"multi_dot(): tensor {index} must be 2D but got {tensor.ndim}D"
            )

    for index, tensor in enumerate(arrays[1:], 1):
        if tensor.dtype != first.dtype:
            raise RuntimeError(
                "multi_dot(): all tensors must have be the same dtype but "
                f"tensor 0 is {first.dtype} and tensor {index} {tensor.dtype}"
            )
        if tensor.device != first.device:
            raise RuntimeError(
                "multi_dot(): all tensors must be on the same device but "
                f"tensor 0 is on {first.device} and tensor {index} is on {tensor.device}"
            )

    first_was_1d = first.ndim == 1
    last_was_1d = last.ndim == 1
    if first_was_1d:
        arrays[0] = first.unsqueeze(0)
    if last_was_1d:
        arrays[-1] = last.unsqueeze(1)

    for index in range(num_tensors - 1):
        if arrays[index].shape[1] != arrays[index + 1].shape[0]:
            raise RuntimeError(
                f"multi_dot(): tensors {index} and {index + 1} with shapes "
                f"{list(tensors[index].shape)} and {list(tensors[index + 1].shape)} "
                "cannot be multiplied"
            )

    output_shape = [arrays[0].shape[0], arrays[-1].shape[1]]
    if first_was_1d:
        output_shape.pop(0)
    if last_was_1d:
        output_shape.pop(-1)
    return arrays, tuple(output_shape)


def _matrix_chain_order(arrays):
    num_tensors = len(arrays)
    dimensions = [arrays[0].shape[0]] + [array.shape[1] for array in arrays]
    costs = [[0] * num_tensors for _ in range(num_tensors)]
    splits = [[0] * num_tensors for _ in range(num_tensors)]

    for chain_length in range(2, num_tensors + 1):
        for start in range(num_tensors - chain_length + 1):
            end = start + chain_length - 1
            best_cost = None
            for split in range(start, end):
                cost = (
                    costs[start][split]
                    + costs[split + 1][end]
                    + dimensions[start] * dimensions[split + 1] * dimensions[end + 1]
                )
                if best_cost is None or cost < best_cost:
                    best_cost = cost
                    splits[start][end] = split
            costs[start][end] = best_cost
    return splits


def _multiply_chain(arrays, splits, start, end, out=None):
    if start == end:
        return arrays[start]

    split = splits[start][end]
    left = _multiply_chain(arrays, splits, start, split)
    right = _multiply_chain(arrays, splits, split + 1, end)
    if out is not None:
        return _matrix_multiply(left, right, out=out)
    return _matrix_multiply(left, right)


def _multiply_three(arrays, out=None):
    a, b, c = arrays
    rows, inner_ab = a.shape
    inner_bc, columns = c.shape
    left_cost = rows * inner_bc * (inner_ab + columns)
    right_cost = inner_ab * columns * (rows + inner_bc)

    if left_cost > right_cost:
        right = _matrix_multiply(b, c)
        return _matrix_multiply(a, right, out=out)

    left = _matrix_multiply(a, b)
    return _matrix_multiply(left, c, out=out)


def _multi_dot_impl(arrays, out=None):
    num_tensors = len(arrays)
    if num_tensors == 2:
        return _matrix_multiply(arrays[0], arrays[1], out=out)
    if num_tensors == 3:
        return _multiply_three(arrays, out=out)

    splits = _matrix_chain_order(arrays)
    return _multiply_chain(arrays, splits, 0, num_tensors - 1, out=out)


def linalg_multi_dot(tensors):
    logger.debug("GEMS_KUNLUNXIN LINALG_MULTI_DOT")
    arrays, output_shape = _validate_and_prepare(tensors)
    result = _multi_dot_impl(arrays)
    return result.view(output_shape)


def linalg_multi_dot_out(tensors, *, out):
    logger.debug("GEMS_KUNLUNXIN LINALG_MULTI_DOT_OUT")
    arrays, output_shape = _validate_and_prepare(tensors)
    first = arrays[0]
    if out.dtype != first.dtype:
        raise RuntimeError(
            f"multi_dot(): expected out tensor to have dtype {first.dtype} "
            f"but got {out.dtype}"
        )
    if out.device != first.device:
        raise RuntimeError(
            f"multi_dot(): expected out tensor to be on device {first.device} "
            f"but got {out.device}"
        )

    if tuple(out.shape) != output_shape:
        if out.numel() != 0:
            warnings.warn(
                "An output with one or more elements was resized since it had "
                f"shape {list(out.shape)}, which does not match the required "
                f"output shape {list(output_shape)}. This behavior is deprecated, "
                "and in a future PyTorch release outputs will not be resized "
                "unless they have zero elements. You can explicitly reuse an out "
                "tensor t by resizing it, inplace, to zero elements with "
                "t.resize_(0).",
                UserWarning,
                stacklevel=2,
            )
        out.resize_(output_shape)

    matrix_out = out.view(arrays[0].shape[0], arrays[-1].shape[1])
    _multi_dot_impl(arrays, out=matrix_out)
    return out
