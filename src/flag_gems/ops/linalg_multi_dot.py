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

from flag_gems import runtime

logger = logging.getLogger(__name__)

_CPU_KEYSET = torch._C.DispatchKeySet(torch._C.DispatchKey.CPU)
_DEVICE_KEYSET = torch._C.DispatchKeySet(
    getattr(torch._C.DispatchKey, runtime.device.dispatch_key)
)


def _native_mm(left, right, out=None):
    """Run the backend GEMM without re-entering FlagGems' ``mm`` override."""
    keyset = _CPU_KEYSET if left.device.type == "cpu" else _DEVICE_KEYSET
    if out is not None:
        return torch.ops.aten.mm.out.redispatch(keyset, left, right, out=out)
    return torch.ops.aten.mm.default.redispatch(keyset, left, right)


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
        return _native_mm(left, right, out=out)
    return _native_mm(left, right)


def _multiply_three(arrays, out=None):
    a, b, c = arrays
    rows, inner_ab = a.shape
    inner_bc, columns = c.shape
    left_cost = rows * inner_bc * (inner_ab + columns)
    right_cost = inner_ab * columns * (rows + inner_bc)

    if left_cost > right_cost:
        right = _native_mm(b, c)
        return _native_mm(a, right, out=out)

    left = _native_mm(a, b)
    return _native_mm(left, c, out=out)


def _multi_dot_impl(arrays, out=None):
    num_tensors = len(arrays)
    if num_tensors == 2:
        return _native_mm(arrays[0], arrays[1], out=out)
    if num_tensors == 3:
        return _multiply_three(arrays, out=out)

    splits = _matrix_chain_order(arrays)
    return _multiply_chain(arrays, splits, 0, num_tensors - 1, out=out)


def linalg_multi_dot(tensors):
    logger.debug("GEMS LINALG_MULTI_DOT")
    arrays, output_shape = _validate_and_prepare(tensors)
    result = _multi_dot_impl(arrays)
    return result.view(output_shape)


def linalg_multi_dot_out(tensors, *, out):
    logger.debug("GEMS LINALG_MULTI_DOT_OUT")
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
