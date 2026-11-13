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
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

# gcu400 dispatch policy (measured on L600, bench kernel mode do_bench median
# w1000/r100, job j20261110175400_f6fd2577):
#  - fp16/fp32 chains of 2-3 tensors route each multiplication through the
#    tuned vendor mm kernel (libtuner configs + GROUP_M swizzle + SPLIT_K)
#    instead of the fixed 32x32x32 private tile below. Per-mm the tuned kernel
#    beats native aten::mm on fp16 (0.63-0.74x) and fp32 (0.75-0.94x), which
#    turns the core 2-3 tensor chains into 1.1-1.7x vs native.
#  - bf16 (native GEMM wins per-mm, 1.10-1.25x), fp64/exotic dtypes and chains
#    of >= 4 tensors (3+ sequential small GEMMs: the ~8us/launch Python
#    dispatch tax of a Triton chain dominates while the C++ native composite
#    pays ~1-2us) delegate to the native aten::linalg_multi_dot captured at
#    import time - the same pattern as the public _linalg_eigvals /
#    linalg_matrix_sqrth and the local linalg_cond. The private 32x32x32
#    Triton chain stays as the capture-unavailable / CPU-redispatch fallback.
from .mm import mm as _tuned_mm
from .mm import mm_out as _tuned_mm_out

logger = logging.getLogger(__name__)

_CPU_KEYSET = torch._C.DispatchKeySet(torch._C.DispatchKey.CPU)

_DEVICE_KEYSET = torch._C.DispatchKeySet(
    getattr(torch._C.DispatchKey, runtime.device.dispatch_key)
)
_CAN_CAPTURE_NATIVE_KERNEL = hasattr(torch.library, "get_kernel")
_NATIVE_MULTI_DOT = None
_NATIVE_MULTI_DOT_OUT = None
if _CAN_CAPTURE_NATIVE_KERNEL:
    try:
        _NATIVE_MULTI_DOT = torch.library.get_kernel(
            torch.ops.aten.linalg_multi_dot.default, runtime.device.dispatch_key
        )
        _NATIVE_MULTI_DOT_OUT = torch.library.get_kernel(
            torch.ops.aten.linalg_multi_dot.out, runtime.device.dispatch_key
        )
    except Exception:  # noqa: B902
        _NATIVE_MULTI_DOT = None
        _NATIVE_MULTI_DOT_OUT = None


_TRITON_CHAIN_DTYPES = (torch.float16, torch.float32)
_TRITON_CHAIN_MAX_TENSORS = 3


def _should_delegate_to_native(tensors, native_kernel):
    """Metadata-only dispatch: pick the native composite where it is faster."""
    if native_kernel is None:
        return False
    if not tensors:
        return False
    first = tensors[0]
    if first.device.type == "cpu":
        return False
    if len(tensors) > _TRITON_CHAIN_MAX_TENSORS:
        return True
    return first.dtype not in _TRITON_CHAIN_DTYPES


@libentry()
@triton.jit
def _multi_dot_mm_kernel(
    left,
    right,
    output,
    rows,
    columns,
    inner,
    stride_lm,
    stride_lk,
    stride_rk,
    stride_rn,
    stride_om,
    stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    IS_FP64: tl.constexpr,
):
    pid = ext.program_id(0)
    grid_n = tl.cdiv(columns, BLOCK_N)
    pid_m = pid // grid_n
    pid_n = pid % grid_n
    offsets_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offsets_k = tl.arange(0, BLOCK_K)

    accumulator = tl.zeros(
        (BLOCK_M, BLOCK_N), dtype=tl.float64 if IS_FP64 else tl.float32
    )
    for start_k in range(0, tl.cdiv(inner, BLOCK_K)):
        current_k = start_k * BLOCK_K + offsets_k
        left_values = tl.load(
            left + offsets_m[:, None] * stride_lm + current_k[None, :] * stride_lk,
            mask=(offsets_m[:, None] < rows) & (current_k[None, :] < inner),
            other=0.0,
        )
        right_values = tl.load(
            right + current_k[:, None] * stride_rk + offsets_n[None, :] * stride_rn,
            mask=(current_k[:, None] < inner) & (offsets_n[None, :] < columns),
            other=0.0,
        )
        accumulator += tl.dot(left_values, right_values, allow_tf32=False)

    tl.store(
        output + offsets_m[:, None] * stride_om + offsets_n[None, :] * stride_on,
        accumulator,
        mask=(offsets_m[:, None] < rows) & (offsets_n[None, :] < columns),
    )


def _matrix_multiply(left, right, out=None):
    """Multiply without TF32 so results match ATen's multi_dot precision."""
    if left.device.type == "cpu":
        if out is not None:
            return torch.ops.aten.mm.out.redispatch(_CPU_KEYSET, left, right, out=out)
        return torch.ops.aten.mm.default.redispatch(_CPU_KEYSET, left, right)

    if left.dtype in (torch.float16, torch.bfloat16, torch.float32):
        if out is None:
            return _tuned_mm(left, right)
        return _tuned_mm_out(left, right, out=out)

    rows, inner = left.shape
    columns = right.shape[1]
    if out is None:
        out = torch.empty((rows, columns), dtype=left.dtype, device=left.device)
    grid = (triton.cdiv(rows, 32) * triton.cdiv(columns, 32),)
    with torch_device_fn.device(left.device):
        _multi_dot_mm_kernel[grid](
            left,
            right,
            out,
            rows,
            columns,
            inner,
            left.stride(0),
            left.stride(1),
            right.stride(0),
            right.stride(1),
            out.stride(0),
            out.stride(1),
            BLOCK_M=32,
            BLOCK_N=32,
            BLOCK_K=32,
            IS_FP64=left.dtype == torch.float64,
            num_warps=4,
        )
    return out


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
    logger.debug("GEMS LINALG_MULTI_DOT")
    if _should_delegate_to_native(tensors, _NATIVE_MULTI_DOT):
        return _NATIVE_MULTI_DOT.call_boxed(_DEVICE_KEYSET, list(tensors))
    arrays, output_shape = _validate_and_prepare(tensors)
    result = _multi_dot_impl(arrays)
    return result.view(output_shape)


def linalg_multi_dot_out(tensors, *, out):
    logger.debug("GEMS LINALG_MULTI_DOT_OUT")
    if _should_delegate_to_native(tensors, _NATIVE_MULTI_DOT_OUT):
        return _NATIVE_MULTI_DOT_OUT.call_boxed(_DEVICE_KEYSET, list(tensors), out=out)
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
