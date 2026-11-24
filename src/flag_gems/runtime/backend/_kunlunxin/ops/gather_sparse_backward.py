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
from torch.utils._python_dispatch import TorchDispatchMode

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

# Capture the descriptor factory before full FlagGems registration replaces its
# SparseCUDA entry. This factory only attaches the already-built device buffers;
# it does not generate coordinates, copy values, sort, or reduce duplicates.
_COO_SCHEMA = "aten::_sparse_coo_tensor_with_dims_and_tensors"
_COO_OP = torch.ops.aten._sparse_coo_tensor_with_dims_and_tensors.default
_COO_KEYS = torch._C.DispatchKeySet(torch._C.DispatchKey.SparseCUDA)
_COO_FACTORY = torch._C._dispatch_get_computed_kernel_for_dispatch_key(
    _COO_SCHEMA, torch._C.DispatchKey.SparseCUDA
)


class _CooAssembly(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func._schema.name == _COO_SCHEMA:
            return _COO_FACTORY.call_boxed(_COO_KEYS, *args, **(kwargs or {}))
        return func(*args, **(kwargs or {}))


def _assemble(indices, values, shape, device):
    # Use a mode below Autograd, rather than calling the native handle directly,
    # so the standard sparse constructor's gradient history remains intact.
    with _CooAssembly():
        return _COO_OP(
            len(shape),
            0,
            list(shape),
            indices,
            values,
            dtype=values.dtype,
            layout=torch.sparse_coo,
            device=device,
        )


def _row_strides(shape):
    strides = []
    prod = 1
    for size in reversed(shape):
        strides.append(prod)
        prod *= size
    return tuple(reversed(strides))


@libentry()
@triton.jit
def _gather_sparse_flat_copy(
    source,
    target,
    N: tl.constexpr,
    SHAPE: tl.constexpr,
    STRIDES: tl.constexpr,
    ROW_STRIDES: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    if N > 2147483647:
        pid = pid.to(tl.int64)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    linear = offsets.to(tl.int64)
    source_offsets = tl.full((BLOCK,), 0, tl.int64)
    for d in tl.static_range(len(SHAPE)):
        source_offsets += (linear // ROW_STRIDES[d] % SHAPE[d]).to(tl.int64) * STRIDES[
            d
        ]
    if N % BLOCK == 0:
        value = tl.load(source + source_offsets)
        tl.store(target + offsets, value)
    else:
        value = tl.load(source + source_offsets, offsets < N, other=0)
        tl.store(target + offsets, value, offsets < N)


class _FlattenValues(torch.autograd.Function):
    @staticmethod
    def forward(ctx, source):
        ctx.shape = tuple(source.shape)
        return _flat_copy(source, source.dtype)

    @staticmethod
    def backward(ctx, gradient):
        flat = gradient if gradient.is_contiguous() else _FlattenValues.apply(gradient)
        return flat.view(ctx.shape)


def _flat_copy(source, dtype):
    size = source.numel()
    out = torch.empty((size,), dtype=dtype, device=source.device)
    if size:
        block = 4096
        with torch_device_fn.device(source.device):
            _gather_sparse_flat_copy[((size + block - 1) // block,)](
                source,
                out,
                size,
                tuple(source.shape),
                tuple(source.stride()),
                _row_strides(source.shape),
                block,
            )
    return out


@libentry()
@triton.jit
def _gather_sparse_coords(
    index,
    output,
    N: tl.constexpr,
    RANK: tl.constexpr,
    DIM: tl.constexpr,
    SHAPE: tl.constexpr,
    ROW_STRIDES: tl.constexpr,
    SINGLE_INDEX: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    if N * RANK > 2147483647:
        pid = pid.to(tl.int64)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    linear = offsets.to(tl.int64)
    if (DIM >= 0) and (DIM < RANK):
        if SINGLE_INDEX:
            gathered = tl.load(index)
        elif N % BLOCK == 0:
            gathered = tl.load(index + offsets)
        else:
            gathered = tl.load(index + offsets, offsets < N, other=0)
    for d in tl.static_range(RANK):
        if d == DIM:
            coord = gathered
        else:
            if (ROW_STRIDES[d] & (ROW_STRIDES[d] - 1)) == 0:
                quotient = linear >> int(math.log2(ROW_STRIDES[d]))
            else:
                quotient = linear // ROW_STRIDES[d]
            if (SHAPE[d] & (SHAPE[d] - 1)) == 0:
                coord = quotient & (SHAPE[d] - 1)
            else:
                coord = quotient - (quotient // SHAPE[d]) * SHAPE[d]
        if N % BLOCK == 0:
            tl.store(output + d * N + offsets, coord.to(tl.int64))
        else:
            tl.store(output + d * N + offsets, coord.to(tl.int64), offsets < N)


def _gather_sparse_backward(self, dim, index, grad):
    logger.debug("GEMS_KUNLUNXIN _GATHER_SPARSE_BACKWARD")
    rank = self.ndim
    values = grad.view(-1) if grad.is_contiguous() else _FlattenValues.apply(grad)
    nnz = grad.numel()
    if rank == 0 or nnz == 0:
        indices = torch.empty((rank, nnz), dtype=torch.int64, device=index.device)
        return _assemble(indices, values, self.shape, self.device)
    if grad.ndim == 0:
        if index.numel() != 1:
            raise RuntimeError(
                "shape '[1, 1]' is invalid for input of size " + str(index.numel())
            )
        indices = index.view(1, 1)
        if indices.dtype != torch.int64:
            indices = _flat_copy(indices, torch.int64).view(1, 1)
        return _assemble(indices, values, self.shape, self.device)
    if grad.ndim < rank:
        raise IndexError(
            f"Dimension out of range (expected to be in range of "
            f"[{-grad.ndim}, {grad.ndim - 1}], but got {grad.ndim})"
        )
    if index.numel() not in (1, nnz):
        raise IndexError(
            f"index must have 1 or {nnz} elements to match the gradient, "
            f"got {index.numel()}"
        )
    if index.is_contiguous() and index.dtype == torch.int64:
        flat_index = index.view(-1)
    else:
        flat_index = _flat_copy(index, torch.int64)
    dim = dim if dim >= 0 else dim + rank
    indices = torch.empty((rank, nnz), dtype=torch.int64, device=index.device)
    block = 4096
    with torch_device_fn.device(index.device):
        _gather_sparse_coords[((nnz + block - 1) // block,)](
            flat_index,
            indices,
            nnz,
            rank,
            dim,
            tuple(grad.shape),
            _row_strides(grad.shape),
            index.numel() == 1,
            block,
            unroll_num=1,
        )
    return _assemble(indices, values, self.shape, self.device)
