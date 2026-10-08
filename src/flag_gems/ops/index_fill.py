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

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

# Number of shape/stride slots in the generic kernel signature. Unused slots
# are padded with shape 1 / stride 0. The contiguous 2-D kernel is
# rank-independent, so only strided tensors with rank > _MAX_RANK take the
# materialize-and-fill path in _index_fill_high_rank_fallback.
_MAX_RANK = 8
_BLOCK_SIZE = 512
_COPY_BLOCK_SIZE = 1024

_FALLBACK_KEYSET = torch._C.DispatchKeySet(
    torch._C.DispatchKey.CompositeExplicitAutograd
)


def _native_clone(inp):
    # Clone without re-dispatching into FlagGems-registered ops.
    return torch.ops.aten.clone.default.redispatch(_FALLBACK_KEYSET, inp)


def _native_copy_(out, src):
    return torch.ops.aten.copy_.default.redispatch(_FALLBACK_KEYSET, out, src, False)


@libentry()
@triton.jit
def index_fill_kernel(
    out,
    index,
    value,
    N,
    index_len,
    dim_size,
    shape_0,
    shape_1,
    shape_2,
    shape_3,
    shape_4,
    shape_5,
    shape_6,
    shape_7,
    stride_0,
    stride_1,
    stride_2,
    stride_3,
    stride_4,
    stride_5,
    stride_6,
    stride_7,
    DIM: tl.constexpr,
    RANK: tl.constexpr,
    VALUE_IS_TENSOR: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Generic strided path: iterate over the fill space, i.e. all coordinates
    # of `out` with the `DIM`-th coordinate replaced by a position in `index`.
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    linear = offsets.to(tl.int64)
    out_offsets = tl.zeros((BLOCK_SIZE,), dtype=tl.int64)
    shapes = (shape_0, shape_1, shape_2, shape_3, shape_4, shape_5, shape_6, shape_7)
    strides = (
        stride_0,
        stride_1,
        stride_2,
        stride_3,
        stride_4,
        stride_5,
        stride_6,
        stride_7,
    )
    for i in tl.static_range(RANK):
        size = index_len if i == DIM else shapes[i]
        coord = linear % size
        linear = linear // size
        if i == DIM:
            raw_index = tl.load(index + coord, mask=mask, other=0).to(tl.int64)
            valid_index = (raw_index >= -dim_size) & (raw_index < dim_size)
            coord = tl.where(raw_index < 0, raw_index + dim_size, raw_index)
        out_offsets += coord * strides[i]
    if VALUE_IS_TENSOR:
        fill_value = tl.load(value)
    else:
        fill_value = value
    # Out-of-range index entries are skipped silently by the store mask.
    # (PyTorch reports them as an error, but tl.device_assert fails to
    # compile on non-CUDA FlagGems backends, so the check is omitted.)
    tl.store(out + out_offsets, fill_value, mask=mask & valid_index)


@libentry()
@triton.jit
def index_fill_contiguous_kernel(
    out,
    index,
    value,
    outer_index_len,
    index_len,
    dim_size,
    inner_size,
    VALUE_IS_TENSOR: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Contiguous fast path: 2-D grid over (outer x index) rows and inner blocks.
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    inner_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    m_mask = m_offsets < outer_index_len
    index_coord = m_offsets % index_len
    outer_coord = m_offsets // index_len
    raw_index = tl.load(index + index_coord, mask=m_mask, other=0).to(tl.int64)
    valid_index = (raw_index >= -dim_size) & (raw_index < dim_size)
    normalized_index = tl.where(raw_index < 0, raw_index + dim_size, raw_index)
    out_offsets = outer_coord[:, None] * dim_size * inner_size
    out_offsets += normalized_index[:, None] * inner_size
    out_offsets += inner_offsets[None, :]
    store_mask = m_mask[:, None] & (inner_offsets[None, :] < inner_size)
    store_mask = store_mask & valid_index[:, None]
    if VALUE_IS_TENSOR:
        fill_value = tl.load(value)
    else:
        fill_value = value
    # Out-of-range index entries are skipped silently by the store mask.
    # (PyTorch reports them as an error, but tl.device_assert fails to
    # compile on non-CUDA FlagGems backends, so the check is omitted.)
    tl.store(out + out_offsets, fill_value, mask=store_mask)


@libentry()
@triton.jit
def index_fill_copy_kernel(out, inp, N, BLOCK_SIZE: tl.constexpr):
    # Flat copy for the out-of-place fast path. Inside a fully registered
    # FlagGems context, a native clone routes its internal copy_ to the
    # generic pointwise copy kernel (~2.7 TB/s), while this dedicated kernel
    # reaches ~3.4 TB/s.
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    tl.store(out + offsets, tl.load(inp + offsets, mask=mask), mask=mask)


def _index_fill_contiguous_launch(out, dim, index, value, value_is_tensor, fill_numel):
    dim_size = out.size(dim)
    inner_size = 1
    for i in range(dim + 1, out.ndim):
        inner_size *= out.shape[i]
    block_n = 1
    block_m = _BLOCK_SIZE
    if inner_size > 1:
        block_n = min(64, triton.next_power_of_2(inner_size))
        if inner_size <= 4:
            block_m = _BLOCK_SIZE
        else:
            block_m = max(1, _BLOCK_SIZE // block_n)
    outer_index_len = fill_numel // inner_size
    grid = (
        triton.cdiv(outer_index_len, block_m),
        triton.cdiv(inner_size, block_n),
    )
    index_fill_contiguous_kernel[grid](
        out,
        index,
        value,
        outer_index_len,
        index.numel(),
        dim_size,
        inner_size,
        VALUE_IS_TENSOR=value_is_tensor,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
    )


def _index_fill_high_rank_fallback(out, dim, index, value, value_is_tensor, fill_numel):
    # Rare path: strided tensors whose rank exceeds the number of shape/stride
    # slots of the generic kernel. Materialize a contiguous copy, fill it with
    # the rank-independent 2-D kernel, then write back into the strided view.
    # Native index_fill cannot be used here: its composite implementation
    # dispatches internally, which re-enters our registered kernels and
    # recurses under full registration.
    contig = torch.empty(out.shape, dtype=out.dtype, device=out.device)
    _native_copy_(contig, out)
    _index_fill_contiguous_launch(
        contig, dim, index, value, value_is_tensor, fill_numel
    )
    _native_copy_(out, contig)
    return out


def _index_fill_impl(out, dim, index, value, value_is_tensor):
    if out.numel() == 0 or index.numel() == 0:
        return out

    dim_size = out.size(dim)
    fill_numel = out.numel() // dim_size * index.numel()
    with torch_device_fn.device(out.device):
        if out.is_contiguous():
            # The 2-D kernel is rank-independent: inner_size is computed on the
            # host, so contiguous tensors of any rank take this path.
            _index_fill_contiguous_launch(
                out, dim, index, value, value_is_tensor, fill_numel
            )
        elif out.ndim > _MAX_RANK:
            return _index_fill_high_rank_fallback(
                out, dim, index, value, value_is_tensor, fill_numel
            )
        else:
            shapes = list(out.shape)
            strides = list(out.stride())
            while len(shapes) < _MAX_RANK:
                shapes.append(1)
                strides.append(0)
            grid = (triton.cdiv(fill_numel, _BLOCK_SIZE),)
            index_fill_kernel[grid](
                out,
                index,
                value,
                fill_numel,
                index.numel(),
                dim_size,
                *shapes,
                *strides,
                DIM=dim,
                RANK=out.ndim,
                VALUE_IS_TENSOR=value_is_tensor,
                BLOCK_SIZE=_BLOCK_SIZE,
            )
    return out


def _prepare_index(inp, dim, index):
    if inp.ndim == 0:
        raise IndexError("index_fill expects self to have at least one dimension")
    if dim < -inp.ndim or dim >= inp.ndim:
        raise IndexError(
            f"Dimension out of range (expected to be in range of "
            f"[{-inp.ndim}, {inp.ndim - 1}], but got {dim})"
        )
    dim = dim % inp.ndim

    if index.dtype != torch.long:
        raise IndexError("index_fill_(): Expected dtype int64 for index.")
    if index.device != inp.device:
        raise RuntimeError(
            "Expected all tensors to be on the same device, but found at least "
            f"two devices, {inp.device} and {index.device}!"
        )
    if index.ndim > 1:
        raise IndexError("index_fill_(): Index is supposed to be a vector")
    if index.ndim == 0:
        index = index.reshape(1)

    return dim, index


def _prepare_tensor_value(inp, value):
    if value.ndim != 0:
        raise RuntimeError(
            "index_fill_ only supports a 0-dimensional value tensor, "
            f"but got tensor with {value.ndim} dimension(s)."
        )
    if value.device.type == "cpu":
        return False, value.item()
    if value.device != inp.device:
        raise RuntimeError(
            "Expected all tensors to be on the same device, but found at least "
            f"two devices, {inp.device} and {value.device}!"
        )
    return True, value


def index_fill(inp, dim, index, value):
    # Entry for both `index_fill.int_Scalar` and `index_fill.int_Tensor`: the
    # dispatcher routes by value type, so a 0-dimensional tensor value arrives
    # as a Tensor and a plain number as a Python scalar.
    logger.debug("GEMS INDEX_FILL")
    dim, index = _prepare_index(inp, dim, index)
    if isinstance(value, torch.Tensor):
        value_is_tensor, value = _prepare_tensor_value(inp, value)
    else:
        value_is_tensor = False
    if inp.numel() == 0 or index.numel() == 0:
        return _native_clone(inp)
    if inp.is_contiguous():
        # Fast path: allocate the output and copy `inp` with a dedicated flat
        # copy kernel instead of cloning (see index_fill_copy_kernel), then
        # fill the selected positions. Any rank works here.
        out = torch.empty_like(inp)
        with torch_device_fn.device(inp.device):
            grid = (triton.cdiv(out.numel(), _COPY_BLOCK_SIZE),)
            index_fill_copy_kernel[grid](
                out, inp, out.numel(), BLOCK_SIZE=_COPY_BLOCK_SIZE
            )
        return _index_fill_impl(out, dim, index, value, value_is_tensor)
    out = _native_clone(inp)
    return _index_fill_impl(out, dim, index, value, value_is_tensor)


def index_fill_(inp, dim, index, value):
    # Entry for both `index_fill_.int_Scalar` and `index_fill_.int_Tensor`.
    logger.debug("GEMS INDEX_FILL_")
    dim, index = _prepare_index(inp, dim, index)
    if isinstance(value, torch.Tensor):
        value_is_tensor, value = _prepare_tensor_value(inp, value)
    else:
        value_is_tensor = False
    return _index_fill_impl(inp, dim, index, value, value_is_tensor)
