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

from flag_gems.ops.index_copy_ import index_copy as default_index_copy
from flag_gems.ops.index_copy_ import index_copy_ as default_index_copy_
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


def _cfgs():
    return [
        triton.Config({"BLOCK_SIZE": 128}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8),
    ]


@libentry()
@triton.autotune(configs=_cfgs(), key=["N"])
@triton.jit
def index_copy_kernel_1d(
    index_ptr,
    src_ptr,
    out_ptr,
    N,
    inp_stride_dim,
    inp_shape_dim,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    idx = tl.load(index_ptr + offsets, mask=mask, other=0).to(tl.int64)
    src_val = tl.load(src_ptr + offsets, mask=mask, other=0)

    out_offset = idx * inp_stride_dim
    store_mask = mask & (idx >= 0) & (idx < inp_shape_dim)
    tl.store(out_ptr + out_offset, src_val, mask=store_mask)


@libentry()
@triton.autotune(configs=_cfgs(), key=["N"])
@triton.jit
def index_copy_kernel_2d(
    index_ptr,
    src_ptr,
    out_ptr,
    N,
    inp_stride_0,
    inp_stride_1,
    src_stride_0,
    src_stride_1,
    src_shape_1,
    inp_shape_dim,
    dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    src_offset_1 = offsets % src_shape_1
    src_offset_0 = offsets // src_shape_1

    src_linear = src_offset_0 * src_stride_0 + src_offset_1 * src_stride_1

    if dim == 0:
        idx = tl.load(index_ptr + src_offset_0, mask=mask, other=0).to(tl.int64)
        out_linear = idx * inp_stride_0 + src_offset_1 * inp_stride_1
    else:
        idx = tl.load(index_ptr + src_offset_1, mask=mask, other=0).to(tl.int64)
        out_linear = src_offset_0 * inp_stride_0 + idx * inp_stride_1

    store_mask = mask & (idx >= 0) & (idx < inp_shape_dim)
    src_val = tl.load(src_ptr + src_linear, mask=mask, other=0)
    tl.store(out_ptr + out_linear, src_val, mask=store_mask)


@libentry()
@triton.autotune(configs=_cfgs(), key=["N"])
@triton.jit
def index_copy_kernel_3d(
    index_ptr,
    src_ptr,
    out_ptr,
    N,
    inp_stride_0,
    inp_stride_1,
    inp_stride_2,
    src_stride_0,
    src_stride_1,
    src_stride_2,
    src_shape_1,
    src_shape_2,
    inp_shape_dim,
    dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    src_offset_2 = offsets % src_shape_2
    tmp = offsets // src_shape_2
    src_offset_1 = tmp % src_shape_1
    src_offset_0 = tmp // src_shape_1

    src_linear = (
        src_offset_0 * src_stride_0
        + src_offset_1 * src_stride_1
        + src_offset_2 * src_stride_2
    )

    if dim == 0:
        idx = tl.load(index_ptr + src_offset_0, mask=mask, other=0).to(tl.int64)
        out_linear = (
            idx * inp_stride_0
            + src_offset_1 * inp_stride_1
            + src_offset_2 * inp_stride_2
        )
    elif dim == 1:
        idx = tl.load(index_ptr + src_offset_1, mask=mask, other=0).to(tl.int64)
        out_linear = (
            src_offset_0 * inp_stride_0
            + idx * inp_stride_1
            + src_offset_2 * inp_stride_2
        )
    else:
        idx = tl.load(index_ptr + src_offset_2, mask=mask, other=0).to(tl.int64)
        out_linear = (
            src_offset_0 * inp_stride_0
            + src_offset_1 * inp_stride_1
            + idx * inp_stride_2
        )

    store_mask = mask & (idx >= 0) & (idx < inp_shape_dim)
    src_val = tl.load(src_ptr + src_linear, mask=mask, other=0)
    tl.store(out_ptr + out_linear, src_val, mask=store_mask)


def _validate(inp, dim, index, src):
    # Only cheap, host-side (metadata) checks here. The element-wise index
    # bounds check (0 <= index < size) is intentionally NOT performed on the
    # device: under `flag_gems.use_gems()` such tensor ops get re-dispatched to
    # gems Triton kernels and dominate latency on small shapes (~0.9ms). The
    # kernel below instead applies the bounds as a store mask, keeping
    # out-of-range indices from corrupting memory while adding no host overhead.
    assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
    assert index.numel() == src.size(
        dim
    ), "The dimth dimension of source must have the same size as the length of index"
    assert (
        inp.ndim == src.ndim
    ), "Self and source should have the same number of dimensions"
    assert all(
        (inp.size(i) == src.size(i)) or i == dim for i in range(0, inp.ndim)
    ), "src.size(d) == self.size(d) for all dimensions d != dim"


def _launch(inp, dim, index, src):
    N = src.numel()
    if N == 0:
        return inp

    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE"]),)

    if inp.ndim == 1:
        index_copy_kernel_1d[grid](index, src, inp, N, inp.stride(0), inp.size(0))
    elif inp.ndim == 2:
        index_copy_kernel_2d[grid](
            index,
            src,
            inp,
            N,
            inp.stride(0),
            inp.stride(1),
            src.stride(0),
            src.stride(1),
            src.size(1),
            inp.size(dim),
            dim,
        )
    elif inp.ndim == 3:
        index_copy_kernel_3d[grid](
            index,
            src,
            inp,
            N,
            inp.stride(0),
            inp.stride(1),
            inp.stride(2),
            src.stride(0),
            src.stride(1),
            src.stride(2),
            src.size(1),
            src.size(2),
            inp.size(dim),
            dim,
        )
    return inp


_FALLBACK_KEYSET = torch._C.DispatchKeySet(
    torch._C.DispatchKey.CompositeExplicitAutograd
)


def index_copy(inp, dim, index, src):
    logger.debug("GEMS_HYGON INDEX_COPY")
    # The specialized kernels cover up to 3D; fall back to the generic
    # implementation for higher-rank tensors to preserve correctness.
    if inp.ndim > 3:
        return default_index_copy(inp, dim, index, src)
    _validate(inp, dim, index, src)
    dim %= inp.ndim
    # Native clone to avoid re-dispatch into gems copy_/clone kernels under
    # `flag_gems.use_gems()`, which would add significant host overhead.
    out = torch.ops.aten.clone.default.redispatch(_FALLBACK_KEYSET, inp)
    return _launch(out, dim, index, src)


def index_copy_(inp, dim, index, src):
    logger.debug("GEMS_HYGON INDEX_COPY_")
    # The specialized kernels cover up to 3D; fall back to the generic
    # implementation for higher-rank tensors to preserve correctness.
    if inp.ndim > 3:
        return default_index_copy_(inp, dim, index, src)
    _validate(inp, dim, index, src)
    dim %= inp.ndim
    return _launch(inp, dim, index, src)
