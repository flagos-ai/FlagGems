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


@libentry()
@triton.jit
def _hygon_clone_kernel(inp, out, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    value = tl.load(inp + offsets, mask=mask)
    tl.store(out + offsets, value, mask=mask)


def _clone_without_copy_dispatch(inp):
    # Local clone so that re-dispatch into gems copy_/clone kernels under
    # `flag_gems.use_gems()` cannot add its host overhead. Non-contiguous
    # inputs still go through aten, since the flat-index kernel assumes a
    # contiguous layout.
    if not inp.is_contiguous():
        return torch.ops.aten.clone.default.redispatch(_FALLBACK_KEYSET, inp)

    out = torch.empty_like(inp)
    n_elements = inp.numel()
    if n_elements == 0:
        return out

    # 8 elements per lane. With the previous BLOCK_SIZE=256 (2 per lane) the
    # copy was issue-bound at 304 GB/s on Hygon regardless of dtype; this
    # reaches 1311 GB/s (measured, fp16 8192x8192), which is what the hardware
    # actually has.
    block_size = 2048
    grid = (triton.cdiv(n_elements, block_size),)
    _hygon_clone_kernel[grid](inp, out, n_elements, BLOCK_SIZE=block_size, num_warps=8)
    return out


@libentry()
@triton.jit
def _small_index_copy_kernel(
    inp_ptr,
    index_ptr,
    src_ptr,
    out_ptr,
    inp_stride_0,
    src_stride_0,
    C,
    K,
    BLOCK_K: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    # The index is a flat, row-shared vector (see _validate), so a column
    # block can resolve its own winner: the largest k with index[k] == c.
    # Out-of-range or negative entries match no column and drop out exactly
    # like the scatter kernel's store mask.
    row = tl.program_id(axis=0)
    c = tl.program_id(axis=1) * BLOCK_C + tl.arange(0, BLOCK_C)
    c_mask = c < C
    k = tl.arange(0, BLOCK_K)
    # -1 padding can never equal a column index.
    idx = tl.load(index_ptr + k, mask=k < K, other=-1).to(tl.int64)
    winner = tl.max(
        tl.where(c[:, None].to(tl.int64) == idx[None, :], k[None, :] + 1, 0), axis=1
    )
    inp_val = tl.load(inp_ptr + row * inp_stride_0 + c, mask=c_mask, other=0)
    src_val = tl.load(
        src_ptr + row * src_stride_0 + winner - 1,
        mask=c_mask & (winner > 0),
        other=0,
    )
    tl.store(
        out_ptr + row * inp_stride_0 + c,
        tl.where(winner > 0, src_val, inp_val),
        mask=c_mask,
    )


def _try_small_index_copy(inp, dim, index, src):
    # Below ~256K elements the launch and host-gap of the clone + scatter
    # pair dominate the runtime, so one fused kernel that writes every
    # output element exactly once wins on wall clock even though each
    # program re-resolves the index.
    if (
        dim != 1
        or inp.ndim != 2
        or inp.numel() == 0
        or index.numel() == 0
        or index.numel() > 64
        or inp.numel() > 262144
        or not inp.is_contiguous()
        or not src.is_contiguous()
    ):
        return None

    rows, cols = inp.shape
    k_len = src.size(1)
    out = torch.empty_like(inp)
    block_c = min(triton.next_power_of_2(cols), 128)
    block_k = triton.next_power_of_2(k_len)
    grid = (rows, triton.cdiv(cols, block_c))
    _small_index_copy_kernel[grid](
        inp,
        index,
        src,
        out,
        inp.stride(0),
        src.stride(0),
        cols,
        k_len,
        BLOCK_C=block_c,
        BLOCK_K=block_k,
        num_warps=max(1, block_c // 32),
    )
    return out


def index_copy(inp, dim, index, src):
    logger.debug("GEMS_HYGON INDEX_COPY")
    # The specialized kernels cover up to 3D; fall back to the generic
    # implementation for higher-rank tensors to preserve correctness.
    if inp.ndim > 3:
        return default_index_copy(inp, dim, index, src)
    _validate(inp, dim, index, src)
    dim %= inp.ndim
    # Small shapes skip the clone + scatter pair and run on a single fused
    # kernel instead.
    out = _try_small_index_copy(inp, dim, index, src)
    if out is not None:
        return out
    out = _clone_without_copy_dispatch(inp)
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
