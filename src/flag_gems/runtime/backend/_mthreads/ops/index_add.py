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

from flag_gems import runtime
from flag_gems.ops.index_add import _validate_index_add_args
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import dim_compress, libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)

_INDEX_OUT_OF_BOUNDS_MESSAGE = "0 <= index < self.size(dim)"


def _read_index_bounds(index):
    return index.min().item(), index.max().item()


def _resolve_index_for_kernel(index):
    # A contiguous lazy-negative tensor still exposes the un-negated storage
    # to a pointer-based Triton kernel. Materialize only that exceptional case.
    # Calling resolve_neg() from inside use_gems() re-enters FlagGems' Python
    # override and can negate the logical value twice. Toggle the metadata bit
    # off first, then explicitly negate the ordinary physical view.
    if index.is_neg():
        return torch.neg(torch._neg_view(index))
    return index


def _assert_index_in_bounds(index, upper_bound):
    if index.numel() == 0:
        return
    idx_min, idx_max = _read_index_bounds(index)
    if idx_min < 0 or idx_max >= upper_bound:
        raise AssertionError(_INDEX_OUT_OF_BOUNDS_MESSAGE)


def _volume(shape):
    value = 1
    for item in shape:
        value *= int(item)
    return value


def _can_use_contiguous_suffix_path(inp, dim, index, src):
    return (
        src.numel() > 0
        and inp.ndim == src.ndim
        and 0 <= dim < inp.ndim
        and index.ndim == 1
        and index.dtype in (torch.int32, torch.int64)
        and inp.dtype == src.dtype
        and inp.dtype in (torch.float16, torch.float32)
        and index.numel() == src.size(dim)
        and inp.is_contiguous()
        and src.is_contiguous()
        and all(inp.size(i) == src.size(i) for i in range(inp.ndim) if i != dim)
        and _volume(src.shape[dim + 1 :]) > 1
    )


@libentry()
@triton.heuristics(runtime.get_heuristic_config("index_add"))
@triton.jit
def index_add_kernel(
    out_ptr,
    index_ptr,
    src_ptr,
    M,
    N,
    alpha,
    inp_len,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Kernel for index_add operation with autotune.

    After dim_compress, tensors are reshaped so that:
    - inp has shape (M, inp_len) where inp_len is the size of target dimension
    - src has shape (M, N) where N is the size of index

    For each row m and each index position n:
        out[m, index[n]] += alpha * src[m, n]
    """
    pid_m = ext.program_id(axis=0)
    pid_n = ext.program_id(axis=1)

    # Calculate row and column offsets
    rows_offset = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    cols_offset = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]

    # Create masks
    rows_mask = rows_offset < M
    cols_mask = cols_offset < N
    block_mask = rows_mask & cols_mask

    # Load indices for this block of columns
    cur_indices = tl.load(index_ptr + cols_offset, mask=cols_mask, other=0)

    # Calculate offsets into inp/out (which has shape M x inp_len)
    inp_off = rows_offset * inp_len + cur_indices

    # Calculate offsets into src (which has shape M x N)
    src_off = rows_offset * N + cols_offset

    # Load source values
    cur_src = tl.load(src_ptr + src_off, mask=block_mask, other=0.0)

    # Use atomic_add to correctly handle repeated indices in index,
    # aligned with the common op (src/flag_gems/ops/index_add.py).
    # When multiple source elements map to the same output position (duplicate
    # indices), plain load-store would cause race conditions or lost updates.
    # atomic_add guarantees all contributions are accumulated correctly.
    tl.atomic_add(out_ptr + inp_off, alpha * cur_src, mask=block_mask)


@libentry()
@triton.jit
def _index_add_contiguous_suffix_kernel(
    out,
    index,
    src,
    row_count,
    index_len,
    out_dim,
    suffix_size,
    alpha,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    rows = ext.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    cols = ext.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
    row_mask = rows < row_count
    mask = row_mask & (cols < suffix_size)
    edge = rows % index_len
    prefix = rows // index_len
    receiver = tl.load(index + edge, mask=row_mask, other=0).to(tl.int64)
    src_offsets = rows * suffix_size + cols
    out_offsets = (prefix * out_dim + receiver) * suffix_size + cols
    values = tl.load(src + src_offsets, mask=mask, other=0.0)
    tl.atomic_add(out + out_offsets, values * alpha, mask=mask)


def _contiguous_suffix_config(suffix_size):
    block_n = min(512, triton.next_power_of_2(suffix_size))
    return 4, block_n


def _run_contiguous_suffix_path(out, dim, index, src, alpha):
    suffix_size = _volume(src.shape[dim + 1 :])
    row_count = _volume(src.shape[:dim]) * index.numel()
    block_m, block_n = _contiguous_suffix_config(suffix_size)
    grid = (
        triton.cdiv(row_count, block_m),
        triton.cdiv(suffix_size, block_n),
    )
    with torch_device_fn.device(out.device):
        _index_add_contiguous_suffix_kernel[grid](
            out,
            index,
            src,
            row_count,
            index.numel(),
            out.size(dim),
            suffix_size,
            alpha,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
        )
    return out


def index_add(inp, dim, index, src, alpha=1):
    """
    Optimized index_add for mthreads backend.

    self.index_add_(dim, index, source, alpha=1) -> Tensor

    For a 3-D tensor the output is:
        self[index[i], :, :] += alpha * src[i, :, :]  # if dim == 0
        self[:, index[i], :] += alpha * src[:, i, :]  # if dim == 1
        self[:, :, index[i]] += alpha * src[:, :, i]  # if dim == 2
    """
    logger.debug("GEMS_MTHREADS INDEX_ADD")

    dim = _validate_index_add_args(inp, dim, index, src)
    if src.numel() == 0:
        return inp.clone(memory_format=torch.contiguous_format)

    use_contiguous_suffix_path = _can_use_contiguous_suffix_path(
        inp, dim, index, src
    ) and not torch._C._is_alias_of(inp, src)

    # Make inputs contiguous. resolve_neg() is a no-op for normal indices.
    inp = inp.contiguous()
    index = _resolve_index_for_kernel(index).contiguous()
    src = src.contiguous()

    inp_len = inp.size(dim)
    N = index.numel()
    M = src.numel() // N

    # Reject invalid receivers before a pointer kernel can observe them.
    # Use min/max to avoid allocating full-size boolean tensors.
    _assert_index_in_bounds(index, inp_len)

    if use_contiguous_suffix_path:
        out = inp.clone()
        return _run_contiguous_suffix_path(out, dim, index, src, alpha)

    # Move target dim to last position for coalesced memory access
    final_dim = inp.ndim - 1
    if dim != final_dim:
        inp = dim_compress(inp, dim)
        src = dim_compress(src, dim)

    # Clone input for output
    out = inp.clone()

    # Calculate grid with autotune
    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]),
        triton.cdiv(N, meta["BLOCK_N"]),
    )

    with torch_device_fn.device(inp.device):
        index_add_kernel[grid](out, index, src, M, N, alpha, inp_len)

    # Restore original dimension order if needed
    if dim != final_dim:
        order = list(range(out.ndim - 1))
        order.insert(dim, final_dim)
        return out.permute(order).contiguous()
    else:
        return out


def index_add_(inp, dim, index, src, alpha=1):
    """
    In-place version of index_add.
    """
    logger.debug("GEMS_MTHREADS INDEX_ADD_")

    dim = _validate_index_add_args(inp, dim, index, src)
    if src is inp or index is inp:
        raise RuntimeError(
            "input overlaps with source or index; clone the overlapping tensor "
            "before calling index_add_"
        )
    if src.numel() == 0:
        return inp
    if torch._C._is_alias_of(inp, src) or torch._C._is_alias_of(inp, index):
        raise RuntimeError(
            "input overlaps with source or index; clone the overlapping tensor "
            "before calling index_add_"
        )

    use_contiguous_suffix_path = _can_use_contiguous_suffix_path(
        inp, dim, index, src
    ) and not torch._C._is_alias_of(inp, src)

    # Make index and src contiguous. resolve_neg() is a no-op normally.
    index = _resolve_index_for_kernel(index).contiguous()
    src = src.contiguous()

    inp_len = inp.size(dim)
    N = index.numel()
    M = src.numel() // N

    # Reject invalid receivers before a pointer kernel can observe them.
    # Use min/max to avoid allocating full-size boolean tensors.
    _assert_index_in_bounds(index, inp_len)

    if use_contiguous_suffix_path:
        return _run_contiguous_suffix_path(inp, dim, index, src, alpha)

    # Move target dim to last position
    final_dim = inp.ndim - 1

    if dim != final_dim:
        # Need to work on a permuted copy
        inp_work = dim_compress(inp.clone().contiguous(), dim)
        src_work = dim_compress(src, dim)

        # Calculate grid with autotune
        grid = lambda meta: (
            triton.cdiv(M, meta["BLOCK_M"]),
            triton.cdiv(N, meta["BLOCK_N"]),
        )

        with torch_device_fn.device(inp.device):
            index_add_kernel[grid](inp_work, index, src_work, M, N, alpha, inp_len)

        # Restore original dimension order and copy back
        order = list(range(inp_work.ndim - 1))
        order.insert(dim, final_dim)
        inp_work = inp_work.permute(order).contiguous()
        inp.copy_(inp_work)
    else:
        # Can work directly on input if already contiguous
        inp_contig = inp.contiguous()

        # Calculate grid with autotune
        grid = lambda meta: (
            triton.cdiv(M, meta["BLOCK_M"]),
            triton.cdiv(N, meta["BLOCK_N"]),
        )

        with torch_device_fn.device(inp.device):
            index_add_kernel[grid](inp_contig, index, src, M, N, alpha, inp_len)

        # Copy back if input wasn't contiguous
        if not inp.is_contiguous():
            inp.copy_(inp_contig)

    return inp
