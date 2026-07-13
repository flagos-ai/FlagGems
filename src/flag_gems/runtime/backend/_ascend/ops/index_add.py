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
from flag_gems.utils import dim_compress, libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)

_FALLBACK_KEYSET = torch._C.DispatchKeySet(
    torch._C.DispatchKey.CompositeExplicitAutograd
)


@libentry()
@triton.jit
def index_add_kernel(
    inp_ptr,
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
    pid_m = tle.program_id(axis=0)
    pid_n = tle.program_id(axis=1)

    rows_offset = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    cols_offset = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]

    rows_mask = rows_offset < M
    cols_mask = cols_offset < N
    block_mask = rows_mask & cols_mask

    cur_indices = tl.load(index_ptr + cols_offset, mask=cols_mask, other=0)

    inp_off = rows_offset * inp_len + cur_indices
    cur_inp = tl.load(inp_ptr + inp_off, mask=block_mask, other=0.0)

    src_off = rows_offset * N + cols_offset
    cur_src = tl.load(src_ptr + src_off, mask=block_mask, other=0.0)

    result = cur_inp + alpha * cur_src
    tl.store(out_ptr + inp_off, result, mask=block_mask)


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
    ROW_BLOCKS_PER_PROGRAM: tl.constexpr,
):
    pid_row_group = tle.program_id(axis=0)
    pid_n = tle.program_id(axis=1)

    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    col_mask = cols < suffix_size
    active_group = pid_row_group > 0

    # Contiguous tensors are viewed as [prefix, index_len, suffix].
    # Row grouping keeps Ascend launch axes within the device limit.  The first
    # row-group is intentionally unused; it absorbs repeated program-0 execution
    # seen during Ascend lowering while preserving atomic-add semantics.
    for block_offset in tl.static_range(0, ROW_BLOCKS_PER_PROGRAM):
        pid_row = (pid_row_group - 1) * ROW_BLOCKS_PER_PROGRAM + block_offset
        # Keep each atomic update one-dimensional.  The Ascend lowering of a
        # [BLOCK_M, BLOCK_N] atomic tile is substantially slower for wide suffixes.
        for row_offset in tl.static_range(0, BLOCK_M):
            row = pid_row * BLOCK_M + row_offset
            row_mask = active_group & (row < row_count)

            src_dim_idx = row % index_len
            prefix_idx = row // index_len
            dst_dim_idx = tl.load(index + src_dim_idx, mask=row_mask, other=0).to(
                tl.int64
            )
            valid = row_mask & (dst_dim_idx >= 0) & (dst_dim_idx < out_dim)

            src_offsets = row * suffix_size + cols
            out_offsets = (prefix_idx * out_dim + dst_dim_idx) * suffix_size + cols
            values = tl.load(src + src_offsets, mask=row_mask & col_mask, other=0.0)
            tl.atomic_add(
                out + out_offsets,
                values * alpha,
                mask=valid & col_mask,
                sem="relaxed",
            )


@libentry()
@triton.jit
def _index_add_contiguous_suffix_flat_kernel(
    out,
    index,
    src,
    total_count,
    index_len,
    out_dim,
    suffix_size,
    alpha,
    BLOCK_SIZE: tl.constexpr,
    BLOCKS_PER_PROGRAM: tl.constexpr,
):
    pid_group = tle.program_id(axis=0)

    # A 1D layout is safer for leading-dim and tiny-suffix updates.  Program 0
    # is a no-op for the same reason as the row-tiled kernel above.
    active_group = pid_group > 0
    for block_offset in tl.static_range(0, BLOCKS_PER_PROGRAM):
        pid = (pid_group - 1) * BLOCKS_PER_PROGRAM + block_offset
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = active_group & (offsets < total_count)

        cols = offsets % suffix_size
        rows = offsets // suffix_size
        src_dim_idx = rows % index_len
        prefix_idx = rows // index_len
        dst_dim_idx = tl.load(index + src_dim_idx, mask=mask, other=0).to(tl.int64)
        valid = mask & (dst_dim_idx >= 0) & (dst_dim_idx < out_dim)

        src_offsets = rows * suffix_size + cols
        out_offsets = (prefix_idx * out_dim + dst_dim_idx) * suffix_size + cols
        values = tl.load(src + src_offsets, mask=mask, other=0.0)
        tl.atomic_add(out + out_offsets, values * alpha, mask=valid, sem="relaxed")


def _get_block_config(M, N):
    BLOCK_M = 4 if M < 4096 else 8
    BLOCK_N = max(4, min(512, triton.next_power_of_2(N)))
    return BLOCK_M, BLOCK_N


def _volume(shape):
    value = 1
    for item in shape:
        value *= int(item)
    return value


def _assert_index_in_bounds(index, upper_bound):
    # Validate before scatter so an in-place call leaves its input unchanged on error.
    lower, upper = torch.ops.aten.aminmax.default.redispatch(
        _FALLBACK_KEYSET, index, dim=None, keepdim=False
    )
    assert (
        lower.item() >= 0 and upper.item() < upper_bound
    ), "0 <= index < self.size(dim)"


def _can_use_contiguous_suffix_path(inp, dim, index, src):
    if src.numel() == 0:
        return False
    if not (
        inp.ndim == src.ndim
        and 0 <= dim < inp.ndim
        and index.ndim == 1
        and index.dtype in (torch.int32, torch.int64)
        and inp.dtype == src.dtype
        and index.numel() == src.size(dim)
        and inp.is_contiguous()
        and src.is_contiguous()
        and all(inp.size(i) == src.size(i) for i in range(inp.ndim) if i != dim)
    ):
        return False
    return _volume(src.shape[dim + 1 :]) > 1


def _flat_contiguous_suffix_block_config(total_count):
    block_size = 128
    blocks = triton.cdiv(total_count, block_size)
    blocks_per_program = 1
    if blocks > 65535:
        blocks_per_program = triton.next_power_of_2(triton.cdiv(blocks, 65535))
        if blocks_per_program > 16:
            return None
    return block_size, blocks, blocks_per_program


def _run_contiguous_suffix_flat_path(out, dim, index, src, alpha):
    suffix_size = _volume(src.shape[dim + 1 :])
    row_count = _volume(src.shape[:dim]) * index.numel()
    total_count = row_count * suffix_size
    config = _flat_contiguous_suffix_block_config(total_count)
    if config is None:
        return False

    block_size, blocks, blocks_per_program = config
    grid = (triton.cdiv(blocks, blocks_per_program) + 1,)
    with torch_device_fn.device(out.device):
        _index_add_contiguous_suffix_flat_kernel[grid](
            out,
            index,
            src,
            total_count,
            index.numel(),
            out.size(dim),
            suffix_size,
            alpha,
            BLOCK_SIZE=block_size,
            BLOCKS_PER_PROGRAM=blocks_per_program,
        )
    return True


def _contiguous_suffix_block_config(row_count, suffix_size):
    block_m = 16
    block_n = 64 if suffix_size <= 64 else 512
    if triton.cdiv(suffix_size, block_n) > 65535:
        return None

    row_blocks = triton.cdiv(row_count, block_m)
    row_blocks_per_program = 1
    if row_blocks > 65535:
        row_blocks_per_program = triton.next_power_of_2(triton.cdiv(row_blocks, 65535))
        if row_blocks_per_program > 16:
            return None
    return block_m, block_n, row_blocks, row_blocks_per_program


def _run_contiguous_suffix_path(out, dim, index, src, alpha):
    suffix_size = _volume(src.shape[dim + 1 :])
    if dim == 0 or suffix_size < 16:
        return _run_contiguous_suffix_flat_path(out, dim, index, src, alpha)

    row_count = _volume(src.shape[:dim]) * index.numel()
    config = _contiguous_suffix_block_config(row_count, suffix_size)
    if config is None:
        return False

    block_m, block_n, row_blocks, row_blocks_per_program = config

    grid = (
        triton.cdiv(row_blocks, row_blocks_per_program) + 1,
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
            ROW_BLOCKS_PER_PROGRAM=row_blocks_per_program,
        )
    return True


def index_add(inp, dim, index, src, alpha=1):
    logger.debug("GEMS_ASCEND INDEX_ADD")

    inp = inp.contiguous()
    index = index.contiguous()
    src = src.contiguous()

    dim = dim % inp.ndim
    inp_len = inp.size(dim)
    N = index.numel()
    M = src.numel() // N

    normalized_dim = dim % inp.ndim if -inp.ndim <= dim < inp.ndim else dim
    if _can_use_contiguous_suffix_path(inp, normalized_dim, index, src):
        _assert_index_in_bounds(index, inp.size(dim))
        out = inp.clone()
        if _run_contiguous_suffix_path(out, normalized_dim, index, src, alpha):
            return out

    final_dim = inp.ndim - 1
    if dim != final_dim:
        inp = dim_compress(inp, dim)
        src = dim_compress(src, dim)

    out = inp.clone()

    BLOCK_M, BLOCK_N = _get_block_config(M, N)
    grid = (
        triton.cdiv(M, BLOCK_M),
        triton.cdiv(N, BLOCK_N),
    )

    with torch_device_fn.device(inp.device):
        index_add_kernel[grid](
            inp,
            out,
            index,
            src,
            M,
            N,
            alpha,
            inp_len,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
        )

    if dim != final_dim:
        order = list(range(out.ndim - 1))
        order.insert(dim, final_dim)
        return out.permute(order).contiguous()
    else:
        return out


def index_add_(inp, dim, index, src, alpha=1):
    logger.debug("GEMS_ASCEND INDEX_ADD_")

    index = index.contiguous()
    src = src.contiguous()

    dim = dim % inp.ndim
    inp_len = inp.size(dim)
    N = index.numel()
    M = src.numel() // N

    normalized_dim = dim % inp.ndim if -inp.ndim <= dim < inp.ndim else dim
    if _can_use_contiguous_suffix_path(inp, normalized_dim, index, src):
        _assert_index_in_bounds(index, inp.size(dim))
        if _run_contiguous_suffix_path(inp, normalized_dim, index, src, alpha):
            return inp

    final_dim = inp.ndim - 1

    if dim != final_dim:
        inp_work = dim_compress(inp.clone().contiguous(), dim)
        src_work = dim_compress(src, dim)
        out_work = inp_work.clone()

        BLOCK_M, BLOCK_N = _get_block_config(M, N)
        grid = (
            triton.cdiv(M, BLOCK_M),
            triton.cdiv(N, BLOCK_N),
        )

        with torch_device_fn.device(inp.device):
            index_add_kernel[grid](
                inp_work,
                out_work,
                index,
                src_work,
                M,
                N,
                alpha,
                inp_len,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
            )

        order = list(range(out_work.ndim - 1))
        order.insert(dim, final_dim)
        inp_work = out_work.permute(order).contiguous()
        inp.copy_(inp_work)
    else:
        inp_contig = inp.contiguous()
        out_contig = inp_contig.clone()

        BLOCK_M, BLOCK_N = _get_block_config(M, N)
        grid = (
            triton.cdiv(M, BLOCK_M),
            triton.cdiv(N, BLOCK_N),
        )

        with torch_device_fn.device(inp.device):
            index_add_kernel[grid](
                inp_contig,
                out_contig,
                index,
                src,
                M,
                N,
                alpha,
                inp_len,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
            )

        if inp.is_contiguous():
            inp.copy_(out_contig)
        else:
            inp.copy_(out_contig)

    return inp
