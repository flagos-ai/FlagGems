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

from flag_gems.utils.shape_utils import MemOverlap, has_internal_overlapping

logger = logging.getLogger(__name__)

_LASTDIM_W_TARGET_BYTES = 131072  # 2R1W flat streaming sweet spot: ~128KB per tile
_NUM_PROGS = 48
_INT32_MAX = 2**31 - 1


@triton.jit
def select_scatter_kernel(
    out_ptr,
    inp_ptr,
    src_ptr,
    total_elements,
    dim_size,
    dim_prod_post,
    index,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = block_start + offsets < total_elements
    idx = block_start + offsets

    pre_idx = idx // (dim_size * dim_prod_post)
    dim_idx = (idx // dim_prod_post) % dim_size
    post_idx = idx % dim_prod_post

    select_mask = dim_idx == index

    inp_data = tl.load(inp_ptr + idx, mask=mask)

    src_idx = pre_idx * dim_prod_post + post_idx
    src_data = tl.load(src_ptr + src_idx, mask=mask & select_mask)
    result = tl.where(select_mask, src_data, inp_data)
    tl.store(out_ptr + idx, result, mask=mask)


@triton.jit
def _select_scatter_lastdim_kernel(
    out_ptr,
    inp_ptr,
    src_ptr,
    num_lines,
    line_len,
    index,
    K: tl.constexpr,
    W: tl.constexpr,
):
    # Fast path for dim_prod_post == 1: the tensor is num_lines rows of
    # line_len elements; row r takes src[r] at column `index`. Each block is a
    # flat maskless copy of K consecutive rows followed by a K-lane scatter
    # that patches the selected column, so the select costs O(1/line_len)
    # per element instead of a per-element where/div-mod chain. The tail block
    # is anchored with a scalar clamp (idempotent re-write of the overlap),
    # keeping every load/store maskless. The patch store overlaps the copy
    # store's address range; a debug_barrier (store buffer flush) between them
    # is required for ordering — see compiler_traps 2026-11-11 WAW entry.
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)
    total_blocks = tl.cdiv(num_lines, K)
    for block_id in tl.range(pid, total_blocks, num_progs):
        line0 = tl.minimum(block_id * K, num_lines - K)
        offs = line0 * line_len + tl.arange(0, W)
        if K == 1:
            # width-1 vectors are pathological on this backend; patch via scalars
            src_val = tl.load(src_ptr + line0)
            inp_data = tl.load(inp_ptr + offs)
            tl.store(out_ptr + offs, inp_data)
            # scalar stores do not queue on the vector/DMA store path: without
            # a barrier the wide copy store may land after the patch and
            # clobber it (observed ~50% wrong values on (1024, 65536)).
            tl.debug_barrier()
            tl.store(out_ptr + line0 * line_len + index, src_val)
        else:
            lines = line0 + tl.arange(0, K)
            src_vec = tl.load(src_ptr + lines)
            inp_data = tl.load(inp_ptr + offs)
            tl.store(out_ptr + offs, inp_data)
            tl.debug_barrier()
            tl.store(out_ptr + lines * line_len + index, src_vec)


def _is_pow2(n):
    return n > 0 and (n & (n - 1)) == 0


def _launch_lastdim(out, inp, src, dim_size, index, total_elements):
    line_len = dim_size
    num_lines = total_elements // line_len
    itemsize = inp.element_size()
    w_target = _LASTDIM_W_TARGET_BYTES // itemsize
    k_cap = w_target // line_len
    if k_cap < 1:
        k_cap = 1
    if k_cap > num_lines:
        k_cap = num_lines
    # Below ~1 MB of traffic a single program is bandwidth-starved; bias
    # towards >= 16 blocks so small-to-mid shapes keep CTA parallelism.
    # Launch-bound tiny tensors (<= 64 KB) skip the bias: fewer, fatter
    # blocks win there.
    if total_elements * itemsize >= 65536:
        k_grid = num_lines // 16
        if k_grid < 1:
            k_grid = 1
        if k_cap > k_grid:
            k_cap = k_grid
    k = 1 << (k_cap.bit_length() - 1)
    total_blocks = (num_lines + k - 1) // k
    grid = (min(_NUM_PROGS, total_blocks),)
    _select_scatter_lastdim_kernel[grid](
        out,
        inp,
        src,
        num_lines,
        line_len,
        index,
        K=k,
        W=k * line_len,
        num_warps=2,
    )


def select_scatter(inp, src, dim, index):
    logger.debug("GEMS_ENFLAME SELECT_SCATTER")
    assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
    assert index >= -inp.size(dim) and index < inp.size(dim), "Invalid index"
    dim = dim % inp.ndim
    index = index % inp.size(dim)

    valid_shape = list(inp.shape)
    del valid_shape[dim]
    assert (
        list(src.shape) == valid_shape
    ), "Expected src to have a size equal to the slice of self"

    if has_internal_overlapping(inp) == MemOverlap.Yes:
        out = torch.empty(inp.size(), dtype=inp.dtype, device=inp.device)
    elif inp.is_contiguous():
        out = torch.empty_strided(
            inp.size(), inp.stride(), dtype=inp.dtype, device=inp.device
        )
    else:
        # Non-contiguous input: the flat-addressing kernels below require a
        # contiguous output; allocating with inp's strides would scramble the
        # written values (pre-existing issue in the generic path as well).
        out = torch.empty(inp.size(), dtype=inp.dtype, device=inp.device)

    total_elements = inp.numel()
    dim_size = inp.size(dim)

    dim_prod_post = 1
    for d in range(dim + 1, inp.ndim):
        dim_prod_post *= inp.size(d)

    fast_path = (
        dim_prod_post == 1
        and _is_pow2(dim_size)
        and 2 <= dim_size <= 65536
        and total_elements < _INT32_MAX
        and out.is_contiguous()
    )
    if fast_path:
        src_c = src if src.is_contiguous() else src.contiguous()
        inp_c = inp if inp.is_contiguous() else inp.contiguous()
        _launch_lastdim(out, inp_c, src_c, dim_size, index, total_elements)
        return out

    inp = inp.contiguous()
    src = src.contiguous()

    BLOCK_SIZE = 4096
    if total_elements // BLOCK_SIZE > 65535:
        BLOCK_SIZE = 16384
    grid = (triton.cdiv(total_elements, BLOCK_SIZE),)

    select_scatter_kernel[grid](
        out,
        inp,
        src,
        total_elements,
        dim_size,
        dim_prod_post,
        index,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out
