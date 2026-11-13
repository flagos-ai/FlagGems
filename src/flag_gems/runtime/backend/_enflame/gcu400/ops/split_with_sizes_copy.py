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

import functools
import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def split_indexed_copy_kernel(
    out_ptr,
    inp_ptr,
    split_offsets_ptr,
    n_elements,
    dim_size,
    dim_prod_pre: tl.constexpr,
    dim_prod_post: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Copy all split outputs into one flat output storage.

    Output storage layout is the concatenation of each split output in
    split_sizes order. split_offsets_ptr stores prefix sums along split dim.
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    mask = idx < n_elements

    split_start = tl.full((BLOCK_SIZE,), 0, tl.int64)
    split_size = tl.full((BLOCK_SIZE,), 1, tl.int64)
    output_start = tl.full((BLOCK_SIZE,), 0, tl.int64)

    for split_id in tl.static_range(0, NUM_SPLITS):
        dim_start = tl.load(split_offsets_ptr + split_id)
        dim_end = tl.load(split_offsets_ptr + split_id + 1)
        current_split_size = dim_end - dim_start
        current_output_start = dim_start * dim_prod_pre * dim_prod_post
        current_output_end = dim_end * dim_prod_pre * dim_prod_post
        in_current_split = (idx >= current_output_start) & (idx < current_output_end)

        split_start = tl.where(in_current_split, dim_start, split_start)
        split_size = tl.where(in_current_split, current_split_size, split_size)
        output_start = tl.where(in_current_split, current_output_start, output_start)

    local_idx = idx - output_start
    pre_idx = local_idx // (split_size * dim_prod_post)
    split_idx = (local_idx // dim_prod_post) % split_size
    post_idx = local_idx % dim_prod_post

    input_idx = (
        pre_idx * dim_size * dim_prod_post
        + (split_start + split_idx) * dim_prod_post
        + post_idx
    )

    data = tl.load(inp_ptr + input_idx, mask=mask)
    tl.store(out_ptr + idx, data, mask=mask)


@libentry()
@triton.jit
def split_flat_copy_kernel(
    inp_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Flat contiguous copy for small element counts (one block per program)."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    data = tl.load(inp_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, data, mask=mask)


@libentry()
@triton.jit
def split_persistent_copy_kernel(
    inp_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Grid-stride persistent contiguous copy.

    GCU400 dispatches ~50ns per CTA, so a cdiv(n, 1024)-program grid is
    dispatch-floor bound for large n. A grid capped at 48 programs with a wide
    BLOCK_SIZE keeps the copy memory bound instead.
    """
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    stride = num_programs * BLOCK_SIZE
    for start in tl.range(pid * BLOCK_SIZE, n_elements, stride):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        data = tl.load(inp_ptr + offsets, mask=mask)
        tl.store(out_ptr + offsets, data, mask=mask)


@libentry()
@triton.jit(
    do_not_specialize=[
        "blocks_per_split",
        "max_blocks",
        "dim_size",
        "dim_prod_post",
    ]
)
def split_segmented_copy_kernel(
    out_ptr,
    inp_ptr,
    dim_offsets_ptr,
    out_offsets_ptr,
    blocks_per_split,
    max_blocks,
    dim_size,
    dim_prod_post,
    NUM_SPLITS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """
    Segment-slab copy for dim != 0.

    Splits partition the dim, so every (split, pre-index) slab is contiguous in
    both the input and the concatenated output storage. Work item id decomposes
    as (split_id, pre_id, block_id); split scalars are loaded once per
    (program, split) because per-item scalar loads are latency bound here. All
    addressing is affine 1D, avoiding the where-chain / vector div-mod index
    math that miscompiles on this backend.
    """
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    for split_id in tl.static_range(0, NUM_SPLITS):
        dim_start = tl.load(dim_offsets_ptr + split_id)
        dim_end = tl.load(dim_offsets_ptr + split_id + 1)
        split_size = dim_end - dim_start
        slab_elems = split_size * dim_prod_post
        out_split_base = tl.load(out_offsets_ptr + split_id)
        for local_work in tl.range(pid, blocks_per_split, num_programs):
            pre_id = local_work // max_blocks
            block_id = local_work % max_blocks
            elem0 = block_id * BLOCK
            in_base = (pre_id * dim_size + dim_start) * dim_prod_post + elem0
            out_base = out_split_base + pre_id * slab_elems + elem0
            offsets = tl.arange(0, BLOCK)
            mask = (elem0 + offsets) < slab_elems
            data = tl.load(inp_ptr + in_base + offsets, mask=mask)
            tl.store(out_ptr + out_base + offsets, data, mask=mask)


_SMALL_COPY_THRESHOLD = 8192
_PERSISTENT_COPY_MAX_PROGRAMS = 48
_COPY_BLOCK_BYTES = 32768


class _FlatCopyFunction(torch.autograd.Function):
    """Copy semantics of Tensor.clone(): identity gradient, fresh storage."""

    @staticmethod
    def forward(ctx, src):
        out = torch.empty((src.numel(),), dtype=src.dtype, device=src.device)
        if src.numel() > 0:
            _launch_flat_copy(src, out)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        return grad_out


def _launch_flat_copy(src, dst):
    n_elements = src.numel()
    if n_elements <= _SMALL_COPY_THRESHOLD:
        block_size = 1024
        grid = (triton.cdiv(n_elements, block_size),)
        split_flat_copy_kernel[grid](
            src, dst, n_elements, BLOCK_SIZE=block_size, num_warps=4
        )
    else:
        element_size = src.element_size()
        block_size = max(4096, min(16384, _COPY_BLOCK_BYTES // element_size))
        grid = (
            min(_PERSISTENT_COPY_MAX_PROGRAMS, triton.cdiv(n_elements, block_size)),
        )
        split_persistent_copy_kernel[grid](
            src, dst, n_elements, BLOCK_SIZE=block_size, num_warps=4
        )


def _normalize_split_sizes(split_sizes):
    if isinstance(split_sizes, torch.Tensor):
        split_sizes = split_sizes.tolist()

    if hasattr(split_sizes, "__iter__"):
        split_sizes = list(split_sizes)

    return [int(size) for size in split_sizes]


def _normalize_dim(inp, dim):
    assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
    return dim % inp.ndim


def _product(values):
    result = 1
    for value in values:
        result *= value
    return result


def _make_split_offsets(split_sizes, device, dtype=torch.int64):
    offsets = [0]
    for size in split_sizes:
        offsets.append(offsets[-1] + size)
    return torch.tensor(offsets, dtype=dtype, device=device)


@functools.lru_cache(maxsize=128)
def _segmented_metadata(split_sizes, dim_prod_pre, dim_prod_post, device_index):
    device = torch.device("gcu", device_index)
    dim_offsets = _make_split_offsets(split_sizes, device, torch.int32)
    out_offsets = [0]
    for size in split_sizes:
        out_offsets.append(out_offsets[-1] + dim_prod_pre * size * dim_prod_post)
    out_offsets_t = torch.tensor(out_offsets, dtype=torch.int32, device=device)
    return dim_offsets, out_offsets_t


def _launch_segmented_copy(
    out_storage, source, split_sizes, dim, dim_prod_pre, dim_prod_post
):
    dim_size = source.shape[dim]
    device_index = source.device.index
    if device_index is None:
        device_index = torch.gcu.current_device()
    dim_offsets, out_offsets_t = _segmented_metadata(
        tuple(split_sizes), dim_prod_pre, dim_prod_post, device_index
    )

    element_size = source.element_size()
    positive_slabs = [size * dim_prod_post for size in split_sizes if size > 0]
    max_slab_elems = max(positive_slabs) if positive_slabs else 1
    block_size = min(
        _COPY_BLOCK_BYTES // element_size,
        max(256, triton.next_power_of_2(max_slab_elems)),
    )
    max_blocks = triton.cdiv(max_slab_elems, block_size)
    blocks_per_split = dim_prod_pre * max_blocks
    n_work = len(split_sizes) * blocks_per_split
    grid = (min(_PERSISTENT_COPY_MAX_PROGRAMS, n_work),)
    split_segmented_copy_kernel[grid](
        out_storage,
        source,
        dim_offsets,
        out_offsets_t,
        blocks_per_split,
        max_blocks,
        dim_size,
        dim_prod_post,
        NUM_SPLITS=len(split_sizes),
        BLOCK=block_size,
        num_warps=4,
    )


def split_with_sizes_copy(inp, split_sizes, dim=0):
    logger.debug("GEMS SPLIT_WITH_SIZES_COPY")

    dim = _normalize_dim(inp, dim)
    split_sizes = _normalize_split_sizes(split_sizes)

    split_sum = sum(split_sizes)
    assert split_sum == inp.shape[dim], "Invalid split_sizes"

    source = inp if inp.is_contiguous() else inp.contiguous()

    dim_prod_pre = _product(inp.shape[:dim])
    dim_prod_post = _product(inp.shape[dim + 1 :])

    use_flat_copy = (
        dim == 0 and inp.numel() > 0 and not inp.is_complex() and inp.numel() < 2**31
    )
    use_segmented_copy = (
        dim != 0 and inp.numel() > 0 and not inp.is_complex() and inp.numel() < 2**31
    )
    if use_flat_copy:
        output_storage = _FlatCopyFunction.apply(source.reshape(-1))
    elif use_segmented_copy:
        output_storage = torch.empty((inp.numel(),), dtype=inp.dtype, device=inp.device)
        _launch_segmented_copy(
            output_storage, source, split_sizes, dim, dim_prod_pre, dim_prod_post
        )
    elif inp.numel() > 0 and dim != 0:
        # dim != 0 with complex dtype or >= 2**31 elements: int32 offsets and
        # affine addressing no longer cover the domain; keep the legacy path.
        output_storage = torch.empty((inp.numel(),), dtype=inp.dtype, device=inp.device)
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(inp.numel(), BLOCK_SIZE),)
        split_offsets = _make_split_offsets(split_sizes, inp.device)
        split_indexed_copy_kernel[grid](
            output_storage,
            source,
            split_offsets,
            inp.numel(),
            inp.shape[dim],
            dim_prod_pre,
            dim_prod_post,
            NUM_SPLITS=len(split_sizes),
            BLOCK_SIZE=BLOCK_SIZE,
        )
    elif inp.numel() > 0:
        # dim == 0 with complex dtype or >= 2**31 elements: Triton cannot map
        # complex natively and int32 offsets overflow; fall back to clone().
        output_storage = source.reshape(-1).clone()
    else:
        output_storage = torch.empty((inp.numel(),), dtype=inp.dtype, device=inp.device)

    result = []
    element_offset = 0
    elements_per_dim = dim_prod_pre * dim_prod_post
    for size in split_sizes:
        out_shape = list(inp.shape)
        out_shape[dim] = size
        split_numel = size * elements_per_dim
        result.append(
            output_storage[element_offset : element_offset + split_numel].reshape(
                out_shape
            )
        )
        element_offset += split_numel

    return tuple(result)
