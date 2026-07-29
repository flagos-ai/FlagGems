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


def _make_split_offsets(split_sizes, device):
    offsets = [0]
    for size in split_sizes:
        offsets.append(offsets[-1] + size)
    return torch.tensor(offsets, dtype=torch.int64, device=device)


def split_with_sizes_copy(inp, split_sizes, dim=0):
    logger.debug("GEMS SPLIT_WITH_SIZES_COPY")

    dim = _normalize_dim(inp, dim)
    split_sizes = _normalize_split_sizes(split_sizes)

    split_sum = sum(split_sizes)
    assert split_sum == inp.shape[dim], "Invalid split_sizes"

    source = inp if inp.is_contiguous() else inp.contiguous()

    dim_prod_pre = _product(inp.shape[:dim])
    dim_prod_post = _product(inp.shape[dim + 1 :])

    if dim == 0:
        output_storage = source.reshape(-1).clone()
    else:
        output_storage = torch.empty((inp.numel(),), dtype=inp.dtype, device=inp.device)

    if inp.numel() > 0 and dim != 0:
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
