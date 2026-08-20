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

import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)


@libentry()
@triton.jit(
    # runtime dims; avoid per-shape recompilation
    do_not_specialize=[
        "input_dim",
        "index_len",
        "inner_size",
        "numel",
    ]
)
def index_copy_ascend_flat_kernel(
    out_ptr,
    src_ptr,
    index_ptr,
    input_dim,
    index_len,
    inner_size,
    numel,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tle.program_id(axis=0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    offsets_i64 = offsets.to(tl.int64)

    # [outer, index, inner] decomposition
    inner_offset = offsets_i64 % inner_size
    src_row = offsets_i64 // inner_size
    index_offset = src_row % index_len
    outer_offset = src_row // index_len

    dst_index = tl.load(index_ptr + index_offset, mask=mask, other=0)
    valid_index = (dst_index >= 0) & (dst_index < input_dim)

    value = tl.load(src_ptr + offsets_i64, mask=mask, other=0.0)
    dst_offset = (outer_offset * input_dim + dst_index) * inner_size + inner_offset

    # Triton Ascend does not support the mask argument of tl.device_assert.
    tl.store(out_ptr + dst_offset, value, mask=mask & valid_index)


@libentry()
@triton.jit(
    do_not_specialize=[
        "input_dim",
        "index_len",
        "inner_size",
        "row_count",
    ]
)
def index_copy_ascend_row_kernel(
    out_ptr,
    src_ptr,
    index_ptr,
    input_dim,
    index_len,
    inner_size,
    row_count,
    BLOCK_SIZE: tl.constexpr,
):
    # one program per [outer, index] row
    row_id = tle.program_id(axis=0)
    row_id_i64 = row_id.to(tl.int64)

    inner_offsets = tle.program_id(axis=1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    inner_offsets_i64 = inner_offsets.to(tl.int64)

    row_mask = row_id < row_count
    inner_mask = inner_offsets < inner_size

    index_offset = row_id_i64 % index_len
    outer_offset = row_id_i64 // index_len

    dst_index = tl.load(index_ptr + index_offset, mask=row_mask, other=0)
    valid_index = (dst_index >= 0) & (dst_index < input_dim)
    mask = row_mask & inner_mask & valid_index

    src_offsets = row_id_i64 * inner_size + inner_offsets_i64
    dst_offsets = (
        outer_offset * input_dim + dst_index
    ) * inner_size + inner_offsets_i64

    value = tl.load(src_ptr + src_offsets, mask=mask, other=0.0)
    tl.store(out_ptr + dst_offsets, value, mask=mask)


def _select_flat_block_size(numel, inner_size):
    # narrow block keeps more programs in flight for small workloads
    if inner_size == 1 and numel <= 4096:
        return 256
    return 1024


def _launch_index_copy(out, dim, index, src):
    index_len = index.numel()
    inner_size = math.prod(out.shape[dim + 1 :])
    outer_size = math.prod(out.shape[:dim])

    if index_len == 0 or inner_size == 0:
        return

    input_dim = out.size(dim)
    numel = outer_size * index_len * inner_size

    with torch_device_fn.device(out.device):
        if inner_size <= 4:
            # flat kernel for tiny suffixes
            block_size = _select_flat_block_size(numel, inner_size)
            grid = (triton.cdiv(numel, block_size),)
            index_copy_ascend_flat_kernel[grid](
                out,
                src,
                index,
                input_dim,
                index_len,
                inner_size,
                numel,
                BLOCK_SIZE=block_size,
            )
            return

        # row kernel for wider suffixes
        row_count = outer_size * index_len
        block_size = 256
        grid = (row_count, triton.cdiv(inner_size, block_size))
        index_copy_ascend_row_kernel[grid](
            out,
            src,
            index,
            input_dim,
            index_len,
            inner_size,
            row_count,
            BLOCK_SIZE=block_size,
        )


def index_copy(inp, dim, index, src):
    logger.debug("GEMS ASCEND INDEX_COPY")
    dim %= inp.ndim
    out = inp.clone()
    _launch_index_copy(out, dim, index, src)
    return out


def index_copy_(inp, dim, index, src):
    logger.debug("GEMS ASCEND INDEX_COPY_")
    dim %= inp.ndim
    _launch_index_copy(inp, dim, index, src)
    return inp
