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

from flag_gems import runtime
from flag_gems.utils import dim_compress, libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)


@libentry()
@triton.heuristics(runtime.get_heuristic_config("index_select"))
@triton.jit
def index_select_kernel(
    inp,
    out,
    M: tl.constexpr,
    N: tl.constexpr,
    index,
    index_len: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_x = ext.program_id(axis=0)
    pid_y = ext.program_id(axis=1)
    rows_offsets = pid_x * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    rows_mask = rows_offsets < M
    cols_offsets = pid_y * BLOCK_N + tl.arange(0, BLOCK_N)

    out_mask = rows_mask and (cols_offsets < index_len)

    indices = tl.load(index + cols_offsets, mask=(cols_offsets < index_len), other=0)
    inp_off = rows_offsets * N + indices[None, :]
    out_off = rows_offsets * index_len + cols_offsets[None, :]

    selected = tl.load(inp + inp_off, mask=rows_mask, other=0.0)
    tl.store(out + out_off, selected, mask=out_mask)


@libentry()
@triton.jit
def index_select_slice_kernel(
    inp,
    out,
    index,
    D,
    INNER,
    IDXL,
    BLOCK_I: tl.constexpr,
):
    # One program per output slice. Each slice copies INNER contiguous
    # elements: out[o, j, :] = inp[o, index[j], :]. Because the copied
    # run is contiguous (stride 1) with a scalar base, OffsetAnalysis can
    # prove it -> block DMA instead of the strided gather that dim_compress
    # would force by moving the indexed dim to the inner position.
    pid = ext.program_id(axis=0)
    o = pid // IDXL
    j = pid % IDXL
    idx = tl.load(index + j)
    # Advance the base pointers once (embedding-style) so the per-iteration
    # address expression is just `ptr + cols`; this lets OffsetAnalysis keep
    # the contiguous block-DMA form and shaves the small-shape launch floor.
    inp += o * D * INNER + idx * INNER
    out += pid * INNER
    for c in range(0, INNER, BLOCK_I):
        cols = c + tl.arange(0, BLOCK_I)
        mask = cols < INNER
        vals = tl.load(inp + cols, mask=mask, other=0)
        tl.store(out + cols, vals, mask=mask)


def index_select(inp, dim, index):
    logger.debug("GEMS_KUNLUNXIN INDEX_SELECT")
    assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
    assert index.ndim <= 1, "Index should have dimension 1 or 0"

    if index.ndim == 0:
        index = index.unsqueeze(0)
    dim = dim % inp.ndim
    index_len = index.numel()

    inp = inp.contiguous()
    index = index.contiguous()
    shape = list(inp.shape)
    inner = math.prod(shape[dim + 1 :])
    out_shape = shape[:dim] + [index_len] + shape[dim + 1 :]
    out = torch.empty(out_shape, dtype=inp.dtype, device=inp.device)

    if inner > 1:
        # Fast path: indexed dim is not the innermost -> each selected slice
        # is a contiguous run of `inner` elements. Copy whole slices.
        outer = math.prod(shape[:dim])
        dim_size = shape[dim]
        n_slices = outer * index_len
        block_i = min(triton.next_power_of_2(inner), 8192)
        grid = (n_slices,)
        index_select_slice_kernel[grid](
            inp,
            out,
            index,
            dim_size,
            inner,
            index_len,
            BLOCK_I=block_i,
            num_warps=8,
            buffer_size_limit=4096,
        )
        return out

    # Fallback: indexed dim is the innermost (inner == 1) -> genuine gather.
    inp = dim_compress(inp, dim)
    N = shape[dim]
    M = inp.numel() // N
    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]),
        triton.cdiv(index_len, meta["BLOCK_N"]),
    )
    index_select_kernel[grid](inp, out, M, N, index, index_len)
    return out
