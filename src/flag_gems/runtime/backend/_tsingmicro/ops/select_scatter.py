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

try:
    from flag_gems.utils.shape_utils import MemOverlap, has_internal_overlapping
except ImportError:

    class MemOverlap:
        Yes, No = "Yes", "No"

    def has_internal_overlapping(_):
        return MemOverlap.No


logger = logging.getLogger(__name__)
# 不要用 2**24：rel_off 会很大，片内 // % 仍慢且易丢精度
_CHUNK = 65536


@triton.jit
def select_scatter_kernel(
    out_ptr,
    inp_ptr,
    src_ptr,
    chunk_numel,
    base_pre,
    base_dim,
    base_post,
    base_mod,
    base_q,
    dim_size,
    dim_prod_post,
    index,
    LAST_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    rel_off = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = rel_off < chunk_numel
    r = rel_off.to(tl.int32)

    if LAST_DIM:
        lin = base_mod + r
        dim_idx = lin % dim_size
        src_idx = base_q + lin // dim_size
    else:
        db = dim_size * dim_prod_post
        rel_pre, rem = r // db, r % db
        rel_dim, rel_post = rem // dim_prod_post, rem % dim_prod_post
        post_sum = base_post + rel_post
        dim_sum = base_dim + rel_dim + post_sum // dim_prod_post
        dim_idx = dim_sum % dim_size
        src_idx = (
            base_pre + rel_pre + dim_sum // dim_size
        ) * dim_prod_post + post_sum % dim_prod_post

    m = dim_idx == index
    inp_data = tl.load(inp_ptr + rel_off, mask=mask)
    src_data = tl.load(src_ptr + src_idx, mask=mask & m)
    tl.store(out_ptr + rel_off, tl.where(m, src_data, inp_data), mask=mask)


def select_scatter(inp, src, dim, index, chunk_elem=_CHUNK, block_size=1024):
    logger.debug("GEMS_TSINGMICRO SELECT_SCATTER")
    assert -inp.ndim <= dim < inp.ndim and -inp.size(dim) <= index < inp.size(dim)
    dim, index = dim % inp.ndim, index % inp.size(dim)
    assert list(src.shape) == [s for i, s in enumerate(inp.shape) if i != dim]

    if has_internal_overlapping(inp) == MemOverlap.Yes:
        out = torch.empty(inp.size(), dtype=inp.dtype, device=inp.device)
    else:
        out = torch.empty_strided(
            inp.size(), inp.stride(), dtype=inp.dtype, device=inp.device
        )

    inp, src = inp.contiguous(), src.contiguous()
    flat_inp, flat_out = inp.reshape(-1), out.reshape(-1)
    dim_size = inp.size(dim)
    dim_prod_post = 1
    for d in range(dim + 1, inp.ndim):
        dim_prod_post *= inp.size(d)
    last_dim = dim_prod_post == 1
    db = dim_size * dim_prod_post

    pos, total = 0, inp.numel()
    while pos < total:
        n = min(chunk_elem, total - pos)
        rem = pos % db
        select_scatter_kernel[(triton.cdiv(n, block_size),)](
            flat_out[pos:],
            flat_inp[pos:],
            src,
            n,
            pos // db,
            rem // dim_prod_post,
            rem % dim_prod_post,
            pos % dim_size,
            pos // dim_size,
            dim_size,
            dim_prod_post,
            index,
            LAST_DIM=last_dim,
            BLOCK_SIZE=block_size,
        )
        pos += n
    return out
