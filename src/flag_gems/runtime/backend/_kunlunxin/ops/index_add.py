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
from flag_gems.utils import dim_compress, libentry

logger = logging.getLogger(__name__)
# def cfggen():
#     block_m = [1, 2, 4]
#     block_n = [128, 1024, 2048, 4096]
#     configs = [
#         triton.Config({"BLOCK_M": m, "BLOCK_N": n}, num_warps=4)
#         for m in block_m
#         for n in block_n
#     ]
#     return configs


@libentry()
# @triton.autotune(configs=cfggen(), key=["M", "N"])
@triton.heuristics(runtime.get_heuristic_config("index_add"))
# @triton.autotune(
#     configs=[], generate_configs="index_add", op_affiliation="cluster", row_sign="M", col_sign="N",
#     key=["M", "N"],
# )
@triton.jit
def index_add_kernel(
    inp,
    inp_cont,
    index,
    src,
    M: tl.constexpr,
    N: tl.constexpr,
    alpha,
    inp_len,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_x = tl.program_id(axis=0)  # block_x
    pid_y = tl.program_id(axis=1)  # block_y
    rows_offsets = (
        pid_x * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    )  # block_x * BLOCK_M + tl.arange(0, BLOCK_M)
    cols_offsets = pid_y * BLOCK_N + tl.arange(
        0, BLOCK_N
    )  # block_y * BLOCK_N + tl.arange(0, BLOCK_N)

    rows_mask = (
        rows_offsets < M
    )  # rows_mask = block_x * BLOCK_M + tl.arange(0, BLOCK_M) < M
    index_mask = (
        cols_offsets < N
    )  # index_mask = block_y * BLOCK_N + tl.arange(0, BLOCK_N) < N
    block_mask = rows_mask and index_mask  # block_mask = rows_mask and index_mask

    cur_indices = tl.load(
        index + cols_offsets, mask=index_mask, other=0
    )  # cur_indices = tl.load(index + cols_offsets, mask=index_mask, other=0)
    inp_off = (
        rows_offsets * inp_len + cur_indices[None, :]
    )  # inp_off = (block_x * BLOCK_M + tl.arange(0, BLOCK_M)) * M + cur_indices
    cur_inp = tl.load(
        inp + inp_off, mask=block_mask, other=0.0
    )  # cur_inp = tl.load(inp + inp_off, mask=block_mask, other=0.0)
    src_off = (
        rows_offsets * N + cols_offsets[None, :]
    )  # src_off = (block_x * BLOCK_M + tl.arange(0, BLOCK_M)) * N + block_y * BLOCK_N + tl.arange(0, BLOCK_N)
    cur_src = tl.load(
        src + src_off, mask=block_mask, other=0.0
    )  # cur_src = tl.load(src + src_off, mask=block_mask, other=0.0)
    cur_inp += alpha * cur_src

    tl.store(inp_cont + inp_off, cur_inp, mask=block_mask)


@libentry()
@triton.jit
def classify_index_duplicates_kernel(
    index,
    counts,
    duplicate_flag,
    N: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    for block in tl.static_range(0, NUM_BLOCKS):
        offsets = block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N
        indices = tl.load(index + offsets, mask=mask, other=0)
        tl.atomic_add(counts + indices, 1, mask=mask, sem="relaxed")


@libentry()
@triton.jit
def check_index_duplicate_counts_kernel(
    counts,
    duplicate_flag,
    domain_size,
    BLOCK_SIZE: tl.constexpr,
):
    max_count = tl.full((), 0, dtype=tl.int32)
    for start in tl.range(0, domain_size, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        values = tl.load(counts + offsets, mask=offsets < domain_size, other=0)
        max_count = tl.maximum(max_count, tl.max(values, axis=0))
    tl.store(duplicate_flag, (max_count > 1).to(tl.int32))


@libentry()
@triton.jit
def index_add_bfloat16_kernel(
    inp_cont,
    index,
    src,
    M,
    N: tl.constexpr,
    alpha,
    inp_len,
    SOURCE_START: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(axis=0)
    destination = tl.program_id(axis=1)
    offsets = SOURCE_START + tl.arange(0, BLOCK_N)
    source_mask = offsets < SOURCE_START + CHUNK_SIZE
    output_mask = (row < M) & (destination < inp_len)

    indices = tl.load(index + offsets, mask=source_mask, other=0)
    values = tl.load(
        src + row * N + offsets, mask=output_mask & source_mask, other=0.0
    ).to(tl.float32)
    updates = tl.where(source_mask & (indices == destination), values, 0.0)
    original = tl.load(
        inp_cont + row * inp_len + destination, mask=output_mask, other=0.0
    )
    tl.store(
        inp_cont + row * inp_len + destination,
        original + alpha * tl.sum(updates, axis=0),
        mask=output_mask,
    )


def launch_index_add(inp, inp_cont, index, src, M, N, alpha, inp_len):
    if N <= 1:
        has_duplicates = False
    else:
        counts = torch.zeros(inp_len, dtype=torch.int32, device=index.device)
        duplicate_flag = torch.zeros(1, dtype=torch.int32, device=index.device)
        block_size = min(triton.next_power_of_2(N), 1024)
        classify_index_duplicates_kernel[(1,)](
            index,
            counts,
            duplicate_flag,
            N,
            NUM_BLOCKS=triton.cdiv(N, block_size),
            BLOCK_SIZE=block_size,
        )
        check_index_duplicate_counts_kernel[(1,)](
            counts, duplicate_flag, inp_len, BLOCK_SIZE=1024
        )
        has_duplicates = bool(duplicate_flag.item())

    if has_duplicates:
        block_n = min(triton.next_power_of_2(N), 1024)
        for source_start in range(0, N, block_n):
            chunk_size = min(block_n, N - source_start)
            index_add_bfloat16_kernel[(M, inp_len)](
                inp_cont,
                index,
                src,
                M,
                N,
                alpha,
                inp_len,
                SOURCE_START=source_start,
                CHUNK_SIZE=chunk_size,
                BLOCK_N=block_n,
            )
        return

    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]),
        triton.cdiv(N, meta["BLOCK_N"]),
    )
    index_add_kernel[grid](inp, inp_cont, index, src, M, N, alpha, inp_len)


def index_add(inp, dim, index, src, alpha=1):
    logger.debug("GEMS_KUNLUNXIN INDEX_ADD")
    assert ((0 <= index) * (index < inp.size(dim))).equal(
        torch.ones(tuple(index.shape), dtype=torch.bool, device=index.device)
    ), "0 <= index < self.size(dim)"
    assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
    assert index.numel() == src.size(
        dim
    ), "The dimth dimension of source must have the same size as the length of index"
    assert (
        inp.ndim == src.ndim
    ), "Self and source should have the same number of dimensions"
    assert (
        ((inp.size(i) == src.size(i)) or i == dim) for i in range(0, inp.ndim)
    ), "src.size(d) == self.size(d) for all dimensions d != dim"

    inp = inp.contiguous()
    index = index.contiguous()
    src = src.contiguous()

    dim = dim % inp.ndim
    inp_len = inp.size(dim)
    N = index.numel()
    M = src.numel() // N
    fine_dim = inp.ndim - 1
    if dim != fine_dim:
        inp = dim_compress(inp, dim)
        src = dim_compress(src, dim)
    inp_cont = inp.clone()

    launch_index_add(inp, inp_cont, index, src, M, N, alpha, inp_len)
    if dim != fine_dim:
        order = [i for i in range(inp_cont.ndim - 1)]
        order.insert(dim, fine_dim)
        return inp_cont.permute(order).contiguous()
    else:
        return inp_cont


def index_add_(inp, dim, index, src, alpha=1):
    logger.debug("GEMS_KUNLUNXIN INDEX_ADD_")
    assert ((0 <= index) * (index < inp.size(dim))).equal(
        torch.ones(tuple(index.shape), dtype=torch.bool, device=index.device)
    ), "0 <= index < self.size(dim)"
    assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
    assert index.numel() == src.size(
        dim
    ), "The dimth dimension of source must have the same size as the length of index"
    assert (
        inp.ndim == src.ndim
    ), "Self and source should have the same number of dimensions"
    assert (
        ((inp.size(i) == src.size(i)) or i == dim) for i in range(0, inp.ndim)
    ), "src.size(d) == self.size(d) for all dimensions d != dim"

    inp_cont = inp.clone()
    inp_cont = inp_cont.contiguous()
    index = index.contiguous()
    src = src.contiguous()

    dim = dim % inp_cont.ndim
    inp_len = inp_cont.size(dim)
    N = index.numel()
    M = src.numel() // N
    fine_dim = inp_cont.ndim - 1
    if dim != fine_dim:
        inp_cont = dim_compress(inp_cont, dim)
        src = dim_compress(src, dim)

    launch_index_add(inp_cont, inp_cont, index, src, M, N, alpha, inp_len)
    if dim != fine_dim:
        order = [i for i in range(inp_cont.ndim - 1)]
        order.insert(dim, fine_dim)
        inp_cont = inp_cont.permute(order).contiguous()
        inp.copy_(inp_cont)
        return inp
    else:
        inp.copy_(inp_cont)
        return inp

