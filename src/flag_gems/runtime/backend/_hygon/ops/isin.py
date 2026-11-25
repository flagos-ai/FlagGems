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

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import triton_lang_extension as ext
from flag_gems.utils.libentry import libentry

from .unique import _unique2

logger = logging.getLogger(__name__)


def launch_arg(BLOCK_M, BLOCK_N, N, num_warps):
    return BLOCK_M, min(BLOCK_N, triton.next_power_of_2(N)), num_warps


@triton.jit
def isin_by_comparation_impl(
    global_pid,
    in0_ravel_ptr: tl.tensor,
    in1_ravel_ptr: tl.tensor,  # in
    out_ptr: tl.tensor,  # out
    M: int,  # num_tasks
    N: int,  # num_tasks_1
    BLOCK_M: tl.constexpr,  # tile_size
    BLOCK_N: tl.constexpr,  # tile_size_1
    invert: tl.constexpr,
):
    row_off = global_pid * BLOCK_M
    rows = row_off + tl.arange(0, BLOCK_M)[:, None]
    row_mask = rows < M
    out_ptr += rows
    in0_ravel_ptr += rows + tl.zeros([BLOCK_N], dtype=tl.int32)
    in1_ravel_ptr += tl.zeros([BLOCK_M], dtype=tl.int32)[:, None]

    block = tl.full([BLOCK_M, BLOCK_N], value=(1 if invert else 0), dtype=tl.int1)
    in0 = tl.load(in0_ravel_ptr, row_mask, other=0)
    for col_off in range(0, N, BLOCK_N):
        cols = col_off + tl.arange(0, BLOCK_N)[None, :]
        col_mask = cols < N
        mask = row_mask & col_mask
        in1 = tl.load(in1_ravel_ptr + cols, mask, other=0)
        block = tl.where(
            mask,
            tl.where(invert, block & (in0 != in1), block | (in0 == in1)),
            block,
        )
    out = tl.min(block, axis=1) if invert else tl.max(block, axis=1)
    tl.store(out_ptr, out[:, None], row_mask)


@libentry()
@triton.jit
def isin_by_comparation_kernel(
    in0_ravel_ptr: tl.tensor,
    in1_ravel_ptr: tl.tensor,  # in
    out_ptr: tl.tensor,  # out
    M: int,  # num_tasks
    N: int,  # num_tasks_1
    BLOCK_M: tl.constexpr,  # tile_size
    BLOCK_N: tl.constexpr,  # tile_size_1
    tiles_per_cta: int,
    invert: tl.constexpr,
):
    pid = ext.program_id(0)
    ctas_num = ext.num_programs(0)
    # grid-stride-loop style kernel
    for j in range(0, tiles_per_cta):
        global_pid = pid + j * ctas_num
        isin_by_comparation_impl(
            global_pid,
            in0_ravel_ptr,
            in1_ravel_ptr,  # in
            out_ptr,  # out
            M,
            N,
            BLOCK_M,
            BLOCK_N,
            invert,
        )


def isin_by_comparation(
    in0: torch.tensor,
    in1: torch.tensor,
    invert: bool,
):
    in0_ravel = in0.contiguous().ravel()
    in1_ravel = in1.contiguous().ravel()
    M = in0.numel()
    N = in1.numel()
    if M <= 1024:
        BLOCK_M, BLOCK_N, num_warps = launch_arg(1, 256, N, 4)
    elif M <= 3072:
        BLOCK_M, BLOCK_N, num_warps = launch_arg(2, 256, N, 4)
    elif M <= 6144:
        BLOCK_M, BLOCK_N, num_warps = launch_arg(4, 128, N, 4)
    # elif M <= 9216:
    #     BLOCK_M, BLOCK_N, num_warps = launch_arg(4, 256, N, 8)
    else:
        BLOCK_M, BLOCK_N, num_warps = launch_arg(4, 128, N, 4)
    if torch.version.hip is not None:
        num_warps = min(16, num_warps)
    ctas_num = min(65536, triton.cdiv(M, BLOCK_M))
    tiles_per_cta = triton.cdiv(M, BLOCK_M * ctas_num)
    grid = (ctas_num,)
    out = torch.empty_like(in0_ravel, dtype=torch.bool)
    with torch_device_fn.device(in0_ravel.device.index):
        isin_by_comparation_kernel[grid](
            in0_ravel,
            in1_ravel,  # in
            out,  # out
            M,
            N,
            BLOCK_M,
            BLOCK_N,
            tiles_per_cta=tiles_per_cta,
            invert=invert,
            num_warps=num_warps,
        )
    return out.view_as(in0)


@triton.jit
def isin_by_search_impl(
    global_pid,
    in0_ravel_ptr: tl.tensor,
    in1_sorted_ptr: tl.tensor,  # in
    out_ptr: tl.tensor,  # out
    M: int,  # num_tasks
    N: int,  # num_tasks_1
    log_n: tl.constexpr,
    BLOCK_M: tl.constexpr,  # tile_size
    invert: tl.constexpr,
):
    r = tl.arange(0, BLOCK_M)
    i0 = global_pid * BLOCK_M + r
    mask = i0 < M

    # load in0_ravel
    in0_ravel = tl.load(in0_ravel_ptr + i0, mask=mask, other=0)

    # binary search: lower_bound
    out = tl.zeros_like(r).to(tl.int1)
    start = tl.zeros_like(r)
    end = start + N
    while_mask = mask & (start < end)
    for i in range(log_n):
        mid = tl.where(while_mask, start + (end - start) // 2, 0)
        mid_val = tl.load(in1_sorted_ptr + mid, mask=while_mask, other=0)
        out |= while_mask & (mid_val == in0_ravel)
        start = tl.where(while_mask & (mid_val < in0_ravel), mid + 1, start)
        end = tl.where(while_mask & (mid_val > in0_ravel), mid, end)
        while_mask = mask & (start < end)

    # store out
    tl.store(out_ptr + i0, ~out if invert else out, mask=mask)


@libentry()
@triton.jit
def isin_by_search_kernel(
    in0_ravel_ptr: tl.tensor,
    in1_sorted_ptr: tl.tensor,  # in
    out_ptr: tl.tensor,  # out
    M: int,  # num_tasks
    N: int,  # num_tasks_1
    log_n: tl.constexpr,
    BLOCK_M: tl.constexpr,  # tile_size
    tiles_per_cta: int,
    invert: tl.constexpr,
):
    pid = ext.program_id(0)
    ctas_num = ext.num_programs(0)
    # grid-stride-loop style kernel
    for j in range(0, tiles_per_cta):
        global_pid = pid + j * ctas_num
        isin_by_search_impl(
            global_pid,
            in0_ravel_ptr,
            in1_sorted_ptr,  # in
            out_ptr,  # out
            M,
            N,
            log_n,
            BLOCK_M,
            invert,
        )


def isin_by_search(
    in0: torch.tensor,
    in1: torch.tensor,
    invert: bool,
    unique_in0: bool,
    unique_in1: bool,
):
    # unique or sort or ravel
    if unique_in0:
        in0_ravel, unique_order, _ = _unique2(
            in0, sorted=True, return_inverse=True, return_counts=False
        )
    else:
        in0_ravel = in0.contiguous().ravel()
    if unique_in1:
        in1_ravel, _, _ = _unique2(
            in1, sorted=True, return_inverse=False, return_counts=False
        )
    else:
        in1_ravel, _ = torch.sort(in1.ravel())
    # launch kernel func
    M = in0_ravel.numel()
    N = in1_ravel.numel()
    if M <= 1048576:  # 2 ** 20 = 1024 * 1024
        _, BLOCK_M, num_warps = launch_arg(None, 512, M, 8)
    elif M <= 4194304:  # 2 ** 22 = 1024 * 4096
        _, BLOCK_M, num_warps = launch_arg(None, 1024, M, 8)
    elif M <= 8388608:  # 2 ** 23 = 1024 * 8192
        _, BLOCK_M, num_warps = launch_arg(None, 2048, M, 16)
    elif M <= 268435456:  # 2 ** 28 = 1024 * 262144
        _, BLOCK_M, num_warps = launch_arg(None, 4096, M, 32)
    else:
        _, BLOCK_M, num_warps = launch_arg(None, 2048, M, 16)
    if torch.version.hip is not None:
        num_warps = min(16, num_warps)
    log_n = int(math.log2(N)) + 1
    ctas_num = min(65536, triton.cdiv(M, BLOCK_M))
    tiles_per_cta = triton.cdiv(M, BLOCK_M * ctas_num)
    grid = (ctas_num,)
    out = torch.empty_like(in0_ravel, dtype=torch.bool)
    with torch_device_fn.device(in0_ravel.device.index):
        isin_by_search_kernel[grid](
            in0_ravel,
            in1_ravel,  # in
            out,  # out
            M,
            N,
            log_n,
            BLOCK_M,
            tiles_per_cta=tiles_per_cta,
            invert=invert,
            num_warps=num_warps,
        )
    if unique_in0:
        out = torch.gather(out, 0, unique_order.ravel().to(torch.int64))
    return out.view_as(in0)


_ISIN_INTEGER_DTYPES = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)


@libentry()
@triton.jit(do_not_specialize=["value"])
def _isin_tensor_scalar_kernel(
    inp,
    out,
    value,
    N: tl.constexpr,
    SHAPE: tl.constexpr,
    STRIDES: tl.constexpr,
    CONTIGUOUS: tl.constexpr,
    INVERT: tl.constexpr,
    BLOCK: tl.constexpr,
    STREAMING: tl.constexpr,
):
    load_cache: tl.constexpr = ".cg" if STREAMING else ""
    store_cache: tl.constexpr = ".cs" if STREAMING else ""
    pid = tl.program_id(0)
    if N > 2147483647:
        index = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    else:
        index = pid * BLOCK + tl.arange(0, BLOCK)
    if CONTIGUOUS:
        offset = index
    else:
        offset = tl.full((BLOCK,), 0, tl.int64)
        remaining = index
        for axis in tl.static_range(len(SHAPE) - 1, -1, -1):
            coord = remaining % SHAPE[axis]
            remaining = remaining // SHAPE[axis]
            offset += coord.to(tl.int64) * STRIDES[axis]
    if N % BLOCK == 0:
        x = tl.load(inp + offset, cache_modifier=load_cache)
    else:
        x = tl.load(inp + offset, index < N, other=0, cache_modifier=load_cache)
    # Match native Tensor-Scalar equality, including wrapped integer scalars.
    result = x == value.to(x.dtype)
    if INVERT:
        result = ~result
    if N % BLOCK == 0:
        tl.store(out + index, result, cache_modifier=store_cache)
    else:
        tl.store(out + index, result, index < N, cache_modifier=store_cache)


def isin(
    in0,
    in1,
    *,
    assume_unique: bool = False,
    invert: bool = False,
) -> torch.Tensor:
    logger.debug("GEMS_HYGON ISIN")

    if (
        torch.is_tensor(in0)
        and type(in1) is int
        and in0.dtype in _ISIN_INTEGER_DTYPES
        and -(1 << 63) <= in1 < (1 << 63)
    ):
        out = torch.empty(in0.shape, dtype=torch.bool, device=in0.device)
        count = in0.numel()
        if count:
            contiguous = in0.is_contiguous()
            shape = () if contiguous else tuple(in0.shape)
            strides = () if contiguous else tuple(in0.stride())
            streaming = contiguous and count >= 1 << 24
            if not contiguous or count <= 8192:
                block, warps = 512, 4
            elif in0.dtype == torch.int16:
                block, warps = (4096, 8) if streaming else (2048, 4)
            elif in0.dtype == torch.int32:
                block, warps = (2048, 8) if streaming else (1024, 4)
            else:
                block, warps = 512, 4
            with torch_device_fn.device(in0.device):
                _isin_tensor_scalar_kernel[(triton.cdiv(count, block),)](
                    in0,
                    out,
                    in1,
                    count,
                    shape,
                    strides,
                    contiguous,
                    invert,
                    BLOCK=block,
                    STREAMING=streaming,
                    num_warps=warps,
                )
        return out

    if not torch.is_tensor(in0):
        assert torch.is_tensor(in1)
        in0 = torch.tensor(in0, device=in1.device)
    elif not torch.is_tensor(in1):
        assert torch.is_tensor(in0)
        in1 = torch.tensor(in1, device=in0.device)
    if in0.numel() == 0 or in1.numel() == 0:
        return torch.full_like(in0, invert, dtype=torch.bool)
    elif in0.numel() <= 12288 and in1.numel() <= 12288:  # 1024 * 12
        return isin_by_comparation(in0, in1, invert)
    elif assume_unique or in1.numel() <= 4194304:  # 1024 * 4096
        return isin_by_search(in0, in1, invert, unique_in0=False, unique_in1=False)
    else:
        return isin_by_search(in0, in1, invert, unique_in0=False, unique_in1=True)
