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
import os

import torch
import triton
import triton.language as tl
from _kunlunxin.utils.codegen_config_utils import CodeGenConfig

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import triton_lang_extension as ext
from flag_gems.utils.libentry import libentry

from ..utils.pointwise_dynamic import pointwise_dynamic
from .all import all as tensor_all
from .all import reduce_all
from .any import any as tensor_any
from .any import reduce_any
from .sort import sort_stable
from .unique import _unique2

logger = logging.getLogger(__name__)


# Scalar fast path: isin(elements, test_elements) with a SINGLE scalar
# test_element reduces to an elementwise compare (elements == scalar), which is
# far cheaper than the generic binary-search / comparison kernels. The original
# override always routed the (tensor, scalar) variant into isin_by_search with
# N=1 -> full binary-search machinery for a trivial equality -> gems speedup
# only ~0.1-0.5 (harness/perf_ir_3/ir-isin_tensor_scalar-dev6.log). Reuse the
# tuned kunlunxin compare config (same as eq/ne) on a pointwise_dynamic kernel.
# Integer-exact compare (no float32 cast) to preserve isin's exact-match
# semantics.
_scalar_config = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    isCloseMemoryAsync=False,
    kunlunAutoGrid=True,
    unroll_num=8,
)


@pointwise_dynamic(
    is_tensor=[True, False],
    promotion_methods=[(0, 1, "ALWAYS_BOOL")],
    config=_scalar_config,
)
@triton.jit
def isin_scalar_eq_func(x, y):
    return x == y


@pointwise_dynamic(
    is_tensor=[True, False],
    promotion_methods=[(0, 1, "ALWAYS_BOOL")],
    config=_scalar_config,
)
@triton.jit
def isin_scalar_ne_func(x, y):
    return x != y


@pointwise_dynamic(
    is_tensor=[True, False],
    promotion_methods=[(0, 1, "ALWAYS_BOOL")],
    config=_scalar_config,
)
@triton.jit
def isin_empty_func(x, invert):
    return invert


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
        mask = row_mask and col_mask
        in1 = tl.load(in1_ravel_ptr + cols, mask, other=0)
        block = tl.where(
            mask,
            tl.where(invert, block and (in0 != in1), block or (in0 == in1)),
            invert,
        )
    out = tl.reduce(block, axis=1, combine_fn=(reduce_all if invert else reduce_any))
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
    elif M <= 9216:
        BLOCK_M, BLOCK_N, num_warps = launch_arg(4, 256, N, 8)
    else:
        BLOCK_M, BLOCK_N, num_warps = launch_arg(4, 128, N, 4)
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
    in0_ravel = tl.load(in0_ravel_ptr + i0, mask=mask)

    # binary search: lower_bound
    out = tl.zeros_like(r).to(tl.int1)
    start = tl.zeros_like(r)
    end = start + N
    while_mask = start < end
    for i in range(log_n):
        mid = tl.where(while_mask, start + (end - start) // 2, 0)
        mid_val = tl.load(in1_sorted_ptr + mid, mask=while_mask)
        out = tl.where(while_mask, out or (mid_val == in0_ravel), out)  # found
        start = tl.where(while_mask and (mid_val < in0_ravel), mid + 1, start)
        end = tl.where(while_mask and (mid_val > in0_ravel), mid, end)
        while_mask = start < end

    # store out
    out_offset = tl.where(mask, i0, M + 1)
    tl.store(out_ptr + out_offset, not out if invert else out, mask=mask)


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
    unique_in1: bool,
):
    in0_ravel = in0.contiguous().ravel()
    if unique_in1:
        # print("hit _unique2!!!")
        in1_ravel, _, _ = _unique2(
            in1, sorted=True, return_inverse=False, return_counts=False
        )
    else:
        in1_ravel, _ = sort_stable(in1.ravel(), stable=True)
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
    log_n = int(math.log2(N)) + 1
    ctas_num = min(65536, triton.cdiv(M, BLOCK_M))
    tiles_per_cta = triton.cdiv(M, BLOCK_M * ctas_num)
    # print(f"M = {M}")
    # print(f"BLOCK_M = {BLOCK_M}")
    # print(f"ctas_num = {ctas_num}")
    # print(f"tiles_per_cta = {tiles_per_cta}")
    grid = (ctas_num,)
    out = torch.empty_like(in0_ravel, dtype=torch.bool)
    with torch_device_fn.device(in0_ravel.device.index):
        os.environ["TRITONXPU_OTHER_SIM"] = "1"
        os.environ["TRITONXPU_STORE_MASK_SIM"] = "1"
        os.environ["TRITONXPU_INTERLEAVE"] = "0"
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
            isCloseUnrollControl=True,
        )
        if "TRITONXPU_OTHER_SIM" in os.environ:
            del os.environ["TRITONXPU_OTHER_SIM"]
        if "TRITONXPU_STORE_MASK_SIM" in os.environ:
            del os.environ["TRITONXPU_STORE_MASK_SIM"]
        if "TRITONXPU_INTERLEAVE" in os.environ:
            del os.environ["TRITONXPU_INTERLEAVE"]

    return out.view_as(in0)


_BITMAP_MAX_RANGE = 1 << 17  # 128K slots * 1B = 128KB device bitmap


@libentry()
@triton.jit
def _isin_bitmap_mark_kernel(
    in1_ptr,
    bit_ptr,
    n1,
    min_val: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = ext.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n1
    val = tl.load(in1_ptr + offsets, mask=mask).to(tl.int64)
    idx = val - min_val  # int64 arithmetic; range-checked by the caller cond
    tl.store(bit_ptr + idx, 1, mask=mask)


@libentry()
@triton.jit
def _isin_bitmap_query_kernel(
    in0_ptr,
    bit_ptr,
    out_ptr,
    n0,
    range_size,
    min_val: tl.constexpr,
    invert: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = ext.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n0
    val = tl.load(in0_ptr + offsets, mask=mask).to(tl.int64)
    idx = val - min_val  # int64 (negative -> masked below)
    # clamp the index so the load can never touch out-of-range memory
    # (XPU masked loads ignore the `other` value on this backend), then
    # re-apply the range check as a pure arithmetic predicate.
    in_range = (idx >= 0) & (idx < range_size)
    safe_idx = tl.minimum(tl.maximum(idx, 0), range_size - 1)
    hit = tl.load(bit_ptr + safe_idx, mask=mask) == 1
    hit = hit & in_range
    out = hit != invert
    tl.store(out_ptr + offsets, out, mask=mask)


def isin_by_bitmap(in0, in1, invert):
    """Value-range bitmap direct lookup for integer dtypes with a compact
    value span. isin(elements, test_elements) on integers only needs to know
    which distinct values of test_elements exist; when the value range is
    small the whole set membership reduces to a tiny mark+query pair of
    kernels (O(M+N), no sort/unique/binary-search).

    Falls back (returns None) when the value span is too large or the dtype
    is non-integer, so the caller keeps the existing sort+binary-search path.
    """
    if not (in0.is_floating_point() or in1.is_floating_point()):
        # values as int64 (index arithmetic must be exact; never fp)
        in1_flat = in1.ravel().to(torch.int64)
        lo = int(in1_flat.min().item())
        hi = int(in1_flat.max().item())
        range_size = hi - lo + 1
        if range_size <= _BITMAP_MAX_RANGE:
            bit = torch.zeros(range_size, dtype=torch.uint8, device=in1.device)
            n1 = in1_flat.numel()
            BLOCK = 1024
            grid1 = (triton.cdiv(n1, BLOCK),)
            with torch_device_fn.device(in1.device.index):
                _isin_bitmap_mark_kernel[grid1](
                    in1_flat, bit, n1, min_val=lo, BLOCK_SIZE=BLOCK
                )
            in0_flat = in0.ravel().to(torch.int64)
            n0 = in0_flat.numel()
            out = torch.empty(n0, dtype=torch.bool, device=in0.device)
            grid0 = (triton.cdiv(n0, BLOCK),)
            with torch_device_fn.device(in0.device.index):
                _isin_bitmap_query_kernel[grid0](
                    in0_flat, bit, out, n0, range_size, min_val=lo,
                    invert=invert, BLOCK_SIZE=BLOCK,
                )
            return out.view_as(in0)
    return None


def isin(
    in0,
    in1,
    *,
    assume_unique: bool = False,
    invert: bool = False,
) -> torch.Tensor:
    logger.debug("GEMS_KUNLUNXIN ISIN")
    if not torch.is_tensor(in0):
        assert torch.is_tensor(in1)
        if invert:
            return tensor_all(isin_scalar_ne_func(in1, in0))
        return tensor_any(isin_scalar_eq_func(in1, in0))
    elif not torch.is_tensor(in1):
        assert torch.is_tensor(in0)
        in1 = torch.full((), in1, device=in0.device)
    if in0.numel() == 0:
        return torch.empty_like(in0, dtype=torch.bool)
    if in1.numel() == 0:
        return isin_empty_func(in0, invert)
    elif in1.numel() == 1:
        # (tensor, scalar) fast path: isin == elementwise compare with the
        # single test element. Output shape follows in0.
        scalar_val = in1.ravel()[0].item()
        if invert:
            return isin_scalar_ne_func(in0, scalar_val)
        return isin_scalar_eq_func(in0, scalar_val)
    bitmap_out = isin_by_bitmap(in0, in1, invert)
    if bitmap_out is not None:
        return bitmap_out
    if in0.numel() <= 2048 and in1.numel() <= 2048:
        return isin_by_comparation(in0, in1, invert)
    if assume_unique or in1.numel() <= 4194304:
        return isin_by_search(in0, in1, invert, unique_in1=False)
    return isin_by_search(in0, in1, invert, unique_in1=True)
