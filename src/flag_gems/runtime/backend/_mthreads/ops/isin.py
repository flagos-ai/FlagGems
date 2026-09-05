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

import math
import numbers

import torch
import triton
import triton.language as tl

from flag_gems.ops.all import reduce_all
from flag_gems.ops.any import reduce_any
from flag_gems.ops.unique import _unique2
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import pointwise_dynamic
from flag_gems.utils import triton_lang_extension as ext
from flag_gems.utils.libentry import libentry


# A scalar test set has no set-operation work to do: isin(x, scalar) is an
# elementwise equality (or inequality for invert=True).  Keep the scalar as a
# kernel argument so the generated pointwise kernel performs exactly one input
# load, one compare, and one bool store.  In particular, do not use the common
# eq implementation here: it casts integer operands to fp32 and therefore
# loses equality for values above 2**24.
@pointwise_dynamic(
    is_tensor=[True, False],
    promotion_methods=[(0, 1, "ALWAYS_BOOL")],
)
@triton.jit
def isin_scalar_eq_func(x, y):
    return x == y


@pointwise_dynamic(
    is_tensor=[True, False],
    promotion_methods=[(0, 1, "ALWAYS_BOOL")],
)
@triton.jit
def isin_scalar_ne_func(x, y):
    return x != y


@pointwise_dynamic(
    is_tensor=[True, False],
    promotion_methods=[(0, 1, "ALWAYS_BOOL")],
)
@triton.jit
def isin_scalar_eq_float_func(x, y):
    return x.to(tl.float32) == y.to(tl.float32)


@pointwise_dynamic(
    is_tensor=[True, False],
    promotion_methods=[(0, 1, "ALWAYS_BOOL")],
)
@triton.jit
def isin_scalar_ne_float_func(x, y):
    return x.to(tl.float32) != y.to(tl.float32)


def _normalize_integer_scalar(value, dtype):
    """Match MUSA aten's Python-int scalar conversion without a device copy."""
    if not isinstance(value, numbers.Integral) or isinstance(value, bool):
        return value
    value = int(value)
    if dtype not in (
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    ):
        return value
    if value < -(1 << 63) or value >= (1 << 64):
        raise OverflowError("Python int is out of range for an integral tensor")
    # MUSA aten promotes Python integers to its int32 scalar representation
    # for sub-int64 tensors. int64 elements retain an int64 scalar so large
    # integers remain exact. This mirrors the native Tensor_Scalar overload.
    bits = 64 if dtype == torch.int64 else 32
    narrowed = value & ((1 << bits) - 1)
    if narrowed >= (1 << (bits - 1)):
        narrowed -= 1 << bits
    return narrowed


@libentry()
@triton.jit(do_not_specialize=["scalar"])
def isin_scalar_1d_kernel(
    in_ptr: tl.tensor,
    out_ptr: tl.tensor,
    numel: int,
    scalar,
    BLOCK_SIZE: tl.constexpr,
    invert: tl.constexpr,
    IS_FLOAT: tl.constexpr,
):
    """Contiguous streaming kernel: one input load, compare, and bool store."""
    pid = ext.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    values = tl.load(in_ptr + offsets, mask=mask, other=0)
    values = values.to(tl.float32) if IS_FLOAT else values
    result = values != scalar if invert else values == scalar
    tl.store(out_ptr + offsets, result, mask=mask)


def _isin_scalar_contiguous(in0, scalar, invert):
    numel = in0.numel()
    out = torch.empty_like(in0, dtype=torch.bool)
    block_size = 1024
    grid = (triton.cdiv(numel, block_size),)
    with torch_device_fn.device(in0.device.index):
        isin_scalar_1d_kernel[grid](
            in0,
            out,
            numel,
            scalar,
            BLOCK_SIZE=block_size,
            invert=invert,
            IS_FLOAT=in0.is_floating_point(),
            num_warps=4,
        )
    return out


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
    tl.store(out_ptr + i0, not out if invert else out, mask=mask)


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
        _, BLOCK_M, num_warps = launch_arg(None, 2048, M, 8)
    elif M <= 268435456:  # 2 ** 28 = 1024 * 262144
        _, BLOCK_M, num_warps = launch_arg(None, 4096, M, 8)
    else:
        _, BLOCK_M, num_warps = launch_arg(None, 2048, M, 8)
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


def isin_tensor_scalar(
    in0,
    in1,
    *,
    assume_unique: bool = False,
    invert: bool = False,
) -> torch.Tensor:
    # Preserve Python scalars as scalars all the way to the pointwise kernel.
    # The previous implementation first materialized a one-element device
    # tensor, then routed through comparison/search (including reduction or
    # sort/binary-search machinery).  assume_unique is irrelevant for a
    # singleton set.
    if torch.is_tensor(in0) and (in0.dtype == torch.bool or isinstance(in1, bool)):
        raise RuntimeError("Unsupported input type encountered for isin(): Bool")
    if (
        torch.is_tensor(in0)
        and not in0.is_complex()
        and in0.dtype != torch.bool
        and isinstance(in1, numbers.Real)
    ):
        if in0.numel() == 0:
            return torch.zeros_like(in0, dtype=torch.bool)
        in1 = _normalize_integer_scalar(in1, in0.dtype)
        # The 1-D kernel uses Triton's native index type; keep very large
        # tensors on the stride-aware generator, which can disable block
        # pointers when a 32-bit index would overflow.
        if (
            in0.is_contiguous()
            and in0.numel() <= torch.iinfo(torch.int32).max
            and (in0.is_floating_point() or isinstance(in1, numbers.Integral))
        ):
            return _isin_scalar_contiguous(in0, in1, invert)
        if invert:
            if in0.is_floating_point():
                return isin_scalar_ne_float_func(in0, in1)
            return isin_scalar_ne_func(in0, in1)
        if in0.is_floating_point():
            return isin_scalar_eq_float_func(in0, in1)
        return isin_scalar_eq_func(in0, in1)

    return _isin_mthreads_original(in0, in1, assume_unique=assume_unique, invert=invert)


def _isin_mthreads_original(
    in0,
    in1,
    *,
    assume_unique: bool = False,
    invert: bool = False,
) -> torch.Tensor:
    if not torch.is_tensor(in0):
        assert torch.is_tensor(in1)
        in0 = torch.tensor(in0, device=in1.device)
    elif not torch.is_tensor(in1):
        assert torch.is_tensor(in0)
        in1 = torch.tensor(in1, device=in0.device)
    if in0.numel() == 0 or in1.numel() == 0:
        return torch.zeros_like(in0, dtype=torch.bool)
    elif in0.numel() <= 12288 and in1.numel() <= 12288:  # 1024 * 12
        return isin_by_comparation(in0, in1, invert)
    elif assume_unique or in1.numel() <= 4194304:  # 1024 * 4096
        return isin_by_search(in0, in1, invert, unique_in0=False, unique_in1=False)
    else:
        return isin_by_search(in0, in1, invert, unique_in0=False, unique_in1=True)


def isin(
    in0,
    in1,
    *,
    assume_unique: bool = False,
    invert: bool = False,
) -> torch.Tensor:
    if torch.is_tensor(in0) and not torch.is_tensor(in1):
        return isin_tensor_scalar(in0, in1, assume_unique=assume_unique, invert=invert)
    return _isin_mthreads_original(in0, in1, assume_unique=assume_unique, invert=invert)
