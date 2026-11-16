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


# ---------------------------------------------------------------------------
# Raw 1D fast path for `isin(elements, python_scalar)` (aten::isin.Tensor_Scalar
# with a non-tensor `test_elements`).
#
# Why a raw kernel instead of the pointwise_dynamic wrappers above:
# the pw wrapper is host-bound for the whole official matrix -- on XPU 6 the
# device kernel of `isin_scalar_eq_func` is ~7.6 us (XPURT_PROF: 50 calls /
# 0.381 ms, harness/perf_ir/kernel_time_isin_tensor_scalar_pre.log) while the
# end-to-end call costs ~0.156-0.195 ms.  The wrapper overhead is ~92 us even
# for `torch.neg` (native launch is ~10 us), plus ~30-60 us of extra host work
# that only this op pays: `torch.full((), v, device=...)` (an extra device
# launch, itself dispatching through the gems `full` wrapper) followed by a
# blocking `in1.ravel()[0].item()` device->host sync.  Both are avoidable
# because the scalar is already a host-side Python object.
#
# Semantics reproduced (verified against the torch CPU reference used by the
# harness, `--ref cpu`): `isin(x, s)` is exactly `x == s` under ATen's
# "wrapped number" scalar promotion, i.e. the Python scalar is folded into the
# *tensor's* dtype for integral tensors and for float tensors, while a Python
# float next to an integral tensor compares in the default float dtype
# (`torch.result_type(int32, 0.5) == torch.float32`).  Evidence (CPU, this
# torch build): `isin(uint8_255, -1) -> True`, `isin(int8_-1, 255) -> True`,
# `isin(int32_0, 2**40) -> True`, `isin(int16_5, 70000) -> False`,
# `isin(fp16_inf, 1e30) -> True` -- all identical to `x == s`.  The pw kernels
# compare in Triton's own promotion domain, which differs from ATen: the
# pre-change matrix probe (harness/results/isin_tensor_scalar/probe_matrix_pre.log)
# reports 30 mismatches, e.g. uint8 vs -1 / int32 vs 2**40 / int32 vs 2**63-1
# / fp16 + bf16 vs 1e30 -- because the non-specialized `val0` arg also freezes
# its element type at the first call of the process (same trap as multiply_).
# Folding the scalar on the *host* into the comparison domain and passing it as
# a `tl.constexpr` removes both problems at once.
# ---------------------------------------------------------------------------
_ISIN_SCALAR_MAX_BLOCK = 131072
_ISIN_SCALAR_UNROLL_NUM = 16
_ISIN_SCALAR_BUFFER_SIZE_LIMIT = 8192
_ISIN_SCALAR_MEMORY_ASYNC = False


def _isin_scalar_pick_block(n_elements):
    # Same bucketing as the other flat XPU pointwise kernels (special_erfcx):
    # unmasked when the length divides the tile exactly (the masked memory path
    # costs ~2x on this backend), masked fallback otherwise.
    if n_elements >= 1_048_576 and n_elements % _ISIN_SCALAR_MAX_BLOCK == 0:
        return _ISIN_SCALAR_MAX_BLOCK, 32, False
    if n_elements >= 262_144 and n_elements % 32768 == 0:
        return 32768, 8, False
    if n_elements >= 16384 and n_elements % 16384 == 0:
        return 16384, 8, False
    if n_elements <= 65536:
        return 2048, 4, True
    return 16384, 8, True


_ISIN_INT_WRAP = {
    torch.int8: (8, True),
    torch.uint8: (8, False),
    torch.int16: (16, True),
    torch.int32: (32, True),
    torch.int64: (64, True),
}
# float dtypes handled by the fast path (float8/exotic kinds keep the generic
# pointwise_dynamic route)
_ISIN_SCALAR_FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def _isin_scalar_wrap(scalar, dtype):
    """Fold a Python scalar into `dtype` exactly like ATen's wrapped-number
    scalar promotion does (integral wrap-around included), host-side only."""
    if isinstance(scalar, bool):
        scalar = int(scalar)
    if dtype.is_floating_point:
        # single rounding Python-float -> dtype (fp64 on the host is exact
        # enough to avoid the fp64->fp32->fp16 double-rounding trap)
        return float(torch.tensor(float(scalar), dtype=torch.float64).to(dtype).item())
    bits, signed = _ISIN_INT_WRAP[dtype]
    value = int(scalar) & ((1 << bits) - 1)
    if signed and value >= (1 << (bits - 1)):
        value -= 1 << bits
    return value


@triton.jit
def _isin_scalar_cmp_kernel(
    in0_ptr,
    out_ptr,
    n_elements,
    S: tl.constexpr,
    INVERT: tl.constexpr,
    FP_CAST: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = ext.program_id(0)
    offset = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offset < n_elements
    x = tl.load(in0_ptr + offset, mask=mask, other=0)
    if FP_CAST:
        x = x.to(tl.float32)
    if INVERT:
        result = x != S
    else:
        result = x == S
    tl.store(out_ptr + offset, result, mask=mask)


@triton.jit
def _isin_scalar_cmp_kernel_unmasked(
    in0_ptr,
    out_ptr,
    S: tl.constexpr,
    INVERT: tl.constexpr,
    FP_CAST: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = ext.program_id(0)
    offset = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(in0_ptr + offset)
    if FP_CAST:
        x = x.to(tl.float32)
    if INVERT:
        result = x != S
    else:
        result = x == S
    tl.store(out_ptr + offset, result)


def _isin_scalar_raw(in0, scalar, invert):
    """isin(in0, python_scalar) via a flat compare kernel.

    Returns None when the input layout / dtype is outside the fast path (empty,
    bool, non-contiguous) so the caller keeps the generic pointwise_dynamic
    path unchanged.
    """
    if not isinstance(scalar, (bool, int, float)):
        # complex / exotic Scalar kinds keep the generic fallback behaviour
        return None
    if not torch.is_tensor(in0) or in0.dtype == torch.bool:
        return None
    n_elements = in0.numel()
    if n_elements == 0 or not in0.is_contiguous():
        return None
    if in0.is_floating_point():
        if in0.dtype not in _ISIN_SCALAR_FLOAT_DTYPES:
            return None  # float8 / exotic float kinds keep the generic route
        cmp_dtype = in0.dtype
        fp_cast = False
    elif in0.dtype in _ISIN_INT_WRAP:
        if isinstance(scalar, float):
            # ATen: python float next to an integral tensor -> default fp32
            cmp_dtype = torch.float32
            fp_cast = True
        else:
            cmp_dtype = in0.dtype
            fp_cast = False
    else:
        return None
    value = _isin_scalar_wrap(scalar, cmp_dtype)
    out = torch.empty(in0.shape, dtype=torch.bool, device=in0.device)
    block_size, num_warps, masked = _isin_scalar_pick_block(n_elements)
    grid = (
        triton.cdiv(n_elements, block_size) if masked else n_elements // block_size,
    )
    with torch_device_fn.device(in0.device.index):
        if masked:
            _isin_scalar_cmp_kernel[grid](
                in0,
                out,
                n_elements,
                S=value,
                INVERT=invert,
                FP_CAST=fp_cast,
                BLOCK=block_size,
                num_warps=num_warps,
                unroll_num=_ISIN_SCALAR_UNROLL_NUM,
                buffer_size_limit=_ISIN_SCALAR_BUFFER_SIZE_LIMIT,
                isCloseMemoryAsync=_ISIN_SCALAR_MEMORY_ASYNC,
            )
        else:
            _isin_scalar_cmp_kernel_unmasked[grid](
                in0,
                out,
                S=value,
                INVERT=invert,
                FP_CAST=fp_cast,
                BLOCK=block_size,
                num_warps=num_warps,
                unroll_num=_ISIN_SCALAR_UNROLL_NUM,
                buffer_size_limit=_ISIN_SCALAR_BUFFER_SIZE_LIMIT,
                isCloseMemoryAsync=_ISIN_SCALAR_MEMORY_ASYNC,
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
                    in0_flat,
                    bit,
                    out,
                    n0,
                    range_size,
                    min_val=lo,
                    invert=invert,
                    BLOCK_SIZE=BLOCK,
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
        # aten::isin.Tensor_Scalar: the Python scalar stays on the host, so a
        # flat compare kernel can be launched directly (no `torch.full` device
        # round-trip, no blocking `.item()`).  Returns None for layouts/dtypes
        # outside the fast path (empty, bool, non-contiguous, complex scalars),
        # which keep the pointwise_dynamic route below unchanged.
        fast_out = _isin_scalar_raw(in0, in1, invert)
        if fast_out is not None:
            return fast_out
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
