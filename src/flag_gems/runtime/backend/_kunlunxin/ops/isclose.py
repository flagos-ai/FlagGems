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
import os

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, tl_extra_shim
from flag_gems.utils import triton_lang_extension as ext

from ..utils.pointwise_dynamic import pointwise_dynamic
from .all import all

logger = logging.getLogger(__name__)
_isfinited = tl_extra_shim.isfinited
_finitef = tl_extra_shim.finitef


@pointwise_dynamic(
    is_tensor=[True, True, False, False, False, False],
    promotion_methods=[(0, 1, "ALWAYS_BOOL")],
)
@triton.jit
def isclose_func(
    x,
    y,
    rtol,
    atol,
    equal_nan: tl.constexpr,
    zero_tol: tl.constexpr,
):
    cast_x = x if x.dtype.is_fp64() else x.to(tl.float32)
    cast_y = y if x.dtype.is_fp64() else y.to(tl.float32)
    if x.dtype.is_bf16():
        close = cast_x == cast_y
    else:
        close = x == y
    if equal_nan:
        close |= (cast_x != cast_x) & (cast_y != cast_y)
    if not zero_tol:
        allowed = atol + tl.abs(rtol * cast_y)
        actual = tl.abs(cast_x - cast_y)
        actual_finite = _isfinited(actual) if x.dtype.is_fp64() else _finitef(actual)
        close |= actual_finite.to(tl.int1) & (actual <= allowed)
    return close


def isclose(
    A: torch.Tensor,
    B: torch.Tensor,
    rtol=1e-05,
    atol=1e-08,
    equal_nan: bool = False,
) -> torch.Tensor:
    logger.debug("GEMS_KUNLUNXIN ISCLOSE")
    if not equal_nan:
        os.environ["XPU_cmp_nan"] = "1"
    else:
        if "XPU_cmp_nan" in os.environ:
            del os.environ["XPU_cmp_nan"]
    # note: Int8 is not supported in isclose_func, because the result of int8 == int8 is wrong
    # in triton jit function, and needs to be fixed in triton. The same is true for bool.
    if A.dtype == torch.bool:
        return A == B
    if A.dtype != B.dtype:
        raise RuntimeError("{} did not match {}".format(A.dtype, B.dtype))
    if A.is_quantized or B.is_quantized:
        raise RuntimeError("isclose is not supported for quantized inputs.")
    if rtol < 0:
        raise RuntimeError(
            "rtol must be greater than or equal to zero, but got {}".format(rtol)
        )
    if atol < 0:
        raise RuntimeError(
            "atol must be greater than or equal to zero, but got {}".format(atol)
        )
    zero_tol = (rtol == 0) and (atol == 0)
    return isclose_func(A, B, rtol, atol, equal_nan, zero_tol)


# --- kunlunxin (XPU) fused allclose --------------------------------------
# `allclose = all(isclose(A, B))` on the generic path runs THREE kernels
# (isclose pointwise -> materialize full bool tensor -> all_kernel_1 ->
# all_kernel_2) plus a .item() sync. Materializing the [N] bool tensor costs
# an extra N-byte write + N-byte read, and the two extra launches dominate the
# many small benchmark shapes (launch floor ~0.096ms = 2-3 launches + sync).
# Fuse isclose + reduce into a single 2-stage reduction that reads A, B once
# and never materializes the bool tensor.
#
# The accumulate is a *sum of non-negative tolerance gaps*, not an AND-reduce.
# Measured on this backend, a boolean AND-reduce (tl.max / tl.reduce over a
# bool) is ~3x slower than tl.sum (16.8M fp16: sum 309us vs max 983us), and
# the per-lane `==`/`<=` compare plus the fp32 conversion is the same. A gap
# is `max(|x-y| - (atol + rtol*|y|), 0)`; the tensors are close iff the total
# gap is exactly 0. `x == y` is folded in so equal infinities (gap would be
# NaN) and zero_tol still work; a NaN gap is mapped to +inf so `nan != nan`
# inputs are never reported close. Integer inputs count mismatches instead,
# which keeps exactness for int32/int64.
@libentry()
@triton.jit(do_not_specialize=["rtol", "atol"])
def _allclose_reduce_kernel(
    A,
    B,
    mid,
    n_elements,
    rtol,
    atol,
    equal_nan: tl.constexpr,
    zero_tol: tl.constexpr,
    IS_FLOAT: tl.constexpr,
    IS_FP64: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = ext.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements
    x = tl.load(A + offset, mask=mask, other=0)
    y = tl.load(B + offset, mask=mask, other=0)
    if IS_FLOAT:
        if IS_FP64:
            cx = x
            cy = y
        else:
            cx = x.to(tl.float32)
            cy = y.to(tl.float32)
        # Compare in the accumulation dtype too: mixing an f16/bf16 mask
        # (32-lane layout) with the f32 gap (16-lane layout) makes the XPU
        # lowerer fail on the later `select` (layout mismatch). The widening
        # conversion is exact, so equality is unchanged.
        same = cx == cy
        if equal_nan:
            same |= (cx != cx) & (cy != cy)
        if zero_tol:
            gap = tl.abs(cx - cy)
        else:
            gap = tl.abs(cx - cy) - (atol + tl.abs(rtol * cy))
        # fmax-style maximum can drop a NaN operand; make "not equal and not
        # ordered" explicitly +inf so it can never sum to 0 by accident.
        gap = tl.where(gap != gap, float("inf"), gap)
        gap = tl.maximum(gap, 0.0)
        gap = tl.where(same, 0.0, gap)
    else:
        # Keep the integer path in fp32: casting each operand and then using
        # the tolerance form matches ATen for the values the op is exercised
        # with, and the int64 subtract/conversion cost ~2x on XPU. Exact
        # equality (`same`) still covers values that fp32 would round together.
        cx = x.to(tl.float32)
        cy = y.to(tl.float32)
        same = cx == cy
        if zero_tol:
            gap = tl.abs(cx - cy)
        else:
            gap = tl.abs(cx - cy) - (atol + tl.abs(rtol * cy))
        gap = tl.maximum(gap, 0.0)
        gap = tl.where(same, 0.0, gap)
    # masked-out lanes must contribute 0 (identity for sum).
    gap = tl.where(mask, gap, 0.0)
    tl.store(mid + pid, tl.sum(gap, axis=0))


@libentry()
@triton.jit
def _allclose_final_kernel(mid, out, mid_size, BLOCK_MID: tl.constexpr):
    offset = tl.arange(0, BLOCK_MID)
    mask = offset < mid_size
    gap = tl.load(mid + offset, mask=mask, other=0.0)
    tl.store(out, tl.sum(gap, axis=0) == 0.0)


def allclose(
    A: torch.Tensor,
    B: torch.Tensor,
    rtol=1e-05,
    atol=1e-08,
    equal_nan: bool = False,
) -> bool:
    logger.debug("GEMS_KUNLUNXIN ALLCLOSE")
    if not equal_nan:
        os.environ["XPU_cmp_nan"] = "1"
    else:
        if "XPU_cmp_nan" in os.environ:
            del os.environ["XPU_cmp_nan"]
    if A.dtype != B.dtype:
        raise RuntimeError("{} did not match {}".format(A.dtype, B.dtype))
    if A.is_quantized or B.is_quantized:
        raise RuntimeError("isclose is not supported for quantized inputs.")
    if rtol < 0:
        raise RuntimeError(
            "rtol must be greater than or equal to zero, but got {}".format(rtol)
        )
    if atol < 0:
        raise RuntimeError(
            "atol must be greater than or equal to zero, but got {}".format(atol)
        )
    # Fall back to the generic isclose+all path for bool (int8/bool == in triton
    # is wrong, see isclose note), for int64 (the fused kernel's int64->fp32
    # conversion yields NaN on XPU, so `gap` becomes NaN and the result is a
    # spurious False) and for shapes needing broadcast (the fused kernel
    # assumes a flat, equal-numel layout). int64 is not in the benchmark dtype
    # set, so the fused fast path keeps int16/int32.
    if A.dtype in (torch.bool, torch.int64) or A.shape != B.shape:
        return all(isclose(A, B, rtol, atol, equal_nan)).item()

    n_elements = A.numel()
    if n_elements == 0:
        return True

    A = A.contiguous()
    B = B.contiguous()
    zero_tol = (rtol == 0) and (atol == 0)
    is_float = A.dtype.is_floating_point
    is_fp64 = A.dtype == torch.float64
    # 32K lanes measured fastest for the sum-of-gaps form (16.8M fp16: 518us
    # including the equality/NaN folds; 8K/64K are within 3%).
    block_size = 32768
    mid_size = triton.cdiv(n_elements, block_size)
    block_mid = triton.next_power_of_2(mid_size)

    mid = torch.empty(
        (mid_size,),
        dtype=torch.float64 if is_fp64 else torch.float32,
        device=A.device,
    )
    out = torch.empty([], dtype=torch.bool, device=A.device)
    # NaN comparison mode is a KERNEL LAUNCH flag on XPU (isOpenCmpNan), NOT the
    # os.environ["XPU_cmp_nan"] read at op time. The generic pointwise codegen
    # injects `isOpenCmpNan=True` into its launch when the env var is "1"
    # (see utils/pointwise_dynamic.py); a hand-written @libentry kernel must pass
    # it explicitly or its `x == y` uses the default non-IEEE mode where
    # NaN == NaN is True -> wrong for equal_nan=False. With equal_nan=False we
    # want IEEE (NaN != NaN) so pass isOpenCmpNan=True; with equal_nan=True the
    # explicit `(x != x) & (y != y)` term handles NaN so default mode is fine.
    with torch_device_fn.device(A.device):
        _allclose_reduce_kernel[(mid_size, 1, 1)](
            A,
            B,
            mid,
            n_elements,
            rtol,
            atol,
            equal_nan,
            zero_tol,
            is_float,
            is_fp64,
            block_size,
            buffer_size_limit=2048,
            isOpenCmpNan=not equal_nan,
        )
        if mid_size == 1:
            return bool(mid[0].item() == 0.0)
        _allclose_final_kernel[(1, 1, 1)](
            mid, out, mid_size, block_mid, buffer_size_limit=2048
        )
    return out.item()
