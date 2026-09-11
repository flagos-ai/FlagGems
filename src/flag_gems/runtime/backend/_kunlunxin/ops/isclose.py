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

from ..utils.block_size_utils import get_block_size_1d
from ..utils.pointwise_dynamic import pointwise_dynamic
from .all import all, all_kernel_2, reduce_all

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
# Fuse isclose + AND-reduce into a single 2-stage reduction that reads A, B
# once and never materializes the bool tensor: stage1 computes isclose inline
# per BLOCK chunk and AND-reduces to one bool per program; stage2 (reused
# all_kernel_2) AND-reduces the per-chunk bools. Saves the bool write+read
# (~20-33% of traffic) and one launch. The isclose math is byte-for-byte the
# same as isclose_func so correctness (incl. equal_nan / zero_tol / fp64 /
# bf16 branches) is preserved.
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
    BLOCK_SIZE: tl.constexpr,
):
    pid = ext.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements
    x = tl.load(A + offset, mask=mask, other=0)
    y = tl.load(B + offset, mask=mask, other=0)
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
    # masked-out lanes must be True (identity for AND).
    close = tl.where(mask, close, True)
    result = tl.reduce(close, axis=0, combine_fn=reduce_all)
    tl.store(mid + pid, result)


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
    # is wrong, see isclose note) and for shapes needing broadcast (the fused
    # kernel assumes a flat, equal-numel layout).
    if A.dtype == torch.bool or A.shape != B.shape:
        return all(isclose(A, B, rtol, atol, equal_nan)).item()

    n_elements = A.numel()
    if n_elements == 0:
        return True

    A = A.contiguous()
    B = B.contiguous()
    zero_tol = (rtol == 0) and (atol == 0)
    block_size = get_block_size_1d(n_elements, A.element_size())
    mid_size = triton.cdiv(n_elements, block_size)
    block_mid = triton.next_power_of_2(mid_size)

    mid = torch.empty((mid_size,), dtype=torch.bool, device=A.device)
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
            block_size,
            buffer_size_limit=2048,
            isOpenCmpNan=not equal_nan,
        )
        if mid_size == 1:
            return mid.reshape([]).item()
        all_kernel_2[(1, 1, 1)](mid, out, mid_size, block_mid, buffer_size_limit=2048)
    return out.item()
