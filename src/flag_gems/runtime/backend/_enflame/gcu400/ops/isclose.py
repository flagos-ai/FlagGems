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

from flag_gems.ops.all import all
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)

# gcu400: 48 SPCs, mirrors the enflame CodeGenConfig.max_grid_size.
_NUM_SPC = 48
# Per-CTA streaming tile (elements), keyed by element size so that two input
# tiles plus the int1 accumulator stay within the per-CTA DSM concurrency
# budget; the 1G/16M sweeps favored w2 with a doubled tile for 2-byte dtypes,
# a quarter tile for bf16 (the only dtype comparing after the fp32 cast,
# which is latency-bound on the wider tiles), and w2 for 4/8-byte dtypes.
_BLOCK_2BYTE = 65536
_BLOCK_4BYTE = 32768
_BLOCK_8BYTE = 16384
_BF16_BLOCK = 16384


@triton.jit
def _reduce_all(a, b):
    return a & b


@triton.jit
def _isclose_predicate(
    x,
    y,
    rtol,
    atol,
    equal_nan: tl.constexpr,
    zero_tol: tl.constexpr,
):
    # Scalar isclose predicate, kept line-for-line identical to the
    # pointwise_dynamic isclose_func body (including the bf16 fp32-cast
    # comparison and the fp64 passthrough) so the fused kernels inherit the
    # exact same numerics.
    cast_x = x if x.dtype.is_fp64() else x.to(tl.float32)
    cast_y = y if x.dtype.is_fp64() else y.to(tl.float32)
    # Keep the equality check in the native dtype: fp16/bf16 -> fp32 is an exact bit extension,
    # so the promoted comparison is equivalent to a direct one, saving a wider scalar comparison pass.
    close = x == y
    if equal_nan:
        close |= (cast_x != cast_x) & (cast_y != cast_y)
    if not zero_tol:
        allowed = atol + tl.abs(rtol * cast_y)
        actual = tl.abs(cast_x - cast_y)
        # Only need to rule out actual == +inf (in which case allowed may also be inf and be wrongly judged close);
        # Use a vector comparison instead of a per-element finitef libm call, avoiding a break in the vector pipeline.
        close |= (actual < float("inf")) & (actual <= allowed)
    return close


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
    close = x == y
    if equal_nan:
        close |= (cast_x != cast_x) & (cast_y != cast_y)
    if not zero_tol:
        allowed = atol + tl.abs(rtol * cast_y)
        actual = tl.abs(cast_x - cast_y)
        close |= (actual < float("inf")) & (actual <= allowed)
    return close


def _is_fused_fast_path(A, B):
    return (
        A.dtype != torch.bool
        and not A.dtype.is_complex
        and A.dtype != torch.float64
        and A.element_size() >= 2
        and A.shape == B.shape
        and A.is_contiguous()
        and B.is_contiguous()
        and A.numel() > 0
    )


def _fused_launch_config(A):
    if A.dtype == torch.bfloat16:
        block, num_warps = _BF16_BLOCK, 4
    else:
        block = {2: _BLOCK_2BYTE, 4: _BLOCK_4BYTE, 8: _BLOCK_8BYTE}[A.element_size()]
        num_warps = 2
    N = A.numel()
    if N < block:
        # Small workload: shrink the tile to the workload so the whole range
        # is one mask-free EVEN block at full warp occupancy instead of
        # scanning a masked 2x/16x oversized tile.
        block = max(triton.next_power_of_2(N), 1024)
        num_warps = 4
    return block, num_warps


def isclose(
    A: torch.Tensor,
    B: torch.Tensor,
    rtol=1e-05,
    atol=1e-08,
    equal_nan: bool = False,
) -> torch.Tensor:
    logger.debug("GEMS_ENFLAME ISCLOSE")
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
    if _is_fused_fast_path(A, B):
        N = A.numel()
        out = torch.empty_like(A, dtype=torch.bool)
        block, num_warps = _fused_launch_config(A)
        grid = (min(triton.cdiv(N, block), _NUM_SPC),)
        with torch_device_fn.device(A.device.index):
            isclose_fast_kernel[grid](
                A,
                B,
                out,
                N,
                rtol,
                atol,
                equal_nan=equal_nan,
                zero_tol=zero_tol,
                EVEN=(N % block == 0),
                BLOCK=block,
                num_warps=num_warps,
            )
        return out
    return isclose_func(A, B, rtol, atol, equal_nan, zero_tol)


@libentry()
@triton.jit(do_not_specialize=["N", "rtol", "atol"])
def allclose_fused_kernel(
    x_ptr,
    y_ptr,
    flag_ptr,
    N,
    rtol,
    atol,
    equal_nan: tl.constexpr,
    zero_tol: tl.constexpr,
    EVEN: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Single-launch allclose: stream x/y tiles, fold the isclose predicate into
    # a block-local AND reduction, and publish "not allclose" via one atomic per
    # CTA.
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    arange = tl.arange(0, BLOCK)
    acc = tl.full([BLOCK], value=1, dtype=tl.int1)
    num_blocks = tl.cdiv(N, BLOCK)
    for block_id in tl.range(pid, num_blocks, num_pids):
        off = block_id * BLOCK + arange
        if EVEN:
            x = tl.load(x_ptr + off)
            y = tl.load(y_ptr + off)
        else:
            mask = off < N
            # other=0 keeps the masked lanes close under both the equality
            # (0 == 0) and the tolerance (|0 - 0| <= atol) predicates, so they
            # never break the AND fold.
            x = tl.load(x_ptr + off, mask=mask, other=0)
            y = tl.load(y_ptr + off, mask=mask, other=0)
        close = _isclose_predicate(x, y, rtol, atol, equal_nan, zero_tol)
        acc = acc & close
    block_true = tl.reduce(acc, axis=0, combine_fn=_reduce_all)
    if not block_true:
        tl.atomic_or(flag_ptr, 1)


@libentry()
@triton.jit(do_not_specialize=["N", "rtol", "atol"])
def isclose_fast_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    N,
    rtol,
    atol,
    equal_nan: tl.constexpr,
    zero_tol: tl.constexpr,
    EVEN: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Single-launch isclose for same-shape contiguous inputs: flat-tile streaming
    # without the block-pointer/boundary-check overhead of the generated
    # pointwise kernel.
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    arange = tl.arange(0, BLOCK)
    num_blocks = tl.cdiv(N, BLOCK)
    for block_id in tl.range(pid, num_blocks, num_pids):
        off = block_id * BLOCK + arange
        if EVEN:
            x = tl.load(x_ptr + off)
            y = tl.load(y_ptr + off)
            close = _isclose_predicate(x, y, rtol, atol, equal_nan, zero_tol)
            tl.store(out_ptr + off, close.to(out_ptr.type.element_ty))
        else:
            mask = off < N
            x = tl.load(x_ptr + off, mask=mask, other=0)
            y = tl.load(y_ptr + off, mask=mask, other=0)
            close = _isclose_predicate(x, y, rtol, atol, equal_nan, zero_tol)
            tl.store(out_ptr + off, close.to(out_ptr.type.element_ty), mask=mask)


def _allclose_fused(A, B, rtol, atol, equal_nan):
    zero_tol = (rtol == 0) and (atol == 0)
    N = A.numel()
    flag = torch.zeros(1, dtype=torch.int32, device=A.device)
    block, num_warps = _fused_launch_config(A)
    grid = (min(triton.cdiv(N, block), _NUM_SPC),)
    with torch_device_fn.device(A.device.index):
        allclose_fused_kernel[grid](
            A,
            B,
            flag,
            N,
            rtol,
            atol,
            equal_nan=equal_nan,
            zero_tol=zero_tol,
            EVEN=(N % block == 0),
            BLOCK=block,
            num_warps=num_warps,
        )
    return flag.item() == 0


def allclose(
    A: torch.Tensor,
    B: torch.Tensor,
    rtol=1e-05,
    atol=1e-08,
    equal_nan: bool = False,
) -> bool:
    logger.debug("GEMS_ENFLAME ALLCLOSE")
    if A.dtype != B.dtype or rtol < 0 or atol < 0 or A.is_quantized or B.is_quantized:
        return all(isclose(A, B, rtol, atol, equal_nan)).item()
    if _is_fused_fast_path(A, B):
        return _allclose_fused(A, B, rtol, atol, equal_nan)
    return all(isclose(A, B, rtol, atol, equal_nan)).item()
