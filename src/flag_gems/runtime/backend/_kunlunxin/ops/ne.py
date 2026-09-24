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

from flag_gems.ops.ne_ import ne_ as _generic_ne_
from flag_gems.ops.ne_ import ne_scalar_ as _generic_ne_scalar_

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)


# NOTE: `kunlunAutoGrid=True` + `unroll_num=8` are what make the sibling tuned
# comparison ops (gt / greater / greater_scalar) reach ~0.23-0.41 on large
# shapes. ne/ne_scalar previously shipped a bare config WITHOUT them and were
# stuck at ~0.14 (gems ~7.95ms vs torch ~1.08ms on the 65536-wide shapes, IR
# baseline `harness/perf_ir_3/ir-ne_scalar-dev3.log`). Adding the two params
# lifts throughput ~1.6x (mirrors greater_scalar, zero algorithm change).
config_ = CodeGenConfig(
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
    promotion_methods=[(0, 1, "ALWAYS_BOOL")],
    config=config_,
)
@triton.jit
def ne_func(x, y):
    return x.to(tl.float32) != y.to(tl.float32)


def ne(A, B):
    logger.debug("GEMS_KUNLUNXIN NE")
    os.environ["TRITONXPU_COMPARE_FUSION"] = "1"
    os.environ["TRITONXPU_FP16_FAST"] = "1"
    res = ne_func(A, B)
    del os.environ["TRITONXPU_COMPARE_FUSION"]
    del os.environ["TRITONXPU_FP16_FAST"]
    return res


@pointwise_dynamic(
    is_tensor=[True, False],
    promotion_methods=[(0, 1, "ALWAYS_BOOL")],
    config=config_,
)
@triton.jit
def ne_func_scalar(x, y):
    return x.to(tl.float32) != y


def ne_scalar(A, B):
    logger.debug("GEMS_KUNLUNXIN NE_SCALAR")
    numel = A.numel()
    dtype = A.dtype
    if A.is_contiguous() and dtype in (torch.float16, torch.float32, torch.bfloat16):
        s = float(B)
        wrapped = torch.tensor(s, dtype=dtype).item()
        if math.isfinite(wrapped):
            # wrapped == torch's wrapped-scalar semantics (compare in the
            # input dtype). Only take the fast path when the wrapped scalar
            # is finite: when |s| overflows the dtype (e.g. 66000 for fp16,
            # 1e300 for fp32) torch wraps it to +/-inf and x != +/-inf must
            # stay on the exact generic compare path (x = +/-inf vs
            # s = +/-inf -> False cannot be expressed by the saturating
            # distance formula, which sees inf-inf = NaN -> True).
            if (
                numel >= _NE_SCALAR_FAST_TILE * _NE_SCALAR_MIN_GRID
                and numel % _NE_SCALAR_FAST_TILE == 0
            ):
                # exact-multiple flat tiles (grid >= MIN_GRID): no mask, no
                # i1 -- a saturating fp32 store + vendor bool conversion.
                return _ne_scalar_native(A, float(wrapped), numel, masked=False)
            if numel >= _NE_SCALAR_MASKED_MIN and numel % _NE_SCALAR_FAST_TILE != 0:
                # non-multiple mid sizes (e.g. 2.56M, [10000,256]): flat
                # tiles with a real tail mask. The mask is genuine (tail
                # elements), so the masked-memory path is the only penalty.
                return _ne_scalar_native(A, float(wrapped), numel, masked=True)
            # Small / thin shapes (numel < FAST_TILE): route to the native
            # fused kernel with an adaptive small TILE instead of the
            # ~10x-slower generic pointwise path. Tile hugs numel (next_pow2,
            # floor 1024) so we don't launch a 131072-lane program for a
            # few-K-element tensor. Same fix as le_scalar's small-shape branch.
            if 0 < numel < _NE_SCALAR_FAST_TILE:
                tile = min(
                    _NE_SCALAR_FAST_TILE, max(1024, triton.next_power_of_2(numel))
                )
                return _ne_scalar_native(
                    A, float(wrapped), numel, masked=(numel % tile != 0), tile=tile
                )
    # Like gt_scalar / greater_scalar, the scalar path must NOT set
    # TRITONXPU_COMPARE_FUSION / TRITONXPU_FP16_FAST: for tensor-vs-scalar the
    # fusion env vars make the compiler emit an fp16 compare that trips
    # `arith.cmpf same-type` and overflows uni_sram -> compile failure.
    res = ne_func_scalar(A, B)
    return res


# ---------------------------------------------------------------------------
# ne_scalar: native fused in-dtype compare (x != scalar.to(DTYPE)) under
# TRITONXPU_COMPARE_FUSION=1, bool written directly. Gate admits only finite
# wrapped scalars exactly representable in A.dtype (bit-identical to torch,
# incl. ne(NaN,s)==True).
_NE_SCALAR_FAST_TILE = 131072
_NE_SCALAR_MIN_GRID = 128
_NE_SCALAR_MASKED_MIN = 1 << 20

# ---------------------------------------------------------------------------
# bf16 note: no native bf16 compare on xpu3 -> widen to fp32, large shapes
# stay compiler-bound; fp16/fp32 reach the fused path.


@triton.jit
def ne_scalar_native_kernel(
    out_ptr, x_ptr, scalar, TILE: tl.constexpr, DTYPE: tl.constexpr
):
    pid = tl.program_id(0)
    tid = pid * TILE + tl.arange(0, TILE)
    x = tl.load(x_ptr + tid)
    r = x != scalar.to(DTYPE)
    tl.store(out_ptr + tid, r.to(tl.int8))


@triton.jit
def ne_scalar_native_masked_kernel(
    out_ptr, x_ptr, scalar, numel, TILE: tl.constexpr, DTYPE: tl.constexpr
):
    pid = tl.program_id(0)
    tid = pid * TILE + tl.arange(0, TILE)
    mask = tid < numel
    x = tl.load(x_ptr + tid, mask=mask)
    r = x != scalar.to(DTYPE)
    tl.store(out_ptr + tid, r.to(tl.int8), mask=mask)


def _ne_scalar_native(A, scalar, numel, masked, tile=_NE_SCALAR_FAST_TILE):
    # Single-kernel native compare: writes the bool result directly, so the
    # fp32 intermediate buffer and _copy_from pass of the saturating recipe
    # are gone. Env must be set before the (first) launch so the fusion pass
    # sees it at compile time (same set/del pattern as ne above).
    out = torch.empty_like(A, dtype=torch.bool)
    x = A.reshape(-1)
    grid = (math.ceil(numel / tile),) if masked else (numel // tile,)
    DTYPE = {
        torch.float16: tl.float16,
        torch.bfloat16: tl.bfloat16,
        torch.float32: tl.float32,
    }[A.dtype]
    os.environ["TRITONXPU_COMPARE_FUSION"] = "1"
    os.environ["TRITONXPU_FP16_FAST"] = "1"
    try:
        if masked:
            ne_scalar_native_masked_kernel[grid](
                out,
                x,
                scalar,
                numel,
                TILE=tile,
                DTYPE=DTYPE,
                num_warps=4,
                buffer_size_limit=8192,
                unroll_num=16,
                isCloseMemoryAsync=False,
            )
        else:
            ne_scalar_native_kernel[grid](
                out,
                x,
                scalar,
                TILE=tile,
                DTYPE=DTYPE,
                num_warps=4,
                buffer_size_limit=8192,
                unroll_num=16,
                isCloseMemoryAsync=False,
            )
    finally:
        del os.environ["TRITONXPU_COMPARE_FUSION"]
        del os.environ["TRITONXPU_FP16_FAST"]
    return out


# ---------------------------------------------------------------------------
# ne_ (in-place x.ne_(y)): stores the saturating distance t = min(1,|x-y|*1e64)
# back into x (ne = t, no i1; ne(NaN,y)==True). In-place-safe config avoids the
# async-copy noc-idle-timeout deadlock under aliasing.
# Same gates as eq_: fast unmasked flat tiles only for fp16/fp32 contiguous
# same-shape tensors with exact-multiple numel and grid >= MIN_GRID; all
# other dtypes/shapes/aliasing fall into the DEFAULT-promotion pointwise
# in-place kernel under the in-place-safe config; non-float/non-contiguous
# keep the previous generic in-place path, behavior unchanged.
config_inplace_ = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    kunlunAutoGrid=True,
    unroll_num=8,
)


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")], config=config_inplace_)
@triton.jit
def ne_func_tensor_inplace(x, y):
    t = (x.to(tl.float32) - y.to(tl.float32)) * 1.0e32
    t = t * 1.0e32
    t = tl.abs(t)
    t = tl.minimum(1.0, t)
    return t


def ne_(A, B):
    logger.debug("GEMS_KUNLUNXIN NE_ TENSOR")
    if A.device != B.device:
        B = B.to(A.device)
    numel = A.numel()
    if A.is_contiguous() and A.dtype in (torch.float16, torch.float32, torch.bfloat16):
        if (
            A.dtype in (torch.float16, torch.float32)
            and B.is_contiguous()
            and B.dtype == A.dtype
            and A.shape == B.shape
            and numel >= _NE_TENSOR_INPLACE_FAST_TILE * _NE_TENSOR_INPLACE_MIN_GRID
            and numel % _NE_TENSOR_INPLACE_FAST_TILE == 0
        ):
            # exact-multiple flat tiles: no mask at all; grid fixed.
            return _ne_tensor_inplace_fast(A, B, numel)
        ne_func_tensor_inplace(A, B, out0=A)
        return A
    # Everything else (non-float dtype, non-contiguous, ...) keeps the
    # original generic in-place path, behavior unchanged.
    return _generic_ne_(A, B)


# in-place alias safety: the fast kernel writes into the SAME tensor it
# reads, so it must keep the DEFAULT isCloseMemoryAsync (True = async copy
# closed); passing False with in-place aliasing is the documented "noc idle
# timeout" deadlock, same as _eq_tensor_inplace_fast.
_NE_TENSOR_INPLACE_FAST_TILE = 131072
_NE_TENSOR_INPLACE_MIN_GRID = 128


@triton.jit
def ne_tensor_inplace_fast_kernel(x_ptr, y_ptr, TILE: tl.constexpr):
    pid = tl.program_id(0)
    tid = pid * TILE + tl.arange(0, TILE)
    x = tl.load(x_ptr + tid)
    y = tl.load(y_ptr + tid)
    t = (x.to(tl.float32) - y.to(tl.float32)) * 1.0e32
    t = t * 1.0e32
    t = tl.abs(t)
    t = tl.minimum(1.0, t)
    tl.store(x_ptr + tid, t)


def _ne_tensor_inplace_fast(A, B, numel):
    grid = (numel // _NE_TENSOR_INPLACE_FAST_TILE,)
    ne_tensor_inplace_fast_kernel[grid](
        A,
        B,
        TILE=_NE_TENSOR_INPLACE_FAST_TILE,
        num_warps=4,
        buffer_size_limit=8192,
        unroll_num=16,
        isCloseMemoryAsync=True,
    )
    return A


# ---------------------------------------------------------------------------
# ne_scalar_ (in-place x.ne_(s)): same saturating-distance recipe as ne_,
# gated on finite representable scalar.
@pointwise_dynamic(
    is_tensor=[True, False],
    promotion_methods=[(0, 1, "DEFAULT")],
    config=config_inplace_,
)
@triton.jit
def ne_func_scalar_inplace(x, y):
    t = (x.to(tl.float32) - y) * 1.0e32
    t = t * 1.0e32
    t = tl.abs(t)
    t = tl.minimum(1.0, t)
    return t


def ne_scalar_(A, B):
    logger.debug("GEMS_KUNLUNXIN NE_ SCALAR")
    numel = A.numel()
    dtype = A.dtype
    if (
        A.is_contiguous()
        and dtype in (torch.float16, torch.float32, torch.bfloat16)
        and float(B) == float(torch.tensor(float(B), dtype=dtype).item())
    ):
        wrapped = float(torch.tensor(float(B), dtype=dtype).item())
        if math.isfinite(wrapped):
            if (
                dtype in (torch.float16, torch.float32)
                and numel >= _NE_SCALAR_INPLACE_FAST_TILE * _NE_SCALAR_INPLACE_MIN_GRID
                and numel % _NE_SCALAR_INPLACE_FAST_TILE == 0
            ):
                # exact-multiple flat tiles: no mask at all; grid fixed.
                return _ne_scalar_inplace_fast(A, wrapped)
            ne_func_scalar_inplace(A, B, out0=A)
            return A
    # Everything else (non-representable scalar, non-finite scalar, non-float
    # dtype, non-contiguous, ...) keeps the original generic in-place path,
    # behavior unchanged.
    return _generic_ne_scalar_(A, B)


# in-place alias safety: same as _ne_tensor_inplace_fast above.
_NE_SCALAR_INPLACE_FAST_TILE = 131072
_NE_SCALAR_INPLACE_MIN_GRID = 128


@triton.jit
def ne_scalar_inplace_fast_kernel(x_ptr, scalar, TILE: tl.constexpr):
    pid = tl.program_id(0)
    tid = pid * TILE + tl.arange(0, TILE)
    x = tl.load(x_ptr + tid)
    t = (x.to(tl.float32) - scalar) * 1.0e32
    t = t * 1.0e32
    t = tl.abs(t)
    t = tl.minimum(1.0, t)
    tl.store(x_ptr + tid, t)


def _ne_scalar_inplace_fast(A, scalar):
    grid = (A.numel() // _NE_SCALAR_INPLACE_FAST_TILE,)
    ne_scalar_inplace_fast_kernel[grid](
        A,
        scalar,
        TILE=_NE_SCALAR_INPLACE_FAST_TILE,
        num_warps=4,
        buffer_size_limit=8192,
        unroll_num=16,
        isCloseMemoryAsync=True,
    )
    return A
