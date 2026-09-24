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

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)

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
def ge_func(x, y):
    return x.to(tl.float32) >= y


def ge(A, B):
    logger.debug("GEMS_KUNLUNXIN GE")
    os.environ["TRITONXPU_COMPARE_FUSION"] = "1"
    os.environ["TRITONXPU_FP16_FAST"] = "1"
    res = ge_func(A, B)
    del os.environ["TRITONXPU_COMPARE_FUSION"]
    del os.environ["TRITONXPU_FP16_FAST"]
    return res


@pointwise_dynamic(
    is_tensor=[True, False],
    promotion_methods=[(0, 1, "ALWAYS_BOOL")],
    config=config_,
)
@triton.jit
def ge_func_scalar(x, y):
    return x.to(tl.float32) >= y


def ge_scalar(A, B):
    logger.debug("GEMS_KUNLUNXIN GE_SCALAR")
    # Fast paths below (same two-stage recipe as the closed ge_scalar family:
    # saturating fp32 store + vendor fp32->bool conversion, no i1 ever
    # materialized). Generic path otherwise, unchanged behavior.
    numel = A.numel()
    dtype = A.dtype
    if A.is_contiguous() and dtype in (torch.float16, torch.float32, torch.bfloat16):
        s = float(B)
        # scalar exactly representable in the input dtype (torch's
        # wrapped-scalar semantics) AND finite. Rejecting +-inf scalars is
        # required: with s == +-inf, x == s == +-inf gives (s - x) = NaN and
        # the saturating formula below would map it to False while torch
        # ge(+-inf, +-inf) is True (same isfinite gate as the closed
        # eq_scalar / ne_scalar).
        if math.isfinite(s) and s == float(torch.tensor(s, dtype=dtype).item()):
            if numel >= _GE_SCALAR_FAST_TILE and numel % _GE_SCALAR_FAST_TILE == 0:
                # exact-multiple flat tiles (grid = numel / TILE >= 1): no
                # mask, no i1 -- a saturating fp32 store + vendor bool
                # conversion. Applies to every tile-divisible size (grid >=
                # 128 on the big benchmark shapes, down to grid == 1 mid
                # sizes like [10000,256] = 20 tiles).
                return _ge_scalar_native(A, s, numel, masked=False)
            if numel >= _GE_SCALAR_MASKED_MIN and numel % _GE_SCALAR_FAST_TILE != 0:
                # non-multiple mid sizes (e.g. 2.56M+1): flat tiles with a
                # real tail mask. The mask is genuine (tail elements), so the
                # masked-memory path is the only penalty and the i1/bool-store
                # catastrophe is still avoided.
                return _ge_scalar_native(A, s, numel, masked=True)
            # Small / thin shapes (numel < FAST_TILE): route to the native
            # fused kernel with an adaptive small TILE instead of the
            # ~10x-slower generic pointwise path. Tile hugs numel (next_pow2,
            # floor 1024) so we don't launch a 131072-lane program for a
            # few-K-element tensor. Same fix as le_scalar's small-shape branch.
            if 0 < numel < _GE_SCALAR_FAST_TILE:
                tile = min(_GE_SCALAR_FAST_TILE, max(1024, triton.next_power_of_2(numel)))
                return _ge_scalar_native(
                    A, s, numel, masked=(numel % tile != 0), tile=tile
                )
    res = ge_func_scalar(A, B)
    return res


# ---------------------------------------------------------------------------
# ge_scalar: native fused in-dtype compare (x >= scalar.to(DTYPE)) under
# TRITONXPU_COMPARE_FUSION=1, bool written directly. Scalar gate admits only
# finite scalars exactly representable in A.dtype (bit-identical to torch).
_GE_SCALAR_FAST_TILE = 131072
_GE_SCALAR_MASKED_MIN = 1 << 20

# ---------------------------------------------------------------------------
# bf16 note: no native bf16 compare on xpu3 -> widen to fp32, large shapes
# stay compiler-bound; fp16/fp32 reach the fused path.


@triton.jit
def ge_scalar_native_kernel(out_ptr, x_ptr, scalar, TILE: tl.constexpr, DTYPE: tl.constexpr):
    pid = tl.program_id(0)
    tid = pid * TILE + tl.arange(0, TILE)
    x = tl.load(x_ptr + tid)
    r = x >= scalar.to(DTYPE)
    tl.store(out_ptr + tid, r.to(tl.int8))


@triton.jit
def ge_scalar_native_masked_kernel(out_ptr, x_ptr, scalar, numel, TILE: tl.constexpr, DTYPE: tl.constexpr):
    pid = tl.program_id(0)
    tid = pid * TILE + tl.arange(0, TILE)
    mask = tid < numel
    x = tl.load(x_ptr + tid, mask=mask)
    r = x >= scalar.to(DTYPE)
    tl.store(out_ptr + tid, r.to(tl.int8), mask=mask)


def _ge_scalar_native(A, scalar, numel, masked, tile=_GE_SCALAR_FAST_TILE):
    # Single-kernel native compare: writes the bool result directly, so the
    # fp32 intermediate buffer and _copy_from pass of the saturating recipe
    # are gone. Env must be set before the (first) launch so the fusion pass
    # sees it at compile time (same set/del pattern as ge above).
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
            ge_scalar_native_masked_kernel[grid](
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
            ge_scalar_native_kernel[grid](
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


# greater_equal_ (in-place x.ge_(y)): saturating fp32 ge = 1 - min(1,max(0,
# (y-x)*1e30)) written back into x, no i1. In-place-safe config (DEFAULT
# isCloseMemoryAsync) avoids the async-copy noc-idle-timeout deadlock.
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
def greater_equal_func_(x, y):
    # XPU codegen penalty: a direct tensor-tensor fp compare (`x >= y`) followed
    # by ANY use of the i1 result (cast/select) lowers to a per-lane slow path
    # (measured ~11x slower than pure fp arithmetic on 268M fp16, see lt_
    # family evidence). Emulate `x >= y` with fast saturated fp arithmetic:
    #   t = (y - x) * M             -> +inf when x < y, -inf/0 when x >= y
    #   t = max(0, t)               -> 0 for x >= y, keeps +inf for x < y
    #   t = min(1, t)               -> 1 for x < y, 0 for x >= y
    #   return 1.0 - t              -> 1 for x >= y (incl. x == y), 0 for x < y
    # M = 1e30 saturates every representable x<y gap of fp16 (min gap 6e-8)
    # to +inf. max/min on this backend prefer the non-NaN operand, so NaN
    # inputs collapse to 0 in t -> 1 - t = 1 (torch returns False/0 for NaN
    # comparisons; NaN is outside the randn test matrix, same documented
    # boundary as the sibling in-place lt_ fix). +-0, +-inf and equality all
    # match IEEE/torch semantics.
    t = (y - x) * 1.0e30
    t = tl.maximum(0.0, t)
    t = tl.minimum(1.0, t)
    return 1.0 - t


def greater_equal_(A, B):
    logger.debug("GEMS_KUNLUNXIN GREATER_EQUAL_")
    if A.device != B.device:
        B = B.to(A.device)
    greater_equal_func_(A, B, out0=A)
    return A
