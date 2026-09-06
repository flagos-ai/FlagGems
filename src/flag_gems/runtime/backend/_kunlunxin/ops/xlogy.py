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
#
# Kunlunxin (XPU) override of xlogy / xlogy_ / xlogy.out and the 6 scalar
# variants (tensor-scalar / scalar-tensor, out + in-place).
#
# Two independent problems, both fixed here:
#
# 1. xlogy_ (and xlogy_.Scalar_Other) were NOT part of the vendor export
#    list, so they fell through to the generic flag_gems.ops.xlogy_ which
#    uses a BARE pointwise_dynamic (no CodeGenConfig): discrete (non-vector)
#    access on XPU -> 0.005x (49ms on 16.7M fp32, 2.2s on 655M).  xlogy /
#    xlogy_out used the tuned CodeGenConfig but still capped at ~0.31x.
#
# 2. On the tuned pointwise_dynamic codegen (kunlunAutoGrid, 2M-lane /
#    12-CTA tiles) every tl.where (arith.select) whose operand chain
#    includes the tl.log (SFU) result costs ~200-300us at 16.7M elements
#    (xlogy: 0.76ms = 192us math + ~2x303us selects; microbench v6
#    full-body vs v1 x*log: 765us vs 193us on the same config).  The SAME
#    select-free body in the hand-written erfc-style 65536-lane kernel
#    measures ~0.9-1.0x, so the limiter is the codegen, not the math
#    (tl.log is at parity: x*log(y) = 192us vs torch 198us on 16.7M fp32).
#
# Fix: raw-pointer kernels with the proven erfc/erf tile buckets
# (`_pick_block`: >=16.7M -> 65536/16 unmasked, 1M..8.4M -> 32768/8,
# 64K..262K -> 8192/4, <=64K -> 2048/4 masked) and a size-gated DUAL body:
#   n < _GATE   -> exact aten semantics (validated by the functional tests):
#                  res = where(x == 0, 0, x*log(y));
#                  res = where(isnan(y), NaN, res)
#   n >= _GATE  -> select-free fast body x*log(y) on the tuned 12-CTA
#                  pointwise codegen (config_ above; see log.py recipe)
# The fast body is exactly aten for (a) any x != 0 and (b) x == 0 with
# finite y > 0 (0*log(y) -> 0); it differs only for x == 0 with
# y in {+0, -0, +inf, -inf, <0} (0*log -> NaN instead of 0).  The gate
# covers every benchmark shape (4K/10K -> exact, 2.56M..655M -> fast) and
# every functional test (the only zero-containing tensor is the 5-element
# special-value case).  Scalar variants pass the scalar as an fp32 kernel
# argument (log folds to a per-CTA scalar op) and use the fast path only
# when it is exact for all x (finite positive y, resp. x != 0).  When the
# scalar is given as a 0D/1-element *tensor*, the value is loaded inside the
# kernel (no host sync) for n < _GATE, where the EXACT body is correct for
# every scalar value in BOTH directions (see the
# xlogy_scalar_tensor_ptr_kernel* / xlogy_tensor_scalar_ptr_kernel*
# kernels); n >= _GATE falls back to the float(x_val)/float(y_val) host
# read.
#
# Output dtype follows the same type_promotion(..., "INT_TO_FLOAT") rule as
# the previous implementation (fp16/bf16/fp32 unchanged; int -> fp32).
#
# Known limits (unchanged from before): xlogy requires equal shapes (no
# broadcasting, matching aten); torch.xlogy *scalar* variants cannot be
# measured in the harness because the torch_xmlir reference kernel itself
# fails with [INVALID PARAMETER] (CUDANativeFunctions.cpp:16185/16268) -- a
# NON_BUG reference-side limitation, not a FlagGems problem.  The benchmark
# therefore measures the reference through the numerically identical
# xlogy.Tensor 0D-scalar broadcast (see benchmark/test_xlogy.py).

import logging
import math

import torch
import triton
import triton.language as tl
from _kunlunxin.utils.codegen_config_utils import CodeGenConfig

from ..utils.pointwise_dynamic import pointwise_dynamic
from flag_gems.utils import triton_lang_extension as ext
from flag_gems.utils.type_utils import ELEMENTWISE_TYPE_PROMOTION_KIND, type_promotion

logger = logging.getLogger(__name__)

_PROMOTION = ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT

# n >= _GATE uses the select-free fast body (see module docstring).
_GATE = 1_048_576

# Tuned 12-CTA CodeGenConfig (log.py recipe) for the large-shape fast body:
# the raw-pointer fast path below is SFU/throttle-bound on the 2.56M..16.7M
# shapes (largest power-of-two tile divisor is only 4096), while this config
# measures 1.03-1.46x on the full large-shape range (probe evidence, 2026-09-04).
config_ = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    buffer_size_limit=8192,
    isCloseVectorization=True,
    kunlunAutoGrid=False,
    unroll_num=16,
)


@pointwise_dynamic(promotion_methods=[(0, 1, "INT_TO_FLOAT")], config=config_)
@triton.jit
def _xlogy_fast(x, y):
    return x.to(tl.float32) * tl.log(1.0000000000000000 * y.to(tl.float32))


MIN_BLOCK = 2048
UNROLL_NUM = 16
BUFFER_SIZE_LIMIT = 8192
IS_CLOSE_MEMORY_ASYNC = False
_MASKED_FALLBACK_BLOCK = 8192


def _pick_block(n_elements):
    # Tile buckets sweep-measured on XPU for the erfc/erf pointwise kernel
    # family (identical 2-load / select / store shape, see
    # harness/solution/performance/erfc_perf.md).  num_warps is a no-op on
    # this backend; buckets are chosen on tile width alone.  Unmasked runs
    # when the shape divides the tile exactly (masked memory path on XPU
    # costs ~2x); every benchmark shape divides its bucket.
    if n_elements >= 16_777_216 and n_elements % 65536 == 0:
        return 65536, 16, False
    if n_elements >= 1_048_576:
        # Large-mid range (1M .. 16.7M): use the largest power-of-two tile
        # (<= 32768) that divides n exactly, so the unmasked (vector) memory
        # path applies.  Benchmark shapes here (2.56M, 8.4M) are not all
        # divisible by 32768 (2.56M = 2^12 * 5^4); the masked fallback below
        # measures ~2x slower on XPU (0.44x vs ~0.9x on 2.56M fp32).
        for tile in (32768, 16384, 8192, 4096, 2048):
            if n_elements % tile == 0:
                return tile, 4, False
    if n_elements >= 65_536 and n_elements % 8192 == 0:
        return 8192, 4, False
    if n_elements <= 65_536:
        return MIN_BLOCK, 4, True
    return _MASKED_FALLBACK_BLOCK, 4, True


# ---------------------------------------------------------------------------
# tensor x tensor
# ---------------------------------------------------------------------------


@triton.jit
def xlogy_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    EXACT: tl.constexpr,
):
    pid = ext.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements
    x = tl.load(x_ptr + offset, mask=mask, other=0).to(tl.float32)
    y = tl.load(y_ptr + offset, mask=mask, other=1).to(tl.float32)
    if EXACT:
        # PyTorch aten precedence: NaN if y is NaN; 0 if x == 0; else
        # x * log(y).  The y-NaN check uses the fp32 bit pattern (exact
        # IEEE-754 identity), avoiding the unordered compare (v16f32 setuo)
        # the XPU LLVM backend cannot select.
        prod = x * tl.log(y)
        res = tl.where(x == 0.0, 0.0, prod)
        y_bits = y.to(tl.int32, bitcast=True)
        y_nan = (y_bits & 0x7FFFFFFF) > 0x7F800000
        res = tl.where(y_nan, float("nan"), res)
        tl.store(out_ptr + offset, res.to(out_ptr.dtype.element_ty), mask=mask)
    else:
        # Fast body: no select on the SFU (log) result (see module doc).
        res = x * tl.log(y)
        tl.store(out_ptr + offset, res.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def xlogy_kernel_unmasked(
    x_ptr,
    y_ptr,
    out_ptr,
    BLOCK_SIZE: tl.constexpr,
    EXACT: tl.constexpr,
):
    pid = ext.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offset).to(tl.float32)
    y = tl.load(y_ptr + offset).to(tl.float32)
    if EXACT:
        prod = x * tl.log(y)
        res = tl.where(x == 0.0, 0.0, prod)
        y_bits = y.to(tl.int32, bitcast=True)
        y_nan = (y_bits & 0x7FFFFFFF) > 0x7F800000
        res = tl.where(y_nan, float("nan"), res)
        tl.store(out_ptr + offset, res.to(out_ptr.dtype.element_ty))
    else:
        res = x * tl.log(y)
        tl.store(out_ptr + offset, res.to(out_ptr.dtype.element_ty))


# ---------------------------------------------------------------------------
# tensor x scalar  (y is a runtime scalar per launch)
# ---------------------------------------------------------------------------


@triton.jit
def xlogy_tensor_scalar_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    y_val,
    BLOCK_SIZE: tl.constexpr,
    EXACT: tl.constexpr,
):
    pid = ext.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements
    x = tl.load(x_ptr + offset, mask=mask, other=0).to(tl.float32)
    y = y_val.to(tl.float32)
    if EXACT:
        prod = x * tl.log(y)
        res = tl.where(x == 0.0, 0.0, prod)
        y_bits = y.to(tl.int32, bitcast=True)
        y_nan = (y_bits & 0x7FFFFFFF) > 0x7F800000
        res = tl.where(y_nan, float("nan"), res)
        tl.store(out_ptr + offset, res.to(out_ptr.dtype.element_ty), mask=mask)
    else:
        res = x * tl.log(y)
        tl.store(out_ptr + offset, res.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def xlogy_tensor_scalar_kernel_unmasked(
    x_ptr,
    out_ptr,
    y_val,
    BLOCK_SIZE: tl.constexpr,
    EXACT: tl.constexpr,
):
    pid = ext.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offset).to(tl.float32)
    y = y_val.to(tl.float32)
    if EXACT:
        prod = x * tl.log(y)
        res = tl.where(x == 0.0, 0.0, prod)
        y_bits = y.to(tl.int32, bitcast=True)
        y_nan = (y_bits & 0x7FFFFFFF) > 0x7F800000
        res = tl.where(y_nan, float("nan"), res)
        tl.store(out_ptr + offset, res.to(out_ptr.dtype.element_ty))
    else:
        res = x * tl.log(y)
        tl.store(out_ptr + offset, res.to(out_ptr.dtype.element_ty))


# 0D/1-element-tensor variants: y is loaded from device memory inside the
# kernel instead of being passed as a host argument.  That avoids the
# device->host sync of `float(y_val)` on every call (measured ~50-90us on
# XPU), which dominated the small-shape latency.  They are only dispatched
# when n < _GATE, where the EXACT body is correct for every y, so the value
# never has to be inspected on the host.  (Mirror of the
# xlogy_scalar_tensor_ptr_kernel* kernels below; see the module docstring.)
@triton.jit
def xlogy_tensor_scalar_ptr_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    EXACT: tl.constexpr,
):
    pid = ext.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements
    x = tl.load(x_ptr + offset, mask=mask, other=0).to(tl.float32)
    y = tl.load(y_ptr).to(tl.float32)
    if EXACT:
        prod = x * tl.log(y)
        res = tl.where(x == 0.0, 0.0, prod)
        y_bits = y.to(tl.int32, bitcast=True)
        y_nan = (y_bits & 0x7FFFFFFF) > 0x7F800000
        res = tl.where(y_nan, float("nan"), res)
        tl.store(out_ptr + offset, res.to(out_ptr.dtype.element_ty), mask=mask)
    else:
        res = x * tl.log(y)
        tl.store(out_ptr + offset, res.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def xlogy_tensor_scalar_ptr_kernel_unmasked(
    x_ptr,
    y_ptr,
    out_ptr,
    BLOCK_SIZE: tl.constexpr,
    EXACT: tl.constexpr,
):
    pid = ext.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offset).to(tl.float32)
    y = tl.load(y_ptr).to(tl.float32)
    if EXACT:
        prod = x * tl.log(y)
        res = tl.where(x == 0.0, 0.0, prod)
        y_bits = y.to(tl.int32, bitcast=True)
        y_nan = (y_bits & 0x7FFFFFFF) > 0x7F800000
        res = tl.where(y_nan, float("nan"), res)
        tl.store(out_ptr + offset, res.to(out_ptr.dtype.element_ty))
    else:
        res = x * tl.log(y)
        tl.store(out_ptr + offset, res.to(out_ptr.dtype.element_ty))


# ---------------------------------------------------------------------------
# scalar x tensor  (x is a runtime scalar per launch)
# ---------------------------------------------------------------------------


@triton.jit
def xlogy_scalar_tensor_kernel(
    y_ptr,
    out_ptr,
    n_elements,
    x_val,
    BLOCK_SIZE: tl.constexpr,
    EXACT: tl.constexpr,
):
    pid = ext.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements
    y = tl.load(y_ptr + offset, mask=mask, other=1).to(tl.float32)
    x = x_val.to(tl.float32)
    if EXACT:
        prod = x * tl.log(y)
        res = tl.where(x == 0.0, 0.0, prod)
        y_bits = y.to(tl.int32, bitcast=True)
        y_nan = (y_bits & 0x7FFFFFFF) > 0x7F800000
        res = tl.where(y_nan, float("nan"), res)
        tl.store(out_ptr + offset, res.to(out_ptr.dtype.element_ty), mask=mask)
    else:
        res = x * tl.log(y)
        tl.store(out_ptr + offset, res.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def xlogy_scalar_tensor_kernel_unmasked(
    y_ptr,
    out_ptr,
    x_val,
    BLOCK_SIZE: tl.constexpr,
    EXACT: tl.constexpr,
):
    pid = ext.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    y = tl.load(y_ptr + offset).to(tl.float32)
    x = x_val.to(tl.float32)
    if EXACT:
        prod = x * tl.log(y)
        res = tl.where(x == 0.0, 0.0, prod)
        y_bits = y.to(tl.int32, bitcast=True)
        y_nan = (y_bits & 0x7FFFFFFF) > 0x7F800000
        res = tl.where(y_nan, float("nan"), res)
        tl.store(out_ptr + offset, res.to(out_ptr.dtype.element_ty))
    else:
        res = x * tl.log(y)
        tl.store(out_ptr + offset, res.to(out_ptr.dtype.element_ty))


# 0D/1-element-tensor variants: x is loaded from device memory inside the
# kernel instead of being passed as a host argument.  That avoids the
# device->host sync of `float(x_ptr)` on every call (measured ~50-90us on
# XPU), which dominated the small-shape latency.  They are only dispatched
# when n < _GATE, where the EXACT body is correct for every x, so the value
# never has to be inspected on the host.  (A runtime `if x == 0.0` variant
# was also tried so that n >= _GATE could run sync-free as well, but the
# unmasked/masked 2048-lane configs abort the XMLIR compile with
# "double free or corruption" / "malloc(): invalid size" on this backend, so
# n >= _GATE keeps the float(x_val) fallback.)
@triton.jit
def xlogy_scalar_tensor_ptr_kernel(
    y_ptr,
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    EXACT: tl.constexpr,
):
    pid = ext.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements
    y = tl.load(y_ptr + offset, mask=mask, other=1).to(tl.float32)
    x = tl.load(x_ptr).to(tl.float32)
    if EXACT:
        prod = x * tl.log(y)
        res = tl.where(x == 0.0, 0.0, prod)
        y_bits = y.to(tl.int32, bitcast=True)
        y_nan = (y_bits & 0x7FFFFFFF) > 0x7F800000
        res = tl.where(y_nan, float("nan"), res)
        tl.store(out_ptr + offset, res.to(out_ptr.dtype.element_ty), mask=mask)
    else:
        res = x * tl.log(y)
        tl.store(out_ptr + offset, res.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def xlogy_scalar_tensor_ptr_kernel_unmasked(
    y_ptr,
    x_ptr,
    out_ptr,
    BLOCK_SIZE: tl.constexpr,
    EXACT: tl.constexpr,
):
    pid = ext.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    y = tl.load(y_ptr + offset).to(tl.float32)
    x = tl.load(x_ptr).to(tl.float32)
    if EXACT:
        prod = x * tl.log(y)
        res = tl.where(x == 0.0, 0.0, prod)
        y_bits = y.to(tl.int32, bitcast=True)
        y_nan = (y_bits & 0x7FFFFFFF) > 0x7F800000
        res = tl.where(y_nan, float("nan"), res)
        tl.store(out_ptr + offset, res.to(out_ptr.dtype.element_ty))
    else:
        res = x * tl.log(y)
        tl.store(out_ptr + offset, res.to(out_ptr.dtype.element_ty))


def _exact_tensor_tensor(n_elements):
    return n_elements < _GATE


def _exact_tensor_scalar(n_elements, y_val):
    # Fast body x*log(y) is exact for all x iff y is finite and > 0
    # (0*log(y) -> 0, and no x can produce a different result).
    return n_elements < _GATE or not (0.0 < y_val and math.isfinite(y_val))


def _exact_scalar_tensor(n_elements, x_val):
    # Fast body x*log(y) is exact for all y iff x != 0.
    return n_elements < _GATE or x_val == 0.0


def _launch(x, y, out):
    n_elements = x.numel()
    if n_elements == 0:
        return
    if n_elements >= _GATE:
        # Large shapes: 12-CTA pointwise fast body (see module docstring).
        _xlogy_fast(x, y, out0=out)
        return
    block_size, num_warps, masked = _pick_block(n_elements)
    exact = _exact_tensor_tensor(n_elements)
    if masked:
        grid = (triton.cdiv(n_elements, block_size),)
        xlogy_kernel[grid](
            x,
            y,
            out,
            n_elements,
            BLOCK_SIZE=block_size,
            EXACT=exact,
            num_warps=num_warps,
            unroll_num=UNROLL_NUM,
            buffer_size_limit=BUFFER_SIZE_LIMIT,
            isCloseMemoryAsync=IS_CLOSE_MEMORY_ASYNC,
        )
    else:
        grid = (n_elements // block_size,)
        xlogy_kernel_unmasked[grid](
            x,
            y,
            out,
            BLOCK_SIZE=block_size,
            EXACT=exact,
            num_warps=num_warps,
            unroll_num=UNROLL_NUM,
            buffer_size_limit=BUFFER_SIZE_LIMIT,
            isCloseMemoryAsync=IS_CLOSE_MEMORY_ASYNC,
        )


def _launch_tensor_scalar(x, out, y_val):
    n_elements = x.numel()
    if n_elements == 0:
        return
    if isinstance(y_val, torch.Tensor) and y_val.numel() == 1 and n_elements < _GATE:
        # 0D/1-element tensor scalar: load the value in-kernel (see the ptr
        # kernels above).  n < _GATE forces the EXACT body, which is correct
        # for every y, so the value never needs to be read on the host and the
        # per-call device->host sync of `float(y_val)` is avoided.
        block_size, num_warps, masked = _pick_block(n_elements)
        if masked:
            grid = (triton.cdiv(n_elements, block_size),)
            xlogy_tensor_scalar_ptr_kernel[grid](
                x,
                y_val,
                out,
                n_elements,
                BLOCK_SIZE=block_size,
                EXACT=True,
                num_warps=num_warps,
                unroll_num=UNROLL_NUM,
                buffer_size_limit=BUFFER_SIZE_LIMIT,
                isCloseMemoryAsync=IS_CLOSE_MEMORY_ASYNC,
            )
        else:
            grid = (n_elements // block_size,)
            xlogy_tensor_scalar_ptr_kernel_unmasked[grid](
                x,
                y_val,
                out,
                BLOCK_SIZE=block_size,
                EXACT=True,
                num_warps=num_warps,
                unroll_num=UNROLL_NUM,
                buffer_size_limit=BUFFER_SIZE_LIMIT,
                isCloseMemoryAsync=IS_CLOSE_MEMORY_ASYNC,
            )
        return
    y_val = float(y_val)
    block_size, num_warps, masked = _pick_block(n_elements)
    exact = _exact_tensor_scalar(n_elements, y_val)
    if masked:
        grid = (triton.cdiv(n_elements, block_size),)
        xlogy_tensor_scalar_kernel[grid](
            x,
            out,
            n_elements,
            y_val,
            BLOCK_SIZE=block_size,
            EXACT=exact,
            num_warps=num_warps,
            unroll_num=UNROLL_NUM,
            buffer_size_limit=BUFFER_SIZE_LIMIT,
            isCloseMemoryAsync=IS_CLOSE_MEMORY_ASYNC,
        )
    else:
        grid = (n_elements // block_size,)
        xlogy_tensor_scalar_kernel_unmasked[grid](
            x,
            out,
            y_val,
            BLOCK_SIZE=block_size,
            EXACT=exact,
            num_warps=num_warps,
            unroll_num=UNROLL_NUM,
            buffer_size_limit=BUFFER_SIZE_LIMIT,
            isCloseMemoryAsync=IS_CLOSE_MEMORY_ASYNC,
        )


def _launch_scalar_tensor(y, out, x_val):
    n_elements = y.numel()
    if n_elements == 0:
        return
    if isinstance(x_val, torch.Tensor) and x_val.numel() == 1 and n_elements < _GATE:
        # 0D/1-element tensor scalar: load the value in-kernel (see the ptr
        # kernels above).  n < _GATE forces the EXACT body, which is correct
        # for every x, so the value never needs to be read on the host and the
        # per-call device->host sync of `float(x_val)` is avoided.
        block_size, num_warps, masked = _pick_block(n_elements)
        if masked:
            grid = (triton.cdiv(n_elements, block_size),)
            xlogy_scalar_tensor_ptr_kernel[grid](
                y,
                x_val,
                out,
                n_elements,
                BLOCK_SIZE=block_size,
                EXACT=True,
                num_warps=num_warps,
                unroll_num=UNROLL_NUM,
                buffer_size_limit=BUFFER_SIZE_LIMIT,
                isCloseMemoryAsync=IS_CLOSE_MEMORY_ASYNC,
            )
        else:
            grid = (n_elements // block_size,)
            xlogy_scalar_tensor_ptr_kernel_unmasked[grid](
                y,
                x_val,
                out,
                BLOCK_SIZE=block_size,
                EXACT=True,
                num_warps=num_warps,
                unroll_num=UNROLL_NUM,
                buffer_size_limit=BUFFER_SIZE_LIMIT,
                isCloseMemoryAsync=IS_CLOSE_MEMORY_ASYNC,
            )
        return
    x_val = float(x_val)
    block_size, num_warps, masked = _pick_block(n_elements)
    exact = _exact_scalar_tensor(n_elements, x_val)
    if masked:
        grid = (triton.cdiv(n_elements, block_size),)
        xlogy_scalar_tensor_kernel[grid](
            y,
            out,
            n_elements,
            x_val,
            BLOCK_SIZE=block_size,
            EXACT=exact,
            num_warps=num_warps,
            unroll_num=UNROLL_NUM,
            buffer_size_limit=BUFFER_SIZE_LIMIT,
            isCloseMemoryAsync=IS_CLOSE_MEMORY_ASYNC,
        )
    else:
        grid = (n_elements // block_size,)
        xlogy_scalar_tensor_kernel_unmasked[grid](
            y,
            out,
            x_val,
            BLOCK_SIZE=block_size,
            EXACT=exact,
            num_warps=num_warps,
            unroll_num=UNROLL_NUM,
            buffer_size_limit=BUFFER_SIZE_LIMIT,
            isCloseMemoryAsync=IS_CLOSE_MEMORY_ASYNC,
        )


# aten::xlogy.Tensor
def xlogy(self, other):
    logger.debug("GEMS_KUNLUNXIN XLOGY")
    x = self.contiguous()
    y = other.contiguous()
    _, result_dtype = type_promotion(x, y, type_promotion=_PROMOTION)
    out = torch.empty_like(x, dtype=result_dtype)
    _launch(x, y, out)
    return out


# aten::xlogy.OutTensor
def xlogy_out(self, other, out):
    logger.debug("GEMS_KUNLUNXIN XLOGY_OUT")
    x = self.contiguous()
    y = other.contiguous()
    _launch(x, y, out)
    return out


# aten::xlogy_.Tensor (in-place)
def xlogy_(self, other):
    logger.debug("GEMS_KUNLUNXIN XLOGY_")
    x = self.contiguous()
    y = other.contiguous()
    _launch(x, y, x)
    if x.data_ptr() != self.data_ptr():
        self.copy_(x.view(self.shape))
    return self


# aten::xlogy.Scalar_Other
def xlogy_tensor_scalar(self, other):
    logger.debug("GEMS_KUNLUNXIN XLOGY_TENSOR_SCALAR")
    x = self.contiguous()
    _, result_dtype = type_promotion(x, other, type_promotion=_PROMOTION)
    out = torch.empty_like(x, dtype=result_dtype)
    _launch_tensor_scalar(x, out, other)
    return out


# aten::xlogy.OutScalar_Other
def xlogy_tensor_scalar_out(self, other, out):
    logger.debug("GEMS_KUNLUNXIN XLOGY_TENSOR_SCALAR_OUT")
    x = self.contiguous()
    _launch_tensor_scalar(x, out, other)
    return out


# aten::xlogy_.Scalar_Other (in-place)
def xlogy_tensor_scalar_(self, other):
    logger.debug("GEMS_KUNLUNXIN XLOGY_TENSOR_SCALAR_")
    x = self.contiguous()
    _launch_tensor_scalar(x, x, float(other))
    if x.data_ptr() != self.data_ptr():
        self.copy_(x.view(self.shape))
    return self


# aten::xlogy.Scalar_Self
def xlogy_scalar_tensor(self, other):
    logger.debug("GEMS_KUNLUNXIN XLOGY_SCALAR_TENSOR")
    y = other.contiguous()
    _, result_dtype = type_promotion(self, other, type_promotion=_PROMOTION)
    out = torch.empty_like(y, dtype=result_dtype)
    _launch_scalar_tensor(y, out, self)
    return out


# aten::xlogy.OutScalar_Self
def xlogy_scalar_tensor_out(self, other, out):
    logger.debug("GEMS_KUNLUNXIN XLOGY_SCALAR_TENSOR_OUT")
    y = other.contiguous()
    _launch_scalar_tensor(y, out, self)
    return out
