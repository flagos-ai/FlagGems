# Kunlunxin (XPU) override of less_equal / less_equal_scalar (== le / le_scalar,
# i.e. x <= y). Tuned CodeGenConfig + native fused compare fast paths; the
# shared rationale (fusion, M=1e38, NaN boundary) lives in le.py.
import logging
import math
import os

import torch
import triton
import triton.language as tl
from _kunlunxin.utils.codegen_config_utils import CodeGenConfig

from flag_gems.ops.less_equal_ import less_equal_ as _generic_less_equal_
from flag_gems.runtime import device

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)
device = device.name


config_ = CodeGenConfig(
    1024,
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
def less_equal_func(x, y):
    return x.to(tl.float32) <= y


def less_equal(A, B):
    logger.debug("GEMS_KUNLUNXIN LESS_EQUAL")
    # Fast path for small/mid contiguous same-shape float tensors
    # (numel <= 2^18): saturating fp32 {0,1} then cast to bool. Generic
    # fused compare path otherwise.
    numel = A.numel()
    if (
        A.dtype in (torch.float16, torch.float32, torch.bfloat16)
        and A.dtype == B.dtype
        and A.is_contiguous()
        and B.is_contiguous()
        and A.shape == B.shape
        and 0 < numel <= _LESS_EQUAL_TENSOR_FAST_MAX
    ):
        if numel % _LESS_EQUAL_TENSOR_FAST_TILE == 0:
            # exact-multiple flat tiles (grid = numel / TILE >= 1): no mask.
            return _less_equal_tensor_fast(
                A, B, (numel // _LESS_EQUAL_TENSOR_FAST_TILE,)
            )
        # non-multiple mids / sub-tile sizes: flat tiles with a real tail
        # mask (grid = ceil(numel / TILE) >= 1 for numel >= 1).
        return _less_equal_tensor_fast_masked(A, B, numel)
    os.environ["TRITONXPU_COMPARE_FUSION"] = "1"
    os.environ["TRITONXPU_FP16_FAST"] = "1"
    res = less_equal_func(A, B)
    del os.environ["TRITONXPU_COMPARE_FUSION"]
    del os.environ["TRITONXPU_FP16_FAST"]
    return res


# ---------------------------------------------------------------------------
# less_equal tensor-tensor fast path (fp16/fp32/bf16, contiguous, same shape,
# numel <= 2^18): native fused compare (x <= y) writing bool directly, no fp32
# buffer and no _copy_from. NaN <= y -> False (matches torch).
_LESS_EQUAL_TENSOR_FAST_TILE = 131072
_LESS_EQUAL_TENSOR_FAST_MAX = 1 << 18


@triton.jit
def less_equal_tensor_native_kernel(out_ptr, x_ptr, y_ptr, TILE: tl.constexpr):
    pid = tl.program_id(0)
    tid = pid * TILE + tl.arange(0, TILE)
    r = tl.load(x_ptr + tid) <= tl.load(y_ptr + tid)
    tl.store(out_ptr + tid, r.to(tl.int8))


@triton.jit
def less_equal_tensor_native_masked_kernel(
    out_ptr, x_ptr, y_ptr, numel, TILE: tl.constexpr
):
    pid = tl.program_id(0)
    tid = pid * TILE + tl.arange(0, TILE)
    mask = tid < numel
    r = tl.load(x_ptr + tid, mask=mask) <= tl.load(y_ptr + tid, mask=mask)
    tl.store(out_ptr + tid, r.to(tl.int8), mask=mask)


def _less_equal_tensor_native(A, B, numel, masked):
    # Native fused compare (x <= y, same dtype) writing bool directly -- no fp32
    # buffer, no _copy_from. Env set at compile time so the fusion pass sees it
    # (same pattern as less_equal_scalar's native path).
    out = torch.empty_like(A, dtype=torch.bool)
    x = A.reshape(-1)
    y = B.reshape(-1)
    tile = _LESS_EQUAL_TENSOR_FAST_TILE
    grid = (math.ceil(numel / tile),) if masked else (numel // tile,)
    os.environ["TRITONXPU_COMPARE_FUSION"] = "1"
    os.environ["TRITONXPU_FP16_FAST"] = "1"
    try:
        if masked:
            less_equal_tensor_native_masked_kernel[grid](
                out,
                x,
                y,
                numel,
                TILE=tile,
                num_warps=4,
                buffer_size_limit=8192,
                unroll_num=16,
                isCloseMemoryAsync=False,
            )
        else:
            less_equal_tensor_native_kernel[grid](
                out,
                x,
                y,
                TILE=tile,
                num_warps=4,
                buffer_size_limit=8192,
                unroll_num=16,
                isCloseMemoryAsync=False,
            )
    finally:
        del os.environ["TRITONXPU_COMPARE_FUSION"]
        del os.environ["TRITONXPU_FP16_FAST"]
    return out


def _less_equal_tensor_fast(A, B, grid):
    return _less_equal_tensor_native(A, B, A.numel(), masked=False)


def _less_equal_tensor_fast_masked(A, B, numel):
    return _less_equal_tensor_native(A, B, numel, masked=True)


@pointwise_dynamic(
    is_tensor=[True, False],
    promotion_methods=[(0, 1, "ALWAYS_BOOL")],
    config=config_,
)
@triton.jit
def less_equal_func_scalar(x, y):
    return x.to(tl.float32) <= y


def less_equal_scalar(A, B):
    logger.debug("GEMS_KUNLUNXIN LESS_EQUAL_SCALAR")
    # Fast paths below (same two-stage recipe as the closed le_scalar family:
    # saturating fp32 store + vendor fp32->bool conversion, no i1 ever
    # materialized). Generic path otherwise, unchanged behavior.
    numel = A.numel()
    dtype = A.dtype
    if (
        A.is_contiguous()
        and dtype in (torch.float16, torch.float32, torch.bfloat16)
        and float(B) == float(torch.tensor(float(B), dtype=dtype).item())
    ):
        if (
            numel >= _LESS_EQUAL_SCALAR_FAST_TILE
            and numel % _LESS_EQUAL_SCALAR_FAST_TILE == 0
        ):
            # exact-multiple flat tiles (grid = numel / TILE >= 1): no mask, no
            # i1 -- a saturating fp32 store + vendor bool conversion. Applies
            # to every tile-divisible size (grid >= 128 on the big benchmark
            # shapes, down to grid == 1 mid sizes like [10000,256] = 20 tiles).
            return _less_equal_scalar_native(A, float(B), numel, masked=False)
        if (
            numel >= _LESS_EQUAL_SCALAR_MASKED_MIN
            and numel % _LESS_EQUAL_SCALAR_FAST_TILE != 0
        ):
            # non-multiple mid sizes (e.g. 2.56M+1): flat tiles with a real
            # tail mask. The mask is genuine (tail elements), so the
            # masked-memory path is the only penalty and the i1/bool-store
            # catastrophe is still avoided.
            return _less_equal_scalar_native(A, float(B), numel, masked=True)
        # Small / thin shapes (numel < FAST_TILE): route to the native fused
        # kernel with an adaptive small TILE instead of the ~10x-slower
        # generic pointwise path. Tile hugs numel (next_pow2, floor 1024) so
        # we don't launch a 131072-lane program for a few-K-element tensor.
        # Same fix as le_scalar's small-shape branch.
        if 0 < numel < _LESS_EQUAL_SCALAR_FAST_TILE:
            tile = min(
                _LESS_EQUAL_SCALAR_FAST_TILE,
                max(1024, triton.next_power_of_2(numel)),
            )
            return _less_equal_scalar_native(
                A, float(B), numel, masked=(numel % tile != 0), tile=tile
            )
    res = less_equal_func_scalar(A, B)
    return res


_LESS_EQUAL_SCALAR_FAST_TILE = 131072
_LESS_EQUAL_SCALAR_MASKED_MIN = 1 << 20

# ---------------------------------------------------------------------------
# Native dtype-compare fast path (option A, mirrors le_scalar). less_equal is
# byte-identical to le (x <= s), so le_scalar's probe-verified native compare
# correctness carries over: the scalar is downcast to the input dtype
# in-kernel (`scalar.to(DTYPE)`) so both operands share the input dtype; with
# TRITONXPU_COMPARE_FUSION=1 the backend fuses CmpFOp(i1)+ExtUI(i8)+Store into
# one vendor compare-store intrinsic (3 B/elem, one pass) instead of the
# two-stage saturating recipe's fp32 buffer + _copy_from (~11 B/elem).


@triton.jit
def less_equal_scalar_native_kernel(
    out_ptr, x_ptr, scalar, TILE: tl.constexpr, DTYPE: tl.constexpr
):
    pid = tl.program_id(0)
    tid = pid * TILE + tl.arange(0, TILE)
    x = tl.load(x_ptr + tid)
    r = x <= scalar.to(DTYPE)
    tl.store(out_ptr + tid, r.to(tl.int8))


@triton.jit
def less_equal_scalar_native_masked_kernel(
    out_ptr, x_ptr, scalar, numel, TILE: tl.constexpr, DTYPE: tl.constexpr
):
    pid = tl.program_id(0)
    tid = pid * TILE + tl.arange(0, TILE)
    mask = tid < numel
    x = tl.load(x_ptr + tid, mask=mask)
    r = x <= scalar.to(DTYPE)
    tl.store(out_ptr + tid, r.to(tl.int8), mask=mask)


def _less_equal_scalar_native(
    A, scalar, numel, masked, tile=_LESS_EQUAL_SCALAR_FAST_TILE
):
    # Single-kernel native compare: writes the bool result directly, so the
    # fp32 intermediate buffer and _copy_from pass of the saturating recipe
    # are gone. Env must be set before the (first) launch so the fusion pass
    # sees it at compile time (same set/del pattern as less_equal above).
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
            less_equal_scalar_native_masked_kernel[grid](
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
            less_equal_scalar_native_kernel[grid](
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
# less_equal_ / less_equal_scalar_ (in-place, x.less_equal_(y/s)): saturating
# fp32 recipe le = 1 - min(1, max(0, (x-y)*1e64)) written back into x, no i1.
# In-place-safe config (DEFAULT isCloseMemoryAsync) avoids the noc-idle-timeout
# deadlock that async-copy has under aliasing. NaN/subnormal follow le family.
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
def less_equal_func_tensor_inplace(x, y):
    t = (x.to(tl.float32) - y.to(tl.float32)) * 1.0e32
    t = t * 1.0e32
    t = tl.maximum(0.0, t)
    t = tl.minimum(1.0, t)
    return 1.0 - t


def less_equal_(A, B):
    logger.debug("GEMS_KUNLUNXIN LESS_EQUAL_ TENSOR")
    if A.device != B.device:
        if A.device.type == device:
            B = B.to(A.device)
        else:
            A = A.to(B.device)
    numel = A.numel()
    if A.is_contiguous() and A.dtype in (torch.float16, torch.float32, torch.bfloat16):
        if (
            A.dtype in (torch.float16, torch.float32)
            and B.is_contiguous()
            and B.dtype == A.dtype
            and A.shape == B.shape
            and numel
            >= _LESS_EQUAL_TENSOR_INPLACE_FAST_TILE
            * _LESS_EQUAL_TENSOR_INPLACE_MIN_GRID
            and numel % _LESS_EQUAL_TENSOR_INPLACE_FAST_TILE == 0
        ):
            # exact-multiple flat tiles: no mask at all; grid fixed.
            return _less_equal_tensor_inplace_fast(A, B, numel)
        less_equal_func_tensor_inplace(A, B, out0=A)
        return A
    # Everything else (non-float dtype, non-contiguous, ...) keeps the
    # original generic in-place path, behavior unchanged.
    return _generic_less_equal_(A, B)


# in-place alias safety: the fast kernel writes into the SAME tensor it
# reads, so it must keep the DEFAULT isCloseMemoryAsync (True = async copy
# closed); passing False with in-place aliasing is the documented "noc idle
# timeout" deadlock, same as le.py's/gt.py's in-place fast path note.
_LESS_EQUAL_TENSOR_INPLACE_FAST_TILE = 131072
_LESS_EQUAL_TENSOR_INPLACE_MIN_GRID = 128


@triton.jit
def less_equal_tensor_inplace_fast_kernel(x_ptr, y_ptr, TILE: tl.constexpr):
    pid = tl.program_id(0)
    tid = pid * TILE + tl.arange(0, TILE)
    x = tl.load(x_ptr + tid)
    y = tl.load(y_ptr + tid)
    t = (x.to(tl.float32) - y.to(tl.float32)) * 1.0e32
    t = t * 1.0e32
    t = tl.maximum(0.0, t)
    t = tl.minimum(1.0, t)
    tl.store(x_ptr + tid, 1.0 - t)


def _less_equal_tensor_inplace_fast(A, B, numel):
    grid = (numel // _LESS_EQUAL_TENSOR_INPLACE_FAST_TILE,)
    less_equal_tensor_inplace_fast_kernel[grid](
        A,
        B,
        TILE=_LESS_EQUAL_TENSOR_INPLACE_FAST_TILE,
        num_warps=4,
        buffer_size_limit=8192,
        unroll_num=16,
        isCloseMemoryAsync=True,
    )
    return A


@pointwise_dynamic(
    is_tensor=[True, False],
    promotion_methods=[(0, 1, "DEFAULT")],
    config=config_inplace_,
)
@triton.jit
def less_equal_func_scalar_inplace(x, y):
    t = (x.to(tl.float32) - y) * 1.0e32
    t = t * 1.0e32
    t = tl.maximum(0.0, t)
    t = tl.minimum(1.0, t)
    return 1.0 - t


def less_equal_scalar_(A, B):
    logger.debug("GEMS_KUNLUNXIN LESS_EQUAL_ SCALAR")
    numel = A.numel()
    if (
        A.is_contiguous()
        and A.dtype in (torch.float16, torch.float32, torch.bfloat16)
        and float(B) == float(torch.tensor(float(B), dtype=A.dtype).item())
    ):
        if (
            A.dtype in (torch.float16, torch.float32)
            and numel
            >= _LESS_EQUAL_SCALAR_INPLACE_FAST_TILE
            * _LESS_EQUAL_SCALAR_INPLACE_MIN_GRID
            and numel % _LESS_EQUAL_SCALAR_INPLACE_FAST_TILE == 0
        ):
            # exact-multiple flat tiles: no mask at all; grid fixed.
            return _less_equal_scalar_inplace_fast(A, float(B))
        less_equal_func_scalar_inplace(A, B, out0=A)
        return A
    return less_equal_func_scalar(A, B, out0=A)


# in-place alias safety: same as _LESS_EQUAL_TENSOR_INPLACE_FAST_TILE note
# (write into the SAME tensor it reads -> DEFAULT isCloseMemoryAsync).
_LESS_EQUAL_SCALAR_INPLACE_FAST_TILE = 131072
_LESS_EQUAL_SCALAR_INPLACE_MIN_GRID = 128


@triton.jit
def less_equal_scalar_inplace_fast_kernel(x_ptr, scalar, TILE: tl.constexpr):
    pid = tl.program_id(0)
    tid = pid * TILE + tl.arange(0, TILE)
    x = tl.load(x_ptr + tid)
    t = (x.to(tl.float32) - scalar) * 1.0e32
    t = t * 1.0e32
    t = tl.maximum(0.0, t)
    t = tl.minimum(1.0, t)
    tl.store(x_ptr + tid, 1.0 - t)


def _less_equal_scalar_inplace_fast(A, scalar):
    grid = (A.numel() // _LESS_EQUAL_SCALAR_INPLACE_FAST_TILE,)
    less_equal_scalar_inplace_fast_kernel[grid](
        A,
        scalar,
        TILE=_LESS_EQUAL_SCALAR_INPLACE_FAST_TILE,
        num_warps=4,
        buffer_size_limit=8192,
        unroll_num=16,
        isCloseMemoryAsync=True,
    )
    return A
