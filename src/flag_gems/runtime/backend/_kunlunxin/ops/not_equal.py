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
    buffer_size_limit=4096,
)


@pointwise_dynamic(
    promotion_methods=[(0, 1, "ALWAYS_BOOL")],
    config=config_,
)
@triton.jit
def not_equal_func(x, y):
    return x.to(tl.float32) != y.to(tl.float32)


def not_equal(A, B):
    logger.debug("GEMS_KUNLUNXIN NOT_EQUAL")
    numel = A.numel()
    if (
        A.dtype in (torch.float16, torch.float32, torch.bfloat16)
        and A.dtype == B.dtype
        and A.is_contiguous()
        and B.is_contiguous()
        and A.shape == B.shape
        and 0 < numel <= _NOT_EQUAL_TENSOR_FAST_MAX
    ):
        if numel <= _NOT_EQUAL_TENSOR_SMALL_MAX:
            return _not_equal_tensor_fast(A, B, numel, _NOT_EQUAL_TENSOR_TILE_SMALL)
        return _not_equal_tensor_fast(A, B, numel, _NOT_EQUAL_TENSOR_TILE_MID)
    os.environ["TRITONXPU_COMPARE_FUSION"] = "1"
    os.environ["TRITONXPU_FP16_FAST"] = "1"
    res = not_equal_func(A, B)
    del os.environ["TRITONXPU_COMPARE_FUSION"]
    del os.environ["TRITONXPU_FP16_FAST"]
    return res


# ---------------------------------------------------------------------------
_NOT_EQUAL_TENSOR_TILE_SMALL = 2048
_NOT_EQUAL_TENSOR_SMALL_MAX = 16384
_NOT_EQUAL_TENSOR_TILE_MID = 8192
_NOT_EQUAL_TENSOR_FAST_MAX = 65536


@triton.jit
def not_equal_tensor_fast_kernel(x_ptr, y_ptr, out_ptr, n_elements, TILE: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * TILE + tl.arange(0, TILE)
    mask = offset < n_elements
    x = tl.load(x_ptr + offset, mask=mask, other=0).to(tl.float32)
    y = tl.load(y_ptr + offset, mask=mask, other=0).to(tl.float32)
    tl.store(out_ptr + offset, x != y, mask=mask)


@triton.jit
def not_equal_tensor_fast_unmasked_kernel(x_ptr, y_ptr, out_ptr, TILE: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * TILE + tl.arange(0, TILE)
    x = tl.load(x_ptr + offset).to(tl.float32)
    y = tl.load(y_ptr + offset).to(tl.float32)
    tl.store(out_ptr + offset, x != y)


def _not_equal_tensor_fast(A, B, numel, TILE):
    os.environ["TRITONXPU_COMPARE_FUSION"] = "1"
    os.environ["TRITONXPU_FP16_FAST"] = "1"
    out = torch.empty_like(A, dtype=torch.bool)
    try:
        if numel % TILE == 0:
            not_equal_tensor_fast_unmasked_kernel[(numel // TILE,)](
                A,
                B,
                out,
                TILE=TILE,
                num_warps=4,
                buffer_size_limit=8192,
                unroll_num=16,
                isCloseMemoryAsync=False,
            )
        else:
            # non-multiple of the bucket (e.g. (1024,1) with TILE=2048): a
            # single block with a real tail mask. The mask covers genuine
            # elements only.
            not_equal_tensor_fast_kernel[(triton.cdiv(numel, TILE),)](
                A,
                B,
                out,
                numel,
                TILE=TILE,
                num_warps=4,
                buffer_size_limit=8192,
                unroll_num=16,
                isCloseMemoryAsync=False,
            )
        return out
    finally:
        del os.environ["TRITONXPU_COMPARE_FUSION"]
        del os.environ["TRITONXPU_FP16_FAST"]


@pointwise_dynamic(
    is_tensor=[True, False],
    promotion_methods=[(0, 1, "ALWAYS_BOOL")],
    config=config_,
)
@triton.jit
def not_equal_func_scalar(x, y):
    return x.to(tl.float32) != y


def not_equal_scalar(A, B):
    logger.debug("GEMS_KUNLUNXIN NOT_EQUAL_SCALAR")
    # not_equal.Scalar is an exact alias of ne.Scalar (`torch.not_equal` ==
    # `torch.ne`; same ATen semantics: a != b element-wise, NaN-aware). The
    # generic scalar-compare path (not_equal_func_scalar) materializes
    # `arith.cmpf -> i1 -> bool store` per lane, which the XPU backend lowers
    # to the same i1 slow path that doomed the closed ne_scalar (baseline
    # 2026-08-14, XPU 7: 17.6ms vs 1.09ms on [10000,65536]). Take the closed
    # ne_scalar fast path below (same two-stage saturating recipe,
    # `harness/solution/performance/not_equal_scalar_perf.md`) whenever
    # applicable; generic path otherwise, behavior unchanged.
    numel = A.numel()
    dtype = A.dtype
    if (
        A.is_contiguous()
        and dtype in (torch.float16, torch.float32, torch.bfloat16)
        and numel >= _NOT_EQUAL_SCALAR_MASKED_MIN
    ):
        # Only build the wrapped scalar (a `torch.tensor(...).item()`
        # roundtrip, ~3us host) for candidate sizes; the small-shape generic
        # path below must stay free of host overhead (measured 0.49-0.82x
        # regression on (64,64)/(10000,1)/(100,1,100) when it ran every call).
        s = float(B)
        wrapped = torch.tensor(s, dtype=dtype).item()
        if math.isfinite(wrapped):
            if (
                numel % _NOT_EQUAL_SCALAR_FAST_TILE == 0
                and numel >= _NOT_EQUAL_SCALAR_FAST_TILE * _NOT_EQUAL_SCALAR_MIN_GRID
            ):
                # exact-multiple flat tiles (grid >= MIN_GRID): no mask, no
                # i1 -- a saturating fp32 store + vendor bool conversion.
                return _not_equal_scalar_fast(
                    A, float(wrapped), (numel // _NOT_EQUAL_SCALAR_FAST_TILE,)
                )
            if numel % _NOT_EQUAL_SCALAR_FAST_TILE != 0:
                # non-multiple mid sizes (e.g. 2.56M, [10000,256]): flat
                # tiles with a real tail mask. The mask is genuine (tail
                # elements), so the masked-memory path is the only penalty.
                return _not_equal_scalar_fast_masked(A, float(wrapped), numel)
    # Like ne_scalar / gt_scalar, the scalar path must NOT set
    # TRITONXPU_COMPARE_FUSION / TRITONXPU_FP16_FAST: for tensor-vs-scalar the
    # fusion env vars make the compiler emit an fp16 compare that trips
    # `arith.cmpf same-type` and overflows uni_sram -> compile failure.
    res = not_equal_func_scalar(A, B)
    return res


_NOT_EQUAL_SCALAR_FAST_TILE = 131072
_NOT_EQUAL_SCALAR_MIN_GRID = 128
_NOT_EQUAL_SCALAR_MASKED_MIN = 1 << 20


@triton.jit
def not_equal_scalar_fast_kernel(out_ptr, x_ptr, scalar, TILE: tl.constexpr):
    pid = tl.program_id(0)
    tid = pid * TILE + tl.arange(0, TILE)
    x = tl.load(x_ptr + tid).to(tl.float32)
    d = tl.abs(x - scalar)
    t = tl.minimum(1.0, d * 1.0e30 * 1.0e15)
    tl.store(out_ptr + tid, t)


def _not_equal_scalar_fast(A, scalar, grid):
    out32 = torch.empty_like(A, dtype=torch.float32)
    not_equal_scalar_fast_kernel[grid](
        out32,
        A,
        scalar,
        TILE=_NOT_EQUAL_SCALAR_FAST_TILE,
        num_warps=4,
        buffer_size_limit=8192,
        unroll_num=16,
        isCloseMemoryAsync=False,
    )
    # ``torch.ops.aten._copy_from`` is an explicit ATen fallback (forbidden);
    # ``to(torch.bool)`` lowers to the same vendor fp32->bool conversion
    # (measured 0.048ms vs 0.049ms on 16M elements, XPU 4).
    return out32.to(torch.bool)


@triton.jit
def not_equal_scalar_fast_masked_kernel(
    out_ptr, y_ptr, scalar, numel, TILE: tl.constexpr
):
    pid = tl.program_id(0)
    tid = pid * TILE + tl.arange(0, TILE)
    mask = tid < numel
    y = tl.load(y_ptr + tid, mask=mask).to(tl.float32)
    d = tl.abs(y - scalar)
    t = tl.minimum(1.0, d * 1.0e30 * 1.0e15)
    tl.store(out_ptr + tid, t, mask=mask)


def _not_equal_scalar_fast_masked(A, scalar, numel):
    out32 = torch.empty_like(A, dtype=torch.float32)
    grid = (math.ceil(numel / _NOT_EQUAL_SCALAR_FAST_TILE),)
    not_equal_scalar_fast_masked_kernel[grid](
        out32,
        A,
        scalar,
        numel,
        TILE=_NOT_EQUAL_SCALAR_FAST_TILE,
        num_warps=4,
        buffer_size_limit=8192,
        unroll_num=16,
        isCloseMemoryAsync=False,
    )
    # ``torch.ops.aten._copy_from`` is an explicit ATen fallback (forbidden);
    # ``to(torch.bool)`` lowers to the same vendor fp32->bool conversion
    # (measured 0.048ms vs 0.049ms on 16M elements, XPU 4).
    return out32.to(torch.bool)
