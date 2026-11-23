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
    numel = A.numel()
    dtype = A.dtype
    if (
        A.is_contiguous()
        and dtype in (torch.float16, torch.float32, torch.bfloat16)
        and numel >= _NOT_EQUAL_SCALAR_MASKED_MIN
    ):
        s = float(B)
        wrapped = float(torch.tensor(s, dtype=dtype).item())
        if math.isfinite(wrapped):
            tile = (
                _NOT_EQUAL_SCALAR_TILE_F32
                if dtype == torch.float32
                else _NOT_EQUAL_SCALAR_TILE_HALF
            )
            # fp16 has a native vector compare; bf16 does not, but a bf16 value
            # within the fp16 range widens to fp16 losslessly (7-bit mantissa is
            # a subset of fp16's 10-bit), so route it through the fp16 compare
            # too. Only safe when the scalar itself is fp16-finite: otherwise the
            # scalar would round to +/-inf and collide with overflowing inputs.
            use_half = dtype == torch.float16 or (
                dtype == torch.bfloat16 and abs(wrapped) <= _FP16_MAX
            )
            if numel % tile == 0 and numel >= tile * _NOT_EQUAL_SCALAR_MIN_GRID:
                return _not_equal_scalar_fast(
                    A, wrapped, tile, (numel // tile,), use_half
                )
            return _not_equal_scalar_fast_masked(A, wrapped, tile, numel, use_half)
    if dtype in (torch.float16, torch.float32, torch.bfloat16):
        B = float(torch.tensor(float(B), dtype=dtype).item())
    res = not_equal_func_scalar(A, B)
    return res


# Direct scalar compare vectorizes on XPU with TRITONXPU_COMPARE_FUSION=1 (same
# path not_equal's tensor-tensor kernel rides), giving ~2-4x over the old
# branchless-arithmetic kernel. TRITONXPU_FP16_FAST must stay *off*: with it on,
# the fp16 compare trips a TritonXPUDtypeConvert compile failure. Wide tiles win
# (the transfer bandwidth scales with the per-program contiguous run), so f16/
# bf16 use 128K and f32 64K -- the crossover measured on KL3 at 4096x4096.
# fp16 uses a native vector compare against a tl.full constant (0.92); f32 uses
# a plain compare (0.82); bf16 widens to fp16 to reach that fast path (0.87).
_NOT_EQUAL_SCALAR_TILE_F32 = 65536
_NOT_EQUAL_SCALAR_TILE_HALF = 131072
_NOT_EQUAL_SCALAR_MIN_GRID = 8
_NOT_EQUAL_SCALAR_MASKED_MIN = 1 << 20
_FP16_MAX = 65504.0


@triton.jit
def not_equal_scalar_fast_kernel(out_ptr, x_ptr, scalar, TILE: tl.constexpr):
    pid = tl.program_id(0)
    tid = pid * TILE + tl.arange(0, TILE)
    x = tl.load(x_ptr + tid).to(tl.float32)
    tl.store(out_ptr + tid, x != scalar)


@triton.jit
def not_equal_scalar_fast_masked_kernel(
    out_ptr, x_ptr, scalar, numel, TILE: tl.constexpr
):
    pid = tl.program_id(0)
    tid = pid * TILE + tl.arange(0, TILE)
    mask = tid < numel
    x = tl.load(x_ptr + tid, mask=mask).to(tl.float32)
    tl.store(out_ptr + tid, x != scalar, mask=mask)


@triton.jit
def not_equal_scalar_half_kernel(out_ptr, x_ptr, scalar, TILE: tl.constexpr):
    pid = tl.program_id(0)
    tid = pid * TILE + tl.arange(0, TILE)
    x = tl.load(x_ptr + tid).to(tl.float16)
    yv = tl.full([TILE], scalar, tl.float16)
    tl.store(out_ptr + tid, x != yv)


@triton.jit
def not_equal_scalar_half_masked_kernel(
    out_ptr, x_ptr, scalar, numel, TILE: tl.constexpr
):
    pid = tl.program_id(0)
    tid = pid * TILE + tl.arange(0, TILE)
    mask = tid < numel
    x = tl.load(x_ptr + tid, mask=mask).to(tl.float16)
    yv = tl.full([TILE], scalar, tl.float16)
    tl.store(out_ptr + tid, x != yv, mask=mask)


def _set_compare_env():
    prev = (
        os.environ.get("TRITONXPU_COMPARE_FUSION"),
        os.environ.get("TRITONXPU_FP16_FAST"),
    )
    os.environ["TRITONXPU_COMPARE_FUSION"] = "1"
    # Must be off: fp16 direct compare crashes TritonXPUDtypeConvert with it on.
    os.environ["TRITONXPU_FP16_FAST"] = "0"
    return prev


def _restore_compare_env(prev):
    for key, val in zip(("TRITONXPU_COMPARE_FUSION", "TRITONXPU_FP16_FAST"), prev):
        if val is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = val


def _not_equal_scalar_fast(A, scalar, tile, grid, use_half):
    out = torch.empty_like(A, dtype=torch.bool)
    kernel = not_equal_scalar_half_kernel if use_half else not_equal_scalar_fast_kernel
    prev = _set_compare_env()
    try:
        kernel[grid](
            out.view(torch.uint8),
            A,
            scalar,
            TILE=tile,
            **_NOT_EQUAL_FAST_LAUNCH_OPTS,
        )
    finally:
        _restore_compare_env(prev)
    return out


def _not_equal_scalar_fast_masked(A, scalar, tile, numel, use_half):
    out = torch.empty_like(A, dtype=torch.bool)
    grid = (math.ceil(numel / tile),)
    kernel = (
        not_equal_scalar_half_masked_kernel
        if use_half
        else not_equal_scalar_fast_masked_kernel
    )
    prev = _set_compare_env()
    try:
        kernel[grid](
            out.view(torch.uint8),
            A,
            scalar,
            numel,
            TILE=tile,
            **_NOT_EQUAL_FAST_LAUNCH_OPTS,
        )
    finally:
        _restore_compare_env(prev)
    return out


_NOT_EQUAL_FAST_LAUNCH_OPTS = dict(
    num_warps=4,
    buffer_size_limit=8192,
    unroll_num=16,
    isCloseMemoryAsync=False,
)
