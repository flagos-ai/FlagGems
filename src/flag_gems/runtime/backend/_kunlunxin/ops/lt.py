import logging
import math
import os

import torch
import triton
import triton.language as tl
from _kunlunxin.utils.codegen_config_utils import CodeGenConfig

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)

# The flagtree compiler (older triton fork pinned in the CI image) rejects
# TRITONXPU_COMPARE_FUSION on wide (131072-lane) native kernels: it fuses the
# compare into a 32-bit triton_xpu.cmpf result that TritonXPUNormalize refuses
# (OutOfResources: uni_sram), the same failure le.py/less_equal.py document for
# their generic path. Detect it once and keep the fusion env only on the newer
# triton build that legalizes it.
_IS_FLAGTREE = os.path.exists(
    os.path.join(os.path.dirname(triton.__file__), "FLAGTREE_BACKEND")
)

config_ = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    isCloseMemoryAsync=False,
    unroll_num=8,
    kunlunAutoGrid=True,
)


@pointwise_dynamic(
    promotion_methods=[(0, 1, "ALWAYS_BOOL")],
    config=config_,
)
@triton.jit
def lt_func(x, y):
    return x.to(tl.float32) < y


def lt(A, B):
    logger.debug("GEMS_KUNLUNXIN LT")
    # Fast path for contiguous same-shape float tensor-tensor compares: native
    # fused compare (x < y) writing bool directly (mirrors less_equal.py's
    # tensor-tensor path). numel <= 2^18 uses an adaptive one-wave 8-CTA tile;
    # larger tensors use wide unmasked bulk + small masked tail. The generic
    # pointwise path (below) emits a >= 65536-lane masked 1-D tile for large
    # numels, which exhausts uni_sram on the CI (flagtree) compiler.
    numel = A.numel()
    if (
        A.dtype in (torch.float16, torch.float32, torch.bfloat16)
        and A.dtype == B.dtype
        and A.is_contiguous()
        and B.is_contiguous()
        and A.shape == B.shape
        and 0 < numel
    ):
        if numel <= _LT_TENSOR_FAST_MAX:
            return _lt_tensor_native(A, B, numel, _lt_tensor_adaptive_tile(numel))
        return _lt_tensor_large(A, B, numel)
    if not _IS_FLAGTREE:
        os.environ["TRITONXPU_COMPARE_FUSION"] = "1"
    os.environ["TRITONXPU_FP16_FAST"] = "1"
    res = lt_func(A, B)
    if not _IS_FLAGTREE:
        del os.environ["TRITONXPU_COMPARE_FUSION"]
    del os.environ["TRITONXPU_FP16_FAST"]
    return res


# ---------------------------------------------------------------------------
# lt tensor-tensor fast path (fp16/fp32/bf16, contiguous, same shape): native
# fused compare (x < y) writing bool directly, no fp32 buffer and no _copy_from.
# NaN < y -> False (matches torch, unordered compare). The tile is picked by
# _lt_tensor_adaptive_tile; _LT_TENSOR_FAST_TILE is the wide bulk tile of the
# large-tensor path (and the fallback tile value).
_LT_TENSOR_FAST_TILE = 131072
# One-wave grid target (== machine CTA count, measured 8 on P800).
_LT_TENSOR_FAST_GRID = 8
# Floor on the adaptive tile: below ~4096 lanes a single CTA (grid=1) wins over
# an 8-way split.
_LT_TENSOR_FAST_TILE_MIN = 4096
# Mid/small regime upper bound: numel <= 2^18 uses the adaptive tile; larger
# tensors use the wide-bulk + tail path.
_LT_TENSOR_FAST_MAX = 1 << 18
# Ceiling on *masked* 1-D tiles (the backend mis-lowers masked access at
# >= 32768 lanes). Unmasked tiles may be wider (131072).
_LT_TENSOR_MASKED_TILE_MAX = 16384
_LT_BUFFER_SIZE_LIMIT = 8192
_LT_FAST_LAUNCH_OPTS = {
    "buffer_size_limit": _LT_BUFFER_SIZE_LIMIT,
    "unroll_num": 16,
    "isCloseMemoryAsync": False,
}
# buffer_size_limit is NOT safe on *masked* 1-D kernels (NOC idle timeout).
_LT_MASKED_LAUNCH_OPTS = {
    "unroll_num": 16,
    "isCloseMemoryAsync": False,
}


def _lt_tensor_adaptive_tile(numel):
    return max(
        _LT_TENSOR_FAST_TILE_MIN,
        triton.next_power_of_2(
            (numel + _LT_TENSOR_FAST_GRID - 1) // _LT_TENSOR_FAST_GRID
        ),
    )


@triton.jit
def lt_tensor_native_kernel(out_ptr, x_ptr, y_ptr, TILE: tl.constexpr):
    pid = tl.program_id(0)
    tid = pid * TILE + tl.arange(0, TILE)
    r = tl.load(x_ptr + tid) < tl.load(y_ptr + tid)
    tl.store(out_ptr + tid, r.to(tl.int8))


@triton.jit
def lt_tensor_native_masked_kernel(out_ptr, x_ptr, y_ptr, numel, TILE: tl.constexpr):
    pid = tl.program_id(0)
    tid = pid * TILE + tl.arange(0, TILE)
    mask = tid < numel
    # Clamp the load addresses: masked loads still issue memory accesses on XPU
    # and every address must be legal (numel < TILE, e.g. () / (1,)).
    tid_c = tl.minimum(tid, numel - 1)
    r = (tl.load(x_ptr + tid_c) < tl.load(y_ptr + tid_c)) & mask
    tl.store(out_ptr + tid, r.to(tl.int8), mask=mask)


def _lt_tensor_native(A, B, numel, tile=_LT_TENSOR_FAST_TILE):
    # Native fused compare (x < y, same dtype) writing bool directly.
    out = torch.empty_like(A, dtype=torch.bool)
    x = A.reshape(-1)
    y = B.reshape(-1)
    masked = numel % tile != 0
    if masked and tile > _LT_TENSOR_MASKED_TILE_MAX:
        tile = _LT_TENSOR_MASKED_TILE_MAX
    grid = (math.ceil(numel / tile),) if masked else (numel // tile,)
    if not _IS_FLAGTREE:
        os.environ["TRITONXPU_COMPARE_FUSION"] = "1"
    os.environ["TRITONXPU_FP16_FAST"] = "1"
    launch_opts = {} if _IS_FLAGTREE else _LT_FAST_LAUNCH_OPTS
    masked_opts = {} if _IS_FLAGTREE else _LT_MASKED_LAUNCH_OPTS
    try:
        if masked:
            lt_tensor_native_masked_kernel[grid](
                out, x, y, numel, TILE=tile, num_warps=4, **masked_opts
            )
        else:
            lt_tensor_native_kernel[grid](
                out, x, y, TILE=tile, num_warps=4, **launch_opts
            )
    finally:
        if not _IS_FLAGTREE:
            del os.environ["TRITONXPU_COMPARE_FUSION"]
        del os.environ["TRITONXPU_FP16_FAST"]
    return out


@triton.jit
def lt_tensor_native_tail_kernel(
    out_ptr, x_ptr, y_ptr, tail, offset, TILE: tl.constexpr
):
    pid = tl.program_id(0)
    tid = offset + pid * TILE + tl.arange(0, TILE)
    mask = tid < offset + tail
    tid_c = tl.minimum(tid, offset + tail - 1)
    r = (tl.load(x_ptr + tid_c) < tl.load(y_ptr + tid_c)) & mask
    tl.store(out_ptr + tid, r.to(tl.int8), mask=mask)


def _lt_tensor_large(A, B, numel):
    # Large tensors (numel > _LT_TENSOR_FAST_MAX): exact-multiple bulk written
    # by _LT_TENSOR_FAST_TILE-wide *unmasked* kernels, remainder by a small
    # masked tail. Avoids the generic pointwise path's huge masked 1-D tile.
    out = torch.empty_like(A, dtype=torch.bool)
    x = A.reshape(-1)
    y = B.reshape(-1)
    tail = numel % _LT_TENSOR_FAST_TILE
    bulk = numel - tail
    if not _IS_FLAGTREE:
        os.environ["TRITONXPU_COMPARE_FUSION"] = "1"
    os.environ["TRITONXPU_FP16_FAST"] = "1"
    launch_opts = {} if _IS_FLAGTREE else _LT_FAST_LAUNCH_OPTS
    masked_opts = {} if _IS_FLAGTREE else _LT_MASKED_LAUNCH_OPTS
    try:
        if bulk:
            lt_tensor_native_kernel[(bulk // _LT_TENSOR_FAST_TILE,)](
                out, x, y, TILE=_LT_TENSOR_FAST_TILE, num_warps=4, **launch_opts
            )
        if tail:
            lt_tensor_native_tail_kernel[
                (math.ceil(tail / _LT_TENSOR_MASKED_TILE_MAX),)
            ](
                out,
                x,
                y,
                tail,
                bulk,
                TILE=_LT_TENSOR_MASKED_TILE_MAX,
                num_warps=4,
                **masked_opts,
            )
    finally:
        if not _IS_FLAGTREE:
            del os.environ["TRITONXPU_COMPARE_FUSION"]
        del os.environ["TRITONXPU_FP16_FAST"]
    return out


@pointwise_dynamic(
    is_tensor=[True, False],
    promotion_methods=[(0, 1, "ALWAYS_BOOL")],
    config=config_,
)
@triton.jit
def lt_func_scalar(x, y):
    return x.to(tl.float32) < y


def lt_scalar(A, B):
    logger.debug("GEMS_KUNLUNXIN LT_SCALAR")
    numel = A.numel()
    dtype = A.dtype
    if (
        A.is_contiguous()
        and dtype in (torch.float16, torch.float32, torch.bfloat16)
        and numel >= _LT_SCALAR_MASKED_MIN
    ):
        wrapped = float(torch.tensor(float(B), dtype=dtype).item())
        if math.isfinite(wrapped):
            tile = (
                _LT_SCALAR_TILE_F32 if dtype == torch.float32 else _LT_SCALAR_TILE_HALF
            )
            # fp16 has a native vector compare; a bf16 value within the fp16
            # range widens to fp16 losslessly (7-bit mantissa is a subset of
            # fp16's 10-bit) and preserves ordering, so route it there too. Only
            # safe when the scalar is fp16-finite, else it would round to +/-inf.
            use_half = dtype == torch.float16 or (
                dtype == torch.bfloat16 and abs(wrapped) <= _FP16_MAX
            )
            if numel % tile == 0 and numel >= tile * _LT_SCALAR_MIN_GRID:
                return _lt_scalar_fast(A, wrapped, tile, (numel // tile,), use_half)
            return _lt_scalar_fast_masked(A, wrapped, tile, numel, use_half)
    if dtype in (torch.float16, torch.float32, torch.bfloat16):
        B = float(torch.tensor(float(B), dtype=dtype).item())
    res = lt_func_scalar(A, B)
    return res


# Direct scalar compare vectorizes on XPU with TRITONXPU_COMPARE_FUSION=1 (the
# path lt's tensor-tensor kernel rides). TRITONXPU_FP16_FAST must stay off, or
# the fp16 compare trips a TritonXPUDtypeConvert compile failure. fp16 uses a
# native vector compare against a tl.full constant; f32 a plain compare; bf16
# widens to fp16 to reach that fast path. See not_equal.py for the same recipe.
_LT_SCALAR_TILE_F32 = 65536
_LT_SCALAR_TILE_HALF = 131072
_LT_SCALAR_MIN_GRID = 8
_LT_SCALAR_MASKED_MIN = 1 << 20
_FP16_MAX = 65504.0


@triton.jit
def lt_scalar_fast_kernel(out_ptr, x_ptr, scalar, TILE: tl.constexpr):
    pid = tl.program_id(0)
    tid = pid * TILE + tl.arange(0, TILE)
    x = tl.load(x_ptr + tid).to(tl.float32)
    tl.store(out_ptr + tid, x < scalar)


@triton.jit
def lt_scalar_fast_masked_kernel(out_ptr, x_ptr, scalar, numel, TILE: tl.constexpr):
    pid = tl.program_id(0)
    tid = pid * TILE + tl.arange(0, TILE)
    mask = tid < numel
    x = tl.load(x_ptr + tid, mask=mask).to(tl.float32)
    tl.store(out_ptr + tid, x < scalar, mask=mask)


@triton.jit
def lt_scalar_half_kernel(out_ptr, x_ptr, scalar, TILE: tl.constexpr):
    pid = tl.program_id(0)
    tid = pid * TILE + tl.arange(0, TILE)
    x = tl.load(x_ptr + tid).to(tl.float16)
    yv = tl.full([TILE], scalar, tl.float16)
    tl.store(out_ptr + tid, x < yv)


@triton.jit
def lt_scalar_half_masked_kernel(out_ptr, x_ptr, scalar, numel, TILE: tl.constexpr):
    pid = tl.program_id(0)
    tid = pid * TILE + tl.arange(0, TILE)
    mask = tid < numel
    x = tl.load(x_ptr + tid, mask=mask).to(tl.float16)
    yv = tl.full([TILE], scalar, tl.float16)
    tl.store(out_ptr + tid, x < yv, mask=mask)


def _lt_set_compare_env():
    prev = (
        os.environ.get("TRITONXPU_COMPARE_FUSION"),
        os.environ.get("TRITONXPU_FP16_FAST"),
    )
    os.environ["TRITONXPU_COMPARE_FUSION"] = "1"
    os.environ["TRITONXPU_FP16_FAST"] = "0"
    return prev


def _lt_restore_compare_env(prev):
    for key, val in zip(("TRITONXPU_COMPARE_FUSION", "TRITONXPU_FP16_FAST"), prev):
        if val is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = val


def _lt_scalar_fast(A, scalar, tile, grid, use_half):
    out = torch.empty_like(A, dtype=torch.bool)
    kernel = lt_scalar_half_kernel if use_half else lt_scalar_fast_kernel
    prev = _lt_set_compare_env()
    try:
        kernel[grid](
            out.view(torch.uint8),
            A,
            scalar,
            TILE=tile,
            **_LT_SCALAR_LAUNCH_OPTS,
        )
    finally:
        _lt_restore_compare_env(prev)
    return out


def _lt_scalar_fast_masked(A, scalar, tile, numel, use_half):
    out = torch.empty_like(A, dtype=torch.bool)
    grid = (math.ceil(numel / tile),)
    kernel = lt_scalar_half_masked_kernel if use_half else lt_scalar_fast_masked_kernel
    prev = _lt_set_compare_env()
    try:
        kernel[grid](
            out.view(torch.uint8),
            A,
            scalar,
            numel,
            TILE=tile,
            **_LT_SCALAR_LAUNCH_OPTS,
        )
    finally:
        _lt_restore_compare_env(prev)
    return out


_LT_SCALAR_LAUNCH_OPTS = dict(
    num_warps=4,
    buffer_size_limit=8192,
    unroll_num=16,
    isCloseMemoryAsync=False,
)


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
def lt_func_(x, y):
    t = (y - x) * 1.0e30
    t = tl.maximum(0.0, t)
    t = tl.minimum(1.0, t)
    return t


def lt_(A, B):
    logger.debug("GEMS_KUNLUNXIN LT_")
    if A.device != B.device:
        B = B.to(A.device)
    lt_func_(A, B, out0=A)
    return A


config_scalar_inplace_ = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    kunlunAutoGrid=True,
    unroll_num=8,
    buffer_size_limit=4096,
)


@pointwise_dynamic(
    is_tensor=[True, False],
    promotion_methods=[(0, 1, "DEFAULT")],
    config=config_scalar_inplace_,
)
@triton.jit
def lt_func_scalar_(x, y):
    t = (y - x) * 1.0e30
    t = tl.maximum(0.0, t)
    t = tl.minimum(1.0, t)
    return t


def lt_scalar_(A, B):
    logger.debug("GEMS_KUNLUNXIN LT_SCALAR_")
    numel = A.numel()
    if (
        A.is_contiguous()
        and A.dtype in (torch.float16, torch.float32)
        and numel >= _LT_SCALAR_INPLACE_FAST_TILE
        and numel % _LT_SCALAR_INPLACE_FAST_TILE == 0
        and numel // _LT_SCALAR_INPLACE_FAST_TILE >= _LT_SCALAR_INPLACE_MIN_GRID
        and float(B) == 0.0
    ):
        return _lt_scalar_inplace_fast(A)
    if A.dtype in (torch.float16, torch.float32, torch.bfloat16):
        B = float(torch.tensor(float(B), dtype=A.dtype).item())
    lt_func_scalar_(A, B, out0=A)
    return A


_LT_SCALAR_INPLACE_FAST_TILE = 131072
_LT_SCALAR_INPLACE_MIN_GRID = 128


@triton.jit
def lt_scalar_inplace_fast_kernel(x_ptr, TILE: tl.constexpr):
    pid = tl.program_id(0)
    tid = pid * TILE + tl.arange(0, TILE)
    x = tl.load(x_ptr + tid)
    t = (0.0 - x) * 1.0e30
    t = tl.maximum(0.0, t)
    t = tl.minimum(1.0, t)
    tl.store(x_ptr + tid, t)


def _lt_scalar_inplace_fast(A):
    grid = (A.numel() // _LT_SCALAR_INPLACE_FAST_TILE,)
    lt_scalar_inplace_fast_kernel[grid](
        A,
        TILE=_LT_SCALAR_INPLACE_FAST_TILE,
        num_warps=4,
        buffer_size_limit=8192,
        unroll_num=16,
        isCloseMemoryAsync=True,
    )
    return A


def less_(A, B):
    return lt_(A, B)


def less_scalar_(A, B):
    return lt_scalar_(A, B)
