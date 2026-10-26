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

# The flagtree compiler shipped in the CI image (flagtree 0.7.0+xpu3.6, an
# older triton fork pinned in backends.yaml) rejects TRITONXPU_COMPARE_FUSION
# on the wide (131072-lane) native kernels: it fuses the compare into a 32-bit
# triton_xpu.cmpf result (tensor<131072xui32>) that TritonXPUNormalize refuses
# with OutOfResources: uni_sram -- the same failure le.py documents for its
# generic path -- and the masked kernels crash at runtime when launched with
# the tuned launch opts (kernel exception / NOC idle timeout, same as
# isneginf). Detect it once here and keep the fast config only on the newer
# triton build that legalizes it. TRITONXPU_FP16_FAST stays (it only strips
# the fp32 widening, see le.py).
_IS_FLAGTREE = os.path.exists(
    os.path.join(os.path.dirname(triton.__file__), "FLAGTREE_BACKEND")
)


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
    # Fast path for all contiguous same-shape float tensor-tensor compares:
    # native fused compare (x <= y) writing bool directly. Up to 2^18 elements
    # the tile is adaptive (see _less_equal_tensor_adaptive_tile) so mid sizes
    # get a one-wave 8-CTA grid instead of one 131072-lane program: the fixed
    # 131072 tile left (1024,256) on grid=2 (2/8 CTAs busy) and (64,64,16) on
    # a grid=1 masked kernel with 3/4 of the lanes idle. Larger tensors go
    # through _less_equal_tensor_large (wide unmasked bulk + small masked
    # tail): the generic pointwise path emits a >= 65536-lane masked 1-D tile
    # for large numels, which exhausts uni_sram at compile time on the CI
    # (flagtree) XPU compiler. Generic fused compare path only for
    # non-contiguous / mixed-dtype / differently-shaped inputs.
    numel = A.numel()
    if (
        A.dtype in (torch.float16, torch.float32, torch.bfloat16)
        and A.dtype == B.dtype
        and A.is_contiguous()
        and B.is_contiguous()
        and A.shape == B.shape
        and 0 < numel
    ):
        if numel <= _LESS_EQUAL_TENSOR_FAST_MAX:
            return _less_equal_tensor_native(
                A, B, numel, _less_equal_tensor_adaptive_tile(numel)
            )
        return _less_equal_tensor_large(A, B, numel)
    if not _IS_FLAGTREE:
        os.environ["TRITONXPU_COMPARE_FUSION"] = "1"
    os.environ["TRITONXPU_FP16_FAST"] = "1"
    res = less_equal_func(A, B)
    if not _IS_FLAGTREE:
        del os.environ["TRITONXPU_COMPARE_FUSION"]
    del os.environ["TRITONXPU_FP16_FAST"]
    return res


# ---------------------------------------------------------------------------
# less_equal tensor-tensor fast path (fp16/fp32/bf16, contiguous, same shape):
# native fused compare (x <= y) writing bool directly, no fp32 buffer and no
# _copy_from. NaN <= y -> False (matches torch). The tile is picked by
# _less_equal_tensor_adaptive_tile; _LESS_EQUAL_TENSOR_FAST_TILE is the wide
# bulk tile of the large-tensor path (and the fallback value of the tile
# parameter).
_LESS_EQUAL_TENSOR_FAST_TILE = 131072
# One-wave grid target (== machine CTA count, measured 8 on P800): 8 CTAs beat
# both 1 giant program (7/8 CTA idle) and 16+ small programs (launch wall).
_LESS_EQUAL_TENSOR_FAST_GRID = 8
# Floor on the adaptive tile: at/below ~4096 lanes a single CTA (grid=1) wins
# over an 8-way split (measured n=1024/4096/16384 all best at tile=4096).
_LESS_EQUAL_TENSOR_FAST_TILE_MIN = 4096
# Mid/small tensor regime upper bound: numel <= 2^18 uses the adaptive tile;
# larger tensors use the wide-bulk + tail path below.
_LESS_EQUAL_TENSOR_FAST_MAX = 1 << 18
# Ceiling on *masked* 1-D tiles. The backend mis-lowers masked memory access
# at >= 32768 lanes (the pointwise codegen's _BAD_TILE_SIZE_1D -> 16384
# fallback), so every masked native kernel stays at or below 16384 lanes.
# Unmasked tiles may be wider (131072): on the CI (flagtree) compiler they
# only fit when TRITONXPU_COMPARE_FUSION is off (see _IS_FLAGTREE), which the
# native paths below guarantee on that compiler.
_LESS_EQUAL_TENSOR_MASKED_TILE_MAX = 16384
# Triton XPU buffer budget for the native kernels (same value as the scalar /
# in-place paths below; kept as a named constant per backend convention).
_LESS_EQUAL_BUFFER_SIZE_LIMIT = 8192
# Launch options for the *unmasked* native kernels (kept only on the newer
# triton build; flagtree degrades to a plain launch -- see _IS_FLAGTREE).
# Masked kernels get their own opts below, without buffer_size_limit.
_LESS_EQUAL_FAST_LAUNCH_OPTS = {
    "buffer_size_limit": _LESS_EQUAL_BUFFER_SIZE_LIMIT,
    "unroll_num": 16,
    "isCloseMemoryAsync": False,
}
# buffer_size_limit is NOT safe on *masked* 1-D kernels: launching a masked
# kernel with it set raises a NOC idle timeout on the XPU device (isolated
# tail-kernel probe: buffer_size_limit alone NOCs, on both the local triton
# build and flagtree), while the unmasked flat kernels tolerate it. The
# masked launches keep the other two opts, which the same probe shows are
# safe.
_LESS_EQUAL_MASKED_LAUNCH_OPTS = {
    "unroll_num": 16,
    "isCloseMemoryAsync": False,
}


def _less_equal_tensor_adaptive_tile(numel):
    # Target _LESS_EQUAL_TENSOR_FAST_GRID CTAs for mid sizes, floored at
    # _LESS_EQUAL_TENSOR_FAST_TILE_MIN for tiny tensors. Result is a power of
    # two so the flat kernel keeps its alignment-friendly 1-D form.
    return max(
        _LESS_EQUAL_TENSOR_FAST_TILE_MIN,
        triton.next_power_of_2(
            (numel + _LESS_EQUAL_TENSOR_FAST_GRID - 1) // _LESS_EQUAL_TENSOR_FAST_GRID
        ),
    )


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
    # Clamp the load addresses: masked loads still issue memory accesses on
    # XPU and every address must be legal, so for tensors smaller than one
    # tile (numel < TILE, e.g. the () / (1,) POINTWISE_SHAPES) the OOB lanes
    # read past the allocation. The OOB lane results are zeroed via the
    # explicit `& mask`; the store keeps its guard mask.
    tid_c = tl.minimum(tid, numel - 1)
    r = (tl.load(x_ptr + tid_c) <= tl.load(y_ptr + tid_c)) & mask
    tl.store(out_ptr + tid, r.to(tl.int8), mask=mask)


def _less_equal_tensor_native(A, B, numel, tile=_LESS_EQUAL_TENSOR_FAST_TILE):
    # Native fused compare (x <= y, same dtype) writing bool directly -- no fp32
    # buffer, no _copy_from. Env set at compile time so the fusion pass sees it
    # (same pattern as less_equal_scalar's native path). Exact multiples of the
    # tile run mask-free; all other sizes use a real tail mask (masked tiles
    # are capped at _LESS_EQUAL_TENSOR_MASKED_TILE_MAX -- see its comment).
    out = torch.empty_like(A, dtype=torch.bool)
    x = A.reshape(-1)
    y = B.reshape(-1)
    masked = numel % tile != 0
    if masked and tile > _LESS_EQUAL_TENSOR_MASKED_TILE_MAX:
        tile = _LESS_EQUAL_TENSOR_MASKED_TILE_MAX
    grid = (math.ceil(numel / tile),) if masked else (numel // tile,)
    if not _IS_FLAGTREE:
        os.environ["TRITONXPU_COMPARE_FUSION"] = "1"
    os.environ["TRITONXPU_FP16_FAST"] = "1"
    launch_opts = {} if _IS_FLAGTREE else _LESS_EQUAL_FAST_LAUNCH_OPTS
    masked_opts = {} if _IS_FLAGTREE else _LESS_EQUAL_MASKED_LAUNCH_OPTS
    try:
        if masked:
            less_equal_tensor_native_masked_kernel[grid](
                out,
                x,
                y,
                numel,
                TILE=tile,
                num_warps=4,
                **masked_opts,
            )
        else:
            less_equal_tensor_native_kernel[grid](
                out,
                x,
                y,
                TILE=tile,
                num_warps=4,
                **launch_opts,
            )
    finally:
        if not _IS_FLAGTREE:
            del os.environ["TRITONXPU_COMPARE_FUSION"]
        del os.environ["TRITONXPU_FP16_FAST"]
    return out


@triton.jit
def less_equal_tensor_native_tail_kernel(
    out_ptr, x_ptr, y_ptr, tail, offset, TILE: tl.constexpr
):
    # Tail window of the large-tensor path: writes the range [offset,
    # offset+tail) with small masked tiles (grid = ceil(tail/TILE), same
    # OOB-load clamp as the masked kernel). offset/tail are runtime values so
    # every tail size reuses this one compiled kernel.
    pid = tl.program_id(0)
    tid = offset + pid * TILE + tl.arange(0, TILE)
    mask = tid < offset + tail
    tid_c = tl.minimum(tid, offset + tail - 1)
    r = (tl.load(x_ptr + tid_c) <= tl.load(y_ptr + tid_c)) & mask
    tl.store(out_ptr + tid, r.to(tl.int8), mask=mask)


def _less_equal_tensor_large(A, B, numel):
    # Large tensors (numel > _LESS_EQUAL_TENSOR_FAST_MAX): the exact-multiple
    # bulk is written by _LESS_EQUAL_TENSOR_FAST_TILE-wide *unmasked* kernels
    # (131072 lanes; on the CI flagtree compiler these only compile with
    # TRITONXPU_COMPARE_FUSION off -- see _IS_FLAGTREE), and any remainder by
    # a small masked tail. The generic pointwise path is avoided entirely
    # here: it emits a masked 1-D tile of next_pow2(numel/12) (>= 65536 lanes
    # for these sizes), which fails to compile on flagtree with
    # OutOfResources(uni_sram).
    out = torch.empty_like(A, dtype=torch.bool)
    x = A.reshape(-1)
    y = B.reshape(-1)
    tail = numel % _LESS_EQUAL_TENSOR_FAST_TILE
    bulk = numel - tail
    if not _IS_FLAGTREE:
        os.environ["TRITONXPU_COMPARE_FUSION"] = "1"
    os.environ["TRITONXPU_FP16_FAST"] = "1"
    launch_opts = {} if _IS_FLAGTREE else _LESS_EQUAL_FAST_LAUNCH_OPTS
    masked_opts = {} if _IS_FLAGTREE else _LESS_EQUAL_MASKED_LAUNCH_OPTS
    try:
        if bulk:
            less_equal_tensor_native_kernel[(bulk // _LESS_EQUAL_TENSOR_FAST_TILE,)](
                out,
                x,
                y,
                TILE=_LESS_EQUAL_TENSOR_FAST_TILE,
                num_warps=4,
                **launch_opts,
            )
        if tail:
            less_equal_tensor_native_tail_kernel[
                (math.ceil(tail / _LESS_EQUAL_TENSOR_MASKED_TILE_MAX),)
            ](
                out,
                x,
                y,
                tail,
                bulk,
                TILE=_LESS_EQUAL_TENSOR_MASKED_TILE_MAX,
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
    if not _IS_FLAGTREE:
        os.environ["TRITONXPU_COMPARE_FUSION"] = "1"
    os.environ["TRITONXPU_FP16_FAST"] = "1"
    launch_opts = {} if _IS_FLAGTREE else _LESS_EQUAL_FAST_LAUNCH_OPTS
    masked_opts = {} if _IS_FLAGTREE else _LESS_EQUAL_MASKED_LAUNCH_OPTS
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
                **masked_opts,
            )
        else:
            less_equal_scalar_native_kernel[grid](
                out,
                x,
                scalar,
                TILE=tile,
                DTYPE=DTYPE,
                num_warps=4,
                **launch_opts,
            )
    finally:
        if not _IS_FLAGTREE:
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
    logger.debug("GEMS_KUNLUNXIN LESS_EQUAL__TENSOR")
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
        buffer_size_limit=_LESS_EQUAL_BUFFER_SIZE_LIMIT,
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
    logger.debug("GEMS_KUNLUNXIN LESS_EQUAL__SCALAR")
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
        buffer_size_limit=_LESS_EQUAL_BUFFER_SIZE_LIMIT,
        unroll_num=16,
        isCloseMemoryAsync=True,
    )
    return A
