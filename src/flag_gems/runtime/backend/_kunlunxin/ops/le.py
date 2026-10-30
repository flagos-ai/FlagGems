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

from flag_gems.ops.le_ import le_ as _generic_le_
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
def le_func(x, y):
    return x.to(tl.float32) <= y


def le(A, B):
    logger.debug("GEMS_KUNLUNXIN LE")
    # Native fused-compare fast path (mirrors less_equal.py, identical x <= y
    # semantics). Contiguous same-shape float tensor-tensor compares write the
    # bool result directly -- no fp32 buffer, no _copy_from -- via wide tiled
    # native kernels. Generic pointwise path otherwise.
    #
    # COMPARE_FUSION is what makes the native kernels fast (fuses the i1->i8
    # bool store into a 32-bit vendor cmpf, ~3 B/elem instead of the
    # fp32-buffer path), but the flagtree compiler in the CI image rejects the
    # fused result (tensor<Nxui32>) with OutOfResources on the wide
    # (131072-lane) kernels. Rather than branching on compiler identity
    # (_IS_FLAGTREE), the fast path probes the capability at runtime (see
    # _le_cf_capable) and keeps CF + tuned launch opts on capable compilers,
    # plain no-CF launches otherwise -- no version-branch logic anywhere.
    # TRITONXPU_FP16_FAST stays everywhere: it only strips the .to(fp32) so
    # the compare runs as a native fp16 vcmpf (32-lane) instead of widening to
    # fp32, and it is safe on both compilers.
    numel = A.numel()
    if (
        A.dtype in (torch.float16, torch.float32, torch.bfloat16)
        and A.dtype == B.dtype
        and A.is_contiguous()
        and B.is_contiguous()
        and A.shape == B.shape
        and 0 < numel
    ):
        if numel <= _LE_TENSOR_FAST_MAX:
            return _le_tensor_native(A, B, numel, _le_tensor_adaptive_tile(numel))
        return _le_tensor_large(A, B, numel)
    # Generic path: FP16_FAST only (no COMPARE_FUSION). The generic kernel
    # compares with `<=`, which mis-lowers NaN under CF (see the NaN note
    # below); the fast path's swapped y >= x form is what makes CF safe there,
    # and the generic kernel does not have it, so keep CF off for these rare
    # (non-contiguous / mixed-dtype / differently-shaped) inputs.
    os.environ["TRITONXPU_FP16_FAST"] = "1"
    res = le_func(A, B)
    del os.environ["TRITONXPU_FP16_FAST"]
    return res


# ---------------------------------------------------------------------------
# le tensor-tensor fast path (fp16/fp32/bf16, contiguous, same shape): native
# fused compare writing bool directly, no fp32 buffer / _copy_from. NaN <= y
# -> False (matches torch). Mirrors less_equal.py (identical semantics);
# tiles / masked-ceiling / launch opts are the same values proven there.
#
# NaN note: with TRITONXPU_COMPARE_FUSION=1 the backend lowers ordered `<=`
# (OLE) as NOT(OLT), which is true for NaN, while aten returns False (the
# le_scalar note documents this). IEEE `x <= y` == `y >= x`, and OGE lowers
# correctly, so every tensor kernel below compares the swapped form y >= x
# (isolated probe on this compiler: direct `x <= y` mis-lowered 7/7 NaN lanes
# on fp16/fp32/bf16, swapped `y >= x` matched torch on all dtypes). The swap
# is IEEE-identical, so the no-CF fallback is unaffected.
_LE_TENSOR_FAST_TILE = 131072
# One-wave grid target (== machine CTA count, measured 8 on P800): 8 CTAs beat
# both 1 giant program (7/8 CTA idle) and 16+ small programs (launch wall).
_LE_TENSOR_FAST_GRID = 8
# Floor on the adaptive tile: at/below ~4096 lanes a single CTA (grid=1) wins
# over an 8-way split (measured n=1024/4096/16384 all best at tile=4096).
_LE_TENSOR_FAST_TILE_MIN = 4096
# Mid/small tensor regime upper bound: numel <= 2^18 uses the adaptive tile;
# larger tensors use the wide-bulk + tail path below.
_LE_TENSOR_FAST_MAX = 1 << 18
# Ceiling on *masked* 1-D tiles. The backend mis-lowers masked memory access
# at >= 32768 lanes (the pointwise codegen's _BAD_TILE_SIZE_1D -> 16384
# fallback), so every masked native kernel stays at or below 16384 lanes.
# Unmasked tiles may be wider (131072).
_LE_TENSOR_MASKED_TILE_MAX = 16384
_LE_BUFFER_SIZE_LIMIT = 8192
# buffer_size_limit is NOT safe on *masked* 1-D kernels: launching a masked
# kernel with it set raises a NOC idle timeout on the XPU device (isolated
# probe on both the local triton build and flagtree), while the unmasked flat
# kernels tolerate it. The masked launches keep the other two opts, which the
# same probe shows are safe.
_LE_FAST_LAUNCH_OPTS = {
    "buffer_size_limit": _LE_BUFFER_SIZE_LIMIT,
    "unroll_num": 16,
    "isCloseMemoryAsync": False,
}
_LE_MASKED_LAUNCH_OPTS = {
    "unroll_num": 16,
    "isCloseMemoryAsync": False,
}
# Capability probe state: None = untested, True = COMPARE_FUSION compiles and
# runs on this compiler, False = it fails (flagtree). Cached per process.
_LE_CF_OK = None


def _le_cf_capable(dev):
    # Runtime capability probe -- deliberately NOT a compiler-identity check
    # (no _IS_FLAGTREE-style branch). TRITONXPU_COMPARE_FUSION is the source of
    # the fast path's speed, but the flagtree compiler in the CI image rejects
    # the fused 32-bit cmpf(ui32) result on the wide (131072-lane) kernels with
    # OutOfResources. So the first fast-path dispatch tries to compile+run the
    # exact wide kernel with the fusion flag and observes whether it succeeds.
    # Capable compilers (newer triton builds, incl. the ad274881 widening fix)
    # keep CF + tuned launch opts; incapable ones fall back to plain no-CF
    # launches, which compile on every compiler. The flagtree failure is a
    # compile-time exception (raised by the verify pass before any device
    # work), so it is safe to catch here. The result is cached for the process;
    # a failure degrades to the correct (if slower) no-CF path.
    global _LE_CF_OK
    if _LE_CF_OK is not None:
        return _LE_CF_OK
    _LE_CF_OK = False
    os.environ["TRITONXPU_COMPARE_FUSION"] = "1"
    os.environ["TRITONXPU_FP16_FAST"] = "1"
    try:
        a = torch.empty(_LE_TENSOR_FAST_TILE, dtype=torch.float32, device=dev)
        out = torch.empty(_LE_TENSOR_FAST_TILE, dtype=torch.bool, device=dev)
        le_tensor_native_kernel[(1,)](
            out, a, a, TILE=_LE_TENSOR_FAST_TILE, num_warps=4, **_LE_FAST_LAUNCH_OPTS
        )
        _LE_CF_OK = True
    except Exception:
        _LE_CF_OK = False
    finally:
        del os.environ["TRITONXPU_COMPARE_FUSION"]
        del os.environ["TRITONXPU_FP16_FAST"]
    return _LE_CF_OK


def _le_tensor_adaptive_tile(numel):
    # Target _LE_TENSOR_FAST_GRID CTAs for mid sizes, floored at
    # _LE_TENSOR_FAST_TILE_MIN for tiny tensors. Result is a power of two so
    # the flat kernel keeps its alignment-friendly 1-D form.
    return max(
        _LE_TENSOR_FAST_TILE_MIN,
        triton.next_power_of_2(
            (numel + _LE_TENSOR_FAST_GRID - 1) // _LE_TENSOR_FAST_GRID
        ),
    )


@triton.jit
def le_tensor_native_kernel(out_ptr, x_ptr, y_ptr, TILE: tl.constexpr):
    pid = tl.program_id(0)
    tid = pid * TILE + tl.arange(0, TILE)
    # NaN-safe swapped form (y >= x), see the NaN note above.
    r = tl.load(y_ptr + tid) >= tl.load(x_ptr + tid)
    tl.store(out_ptr + tid, r.to(tl.int8))


@triton.jit
def le_tensor_native_masked_kernel(out_ptr, x_ptr, y_ptr, numel, TILE: tl.constexpr):
    pid = tl.program_id(0)
    tid = pid * TILE + tl.arange(0, TILE)
    mask = tid < numel
    # Clamp the load addresses: masked loads still issue memory accesses on
    # XPU and every address must be legal, so for tensors smaller than one
    # tile (numel < TILE, e.g. the () / (1,) POINTWISE_SHAPES) the OOB lanes
    # read past the allocation. The OOB lane results are zeroed via the
    # explicit `& mask`; the store keeps its guard mask. NaN-safe swapped
    # form (y >= x), see the NaN note above.
    tid_c = tl.minimum(tid, numel - 1)
    r = (tl.load(y_ptr + tid_c) >= tl.load(x_ptr + tid_c)) & mask
    tl.store(out_ptr + tid, r.to(tl.int8), mask=mask)


def _le_tensor_native(A, B, numel, tile=_LE_TENSOR_FAST_TILE):
    # Native fused compare (same dtype) writing bool directly -- no fp32
    # buffer, no _copy_from. CF + tuned opts on capable compilers (runtime
    # probe), plain no-CF launches otherwise. Exact multiples of the tile run
    # mask-free; all other sizes use a real tail mask (masked tiles are capped
    # at _LE_TENSOR_MASKED_TILE_MAX -- see its comment above).
    cf = _le_cf_capable(A.device)
    out = torch.empty_like(A, dtype=torch.bool)
    x = A.reshape(-1)
    y = B.reshape(-1)
    masked = numel % tile != 0
    if masked and tile > _LE_TENSOR_MASKED_TILE_MAX:
        tile = _LE_TENSOR_MASKED_TILE_MAX
    grid = (math.ceil(numel / tile),) if masked else (numel // tile,)
    if cf:
        os.environ["TRITONXPU_COMPARE_FUSION"] = "1"
    os.environ["TRITONXPU_FP16_FAST"] = "1"
    unmasked_opts = _LE_FAST_LAUNCH_OPTS if cf else {}
    masked_opts = _LE_MASKED_LAUNCH_OPTS if cf else {}
    try:
        if masked:
            le_tensor_native_masked_kernel[grid](
                out, x, y, numel, TILE=tile, num_warps=4, **masked_opts
            )
        else:
            le_tensor_native_kernel[grid](
                out, x, y, TILE=tile, num_warps=4, **unmasked_opts
            )
    finally:
        if cf:
            del os.environ["TRITONXPU_COMPARE_FUSION"]
        del os.environ["TRITONXPU_FP16_FAST"]
    return out


@triton.jit
def le_tensor_native_tail_kernel(
    out_ptr, x_ptr, y_ptr, tail, offset, TILE: tl.constexpr
):
    # Tail window of the large-tensor path: writes the range [offset,
    # offset+tail) with small masked tiles (grid = ceil(tail/TILE), same
    # OOB-load clamp as the masked kernel). offset/tail are runtime values so
    # every tail size reuses this one compiled kernel. NaN-safe swapped form
    # (y >= x), see the NaN note above.
    pid = tl.program_id(0)
    tid = offset + pid * TILE + tl.arange(0, TILE)
    mask = tid < offset + tail
    tid_c = tl.minimum(tid, offset + tail - 1)
    r = (tl.load(y_ptr + tid_c) >= tl.load(x_ptr + tid_c)) & mask
    tl.store(out_ptr + tid, r.to(tl.int8), mask=mask)


def _le_tensor_large(A, B, numel):
    # Large tensors (numel > _LE_TENSOR_FAST_MAX): the exact-multiple bulk is
    # written by _LE_TENSOR_FAST_TILE-wide *unmasked* kernels, any remainder by
    # a small masked tail. CF + tuned opts on capable compilers (runtime
    # probe), plain no-CF launches otherwise. The generic pointwise path is
    # avoided entirely here (its masked 1-D tile of next_pow2(numel/12)
    # exhausts uni_sram on the CI flagtree compiler -- see less_equal.py).
    cf = _le_cf_capable(A.device)
    out = torch.empty_like(A, dtype=torch.bool)
    x = A.reshape(-1)
    y = B.reshape(-1)
    tail = numel % _LE_TENSOR_FAST_TILE
    bulk = numel - tail
    if cf:
        os.environ["TRITONXPU_COMPARE_FUSION"] = "1"
    os.environ["TRITONXPU_FP16_FAST"] = "1"
    unmasked_opts = _LE_FAST_LAUNCH_OPTS if cf else {}
    masked_opts = _LE_MASKED_LAUNCH_OPTS if cf else {}
    try:
        if bulk:
            le_tensor_native_kernel[(bulk // _LE_TENSOR_FAST_TILE,)](
                out, x, y, TILE=_LE_TENSOR_FAST_TILE, num_warps=4,
                **unmasked_opts,
            )
        if tail:
            le_tensor_native_tail_kernel[
                (math.ceil(tail / _LE_TENSOR_MASKED_TILE_MAX),)
            ](
                out,
                x,
                y,
                tail,
                bulk,
                TILE=_LE_TENSOR_MASKED_TILE_MAX,
                num_warps=4,
                **masked_opts,
            )
    finally:
        if cf:
            del os.environ["TRITONXPU_COMPARE_FUSION"]
        del os.environ["TRITONXPU_FP16_FAST"]
    return out


@pointwise_dynamic(
    is_tensor=[True, False],
    promotion_methods=[(0, 1, "ALWAYS_BOOL")],
    config=config_,
)
@triton.jit
def le_func_scalar(x, y):
    return x.to(tl.float32) <= y


def le_scalar(A, B):
    logger.debug("GEMS_KUNLUNXIN LE_SCALAR")
    numel = A.numel()
    dtype = A.dtype
    if (
        A.is_contiguous()
        and dtype in (torch.float16, torch.float32, torch.bfloat16)
        and float(B) == float(torch.tensor(float(B), dtype=dtype).item())
    ):
        if numel >= _LE_SCALAR_FAST_TILE and numel % _LE_SCALAR_FAST_TILE == 0:
            return _le_scalar_native(A, float(B), numel, masked=False)
        if numel >= _LE_SCALAR_MASKED_MIN and numel % _LE_SCALAR_FAST_TILE != 0:
            return _le_scalar_native(A, float(B), numel, masked=True)
        # Small / thin shapes (numel < FAST_TILE): route to the native fused
        # kernel with an adaptive small TILE instead of the ~10x-slower generic
        # pointwise path. Tile hugs numel (next_pow2, floor 1024) so we don't
        # launch a 131072-lane program for a few-K-element tensor.
        if 0 < numel < _LE_SCALAR_FAST_TILE:
            tile = min(_LE_SCALAR_FAST_TILE, max(1024, triton.next_power_of_2(numel)))
            return _le_scalar_native(
                A, float(B), numel, masked=(numel % tile != 0), tile=tile
            )
    res = le_func_scalar(A, B)
    return res


_LE_SCALAR_FAST_TILE = 131072
_LE_SCALAR_MASKED_MIN = 1 << 20

# ---------------------------------------------------------------------------
# bf16 note: xpu3 has no native bf16 compare, so bf16 widens to fp32 (16-lane)
# and large shapes stay compiler-bound (~0.56); fp16/fp32 reach the fused path.


@triton.jit
def le_scalar_native_kernel(
    out_ptr, x_ptr, scalar, TILE: tl.constexpr, DTYPE: tl.constexpr
):
    pid = tl.program_id(0)
    tid = pid * TILE + tl.arange(0, TILE)
    x = tl.load(x_ptr + tid)
    # NaN-safety: under TRITONXPU_COMPARE_FUSION=1 the backend lowers ordered
    # `x <= s` (OLE) as NOT(OLT), which is true for NaN, while aten returns
    # False. IEEE `x <= s` == `s >= x`, and OGE lowers correctly (same trick
    # as less_equal_scalar), so swap the operands.
    r = scalar.to(DTYPE) >= x
    tl.store(out_ptr + tid, r.to(tl.int8))


@triton.jit
def le_scalar_native_masked_kernel(
    out_ptr, x_ptr, scalar, numel, TILE: tl.constexpr, DTYPE: tl.constexpr
):
    pid = tl.program_id(0)
    tid = pid * TILE + tl.arange(0, TILE)
    mask = tid < numel
    x = tl.load(x_ptr + tid, mask=mask)
    # Same NaN-safe operand swap as the unmasked kernel above.
    r = scalar.to(DTYPE) >= x
    tl.store(out_ptr + tid, r.to(tl.int8), mask=mask)


def _le_scalar_native(A, scalar, numel, masked, tile=_LE_SCALAR_FAST_TILE):
    # Single-kernel native compare: writes the bool result directly, so the
    # fp32 intermediate buffer and _copy_from pass of the saturating recipe
    # are gone. Env must be set before the (first) launch so the fusion pass
    # sees it at compile time (same set/del pattern as le above).
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
            le_scalar_native_masked_kernel[grid](
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
            le_scalar_native_kernel[grid](
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
# le_ (in-place x.le_(y)): saturating fp32 le = 1 - min(1,max(0,(x-y)*1e64))
# written back into x, no i1. In-place-safe config (DEFAULT isCloseMemoryAsync)
# avoids the noc-idle-timeout deadlock async-copy has under aliasing.
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
def le_func_tensor_inplace(x, y):
    t = (x.to(tl.float32) - y.to(tl.float32)) * 1.0e32
    t = t * 1.0e32
    t = tl.maximum(0.0, t)
    t = tl.minimum(1.0, t)
    return 1.0 - t


def le_(A, B):
    logger.debug("GEMS_KUNLUNXIN LE__TENSOR")
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
            and numel >= _LE_TENSOR_INPLACE_FAST_TILE * _LE_TENSOR_INPLACE_MIN_GRID
            and numel % _LE_TENSOR_INPLACE_FAST_TILE == 0
        ):
            # exact-multiple flat tiles: no mask at all; grid fixed.
            return _le_tensor_inplace_fast(A, B, numel)
        le_func_tensor_inplace(A, B, out0=A)
        return A
    # Everything else (non-float dtype, non-contiguous, ...) keeps the
    # original generic in-place path, behavior unchanged.
    return _generic_le_(A, B)


# in-place alias safety: the fast kernel writes into the SAME tensor it
# reads, so it must keep the DEFAULT isCloseMemoryAsync (True = async copy
# closed); passing False with in-place aliasing is the documented "noc idle
# timeout" deadlock, same as gt.py's in-place fast path note.
_LE_TENSOR_INPLACE_FAST_TILE = 131072
_LE_TENSOR_INPLACE_MIN_GRID = 128


@triton.jit
def le_tensor_inplace_fast_kernel(x_ptr, y_ptr, TILE: tl.constexpr):
    pid = tl.program_id(0)
    tid = pid * TILE + tl.arange(0, TILE)
    x = tl.load(x_ptr + tid)
    y = tl.load(y_ptr + tid)
    t = (x.to(tl.float32) - y.to(tl.float32)) * 1.0e32
    t = t * 1.0e32
    t = tl.maximum(0.0, t)
    t = tl.minimum(1.0, t)
    tl.store(x_ptr + tid, 1.0 - t)


def _le_tensor_inplace_fast(A, B, numel):
    grid = (numel // _LE_TENSOR_INPLACE_FAST_TILE,)
    le_tensor_inplace_fast_kernel[grid](
        A,
        B,
        TILE=_LE_TENSOR_INPLACE_FAST_TILE,
        num_warps=4,
        buffer_size_limit=8192,
        unroll_num=16,
        isCloseMemoryAsync=True,
    )
    return A


# ---------------------------------------------------------------------------
# le_scalar_ (in-place x.le_(scalar)): saturating fp32 recipe le = 1 - min(1,
# max(0, (x-s)*1e64)) written back into x, no i1. In-place-safe config (DEFAULT
# isCloseMemoryAsync) avoids the noc-idle-timeout deadlock that async-copy has
# under aliasing. Mirrors less_equal_scalar_ (same x <= s semantics).
@pointwise_dynamic(
    is_tensor=[True, False],
    promotion_methods=[(0, 1, "DEFAULT")],
    config=config_inplace_,
)
@triton.jit
def le_func_scalar_inplace(x, y):
    t = (x.to(tl.float32) - y) * 1.0e32
    t = t * 1.0e32
    t = tl.maximum(0.0, t)
    t = tl.minimum(1.0, t)
    return 1.0 - t


def le_scalar_(A, B):
    logger.debug("GEMS_KUNLUNXIN LE_SCALAR_")
    numel = A.numel()
    if (
        A.is_contiguous()
        and A.dtype in (torch.float16, torch.float32, torch.bfloat16)
        and float(B) == float(torch.tensor(float(B), dtype=A.dtype).item())
    ):
        if (
            A.dtype in (torch.float16, torch.float32)
            and numel >= _LE_SCALAR_INPLACE_FAST_TILE * _LE_SCALAR_INPLACE_MIN_GRID
            and numel % _LE_SCALAR_INPLACE_FAST_TILE == 0
        ):
            # exact-multiple flat tiles: no mask at all; grid fixed.
            return _le_scalar_inplace_fast(A, float(B))
        le_func_scalar_inplace(A, B, out0=A)
        return A
    return le_func_scalar(A, B, out0=A)


# in-place alias safety: the fast kernel writes into the SAME tensor it reads,
# so it must keep the DEFAULT isCloseMemoryAsync (True = async copy closed);
# passing False with in-place aliasing is the documented noc idle timeout
# deadlock, same as the le_ / gt_ in-place fast path note.
_LE_SCALAR_INPLACE_FAST_TILE = 131072
_LE_SCALAR_INPLACE_MIN_GRID = 128


@triton.jit
def le_scalar_inplace_fast_kernel(x_ptr, scalar, TILE: tl.constexpr):
    pid = tl.program_id(0)
    tid = pid * TILE + tl.arange(0, TILE)
    x = tl.load(x_ptr + tid)
    t = (x.to(tl.float32) - scalar) * 1.0e32
    t = t * 1.0e32
    t = tl.maximum(0.0, t)
    t = tl.minimum(1.0, t)
    tl.store(x_ptr + tid, 1.0 - t)


def _le_scalar_inplace_fast(A, scalar):
    grid = (A.numel() // _LE_SCALAR_INPLACE_FAST_TILE,)
    le_scalar_inplace_fast_kernel[grid](
        A,
        scalar,
        TILE=_LE_SCALAR_INPLACE_FAST_TILE,
        num_warps=4,
        buffer_size_limit=8192,
        unroll_num=16,
        isCloseMemoryAsync=True,
    )
    return A
