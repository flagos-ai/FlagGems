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

"""Kunlunxin (XPU) overrides for the ``_foreach_*`` scalar-family operators.

The generic ``flag_gems.ops._foreach_binary`` executor feeds every element
through a base pointer loaded from a device-side metadata array.  On XPU that
indirection defeats the block-DMA analysis: ``loaded_ptr + idx`` lowers to a
per-element gather (``tensor<1024x!tt.ptr>``, ``fixedStride=INT32_MIN``), which
measures ~0.6 ns/elem -- ~100x slower than the vendor's per-tensor clamp
kernels on a 16.7M-element list (164 ms vs 1.2 ms).

These overrides keep the pointer a *kernel argument* so the XPU backend can
prove the accesses are contiguous (stride=1, block-DMA path) and use:

* ``_foreach16_kernel`` -- one launch processes up to 16 equal-numel tensors
  (the common list shapes), each program walking the same chunk of all tensors
  via ``NT``-guarded unrolled chains.  Launch-bound small shapes win here.
* ``_foreach1_kernel`` -- one launch per tensor for large or heterogeneous
  lists, using the same 12-CTA / big-tile / ``unroll_num`` recipe as the tuned
  ``clamp_max`` pointwise kernel (which matches native on 16.7M elements).

Only the *scalar* (``.Scalar`` / ``.ScalarList``) overloads are overridden
here; ``.List`` / ``.Tensor`` keep the generic multi-tensor executor.
"""

import logging
from typing import Any, List, Optional, Sequence

import torch
import triton
import triton.language as tl
from _kunlunxin.utils.codegen_config_utils import CodeGenConfig
from triton import knobs
from triton.runtime import driver

from flag_gems.runtime import torch_device_fn
from flag_gems.utils.foreach import (
    check_dtype_supported,
    check_tensor_list,
    has_internal_overlap,
    tl_dtype,
)

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# elementwise math (bf16 handled like _kunlunxin/ops/minimum.py: the bf16
# load/convert path is miscompiled on this backend, so compute in f32)
# ---------------------------------------------------------------------------


@triton.jit
def _fmin(a, s):
    if a.dtype == tl.bfloat16:
        return tl.minimum(a.to(tl.float32), s.to(tl.float32))
    return tl.minimum(a, s)


@triton.jit
def _fmax(a, s):
    if a.dtype == tl.bfloat16:
        return tl.maximum(a.to(tl.float32), s.to(tl.float32))
    return tl.maximum(a, s)


@triton.jit
def _fadd(a, s):
    # ATen defines bool addition as logical or.
    if a.dtype == tl.int1:
        return a | s
    return a + s


@triton.jit
def _fsub(a, s):
    return a - s


@triton.jit
def _fmul(a, s):
    if a.dtype == tl.int1:
        return a & s
    return a * s


@triton.jit
def _fdiv(a, s):
    return a / s


@triton.jit
def _fpow(a, s):
    if a.dtype == tl.float64:
        return tl.exp2(tl.log2(a) * s)
    return tl.exp2(tl.log2(a.to(tl.float32)) * s.to(tl.float32))


@triton.jit
def _fpow_rev(a, s):
    # pow.ScalarAndTensor: ``a`` is the exponent list element, ``s`` the base.
    if a.dtype == tl.float64:
        return tl.exp2(tl.log2(s) * a)
    return tl.exp2(tl.log2(s.to(tl.float32)) * a.to(tl.float32))


# ---------------------------------------------------------------------------
# bf16 large-shape route
#
# XPU3 has no bf16 ALU and no bf16 convert instruction (stated verbatim in the
# vendor compiler, triton/backends/xpu/compiler.py); the backend is supposed to
# promote bf16 arithmetic to f32 and fuse the narrowing into the LM-boundary
# load/store.  In a hand-written kernel that fusion does not fire: every
# formulation of ``tl.minimum`` on bf16 measures ~0.0088 ns/elem on a 16.7M
# tensor against a 0.0027 load+store floor (fp16 runs the identical kernel at
# 0.0027).  Eleven source forms, a 72-point launch-config sweep, a compiler-knob
# sweep and four kernel-structure variants all land on the same number.
#
# The ``pointwise_dynamic`` codegen path does *not* pay the penalty -- it runs
# bf16 at 0.0039 ns/elem -- so bf16 is routed through it.  fp16 and fp32 keep
# ``_foreach1_kernel``, which is faster than the codegen path for them:
#
#   16.7M, ns/elem     bf16            fp16            fp32
#   torch              0.0025          0.0024          0.0045
#   _foreach1_kernel   0.0089 (0.28x)  0.0027 (0.89x)  0.0048 (0.94x)
#   pointwise_dynamic  0.0039 (0.64x)  0.0038 (0.65x)  0.0074 (0.61x)
#
# The kernels below call the same ``_f*`` helpers as ``_foreach1_kernel``, so
# the arithmetic is bit-identical between the two routes.
# ---------------------------------------------------------------------------

# clamp.py's tuned recipe for a 1-in/1-out memory-bound elementwise op.
_PW_CONFIG = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    buffer_size_limit=4096,
    isCloseVectorization=False,
    unroll_num=8,
    kunlunAutoGrid=True,
)


def _pw(jit_fn):
    return pointwise_dynamic(
        is_tensor=[True, False],
        promotion_methods=[(0, 1, "DEFAULT")],
        config=_PW_CONFIG,
    )(jit_fn)


@_pw
@triton.jit
def _pw_min(a, s):
    return _fmin(a, s)


@_pw
@triton.jit
def _pw_max(a, s):
    return _fmax(a, s)


@_pw
@triton.jit
def _pw_add(a, s):
    return _fadd(a, s)


@_pw
@triton.jit
def _pw_sub(a, s):
    return _fsub(a, s)


@_pw
@triton.jit
def _pw_mul(a, s):
    return _fmul(a, s)


@_pw
@triton.jit
def _pw_div(a, s):
    return _fdiv(a, s)


@_pw
@triton.jit
def _pw_pow(a, s):
    return _fpow(a, s)


@_pw
@triton.jit
def _pw_pow_rev(a, s):
    return _fpow_rev(a, s)


_PW_FOR = {
    _fmin: _pw_min,
    _fmax: _pw_max,
    _fadd: _pw_add,
    _fsub: _pw_sub,
    _fmul: _pw_mul,
    _fdiv: _pw_div,
    _fpow: _pw_pow,
    _fpow_rev: _pw_pow_rev,
}


# ---------------------------------------------------------------------------
# kernels
# ---------------------------------------------------------------------------


@triton.jit
def _f16_body(
    a0,
    a1,
    a2,
    a3,
    a4,
    a5,
    a6,
    a7,
    a8,
    a9,
    a10,
    a11,
    a12,
    a13,
    a14,
    a15,
    o0,
    o1,
    o2,
    o3,
    o4,
    o5,
    o6,
    o7,
    o8,
    o9,
    o10,
    o11,
    o12,
    o13,
    o14,
    o15,
    s0,
    s1,
    s2,
    s3,
    s4,
    s5,
    s6,
    s7,
    s8,
    s9,
    s10,
    s11,
    s12,
    s13,
    s14,
    s15,
    n,
    fn: tl.constexpr,
    NT: tl.constexpr,
    OT: tl.constexpr,
    BLOCK: tl.constexpr,
    EVEN: tl.constexpr,
):
    """One launch, up to 16 tensors with equal numel.

    Every program walks the same ``BLOCK``-element chunk of each live tensor
    (``NT`` constexpr-guarded), so each pointer stays a kernel argument all the
    way into the load/store -- the block-DMA analysis can prove stride 1.
    """
    off = tl.program_id(0) * BLOCK
    idx = off + tl.arange(0, BLOCK)
    if EVEN:
        if NT > 0:
            tl.store(o0 + idx, fn(tl.load(a0 + idx).to(OT), s0.to(OT)).to(OT))
        if NT > 1:
            tl.store(o1 + idx, fn(tl.load(a1 + idx).to(OT), s1.to(OT)).to(OT))
        if NT > 2:
            tl.store(o2 + idx, fn(tl.load(a2 + idx).to(OT), s2.to(OT)).to(OT))
        if NT > 3:
            tl.store(o3 + idx, fn(tl.load(a3 + idx).to(OT), s3.to(OT)).to(OT))
        if NT > 4:
            tl.store(o4 + idx, fn(tl.load(a4 + idx).to(OT), s4.to(OT)).to(OT))
        if NT > 5:
            tl.store(o5 + idx, fn(tl.load(a5 + idx).to(OT), s5.to(OT)).to(OT))
        if NT > 6:
            tl.store(o6 + idx, fn(tl.load(a6 + idx).to(OT), s6.to(OT)).to(OT))
        if NT > 7:
            tl.store(o7 + idx, fn(tl.load(a7 + idx).to(OT), s7.to(OT)).to(OT))
        if NT > 8:
            tl.store(o8 + idx, fn(tl.load(a8 + idx).to(OT), s8.to(OT)).to(OT))
        if NT > 9:
            tl.store(o9 + idx, fn(tl.load(a9 + idx).to(OT), s9.to(OT)).to(OT))
        if NT > 10:
            tl.store(o10 + idx, fn(tl.load(a10 + idx).to(OT), s10.to(OT)).to(OT))
        if NT > 11:
            tl.store(o11 + idx, fn(tl.load(a11 + idx).to(OT), s11.to(OT)).to(OT))
        if NT > 12:
            tl.store(o12 + idx, fn(tl.load(a12 + idx).to(OT), s12.to(OT)).to(OT))
        if NT > 13:
            tl.store(o13 + idx, fn(tl.load(a13 + idx).to(OT), s13.to(OT)).to(OT))
        if NT > 14:
            tl.store(o14 + idx, fn(tl.load(a14 + idx).to(OT), s14.to(OT)).to(OT))
        if NT > 15:
            tl.store(o15 + idx, fn(tl.load(a15 + idx).to(OT), s15.to(OT)).to(OT))
    else:
        m = idx < n
        if NT > 0:
            tl.store(
                o0 + idx,
                fn(tl.load(a0 + idx, mask=m).to(OT), s0.to(OT)).to(OT),
                mask=m,
            )
        if NT > 1:
            tl.store(
                o1 + idx,
                fn(tl.load(a1 + idx, mask=m).to(OT), s1.to(OT)).to(OT),
                mask=m,
            )
        if NT > 2:
            tl.store(
                o2 + idx,
                fn(tl.load(a2 + idx, mask=m).to(OT), s2.to(OT)).to(OT),
                mask=m,
            )
        if NT > 3:
            tl.store(
                o3 + idx,
                fn(tl.load(a3 + idx, mask=m).to(OT), s3.to(OT)).to(OT),
                mask=m,
            )
        if NT > 4:
            tl.store(
                o4 + idx,
                fn(tl.load(a4 + idx, mask=m).to(OT), s4.to(OT)).to(OT),
                mask=m,
            )
        if NT > 5:
            tl.store(
                o5 + idx,
                fn(tl.load(a5 + idx, mask=m).to(OT), s5.to(OT)).to(OT),
                mask=m,
            )
        if NT > 6:
            tl.store(
                o6 + idx,
                fn(tl.load(a6 + idx, mask=m).to(OT), s6.to(OT)).to(OT),
                mask=m,
            )
        if NT > 7:
            tl.store(
                o7 + idx,
                fn(tl.load(a7 + idx, mask=m).to(OT), s7.to(OT)).to(OT),
                mask=m,
            )
        if NT > 8:
            tl.store(
                o8 + idx,
                fn(tl.load(a8 + idx, mask=m).to(OT), s8.to(OT)).to(OT),
                mask=m,
            )
        if NT > 9:
            tl.store(
                o9 + idx,
                fn(tl.load(a9 + idx, mask=m).to(OT), s9.to(OT)).to(OT),
                mask=m,
            )
        if NT > 10:
            tl.store(
                o10 + idx,
                fn(tl.load(a10 + idx, mask=m).to(OT), s10.to(OT)).to(OT),
                mask=m,
            )
        if NT > 11:
            tl.store(
                o11 + idx,
                fn(tl.load(a11 + idx, mask=m).to(OT), s11.to(OT)).to(OT),
                mask=m,
            )
        if NT > 12:
            tl.store(
                o12 + idx,
                fn(tl.load(a12 + idx, mask=m).to(OT), s12.to(OT)).to(OT),
                mask=m,
            )
        if NT > 13:
            tl.store(
                o13 + idx,
                fn(tl.load(a13 + idx, mask=m).to(OT), s13.to(OT)).to(OT),
                mask=m,
            )
        if NT > 14:
            tl.store(
                o14 + idx,
                fn(tl.load(a14 + idx, mask=m).to(OT), s14.to(OT)).to(OT),
                mask=m,
            )
        if NT > 15:
            tl.store(
                o15 + idx,
                fn(tl.load(a15 + idx, mask=m).to(OT), s15.to(OT)).to(OT),
                mask=m,
            )


# Two entry points over the same body.  Triton's host-side launch path costs
# ~15us fixed plus ~0.9us per kernel argument on XPU (measured: an empty 2-arg
# kernel 0.0151ms, an empty 35-arg kernel 0.0445ms, an empty 50-arg kernel
# 0.0499ms), and at the small list shapes this operator is entirely launch
# bound.  The ``.Scalar`` overloads -- and any ``.ScalarList`` that happens to
# repeat one value -- need only one scalar, so they take the 35-argument entry:
# measured 0.0454ms against 0.0638ms for the 50-argument one on 16 x 4096 fp16.


@triton.jit
def _foreach16_kernel(
    a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15,
    o0, o1, o2, o3, o4, o5, o6, o7, o8, o9, o10, o11, o12, o13, o14, o15,
    s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15,
    n,
    fn: tl.constexpr,
    NT: tl.constexpr,
    OT: tl.constexpr,
    BLOCK: tl.constexpr,
    EVEN: tl.constexpr,
):
    """Per-slot scalars (the general ``.ScalarList`` case)."""
    _f16_body(
        a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15,
        o0, o1, o2, o3, o4, o5, o6, o7, o8, o9, o10, o11, o12, o13, o14, o15,
        s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15,
        n, fn, NT, OT, BLOCK, EVEN,
    )


@triton.jit
def _foreach16s_kernel(
    a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15,
    o0, o1, o2, o3, o4, o5, o6, o7, o8, o9, o10, o11, o12, o13, o14, o15,
    s,
    n,
    fn: tl.constexpr,
    NT: tl.constexpr,
    OT: tl.constexpr,
    BLOCK: tl.constexpr,
    EVEN: tl.constexpr,
):
    """One scalar shared by every slot (the ``.Scalar`` case)."""
    _f16_body(
        a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15,
        o0, o1, o2, o3, o4, o5, o6, o7, o8, o9, o10, o11, o12, o13, o14, o15,
        s, s, s, s, s, s, s, s, s, s, s, s, s, s, s, s,
        n, fn, NT, OT, BLOCK, EVEN,
    )


@triton.jit
def _foreach1_kernel(
    a_ptr,
    o_ptr,
    s,
    num_tasks,
    tiles_per_cta,
    tile_size: tl.constexpr,
    fn: tl.constexpr,
    OT: tl.constexpr,
):
    """One tensor per launch: 12-CTA / big-tile pattern of ``clamp_max``."""
    pid = tl.program_id(0)
    num_ctas = tl.num_programs(0)
    for j in range(0, tiles_per_cta):
        tile_id = pid + j * num_ctas
        tid = tile_id * tile_size + tl.arange(0, tile_size)
        m = tid < num_tasks
        tl.store(
            o_ptr + tid,
            fn(tl.load(a_ptr + tid, mask=m).to(OT), s.to(OT)).to(OT),
            mask=m,
        )


# ---------------------------------------------------------------------------
# launch helpers
# ---------------------------------------------------------------------------

_K16_SLOTS = 16
# Equal-numel single-launch beats 16 per-tensor launches only while the launch
# cost dominates; above ~2^19 elements the 16 unrolled chains lose to the
# 12-CTA big-tile per-tensor kernel (measured crossover ~1M).
_K16_MAX_NUMEL = 1 << 19


def _labels(dtype: torch.dtype) -> str:
    return str(dtype).replace("torch.", "").capitalize()


def _pick_block(n: int) -> int:
    for block in (8192, 4096, 2048, 1024):
        if n % block == 0:
            return block, True
    return 1024, False


# ---------------------------------------------------------------------------
# cached launcher
#
# At the small list shapes this operator is pure launch overhead, and about half
# of it is Triton's host-side binding: on 16 x 4096 fp16 an empty 35-argument
# kernel costs 0.0445ms through ``JITFunction.run`` but 0.0215ms when the
# already-compiled kernel is invoked directly.  The binder re-derives the
# specialization (pointer divisibility, int ==1 / %16) for all 35 arguments on
# every call, then hashes it into a cache key.
#
# This takes the direct route, but only when the specialization is pinned to the
# one case the cached binary was compiled for: every pointer congruent to 0 mod
# 16 and ``n`` divisible by 16 and not 1.  A pointer that is 16-byte aligned is
# also 8-, 4- and 2-byte aligned, so this holds whatever divisibility the
# backend actually asks for; anything outside the guard falls back to
# ``JITFunction.run``.  Entries are keyed per device and per constexpr tuple,
# and a profiling launch hook also forces the fallback so its callbacks still
# fire.
# ---------------------------------------------------------------------------

_RAW_CACHE: dict = {}


def _raw_launch(jit_fn, grid, const_key, tensors, args, num_warps) -> bool:
    """Launch ``jit_fn`` bypassing the binder; False if the caller must not."""
    # The knob holds a ``HookChain``, never ``None``, and the chain defines no
    # ``__bool__``/``__len__`` -- so it is always truthy and ``is not None`` would
    # reject every call.  An empty ``calls`` list means nobody is profiling; any
    # other callable is treated as live hooks and forces the fallback.
    hook = knobs.runtime.launch_enter_hook
    if hook is not None and getattr(hook, "calls", True):
        return False
    for t in tensors:
        if t.data_ptr() % 16:
            return False
    device = torch_device_fn.current_device()
    key = (jit_fn, device, grid, num_warps) + const_key
    entry = _RAW_CACHE.get(key)
    if entry is None:
        kernel = jit_fn[grid](*args, num_warps=num_warps)
        stream = driver.active.get_current_stream(device)
        _RAW_CACHE[key] = (
            kernel.run,
            kernel.function,
            kernel.packed_metadata,
            stream,
        )
        return True
    run, function, metadata, stream = entry
    run(
        grid[0],
        1,
        1,
        stream,
        function,
        metadata,
        None,
        None,
        knobs.runtime.launch_exit_hook,
        *args,
    )
    return True


def _launch16(
    srcs: List[torch.Tensor],
    dsts: List[torch.Tensor],
    vals: List[float],
    fn,
    n: int,
) -> None:
    block, even = _pick_block(n)
    nt = len(srcs)
    nt16 = min(nt, _K16_SLOTS)
    # The dead ``if NT > i`` guards keep the padding slots out of the compiled
    # kernel; the argument values only have to be valid pointers.
    a = list(srcs[:nt16]) + [srcs[0]] * (_K16_SLOTS - nt16)
    # Output pads must not alias a live input (in-place ops write through o_i),
    # so a dedicated per-call scratch is used instead.
    o = list(dsts[:nt16]) + [_scratch_for(dsts[0], n)] * (_K16_SLOTS - nt16)
    svals = list(vals[:nt16]) + [0.0] * (_K16_SLOTS - nt16)
    grid = (triton.cdiv(n, block),)
    live = svals[:nt16]
    ot = tl_dtype(dsts[0].dtype)
    if live and live.count(live[0]) == nt16:
        # One scalar for every slot: 35 kernel arguments instead of 50.
        args = (*a, *o, live[0], n, fn, nt16, ot, block, even)
        if even and n % 16 == 0 and n != 1:
            if _raw_launch(
                _foreach16s_kernel, grid, (fn, nt16, ot, block, even), a + o, args, 8
            ):
                return
        _foreach16s_kernel[grid](*args, num_warps=8)
        return
    _foreach16_kernel[grid](
        *a,
        *o,
        *svals,
        n,
        fn,
        nt16,
        ot,
        block,
        even,
        num_warps=8,
    )


_SCRATCH: dict = {}


def _scratch_for(any_tensor: torch.Tensor, n: int) -> torch.Tensor:
    """Per-(device, n) scratch big enough for n elements (reused)."""
    dev = any_tensor.device
    key = (dev, any_tensor.dtype, n)
    buf = _SCRATCH.get(key)
    if buf is None or buf.numel() < n:
        buf = torch.empty(n, dtype=any_tensor.dtype, device=dev)
        _SCRATCH[key] = buf
    return buf


def _launch1(
    src: torch.Tensor,
    dst: torch.Tensor,
    val: float,
    fn,
) -> None:
    n = src.numel()
    # bf16 has no hardware ALU on XPU3 and the hand-written kernel below runs it
    # at 3.3x the fp16 rate; the codegen path does not pay that penalty.  See
    # the "bf16 large-shape route" block above for the measurements.
    if src.dtype is torch.bfloat16 and dst.dtype is torch.bfloat16:
        pw = _PW_FOR.get(fn)
        if pw is not None:
            pw(src, val, out0=dst)
            return
    # Cap the constexpr tile size: the previous next_pow2(cdiv(n, 12)) grew the
    # tile to ~2M elements for 16M-element tensors, and `tl.arange(0, tile)`
    # with a 2M constexpr tile compiles a gigantic kernel (register spill) that
    # runs ~250x slower than torch on the large shapes.  A fixed 64K tile keeps
    # one compiled kernel for every large size and lets the 12-CTA grid walk
    # more, smaller tiles instead.
    MAX_TILE = 1 << 16
    if n <= MAX_TILE:
        tile_size = triton.next_power_of_2(n)
        num_tiles = 1
    else:
        tile_size = MAX_TILE
        num_tiles = triton.cdiv(n, tile_size)
    num_ctas = 12 if num_tiles > 1 else 1
    tiles_per_cta = triton.cdiv(num_tiles, num_ctas)
    num_warps = 16 if tile_size >= 4096 else (8 if tile_size >= 2048 else 4)
    _foreach1_kernel[(num_ctas,)](
        src,
        dst,
        float(val),
        n,
        tiles_per_cta,
        tile_size,
        fn,
        _torch_to_tl(dst.dtype),
        num_warps=num_warps,
        unroll_num=8,
        buffer_size_limit=4096,
    )


def _torch_to_tl(dtype: torch.dtype):
    return tl_dtype(dtype)


def _launch(
    srcs: List[torch.Tensor],
    dsts: List[torch.Tensor],
    vals: List[float],
    fn,
) -> None:
    numels = [s.numel() for s in srcs]
    n = max(numels)
    if n == 0:
        return
    if len(srcs) <= _K16_SLOTS and all(x == n for x in numels) and n <= _K16_MAX_NUMEL:
        _launch16(srcs, dsts, vals, fn, n)
    else:
        for s, d, v in zip(srcs, dsts, vals):
            _launch1(s, d, v, fn)


# ---------------------------------------------------------------------------
# operator table
# ---------------------------------------------------------------------------

_FLOAT = (torch.float16, torch.bfloat16, torch.float32, torch.float64)
_INT = (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)
ARITH = frozenset(_FLOAT + _INT + (torch.bool,))
NO_BOOL = frozenset(_FLOAT + _INT)

# ATen's wording for the one operator that refuses bool with a dedicated
# message rather than the generic "not implemented" text.
_SUB_BOOL_ERROR = (
    "Subtraction, the `-` operator, with two bool tensors is not supported. "
    "Use the `^` or `logical_xor()` operator instead."
)

# (fn, allowed dtypes for the scalar overloads, dtype error text)
_OPS = {
    "clamp_max": (_fmin, NO_BOOL, None),
    "clamp_min": (_fmax, NO_BOOL, None),
    "minimum": (_fmin, NO_BOOL, None),
    "maximum": (_fmax, NO_BOOL, None),
    "sub": (_fsub, NO_BOOL, _SUB_BOOL_ERROR),
    "mul": (_fmul, ARITH, None),
    "add": (_fadd, ARITH, None),
    "div": (_fdiv, NO_BOOL, None),
    "pow": (_fpow, _FLOAT, None),
}


_RESULT_TYPE: dict = {}


def _result_type(t: torch.Tensor, v) -> torch.dtype:
    """``torch.result_type(t, v)`` memoised on ``(tensor dtype, scalar kind)``.

    A Python scalar enters type promotion as a *wrapped number*: only its
    category (bool / int / float / complex) participates, never its value, so
    the answer is a function of the two dtypes alone.  The lists this operator
    sees are typically 16 tensors of one dtype, and the ATen call costs about as
    much as a whole kernel launch at the small list shapes, so answer the
    repeats from a dict and keep ATen as the one-time oracle.
    """
    key = (t.dtype, type(v))
    out = _RESULT_TYPE.get(key)
    if out is None:
        out = _RESULT_TYPE[key] = torch.result_type(t, v)
    return out


def _resolve_scalars(tensors: Sequence[torch.Tensor], scalars: Any) -> List[float]:
    """Normalise the ``.Scalar`` / ``.ScalarList`` forms into a per-tensor list."""
    if isinstance(scalars, torch.Tensor):
        if scalars.numel() == 1:
            return [scalars.item()] * len(tensors)
        if len(tensors) != scalars.numel():
            raise RuntimeError(
                "Tensor list must have same number of elements as scalar list, "
                f"got {len(tensors)} and {scalars.numel()}"
            )
        return [float(v) for v in scalars.detach().cpu().tolist()]
    if isinstance(scalars, (list, tuple)):
        if len(tensors) != len(scalars):
            raise RuntimeError(
                "Tensor list must have same number of elements as scalar list, "
                f"got {len(tensors)} and {len(scalars)}"
            )
        return [float(v) for v in scalars]
    return [float(scalars)] * len(tensors)


def _apply_scalar(name: str, self, scalars, inplace: bool):
    """The ``.Scalar`` / ``.ScalarList`` overloads of one operator."""
    fn, allowed, dtype_error = _OPS[name]
    self = check_tensor_list(self)
    values = _resolve_scalars(self, scalars)

    results: List[Optional[torch.Tensor]] = [None] * len(self)
    writebacks: List[Any] = []
    # (device, in_dtype, out_dtype) -> [srcs, dsts, vals]
    buckets: dict = {}

    for i, (t, v) in enumerate(zip(self, values)):
        check_dtype_supported(t.dtype, allowed, dtype_error)
        out_dtype = _result_type(t, v)
        if t.is_contiguous():
            src = t
        else:
            src = t.contiguous()
        if inplace:
            if out_dtype != t.dtype:
                raise RuntimeError(
                    f"result type {_labels(out_dtype)} can't be cast to the "
                    f"desired output type {_labels(t.dtype)}"
                )
            if has_internal_overlap(t):
                raise RuntimeError(
                    "unsupported operation: more than one element of the "
                    "written-to tensor refers to a single memory location. "
                    "Please clone() the tensor before performing the operation."
                )
            dst = src
            if dst is not t:
                writebacks.append((t, dst))
            results[i] = t
        else:
            dst = torch.empty_like(src, dtype=out_dtype)
            results[i] = dst
        key = (t.device, src.dtype, out_dtype)
        bucket = buckets.get(key)
        if bucket is None:
            bucket = buckets[key] = ([], [], [])
        bucket[0].append(src)
        bucket[1].append(dst)
        bucket[2].append(float(v))

    for (_dev, _in_dt, _out_dt), (srcs, dsts, vals) in buckets.items():
        _launch(srcs, dsts, vals, fn)

    for dst_view, staged in writebacks:
        dst_view.copy_(staged)

    return None if inplace else results


# ---------------------------------------------------------------------------
# wrappers, one per ATen overload of this batch
# ---------------------------------------------------------------------------


def _make(name, inplace):
    def wrapper(self, scalar):
        logger.debug(
            "GEMS_KUNLUNXIN _FOREACH_%s%s", name.upper(), "_" if inplace else ""
        )
        return _apply_scalar(name, self, scalar, inplace)

    return wrapper


# ``(op, overload, inplace)`` -> module-level function, registered below.
_ENTRIES = [
    ("clamp_max", "Scalar", False),
    ("clamp_max", "Scalar", True),
    ("clamp_max", "ScalarList", False),
    ("clamp_max", "ScalarList", True),
    ("clamp_min", "Scalar", False),
    ("clamp_min", "Scalar", True),
    ("clamp_min", "ScalarList", False),
    ("clamp_min", "ScalarList", True),
    ("minimum", "Scalar", False),
    ("minimum", "Scalar", True),
    ("minimum", "ScalarList", False),
    ("minimum", "ScalarList", True),
    ("maximum", "Scalar", False),
    ("maximum", "Scalar", True),
    ("maximum", "ScalarList", False),
    ("maximum", "ScalarList", True),
    ("sub", "Scalar", True),
    ("sub", "ScalarList", True),
    ("mul", "Scalar", True),
    ("mul", "ScalarList", True),
    ("add", "Scalar", True),
    ("add", "ScalarList", True),
    ("div", "Scalar", True),
    ("div", "ScalarList", True),
    ("pow", "Scalar", True),
    ("pow", "ScalarList", True),
]

_globals = globals()

for _name, _ov, _inplace in _ENTRIES:
    _attr = f"_foreach_{_name}{'_' if _inplace else ''}_{_ov}"
    _fn = _make(_name, _inplace)
    _fn.__name__ = _attr
    _fn.__qualname__ = _attr
    _globals[_attr] = _fn

_clear = ("_name", "_ov", "_inplace", "_attr", "_fn")
for _v in _clear:
    _globals.pop(_v, None)


def registered_wrappers():
    """Map ATen overload key -> wrapper (mirrors the generic module's API)."""
    out = {}
    for name, ov, inplace in _ENTRIES:
        suffix = "_" if inplace else ""
        out[f"_foreach_{name}{suffix}.{ov}"] = _globals[f"_foreach_{name}{suffix}_{ov}"]
    return out


def _foreach_pow_ScalarAndTensor(self, exponent):
    """``pow.ScalarAndTensor``: ``self`` is the scalar base, ``exponent`` the list."""
    logger.debug("GEMS_KUNLUNXIN _FOREACH_POW_SCALARANDTENSOR")
    tensors = check_tensor_list(exponent)
    srcs, dsts, vals = [], [], []
    for t in tensors:
        check_dtype_supported(t.dtype, _FLOAT, None)
        src = t if t.is_contiguous() else t.contiguous()
        dst = torch.empty_like(src, dtype=t.dtype)
        srcs.append(src)
        dsts.append(dst)
        vals.append(float(self))
    _launch(srcs, dsts, vals, _fpow_rev)
    return dsts
