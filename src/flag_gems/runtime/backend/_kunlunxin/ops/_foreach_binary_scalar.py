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

from flag_gems.utils.foreach import (
    check_dtype_supported,
    check_tensor_list,
    has_internal_overlap,
)

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


# ---------------------------------------------------------------------------
# kernels
# ---------------------------------------------------------------------------


@triton.jit
def _foreach16_kernel(
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


def _launch16(
    srcs: List[torch.Tensor],
    dsts: List[torch.Tensor],
    vals: List[float],
    fn,
    n: int,
) -> None:
    from flag_gems.utils.foreach import tl_dtype as _torch_to_tl

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
    _foreach16_kernel[(triton.cdiv(n, block),)](
        *a,
        *o,
        *svals,
        n,
        fn,
        nt16,
        _torch_to_tl(dsts[0].dtype),
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
    if n <= 2048 * 64:
        num_ctas, num_tiles = 1, 1
    else:
        num_ctas, num_tiles = 12, 12
    tile_size = triton.next_power_of_2(triton.cdiv(n, num_tiles))
    if tile_size == 32768:
        tile_size = 16384
        num_tiles = triton.cdiv(n, tile_size)
        num_ctas = num_tiles
    num_tiles = triton.cdiv(n, tile_size)
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
    from flag_gems.utils.foreach import tl_dtype as _t

    return _t(dtype)


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
}


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
        out_dtype = torch.result_type(t, v)
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
    ("mul", "Scalar", True),
    ("add", "ScalarList", True),
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
