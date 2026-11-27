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
# Ascend implementation imported from the tuned operator batch
# host entry points below mirror the generic
# flag_gems names so the backend registrar can pick them up.

"""subtract_ : in-place ``A -= alpha * B`` on Ascend, self-contained Triton.

Semantics reproduced from the reference (``A.sub_(B, alpha=alpha)``):
  * the result is written back into ``A`` storage and ``A`` itself is returned
    (same object / same storage, so the mutation stays observable);
  * ``A``'s dtype is preserved on the output;
  * non-contiguous views are read *and* written through their real element
    strides, so the mutation lands in the storage slots of that view;
  * +/-inf, NaN and signed zeros follow IEEE float arithmetic (no fast-math
    flush, no reordering that could turn ``inf - finite`` into NaN).

Ascend tuning notes (all measured with the level=core harness)
  1. *Tile size was the whole 8x gap.*  The old kernel used BLOCK=4096 and one
     program per block: at (4096, 4096) fp16 that is 102 us vs torch's 19 us
     (0.13x).  The loop was memory-latency bound, not bandwidth bound -- it ran
     at ~100 Gelem/s for every dtype.  BLOCK=16384 with ~160 persistent
     programs (a grid-stride loop) reaches 0.72x on the same shape and ~1.03x
     on 2**30 elements.  fp32 cannot use 16384 (three fp32 streams are 192 KB
     of the 196 KB UB and the compiler reports ``ub overflow``), so fp32 keeps
     BLOCK=8192.
  2. *Small tensors want a small tile.*  4096 elements on a 16384 tile is one
     program on one core (0.86x); BLOCK=2048, two programs, is 1.2x.
  3. *No fp32 round-trip when it cannot help.*  ``A - alpha * B`` used to
     upcast both operands, but for fp32 operands that is a no-op and for 16-bit
     operands with ``alpha == 1`` it only costs UB: the native-dtype path is
     both accurate (one rounding instead of two) and half the tile footprint.
     16-bit dtypes keep the fp32 upcast when ``alpha != 1`` (an fp16
     ``alpha * b`` product would lose ~5e-4 relative, which is at the 1e-3
     test tolerance), and non-float dtypes keep it exactly as before.
  4. *Drop the multiply for alpha == 1* (a constexpr flag): 4% at 2**30.
"""

import torch
import triton
import triton.language as tl

_BLOCK_TINY = 512
_BLOCK_SMALL = 2048
_BLOCK_MID = 4096
_BLOCK_16 = 16384
_BLOCK_32 = 8192
_MAX_PROGRAMS = 160
_BLOCK_STRIDED = 512

_FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


@triton.jit
def _sub_flat(
    A, B, alpha, n, BLOCK: tl.constexpr, ONE: tl.constexpr, UPCAST: tl.constexpr
):
    # Grid-stride: the launch grid is capped at the device's 65535-block limit,
    # so very large tensors are walked by a persistent loop instead of a
    # 65536-program launch (which fails on hardware).
    pid = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    step = tl.num_programs(0) * BLOCK
    base = pid * BLOCK
    while base < n:
        offs = base + lane
        mask = offs < n
        a = tl.load(A + offs, mask=mask, other=0.0)
        b = tl.load(B + offs, mask=mask, other=0.0)
        if UPCAST:
            y = a.to(tl.float32) - alpha * b.to(tl.float32)
            tl.store(A + offs, y.to(A.dtype.element_ty), mask=mask)
        elif ONE:
            tl.store(A + offs, a - b, mask=mask)
        else:
            tl.store(A + offs, a - alpha * b, mask=mask)
        base += step


@triton.jit
def _sub_strided(
    A,
    B,
    alpha,
    n,
    SHAPE: tl.constexpr,
    SA: tl.constexpr,
    SB: tl.constexpr,
    IS_FLOAT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    step = tl.num_programs(0) * BLOCK
    base = pid * BLOCK
    while base < n:
        offs = base + tl.arange(0, BLOCK)
        mask = offs < n
        rem = offs
        oa = tl.zeros((BLOCK,), tl.int64)
        ob = tl.zeros((BLOCK,), tl.int64)
        for i in tl.static_range(len(SHAPE) - 1, -1, -1):
            coord = rem % SHAPE[i]
            rem = rem // SHAPE[i]
            oa += coord.to(tl.int64) * SA[i]
            ob += coord.to(tl.int64) * SB[i]
        a = tl.load(A + oa, mask=mask, other=0.0)
        b = tl.load(B + ob, mask=mask, other=0.0)
        if IS_FLOAT:
            y = a.to(tl.float32) - alpha * b.to(tl.float32)
        else:
            y = a - alpha * b
        tl.store(A + oa, y.to(A.dtype.element_ty), mask=mask)
        base += step


def _collapse(shape, sa, sb):
    """Drop size-1 axes and fold adjacent axes that are dense in both strides."""
    dims = []
    for i in range(len(shape) - 1, -1, -1):
        s = shape[i]
        if s == 1:
            continue
        if dims:
            inner = dims[-1]
            if (sa[i] == inner[0] * inner[1]) and (sb[i] == inner[0] * inner[2]):
                dims[-1] = (inner[0] * s, inner[1], inner[2])
                continue
        dims.append((s, sa[i], sb[i]))
    dims.reverse()
    return ([d[0] for d in dims], [d[1] for d in dims], [d[2] for d in dims])


_PLANS = {}


def _plan(dtype, n, one):
    """(block, grid, flags) for the contiguous in-place path."""
    is_float = dtype in _FLOAT_DTYPES
    # 16-bit operands with alpha != 1 need the fp32 upcast for accuracy; fp32
    # operands never gain from it; non-float dtypes keep the old behaviour.
    upcast = (not is_float) or (dtype is not torch.float32 and not one)
    big = _BLOCK_32 if upcast or dtype is torch.float32 else _BLOCK_16
    if n <= _BLOCK_SMALL:
        block = _BLOCK_TINY
    elif n <= 4 * _BLOCK_MID:
        block = _BLOCK_SMALL
    elif n <= 262144:
        block = _BLOCK_MID
    else:
        block = big
    grid = min((n + block - 1) // block, _MAX_PROGRAMS)
    return block, grid, upcast


def _launch_generic(A, B, alpha, n):
    """Non-contiguous / broadcasting fallback: real element strides, in place."""
    is_float = A.dtype in _FLOAT_DTYPES
    one = (float(alpha) == 1.0) if is_float else (alpha == 1)
    if A.is_contiguous() and B.is_contiguous() and A.shape == B.shape:
        key = (A.dtype, n, one)
        plan = _PLANS.get(key)
        if plan is None:
            plan = _PLANS[key] = _plan(A.dtype, n, one)
        block, grid, upcast = plan
        if is_float:
            _sub_flat[(grid,)](
                A, B, float(alpha), n, BLOCK=block, ONE=one, UPCAST=upcast
            )
        else:
            _sub_flat[(grid,)](
                A, B, float(alpha), n, BLOCK=block, ONE=False, UPCAST=True
            )
        return
    b_off = A.dim() - B.dim()
    bshape = B.shape
    bstride = B.stride()
    sb_full = [0] * A.dim()
    for d in range(A.dim()):
        bd = d - b_off
        if bd >= 0 and bshape[bd] != 1:
            sb_full[d] = bstride[bd]
    shape, sa, sb = _collapse(list(A.shape), list(A.stride()), sb_full)
    if not shape:
        shape, sa, sb = [1], [0], [0]
    _sub_strided[(min((n + _BLOCK_STRIDED - 1) // _BLOCK_STRIDED, 65535),)](
        A,
        B,
        float(alpha) if is_float else int(alpha),
        n,
        SHAPE=tuple(shape),
        SA=tuple(sa),
        SB=tuple(sb),
        IS_FLOAT=is_float,
        BLOCK=_BLOCK_STRIDED,
    )


def run(A, B, *, alpha=1):
    n = A.numel()
    if n == 0:
        return A
    _launch_generic(A, B, alpha, n)
    return A


# --------------------------------------------------------------------------- #
# FlagGems entry points
# --------------------------------------------------------------------------- #
def subtract_(*args, **kwargs):
    return run(*args, **kwargs)
