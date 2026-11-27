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

"""In-place foreach multiply-by-scalar (aten::_foreach_mul_.Scalar) in Triton.

Design notes
------------
* The whole contiguous, same-dtype, same-numel tensor list is handled by ONE
  kernel launch (grid = (blocks, n_tensors)); program_id(1) selects the tensor
  through a dynamic branch so every pointer keeps its own alignment
  specialization.
* The dominant cost for short lists is host dispatch, so the compiled kernel is
  cached and relaunched through the CompiledKernel launcher instead of
  JITFunction.run (no per-call arg binding / specialization / cache-key work).
"""

import logging
import math
import struct

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

_BLOCK = 2048
_BLOCK16 = 4096
_NUM_WARPS = 8
_MAX_D = 8
_MAX_T = 8
_FUSED_DTYPES = (torch.float32, torch.float16, torch.bfloat16)
_INT32_LIMIT = 2147000000


@triton.jit
def _mul_span(p, offs, m, s, EVEN: tl.constexpr):
    if EVEN:
        x = tl.load(p + offs)
        tl.store(p + offs, x * s)
    else:
        x = tl.load(p + offs, mask=m)
        tl.store(p + offs, x * s, mask=m)


@triton.jit
def _mul_same_kernel(
    p0,
    p1,
    p2,
    p3,
    p4,
    p5,
    p6,
    p7,
    n,
    s,
    BLOCK: tl.constexpr,
    NT: tl.constexpr,
    EVEN: tl.constexpr,
):
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)
    offs = pid0 * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    if pid1 == 0:
        _mul_span(p0, offs, m, s, EVEN)
    if pid1 == 1:
        _mul_span(p1, offs, m, s, EVEN)
    if pid1 == 2:
        _mul_span(p2, offs, m, s, EVEN)
    if pid1 == 3:
        _mul_span(p3, offs, m, s, EVEN)
    if pid1 == 4:
        _mul_span(p4, offs, m, s, EVEN)
    if pid1 == 5:
        _mul_span(p5, offs, m, s, EVEN)
    if pid1 == 6:
        _mul_span(p6, offs, m, s, EVEN)
    if pid1 == 7:
        _mul_span(p7, offs, m, s, EVEN)


@triton.jit
def _mul_flat_kernel(x_ptr, n, s_hi, s_lo, BLOCK: tl.constexpr, F64: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    if F64:
        s = s_hi.to(tl.float64) + s_lo.to(tl.float64)
    else:
        s = s_hi
    tl.store(x_ptr + offs, x * s, mask=mask)


@triton.jit
def _mul_strided_kernel(
    x_ptr,
    n,
    s_hi,
    s_lo,
    a0,
    a1,
    a2,
    a3,
    a4,
    a5,
    a6,
    a7,
    b0,
    b1,
    b2,
    b3,
    b4,
    b5,
    b6,
    b7,
    BLOCK: tl.constexpr,
    F64: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    r = offs
    addr = (r % a0) * b0
    r = r // a0
    addr += (r % a1) * b1
    r = r // a1
    addr += (r % a2) * b2
    r = r // a2
    addr += (r % a3) * b3
    r = r // a3
    addr += (r % a4) * b4
    r = r // a4
    addr += (r % a5) * b5
    r = r // a5
    addr += (r % a6) * b6
    r = r // a6
    addr += (r % a7) * b7
    x = tl.load(x_ptr + addr, mask=mask)
    if F64:
        s = s_hi.to(tl.float64) + s_lo.to(tl.float64)
    else:
        s = s_hi
    tl.store(x_ptr + addr, x * s, mask=mask)


_LAUNCH_CACHE = {}


def _stream(dev):
    return torch.cuda.current_stream(dev).cuda_stream


def _split_double(v):
    """Split a double into two fp32 halves that reconstruct it to ~2^-48."""
    v = float(v)
    try:
        hi = struct.unpack("<f", struct.pack("<f", v))[0]
    except (OverflowError, ValueError):
        hi = math.inf if v > 0 else -math.inf
    lo = v - hi
    if not math.isfinite(lo):
        lo = 0.0
    return hi, lo


def _collapse(t):
    """Innermost-first list of (size, stride) with unit dims dropped and
    adjacent mergeable dims fused."""
    sizes = list(t.shape)
    strides = list(t.stride())
    dims = [(int(s), int(st)) for s, st in zip(sizes, strides) if s != 1]
    merged = []
    for s, st in reversed(dims):
        if merged and merged[-1][1] * merged[-1][0] == st:
            ps, pst = merged.pop()
            merged.append((ps * s, st))
        else:
            merged.append((s, st))
    return merged


def _launch_single(t, s):
    n = t.numel()
    _mul_flat_kernel[(-(-n // _BLOCK),)](
        t, n, s, 0.0, BLOCK=_BLOCK, F64=False, num_warps=_NUM_WARPS
    )


def _launch_same(ts, n, s):
    """Fuse a list of contiguous same-dtype tensors that all hold n elements."""
    nt = len(ts)
    t0 = ts[0]
    dt = t0.dtype
    key = (t0.device, dt, nt, n, tuple([t.data_ptr() & 15 for t in ts]))
    ent = _LAUNCH_CACHE.get(key)
    if ent is None:
        blk = _BLOCK if dt is torch.float32 else _BLOCK16
        even = (n % blk) == 0
        mb = -(-n // blk)
        d = t0
        ck = _mul_same_kernel[(mb, nt)](
            ts[0],
            ts[1] if nt > 1 else d,
            ts[2] if nt > 2 else d,
            ts[3] if nt > 3 else d,
            ts[4] if nt > 4 else d,
            ts[5] if nt > 5 else d,
            ts[6] if nt > 6 else d,
            ts[7] if nt > 7 else d,
            n,
            s,
            BLOCK=blk,
            NT=nt,
            EVEN=even,
            num_warps=_NUM_WARPS,
        )
        _LAUNCH_CACHE[key] = (
            ck.run,
            ck.function,
            ck.packed_metadata,
            mb,
            nt,
            blk,
            even,
            t0.device.index,
            tuple([d] * (_MAX_T - nt)),
        )
        return
    run, fn, md, mb, gnt, blk, even, dev, pad = ent
    run(
        mb,
        gnt,
        1,
        _stream(dev),
        fn,
        md,
        None,
        None,
        None,
        *ts,
        *pad,
        n,
        s,
        blk,
        nt,
        even,
    )


def _run_one(t, scalar):
    n = t.numel()
    if n == 0:
        return
    f64 = t.dtype == torch.float64
    hi, lo = _split_double(scalar) if f64 else (float(scalar), 0.0)
    if t.is_contiguous():
        _mul_flat_kernel[(-(-n // _BLOCK),)](
            t, n, hi, lo, BLOCK=_BLOCK, F64=f64, num_warps=_NUM_WARPS
        )
        return
    dims = _collapse(t)
    if len(dims) > _MAX_D:
        # Extremely rare: walk the outermost dimension until rank fits.
        for i in range(t.shape[0]):
            _run_one(t[i], scalar)
        return
    shapes = [d[0] for d in dims] + [1] * (_MAX_D - len(dims))
    strides = [d[1] for d in dims] + [0] * (_MAX_D - len(dims))
    _mul_strided_kernel[(-(-n // _BLOCK),)](
        t,
        n,
        hi,
        lo,
        shapes[0],
        shapes[1],
        shapes[2],
        shapes[3],
        shapes[4],
        shapes[5],
        shapes[6],
        shapes[7],
        strides[0],
        strides[1],
        strides[2],
        strides[3],
        strides[4],
        strides[5],
        strides[6],
        strides[7],
        BLOCK=_BLOCK,
        F64=f64,
        num_warps=_NUM_WARPS,
    )


def _run_general(ts, s):
    buckets = None
    for t in ts:
        if t.numel() == 0:
            continue
        if t.is_contiguous() and t.dtype in _FUSED_DTYPES:
            if buckets is None:
                buckets = {}
            g = buckets.get(t.dtype)
            if g is None:
                buckets[t.dtype] = [t]
            else:
                g.append(t)
        else:
            _run_one(t, s)
    if buckets is None:
        return
    for lst in buckets.values():
        i = 0
        m = len(lst)
        while i < m:
            n = lst[i].numel()
            j = i + 1
            while j < m and j - i < _MAX_T and lst[j].numel() == n:
                j += 1
            if j - i == 1:
                _launch_single(lst[i], s)
            else:
                _launch_same(lst[i:j], n, s)
            i = j


def _foreach_mul__Scalar(self, scalar):
    logger.debug("GEMS_HYGON _FOREACH_MUL__SCALAR")
    ts = self if type(self) is list else list(self)
    if type(scalar) is not float:
        if isinstance(scalar, torch.Tensor):
            scalar = scalar.item()
        if isinstance(scalar, (list, tuple)):
            for i, t in enumerate(ts):
                _run_one(t, scalar[i])
            return None
    s = scalar if type(scalar) is float else float(scalar)
    nt = len(ts)
    if not nt:
        return None
    t0 = ts[0]
    d0 = t0.dtype
    n_el = t0.numel()
    if (
        n_el
        and n_el < _INT32_LIMIT
        and nt <= _MAX_T
        and (d0 is torch.float32 or d0 is torch.float16 or d0 is torch.bfloat16)
    ):
        for t in ts:
            if t.numel() != n_el or not t.is_contiguous() or t.dtype is not d0:
                _run_general(ts, s)
                return None
        _launch_same(ts, n_el, s)
        return None
    _run_general(ts, s)
    return None
