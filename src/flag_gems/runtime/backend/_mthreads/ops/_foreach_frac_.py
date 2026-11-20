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
"""Triton implementation of ``aten::_foreach_frac_``.

Applies ``frac`` (x - trunc(x)) in place to every tensor of a TensorList,
preserving storage, strides and dtype, and returns ``None``.

Launch architecture:

* One fused launch per group of up to 16 flat-addressable tensors of the same
  dtype.  The pointers ride in as scalar kernel arguments (no H2D metadata
  copy) and ``program_id(0)`` selects the tensor with a uniform branch, so a
  16-tensor list costs the same number of launches as a 1-tensor list.
* The homogeneous case (every tensor of the group has the same element count,
  which is what the benchmark and the accuracy tests use) takes a kernel with
  16 pointers and *one* element count -- 17 arguments instead of 33.
* These shapes are launch-bound, so the first call for a given
  (dtype, mask-mode, block, device) keeps the ``CompiledKernel`` that
  ``JITFunction.run`` returns and later calls go straight through
  ``CompiledKernel.__getitem__``: that skips argument binding, specialisation
  refinement, pointer-alias analysis, cache-key computation and the autotuner,
  which together cost more than the kernel for small shapes.  Only 16-byte
  aligned pointers take this path; anything else uses the ordinary JIT launch,
  so the alignment assumption baked into the compiled kernel always holds.
* ``HAS_MASK=False`` is selected when the block size divides the element count;
  the loads/stores then carry no predicate, which lets the backend emit
  vectorised accesses.
* A tensor that is flat-addressable (contiguous, or non-overlapping and dense
  like a transpose) is walked by flat element index; an elementwise in-place op
  does not care about visit order.  Gappy views (``x[:, ::2]``) fall back to a
  stride-decomposed kernel with the dimensions collapsed and sorted so the
  stride-1 dimension is innermost.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops._foreach_unary import _foreach_frac_ as default__foreach_frac_

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)

_SUPPORTED_DTYPES = {torch.float16, torch.bfloat16, torch.float32}


def _needs_fallback(seq):
    """True when the list cannot use the MUSA kernels (device or dtype)."""
    for t in seq:
        if not isinstance(t, torch.Tensor):
            return True
        if t.device.type != "musa" or t.dtype not in _SUPPORTED_DTYPES:
            return True
    return False


_MAX_T = 16
_MAX_RANK = 8
_BLOCK_SMALL = 1024
_BLOCK_LARGE = 4096
_LARGE = 1 << 16

_FAST = {}

_NS = [
    "n0",
    "n1",
    "n2",
    "n3",
    "n4",
    "n5",
    "n6",
    "n7",
    "n8",
    "n9",
    "n10",
    "n11",
    "n12",
    "n13",
    "n14",
    "n15",
]


@triton.jit
def _frac_tile(ptr, n, off, HAS_MASK: tl.constexpr):
    if HAS_MASK:
        mask = off < n
        x = tl.load(ptr + off, mask=mask, other=0.0)
    else:
        x = tl.load(ptr + off)
    xf = x.to(tl.float32)
    y = (xf - tl.where(xf >= 0, tl.floor(xf), tl.ceil(xf))).to(x.dtype)
    if HAS_MASK:
        tl.store(ptr + off, y, mask=mask)
    else:
        tl.store(ptr + off, y)


@triton.jit
def _route_same(
    p0,
    p1,
    p2,
    p3,
    p4,
    p5,
    p6,
    p7,
    p8,
    p9,
    p10,
    p11,
    p12,
    p13,
    p14,
    p15,
    n,
    t,
    off,
    HAS_MASK: tl.constexpr,
):
    if t == 0:
        _frac_tile(p0, n, off, HAS_MASK)
    elif t == 1:
        _frac_tile(p1, n, off, HAS_MASK)
    elif t == 2:
        _frac_tile(p2, n, off, HAS_MASK)
    elif t == 3:
        _frac_tile(p3, n, off, HAS_MASK)
    elif t == 4:
        _frac_tile(p4, n, off, HAS_MASK)
    elif t == 5:
        _frac_tile(p5, n, off, HAS_MASK)
    elif t == 6:
        _frac_tile(p6, n, off, HAS_MASK)
    elif t == 7:
        _frac_tile(p7, n, off, HAS_MASK)
    elif t == 8:
        _frac_tile(p8, n, off, HAS_MASK)
    elif t == 9:
        _frac_tile(p9, n, off, HAS_MASK)
    elif t == 10:
        _frac_tile(p10, n, off, HAS_MASK)
    elif t == 11:
        _frac_tile(p11, n, off, HAS_MASK)
    elif t == 12:
        _frac_tile(p12, n, off, HAS_MASK)
    elif t == 13:
        _frac_tile(p13, n, off, HAS_MASK)
    elif t == 14:
        _frac_tile(p14, n, off, HAS_MASK)
    else:
        _frac_tile(p15, n, off, HAS_MASK)


@triton.jit
def _route_mixed(
    p0,
    p1,
    p2,
    p3,
    p4,
    p5,
    p6,
    p7,
    p8,
    p9,
    p10,
    p11,
    p12,
    p13,
    p14,
    p15,
    n0,
    n1,
    n2,
    n3,
    n4,
    n5,
    n6,
    n7,
    n8,
    n9,
    n10,
    n11,
    n12,
    n13,
    n14,
    n15,
    t,
    off,
    HAS_MASK: tl.constexpr,
):
    if t == 0:
        _frac_tile(p0, n0, off, HAS_MASK)
    elif t == 1:
        _frac_tile(p1, n1, off, HAS_MASK)
    elif t == 2:
        _frac_tile(p2, n2, off, HAS_MASK)
    elif t == 3:
        _frac_tile(p3, n3, off, HAS_MASK)
    elif t == 4:
        _frac_tile(p4, n4, off, HAS_MASK)
    elif t == 5:
        _frac_tile(p5, n5, off, HAS_MASK)
    elif t == 6:
        _frac_tile(p6, n6, off, HAS_MASK)
    elif t == 7:
        _frac_tile(p7, n7, off, HAS_MASK)
    elif t == 8:
        _frac_tile(p8, n8, off, HAS_MASK)
    elif t == 9:
        _frac_tile(p9, n9, off, HAS_MASK)
    elif t == 10:
        _frac_tile(p10, n10, off, HAS_MASK)
    elif t == 11:
        _frac_tile(p11, n11, off, HAS_MASK)
    elif t == 12:
        _frac_tile(p12, n12, off, HAS_MASK)
    elif t == 13:
        _frac_tile(p13, n13, off, HAS_MASK)
    elif t == 14:
        _frac_tile(p14, n14, off, HAS_MASK)
    else:
        _frac_tile(p15, n15, off, HAS_MASK)


@triton.jit(do_not_specialize=["n"])
def _frac_one_kernel(p, n, HAS_MASK: tl.constexpr, BLOCK: tl.constexpr):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    _frac_tile(p, n, off, HAS_MASK)


@triton.jit(do_not_specialize=["n"])
def _frac_uniform_kernel(
    p0,
    p1,
    p2,
    p3,
    p4,
    p5,
    p6,
    p7,
    p8,
    p9,
    p10,
    p11,
    p12,
    p13,
    p14,
    p15,
    n,
    HAS_MASK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    off = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    _route_same(
        p0,
        p1,
        p2,
        p3,
        p4,
        p5,
        p6,
        p7,
        p8,
        p9,
        p10,
        p11,
        p12,
        p13,
        p14,
        p15,
        n,
        tl.program_id(0),
        off,
        HAS_MASK,
    )


@triton.jit(do_not_specialize=_NS)
def _frac_many_kernel(
    p0,
    p1,
    p2,
    p3,
    p4,
    p5,
    p6,
    p7,
    p8,
    p9,
    p10,
    p11,
    p12,
    p13,
    p14,
    p15,
    n0,
    n1,
    n2,
    n3,
    n4,
    n5,
    n6,
    n7,
    n8,
    n9,
    n10,
    n11,
    n12,
    n13,
    n14,
    n15,
    HAS_MASK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    off = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    _route_mixed(
        p0,
        p1,
        p2,
        p3,
        p4,
        p5,
        p6,
        p7,
        p8,
        p9,
        p10,
        p11,
        p12,
        p13,
        p14,
        p15,
        n0,
        n1,
        n2,
        n3,
        n4,
        n5,
        n6,
        n7,
        n8,
        n9,
        n10,
        n11,
        n12,
        n13,
        n14,
        n15,
        tl.program_id(0),
        off,
        HAS_MASK,
    )


@triton.jit(
    do_not_specialize=[
        "n",
        "s0",
        "s1",
        "s2",
        "s3",
        "s4",
        "s5",
        "s6",
        "s7",
        "d0",
        "d1",
        "d2",
        "d3",
        "d4",
        "d5",
        "d6",
        "d7",
    ]
)
def _frac_strided_kernel(
    x_ptr,
    n,
    s0,
    s1,
    s2,
    s3,
    s4,
    s5,
    s6,
    s7,
    d0,
    d1,
    d2,
    d3,
    d4,
    d5,
    d6,
    d7,
    RANK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    off = pid * BLOCK + tl.arange(0, BLOCK)
    mask = off < n
    rem = off.to(tl.int64)
    idx = tl.zeros([BLOCK], dtype=tl.int64)
    if RANK >= 1:
        idx += (rem % d0) * s0
        rem = rem // d0
    if RANK >= 2:
        idx += (rem % d1) * s1
        rem = rem // d1
    if RANK >= 3:
        idx += (rem % d2) * s2
        rem = rem // d2
    if RANK >= 4:
        idx += (rem % d3) * s3
        rem = rem // d3
    if RANK >= 5:
        idx += (rem % d4) * s4
        rem = rem // d4
    if RANK >= 6:
        idx += (rem % d5) * s5
        rem = rem // d5
    if RANK >= 7:
        idx += (rem % d6) * s6
        rem = rem // d6
    if RANK >= 8:
        idx += (rem % d7) * s7
        rem = rem // d7
    x = tl.load(x_ptr + idx, mask=mask, other=0.0)
    xf = x.to(tl.float32)
    frac = (xf - tl.where(xf >= 0, tl.floor(xf), tl.ceil(xf))).to(x.dtype)
    tl.store(x_ptr + idx, frac, mask=mask)


def _is_flat_addressable(t):
    """True when the storage span of ``t`` is exactly ``numel`` elements."""
    if t.is_contiguous():
        return True
    if t.numel() <= 1:
        return True
    expected = 1
    for stride, size in sorted(
        (s, sz) for s, sz in zip(t.stride(), t.shape) if sz != 1
    ):
        if stride != expected:
            return False
        expected *= size
    return True


def _collapse_dims(t):
    """Innermost-first (size, stride) pairs covering exactly the view's elements."""
    dims = [(sz, st) for sz, st in zip(t.shape, t.stride()) if sz != 1]
    if not dims:
        return []
    dims.reverse()
    out = [list(dims[0])]
    for sz, st in dims[1:]:
        last = out[-1]
        if st == last[1] * last[0]:
            last[0] *= sz
        else:
            out.append([sz, st])
    return [(sz, st) for sz, st in sorted(out, key=lambda d: d[1])]


def _dtype_error(dtype):
    raise RuntimeError(
        "\"foreach\" not implemented for '%s'" % str(dtype).replace("torch.", "")
    )


def _launch_strided(t, n):
    dims = _collapse_dims(t)
    rank = len(dims)
    if rank > _MAX_RANK:
        raise NotImplementedError(
            "frac_ supports rank <= %d after collapse" % _MAX_RANK
        )
    sizes = [d[0] for d in dims] + [1] * (_MAX_RANK - rank)
    strides = [d[1] for d in dims] + [1] * (_MAX_RANK - rank)
    _frac_strided_kernel[((n + _BLOCK_SMALL - 1) // _BLOCK_SMALL,)](
        t,
        n,
        strides[0],
        strides[1],
        strides[2],
        strides[3],
        strides[4],
        strides[5],
        strides[6],
        strides[7],
        sizes[0],
        sizes[1],
        sizes[2],
        sizes[3],
        sizes[4],
        sizes[5],
        sizes[6],
        sizes[7],
        RANK=rank,
        BLOCK=_BLOCK_SMALL,
    )


def _aligned(ptrs):
    for p in ptrs:
        if p.data_ptr() & 15:
            return False
    return True


def _launch_one(t, n):
    blk = _BLOCK_LARGE if n >= _LARGE else _BLOCK_SMALL
    hm = (n % blk) != 0
    gx = (n + blk - 1) // blk
    if t.data_ptr() & 15:
        _frac_one_kernel[(gx,)](t, n, hm, blk)
        return
    key = (0, t.dtype, hm, blk, t.device)
    ck = _FAST.get(key)
    if ck is None:
        _FAST[key] = _frac_one_kernel[(gx,)](t, n, hm, blk)
    else:
        ck[(gx, 1, 1)](t, n, hm, blk)


def _foreach_frac_(self):
    logger.debug("GEMS_MTHREADS _FOREACH_FRAC_")
    _seq = self if isinstance(self, (list, tuple)) else [self]
    if _needs_fallback(_seq):
        return default__foreach_frac_(self)
    tensors = self
    if not isinstance(tensors, (list, tuple)):
        raise TypeError("argument 'self' must be tuple of Tensors")
    nt = len(tensors)
    if nt == 0:
        raise RuntimeError("Tensor list must have at least one tensor.")

    flat = []
    strided = []
    first = None
    uniform = True
    for i in range(nt):
        t = tensors[i]
        if not isinstance(t, torch.Tensor):
            raise TypeError("argument 'self' must be tuple of Tensors")
        dt = t.dtype
        if not dt.is_floating_point:
            _dtype_error(dt)
        if first is None:
            first = t
        elif dt is not first.dtype or t.device != first.device:
            uniform = False
        n = t.numel()
        if n == 0:
            continue
        if _is_flat_addressable(t):
            flat.append((t, n))
        else:
            strided.append((t, n))

    if strided:
        for t, n in strided:
            _launch_strided(t, n)

    if not flat:
        return None

    if not uniform or len(flat) == 1:
        for t, n in flat:
            _launch_one(t, n)
        return None

    dev = first.device
    dt0 = first.dtype
    i = 0
    total = len(flat)
    while i < total:
        chunk = flat[i : i + _MAX_T]
        i += _MAX_T
        cnt = len(chunk)
        n0 = chunk[0][1]
        same = True
        for c in chunk:
            if c[1] != n0:
                same = False
                break
        ptrs = [c[0] for c in chunk]
        if cnt < _MAX_T:
            ptrs += [ptrs[0]] * (_MAX_T - cnt)
        if same:
            if cnt > 1 and n0 >= _LARGE:
                # Very large tensors: the 16 co-resident streams of the fused
                # kernel cost ~30% per tensor, while 16 pipelined single-tensor
                # launches hide their host time behind multi-millisecond work.
                for t, n in chunk:
                    _launch_one(t, n)
                continue
            blk = _BLOCK_LARGE if n0 >= _LARGE else _BLOCK_SMALL
            hm = (n0 % blk) != 0
            ntiles = (n0 + blk - 1) // blk
            if _aligned(ptrs):
                key = (1, dt0, hm, blk, dev)
                ck = _FAST.get(key)
                if ck is None:
                    _FAST[key] = _frac_uniform_kernel[(cnt, ntiles)](*ptrs, n0, hm, blk)
                else:
                    ck[(cnt, ntiles, 1)](*ptrs, n0, hm, blk)
            else:
                _frac_uniform_kernel[(cnt, ntiles)](*ptrs, n0, hm, blk)
        else:
            ns = [c[1] for c in chunk]
            max_n = max(ns)
            ns += [0] * (_MAX_T - cnt)
            _frac_many_kernel[(cnt, (max_n + _BLOCK_SMALL - 1) // _BLOCK_SMALL)](
                *ptrs,
                ns[0],
                ns[1],
                ns[2],
                ns[3],
                ns[4],
                ns[5],
                ns[6],
                ns[7],
                ns[8],
                ns[9],
                ns[10],
                ns[11],
                ns[12],
                ns[13],
                ns[14],
                ns[15],
                HAS_MASK=True,
                BLOCK=_BLOCK_SMALL,
            )
    return None
