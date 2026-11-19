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

"""Fused in-place ``_foreach_round_`` (round-half-to-even) for tensor lists.

Semantics reproduced (ATen ``aten::_foreach_round_(Tensor(a!)[] self) -> ()``):

* every element of every tensor in the list is replaced by ``rint(x)``
  (round-half-to-even) in place, preserving storage, stride, shape and dtype;
* no return value;
* integral dtypes are a no-op (``round`` is the identity on integers);
* ``bool`` and complex inputs are rejected the way ATen rejects them.

Performance shape:

* one launch per (device, dtype) group of flat-addressable tensors instead of
  one launch per tensor -- the pointers and element counts travel as plain
  scalar kernel arguments, so no host->device metadata staging happens at all
  (the historical implementation paid a pageable H2D copy per launch);
* element counts and lane indices are 32-bit (64-bit only where pointers
  require it), so per-lane address arithmetic stays narrow;
* contiguity is checked in C first and non-dense layouts are still written
  through their real strides, so no ``contiguous()`` staging copy is needed.
"""

import logging

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice as _libdevice

logger = logging.getLogger(__name__)

_BLOCK = 1024
_BLOCK16 = 2048
_K = 8
_I32_MAX = 2147483647

_TL_DTYPE = {
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
    torch.float32: tl.float32,
    torch.float64: tl.float64,
}

# Bytes-normalized tile: 2048 two-byte elements and 1024 four-byte elements
# both put 32 B per lane at num_warps=4, i.e. two 16 B vector accesses per lane
# instead of one for the 16-bit dtypes.
_BLOCK_OF = {
    torch.float16: _BLOCK16,
    torch.bfloat16: _BLOCK16,
    torch.float32: _BLOCK,
    torch.float64: _BLOCK,
}

# round is the identity on integrals and accepted by ATen for all of them.
_INT_DTYPES = frozenset(
    (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
)

_OVERLAP_MSG = (
    "unsupported operation: more than one element of the written-to tensor "
    "refers to a single memory location. Please clone() the tensor before "
    "performing the operation."
)


@triton.jit
def _round_val(x):
    """rint(x) with the input dtype preserved (fp16/bf16 through fp32)."""
    if x.dtype == tl.float64:
        return _libdevice.rint(x)
    else:
        return _libdevice.rint(x.to(tl.float32)).to(x.dtype)


@triton.jit
def _round_flat_kernel(
    ptr,
    n,
    BLOCK: tl.constexpr,
    DT: tl.constexpr,
    WIDE: tl.constexpr,
    EVEN: tl.constexpr,
):
    if WIDE:
        idx = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    else:
        idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    p = ptr.to(tl.pointer_type(DT))
    if EVEN:
        # numel is a multiple of the tile: every lane is live, so the tile
        # needs neither a bounds compare nor a fill value.
        x = tl.load(p + idx)
        tl.store(p + idx, _round_val(x).to(DT))
    else:
        mask = idx < n
        x = tl.load(p + idx, mask=mask, other=0.0)
        tl.store(p + idx, _round_val(x).to(DT), mask=mask)


@triton.jit
def _round_fused8(
    p0,
    p1,
    p2,
    p3,
    p4,
    p5,
    p6,
    p7,
    n0,
    n1,
    n2,
    n3,
    n4,
    n5,
    n6,
    n7,
    BLOCK: tl.constexpr,
    DT: tl.constexpr,
):
    """One launch for up to eight same-dtype flat-addressable tensors.

    The grid is the concatenation of each tensor's own block range, so the
    program id selects its tensor through the cumulative block bounds.  Empty
    slots have ``n == 0``, which makes their block range empty, so the
    last-matching-slot rule below can never pick them for a live program.
    """
    pid = tl.program_id(0)
    c1 = tl.cdiv(n0, BLOCK)
    c2 = c1 + tl.cdiv(n1, BLOCK)
    c3 = c2 + tl.cdiv(n2, BLOCK)
    c4 = c3 + tl.cdiv(n3, BLOCK)
    c5 = c4 + tl.cdiv(n4, BLOCK)
    c6 = c5 + tl.cdiv(n5, BLOCK)
    c7 = c6 + tl.cdiv(n6, BLOCK)

    addr = p0.to(tl.int64)
    lim = n0.to(tl.int32)
    start = pid * BLOCK
    if pid >= c1:
        addr = p1.to(tl.int64)
        lim = n1.to(tl.int32)
        start = (pid - c1) * BLOCK
    if pid >= c2:
        addr = p2.to(tl.int64)
        lim = n2.to(tl.int32)
        start = (pid - c2) * BLOCK
    if pid >= c3:
        addr = p3.to(tl.int64)
        lim = n3.to(tl.int32)
        start = (pid - c3) * BLOCK
    if pid >= c4:
        addr = p4.to(tl.int64)
        lim = n4.to(tl.int32)
        start = (pid - c4) * BLOCK
    if pid >= c5:
        addr = p5.to(tl.int64)
        lim = n5.to(tl.int32)
        start = (pid - c5) * BLOCK
    if pid >= c6:
        addr = p6.to(tl.int64)
        lim = n6.to(tl.int32)
        start = (pid - c6) * BLOCK
    if pid >= c7:
        addr = p7.to(tl.int64)
        lim = n7.to(tl.int32)
        start = (pid - c7) * BLOCK

    idx = start + tl.arange(0, BLOCK)
    mask = idx < lim
    p = addr.to(tl.pointer_type(DT))
    x = tl.load(p + idx, mask=mask, other=0.0)
    tl.store(p + idx, _round_val(x).to(DT), mask=mask)


@triton.jit
def _round_uniform16(
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
    BLOCK: tl.constexpr,
    DT: tl.constexpr,
    EVEN: tl.constexpr,
):
    """One launch for a uniform list: equal numel, same dtype, flat layout.

    Axis 0 walks the tiles inside a tensor and axis 1 selects the tensor, so no
    division is needed to find the owner of a program, and the element count is
    shared by every slot.  ``t`` is uniform inside a program, so the pointer
    chain costs a handful of scalar compares per program.
    """
    t = tl.program_id(1)
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    addr = p0.to(tl.int64)
    if t >= 1:
        addr = p1.to(tl.int64)
    if t >= 2:
        addr = p2.to(tl.int64)
    if t >= 3:
        addr = p3.to(tl.int64)
    if t >= 4:
        addr = p4.to(tl.int64)
    if t >= 5:
        addr = p5.to(tl.int64)
    if t >= 6:
        addr = p6.to(tl.int64)
    if t >= 7:
        addr = p7.to(tl.int64)
    if t >= 8:
        addr = p8.to(tl.int64)
    if t >= 9:
        addr = p9.to(tl.int64)
    if t >= 10:
        addr = p10.to(tl.int64)
    if t >= 11:
        addr = p11.to(tl.int64)
    if t >= 12:
        addr = p12.to(tl.int64)
    if t >= 13:
        addr = p13.to(tl.int64)
    if t >= 14:
        addr = p14.to(tl.int64)
    if t >= 15:
        addr = p15.to(tl.int64)
    p = addr.to(tl.pointer_type(DT))
    if EVEN:
        x = tl.load(p + idx)
        tl.store(p + idx, _round_val(x).to(DT))
    else:
        mask = idx < n
        x = tl.load(p + idx, mask=mask, other=0.0)
        tl.store(p + idx, _round_val(x).to(DT), mask=mask)


@triton.jit
def _round_strided_kernel(
    ptr,
    n_chunks,
    s0,
    s1,
    s2,
    s3,
    s4,
    s5,
    s6,
    t0,
    t1,
    t2,
    t3,
    t4,
    t5,
    t6,
    inner_size,
    inner_stride,
    BLOCK: tl.constexpr,
    DT: tl.constexpr,
):
    """Write a non-dense (gappy) view in place through its real strides.

    One program covers ``BLOCK`` consecutive elements of the unit-stride inner
    axis; the outer coordinates are recovered from the program id once per
    program, never per element.
    """
    gid = tl.program_id(0)
    pid_o = gid // n_chunks
    pid_i = gid % n_chunks

    rem = pid_o
    off = 0
    q = rem % s0
    rem = rem // s0
    off += q.to(tl.int64) * t0.to(tl.int64)
    q = rem % s1
    rem = rem // s1
    off += q.to(tl.int64) * t1.to(tl.int64)
    q = rem % s2
    rem = rem // s2
    off += q.to(tl.int64) * t2.to(tl.int64)
    q = rem % s3
    rem = rem // s3
    off += q.to(tl.int64) * t3.to(tl.int64)
    q = rem % s4
    rem = rem // s4
    off += q.to(tl.int64) * t4.to(tl.int64)
    q = rem % s5
    rem = rem // s5
    off += q.to(tl.int64) * t5.to(tl.int64)
    q = rem % s6
    off += q.to(tl.int64) * t6.to(tl.int64)

    idx = pid_i.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = idx < inner_size
    pos = off + idx * inner_stride.to(tl.int64)
    p = ptr.to(tl.pointer_type(DT))
    x = tl.load(p + pos, mask=mask, other=0.0)
    tl.store(p + pos, _round_val(x).to(DT), mask=mask)


def _is_dense(t):
    """Flat-addressable: sorted strides (size>1 dims) pack a dense buffer."""
    if t.is_contiguous():
        return True
    if t.numel() <= 1:
        return True
    expected = 1
    items = sorted((s, sz) for s, sz in zip(t.stride(), t.shape) if sz != 1)
    for s, sz in items:
        if s != expected:
            return False
        expected *= sz
    return True


def _strided_layout(t):
    """(inner_size, inner_stride, outer dims, outer_numel) for a gappy view."""
    shape = t.shape
    stride = t.stride()
    rank = len(shape)
    inner = -1
    best = -1
    for d in range(rank):
        sz = shape[d]
        if sz == 1:
            continue
        st = stride[d]
        if st == 1 and sz > best:
            inner = d
            best = sz
    if inner < 0:
        best = -1
        for d in range(rank):
            sz = shape[d]
            if sz == 1:
                continue
            st = stride[d]
            if best < 0 or st < best:
                best = st
                inner = d
    outer = []
    outer_numel = 1
    for d in range(rank):
        if d == inner:
            continue
        sz = shape[d]
        if sz == 1:
            continue
        outer.append((sz, stride[d]))
        outer_numel *= sz
    return shape[inner], stride[inner], outer, outer_numel


def _launch_flat(ptrs, numels, tld, block, wide):
    """One launch per batch: single-tensor kernel, else the fused kernel."""
    m = len(ptrs)
    if m == 1:
        n0 = numels[0]
        _round_flat_kernel[(-(-n0 // block),)](
            ptrs[0], n0, block, tld, wide, (n0 % block) == 0
        )
        return
    if min(numels) == max(numels):
        # Uniform list: one launch for up to 16 tensors, tensor selected by
        # grid axis 1 and the element count shared by every slot.
        n = numels[0]
        even = (n % block) == 0
        chunks = -(-n // block)
        i = 0
        while i < m:
            bp = ptrs[i : i + 16]
            b = len(bp)
            if b < 16:
                bp = bp + [0] * (16 - b)
            _round_uniform16[(chunks, b)](*bp, n, block, tld, even)
            i += b
        return
    i = 0
    while i < m:
        bp = ptrs[i : i + _K]
        bn = numels[i : i + _K]
        b = len(bp)
        if b < _K:
            pad = _K - b
            bp = bp + [0] * pad
            bn = bn + [0] * pad
        total = 0
        for v in bn:
            total += -(-v // block)
        _round_fused8[(total,)](*bp, *bn, block, tld)
        i += b


def _launch_strided(t):
    inner_size, inner_stride, outer, outer_numel = _strided_layout(t)
    if len(outer) > 7:
        raise NotImplementedError("foreach_round_ strided path supports rank <= 8")
    dims = [(1, 0)] * 7
    for i, pair in enumerate(outer):
        dims[i] = pair
    block = _BLOCK_OF[t.dtype]
    n_chunks = -(-inner_size // block)
    (s0, t0), (s1, t1), (s2, t2), (s3, t3), (s4, t4), (s5, t5), (s6, t6) = dims
    _round_strided_kernel[(n_chunks * outer_numel,)](
        t.data_ptr(),
        n_chunks,
        s0,
        s1,
        s2,
        s3,
        s4,
        s5,
        s6,
        t0,
        t1,
        t2,
        t3,
        t4,
        t5,
        t6,
        inner_size,
        inner_stride,
        block,
        _TL_DTYPE[t.dtype],
    )


def _foreach_round_(self):
    logger.debug("GEMS_HYGON _FOREACH_ROUND_")
    if not isinstance(self, (list, tuple)):
        raise TypeError("argument 'self' must be tuple of Tensors")
    if len(self) == 0:
        raise RuntimeError("Tensor list must have at least one tensor.")

    tl_dtype_of = _TL_DTYPE.get
    int_dtypes = _INT_DTYPES
    groups = {}
    wide = []
    strided = []

    for t in self:
        if not isinstance(t, torch.Tensor):
            raise TypeError("argument 'self' must be tuple of Tensors")
        dt = t.dtype
        tld = tl_dtype_of(dt)
        if tld is None:
            if dt in int_dtypes:
                # round on integrals is the identity: nothing to write.
                continue
            raise RuntimeError(
                "\"foreach\" not implemented for '%s'" % str(dt).replace("torch.", "")
            )
        n = t.numel()
        if n == 0:
            continue
        if t.is_contiguous() or _is_dense(t):
            if n > _I32_MAX:
                # 64-bit lane indices are needed: run it on its own.
                wide.append((t, n, tld))
                continue
            key = (t.device, dt)
            g = groups.get(key)
            if g is None:
                groups[key] = ([t.data_ptr()], [n], tld)
            else:
                g[0].append(t.data_ptr())
                g[1].append(n)
            continue
        if torch._debug_has_internal_overlap(t) == 1:
            raise RuntimeError(_OVERLAP_MSG)
        strided.append(t)

    cur = torch.cuda.current_device() if (groups or wide or strided) else None
    if groups:
        for (dev, dt), (ptrs, numels, tld) in groups.items():
            block = _BLOCK_OF[dt]
            dev_idx = dev.index
            if dev_idx is not None and dev_idx != cur:
                with torch.cuda.device(dev):
                    _launch_flat(ptrs, numels, tld, block, False)
            else:
                _launch_flat(ptrs, numels, tld, block, False)

    for t, n, tld in wide:
        dev = t.device
        dev_idx = dev.index
        if dev_idx is not None and dev_idx != cur:
            with torch.cuda.device(dev):
                _launch_flat([t.data_ptr()], [n], tld, _BLOCK_OF[t.dtype], True)
        else:
            _launch_flat([t.data_ptr()], [n], tld, _BLOCK_OF[t.dtype], True)

    for t in strided:
        dev = t.device
        dev_idx = dev.index
        if dev_idx is not None and dev_idx != cur:
            with torch.cuda.device(dev):
                _launch_strided(t)
        else:
            _launch_strided(t)

    return None
