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

"""Triton implementation of aten::_foreach_round_ (in-place list round).

Semantics (verified against ATen on the target):
  * every element of every tensor in the list is replaced by round-half-to-even
    of itself, in place, keeping storage / strides / dtype / object identity;
  * integral dtypes are accepted and left unchanged;
  * bool and complex dtypes raise (NotImplementedError is a RuntimeError);
  * an empty list raises, a non-list argument raises TypeError;
  * the call returns None.

Performance structure (measured): every element is touched once, so the device
side is a memory-bound stream (1.25-1.4 TB/s measured) and the binding cost is
the host launch path: one Triton launch costs ~15-20 us of JIT dispatch on this
stack, exposed once per timed iteration.  Therefore:
  * dense tensors are rounded by the fewest possible launches: one flat launch
    for a single tensor, one slot-chained launch for a short same-dtype list
    (equal numel -> a 17-argument kernel, else one 32-argument kernel);
  * the compiled kernel's own launcher is cached per (dtype, numel class) and
    invoked directly, skipping per-call argument binding, cache-key hashing and
    launch validation; the numel class keeps the divisibility-by-16 attribute
    that the mask vectorisation depends on;
  * generic views use explicit stride arithmetic over the collapsed axes;
  * no temporary tensors, no device metadata, no allocation per call.
Every fast path falls back to an ordinary ``kernel[grid](...)`` launch on any
error, and rounding is idempotent, so a fallback can never corrupt results.
"""

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

try:  # launch hooks of the surrounding runtime, if present
    from triton import knobs as _knobs

    _ENTER = _knobs.runtime.launch_enter_hook
    _EXIT = _knobs.runtime.launch_exit_hook
except Exception:  # pragma: no cover
    _ENTER = None
    _EXIT = None

_BLOCK = 4096
_NUM_WARPS = 8
_SLOTS = 16
# Dense lists at or below this element count are fused into one launch; above
# it the device time exceeds the launch cost and per-tensor launches are kept.
_FUSE_MAX = 1 << 22
_I32_MAX = 1 << 31

_FLOATS = frozenset((torch.float16, torch.bfloat16, torch.float32, torch.float64))
_INTS = frozenset((torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8))

_PTRS = ["p%d" % i for i in range(_SLOTS)]


@triton.jit
def _rbody(p, blk, n, BLOCK: tl.constexpr):
    off = blk * BLOCK + tl.arange(0, BLOCK)
    mask = off < n
    x = tl.load(p + off, mask=mask, other=0.0)
    if x.dtype == tl.float64:
        y = libdevice.rint(x)
    else:
        y = libdevice.rint(x.to(tl.float32)).to(x.dtype)
    tl.store(p + off, y, mask=mask)


@triton.jit
def _round_flat_kernel(p, n, BLOCK: tl.constexpr):
    _rbody(p, tl.program_id(0), n, BLOCK)


@triton.jit
def _round_multi16(
    p0,
    n0,
    p1,
    n1,
    p2,
    n2,
    p3,
    n3,
    p4,
    n4,
    p5,
    n5,
    p6,
    n6,
    p7,
    n7,
    p8,
    n8,
    p9,
    n9,
    p10,
    n10,
    p11,
    n11,
    p12,
    n12,
    p13,
    n13,
    p14,
    n14,
    p15,
    n15,
    BLOCK: tl.constexpr,
):
    """One launch for up to 16 same-dtype dense tensors, unequal sizes allowed.

    Slots are selected by cumulative block ranges; padding slots carry n == 0
    and therefore contribute no blocks.
    """
    pid = tl.program_id(0)
    c = (n0 + BLOCK - 1) // BLOCK
    if pid < c:
        _rbody(p0, pid, n0, BLOCK)
    e = c + (n1 + BLOCK - 1) // BLOCK
    if pid >= c and pid < e:
        _rbody(p1, pid - c, n1, BLOCK)
    c = e
    e = c + (n2 + BLOCK - 1) // BLOCK
    if pid >= c and pid < e:
        _rbody(p2, pid - c, n2, BLOCK)
    c = e
    e = c + (n3 + BLOCK - 1) // BLOCK
    if pid >= c and pid < e:
        _rbody(p3, pid - c, n3, BLOCK)
    c = e
    e = c + (n4 + BLOCK - 1) // BLOCK
    if pid >= c and pid < e:
        _rbody(p4, pid - c, n4, BLOCK)
    c = e
    e = c + (n5 + BLOCK - 1) // BLOCK
    if pid >= c and pid < e:
        _rbody(p5, pid - c, n5, BLOCK)
    c = e
    e = c + (n6 + BLOCK - 1) // BLOCK
    if pid >= c and pid < e:
        _rbody(p6, pid - c, n6, BLOCK)
    c = e
    e = c + (n7 + BLOCK - 1) // BLOCK
    if pid >= c and pid < e:
        _rbody(p7, pid - c, n7, BLOCK)
    c = e
    e = c + (n8 + BLOCK - 1) // BLOCK
    if pid >= c and pid < e:
        _rbody(p8, pid - c, n8, BLOCK)
    c = e
    e = c + (n9 + BLOCK - 1) // BLOCK
    if pid >= c and pid < e:
        _rbody(p9, pid - c, n9, BLOCK)
    c = e
    e = c + (n10 + BLOCK - 1) // BLOCK
    if pid >= c and pid < e:
        _rbody(p10, pid - c, n10, BLOCK)
    c = e
    e = c + (n11 + BLOCK - 1) // BLOCK
    if pid >= c and pid < e:
        _rbody(p11, pid - c, n11, BLOCK)
    c = e
    e = c + (n12 + BLOCK - 1) // BLOCK
    if pid >= c and pid < e:
        _rbody(p12, pid - c, n12, BLOCK)
    c = e
    e = c + (n13 + BLOCK - 1) // BLOCK
    if pid >= c and pid < e:
        _rbody(p13, pid - c, n13, BLOCK)
    c = e
    e = c + (n14 + BLOCK - 1) // BLOCK
    if pid >= c and pid < e:
        _rbody(p14, pid - c, n14, BLOCK)
    c = e
    e = c + (n15 + BLOCK - 1) // BLOCK
    if pid >= c and pid < e:
        _rbody(p15, pid - c, n15, BLOCK)


@triton.jit(do_not_specialize_on_alignment=_PTRS)
def _round_multi16eq(
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
):
    """One launch for up to 16 equal-numel dense tensors (the common list case).

    The grid is ``cnt * blocks_per_tensor``, so slots at or beyond ``cnt`` are
    never selected and may be padded with any valid pointer.  Pointer alignment
    is not specialised: that removes 16 host-side pointer attribute queries per
    launch and costs nothing for the short tensors this path serves.
    """
    pid = tl.program_id(0)
    b = (n + BLOCK - 1) // BLOCK
    if pid < b:
        _rbody(p0, pid, n, BLOCK)
    if pid >= b and pid < 2 * b:
        _rbody(p1, pid - b, n, BLOCK)
    if pid >= 2 * b and pid < 3 * b:
        _rbody(p2, pid - 2 * b, n, BLOCK)
    if pid >= 3 * b and pid < 4 * b:
        _rbody(p3, pid - 3 * b, n, BLOCK)
    if pid >= 4 * b and pid < 5 * b:
        _rbody(p4, pid - 4 * b, n, BLOCK)
    if pid >= 5 * b and pid < 6 * b:
        _rbody(p5, pid - 5 * b, n, BLOCK)
    if pid >= 6 * b and pid < 7 * b:
        _rbody(p6, pid - 6 * b, n, BLOCK)
    if pid >= 7 * b and pid < 8 * b:
        _rbody(p7, pid - 7 * b, n, BLOCK)
    if pid >= 8 * b and pid < 9 * b:
        _rbody(p8, pid - 8 * b, n, BLOCK)
    if pid >= 9 * b and pid < 10 * b:
        _rbody(p9, pid - 9 * b, n, BLOCK)
    if pid >= 10 * b and pid < 11 * b:
        _rbody(p10, pid - 10 * b, n, BLOCK)
    if pid >= 11 * b and pid < 12 * b:
        _rbody(p11, pid - 11 * b, n, BLOCK)
    if pid >= 12 * b and pid < 13 * b:
        _rbody(p12, pid - 12 * b, n, BLOCK)
    if pid >= 13 * b and pid < 14 * b:
        _rbody(p13, pid - 13 * b, n, BLOCK)
    if pid >= 14 * b and pid < 15 * b:
        _rbody(p14, pid - 14 * b, n, BLOCK)
    if pid >= 15 * b and pid < 16 * b:
        _rbody(p15, pid - 15 * b, n, BLOCK)


@triton.jit
def _round_rows_kernel(
    p,
    E,
    S,
    NB,
    o0,
    s0,
    o1,
    s1,
    o2,
    s2,
    o3,
    s3,
    NOUT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Nested view: one row (stride S) per outer index, vectorised inside a row."""
    pid = tl.program_id(0)
    row = pid // NB
    blk = pid - row * NB
    base = 0
    rem = row
    if NOUT >= 1:
        q = rem // o0
        base += q * s0
        rem = rem - q * o0
    if NOUT >= 2:
        q = rem // o1
        base += q * s1
        rem = rem - q * o1
    if NOUT >= 3:
        q = rem // o2
        base += q * s2
        rem = rem - q * o2
    if NOUT >= 4:
        q = rem // o3
        base += q * s3
        rem = rem - q * o3
    col = blk * BLOCK + tl.arange(0, BLOCK)
    mask = col < E
    off = base + col * S
    x = tl.load(p + off, mask=mask, other=0.0)
    if x.dtype == tl.float64:
        y = libdevice.rint(x)
    else:
        y = libdevice.rint(x.to(tl.float32)).to(x.dtype)
    tl.store(p + off, y, mask=mask)


@triton.jit
def _round_flatmod_kernel(
    p,
    n,
    E,
    S,
    o0,
    s0,
    o1,
    s1,
    o2,
    s2,
    o3,
    s3,
    NOUT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Narrow inner axis: flat 1-D grid, one div/mod pair per element."""
    pid = tl.program_id(0)
    idx = pid * BLOCK + tl.arange(0, BLOCK)
    mask = idx < n
    row = idx // E
    col = idx - row * E
    base = 0
    rem = row
    if NOUT >= 1:
        q = rem // o0
        base += q * s0
        rem = rem - q * o0
    if NOUT >= 2:
        q = rem // o1
        base += q * s1
        rem = rem - q * o1
    if NOUT >= 3:
        q = rem // o2
        base += q * s2
        rem = rem - q * o2
    if NOUT >= 4:
        q = rem // o3
        base += q * s3
        rem = rem - q * o3
    off = base + col * S
    x = tl.load(p + off, mask=mask, other=0.0)
    if x.dtype == tl.float64:
        y = libdevice.rint(x)
    else:
        y = libdevice.rint(x.to(tl.float32)).to(x.dtype)
    tl.store(p + off, y, mask=mask)


# ---------------------------------------------------------------------------
# host launch path
# ---------------------------------------------------------------------------

_FAST_ON = True
_FLAT_CACHE = {}
_EQ_CACHE = {}
_STREAM = [None, None]


def _stream():
    st = _STREAM
    if st[0] is None:
        from triton.runtime.driver import driver

        st[0] = driver.active
        st[1] = driver.active.get_current_device()
    return st[0].get_current_stream(st[1])


def _disable_fast():
    global _FAST_ON
    _FAST_ON = False


def _bind(kernel, args):
    """Compile (or fetch) and return (launch, packed_metadata, function)."""
    ck = kernel.warmup(*args, grid=(1, 1, 1), BLOCK=_BLOCK, num_warps=_NUM_WARPS)
    ck[(1, 1, 1)]  # initialise module/function handles
    return (ck.run.launch, ck.packed_metadata, ck.function)


def _launch_flat(t, n):
    ptr = t.data_ptr()
    if _FAST_ON and n < _I32_MAX and not (ptr & 15):
        try:
            key = (t.dtype, n & 15)
            ent = _FLAT_CACHE.get(key)
            if ent is None:
                ent = _FLAT_CACHE[key] = _bind(_round_flat_kernel, (t, n))
            # Raw address: the launcher then skips its per-pointer data_ptr call
            # and mcPointerGetAttribute query (alignment is already checked).
            ent[0](
                (n + _BLOCK - 1) // _BLOCK,
                1,
                1,
                _stream(),
                ent[2],
                None,
                None,
                ent[1],
                None,
                _ENTER,
                _EXIT,
                ptr,
                n,
                _BLOCK,
            )
            return
        except Exception:
            _disable_fast()
    _round_flat_kernel[((n + _BLOCK - 1) // _BLOCK,)](
        t, n, BLOCK=_BLOCK, num_warps=_NUM_WARPS
    )


def _launch_eq(ts, n):
    cnt = len(ts)
    b = (n + _BLOCK - 1) // _BLOCK
    if _FAST_ON:
        try:
            key = (ts[0].dtype, n & 15)
            ent = _EQ_CACHE.get(key)
            if ent is None:
                ps = list(ts)
                while len(ps) < _SLOTS:
                    ps.append(ps[0])
                ent = _EQ_CACHE[key] = _bind(_round_multi16eq, tuple(ps) + (n,))
            addrs = [t.data_ptr() for t in ts]
            while len(addrs) < _SLOTS:
                addrs.append(addrs[0])
            ent[0](
                cnt * b,
                1,
                1,
                _stream(),
                ent[2],
                None,
                None,
                ent[1],
                None,
                _ENTER,
                _EXIT,
                *(tuple(addrs) + (n, _BLOCK)),
            )
            return
        except Exception:
            _disable_fast()
    ps = list(ts)
    while len(ps) < _SLOTS:
        ps.append(ps[0])
    _round_multi16eq[(cnt * b,)](*ps, n, BLOCK=_BLOCK, num_warps=_NUM_WARPS)


def _launch_multi(ts, ns):
    if len(ts) > 1:
        n0 = ns[0]
        same = True
        for n in ns:
            if n != n0:
                same = False
                break
        if same:
            _launch_eq(ts, n0)
            return
    grid = 0
    for n in ns:
        grid += (n + _BLOCK - 1) // _BLOCK
    ps = list(ts)
    cs = list(ns)
    while len(ps) < _SLOTS:
        ps.append(ps[0])
        cs.append(0)
    _round_multi16[(grid,)](
        ps[0],
        cs[0],
        ps[1],
        cs[1],
        ps[2],
        cs[2],
        ps[3],
        cs[3],
        ps[4],
        cs[4],
        ps[5],
        cs[5],
        ps[6],
        cs[6],
        ps[7],
        cs[7],
        ps[8],
        cs[8],
        ps[9],
        cs[9],
        ps[10],
        cs[10],
        ps[11],
        cs[11],
        ps[12],
        cs[12],
        ps[13],
        cs[13],
        ps[14],
        cs[14],
        ps[15],
        cs[15],
        BLOCK=_BLOCK,
        num_warps=_NUM_WARPS,
    )


def _collapse(t):
    """(inner_size, inner_stride, [(outer_size, outer_stride), ...]) or None.

    Axes of size 1 are dropped and equal-stride neighbours are merged after
    sorting by ascending stride, so a visit covers exactly the tensor's element
    set (order is irrelevant for an elementwise in-place op).
    """
    dims = [(int(sz), int(st)) for sz, st in zip(t.shape, t.stride()) if sz != 1]
    if not dims:
        return None
    merged = []
    for sz, st in sorted(dims, key=lambda d: d[1]):
        if merged and merged[-1][1] * merged[-1][0] == st:
            merged[-1] = (merged[-1][0] * sz, merged[-1][1])
        else:
            merged.append((sz, st))
    return merged[0], merged[1:]


def _launch_view(t, n):
    (inner, istride), outer = _collapse(t)
    nout = len(outer)
    if nout > 4:  # exotic high-rank view: recurse along one non-trivial axis
        ax = 0
        for i, sz in enumerate(t.shape):
            if sz > 1:
                ax = i
                break
        for i in range(t.shape[ax]):
            sub = t.narrow(ax, i, 1).squeeze(ax)
            _launch_view(sub, sub.numel())
        return
    if not outer and istride == 1:  # storage-dense
        _launch_flat(t, n)
        return
    try:
        overlap = torch._debug_has_internal_overlap(t) == 1
    except Exception:
        overlap = False
    if overlap:
        raise RuntimeError(
            "unsupported operation: more than one element of the written-to "
            "tensor refers to a single memory location. Please clone() the "
            "tensor before performing the operation."
        )
    o = [1, 1, 1, 1]
    s = [0, 0, 0, 0]
    for i, (sz, st) in enumerate(outer):
        o[i] = sz
        s[i] = st
    nb = (inner + _BLOCK - 1) // _BLOCK
    if inner >= 64:
        _round_rows_kernel[(n // inner * nb,)](
            t,
            inner,
            istride,
            nb,
            o[0],
            s[0],
            o[1],
            s[1],
            o[2],
            s[2],
            o[3],
            s[3],
            NOUT=nout,
            BLOCK=_BLOCK,
            num_warps=_NUM_WARPS,
        )
    else:
        _round_flatmod_kernel[((n + _BLOCK - 1) // _BLOCK,)](
            t,
            n,
            inner,
            istride,
            o[0],
            s[0],
            o[1],
            s[1],
            o[2],
            s[2],
            o[3],
            s[3],
            NOUT=nout,
            BLOCK=_BLOCK,
            num_warps=_NUM_WARPS,
        )


def _one(t):
    dt = t.dtype
    if dt in _INTS:
        return
    if dt not in _FLOATS:
        raise NotImplementedError(
            '"foreach_unary_op_cuda_" not implemented for '
            f"'{str(dt).replace('torch.', '')}'"
        )
    n = t.numel()
    if n == 0:
        return
    if t.is_contiguous():
        _launch_flat(t, n)
    else:
        _launch_view(t, n)


def _dispatch(ts, cs):
    if len(ts) == 1:
        t = ts[0]
        if t.is_contiguous():
            _launch_flat(t, cs[0])
        else:
            _launch_view(t, cs[0])
        return
    dense = []
    dcounts = []
    for i, t in enumerate(ts):
        if t.is_contiguous():
            dense.append(t)
            dcounts.append(cs[i])
        else:
            _launch_view(t, cs[i])
    if not dense:
        return
    if len(dense) == 1:
        _launch_flat(dense[0], dcounts[0])
        return
    # Short lists use the 17-argument equal-numel kernel (its pointer-alignment
    # opt-out is free at that size); above the threshold the device time
    # dominates and a branched multi-slot kernel loses bandwidth, so one flat
    # launch per tensor is kept.
    if sum(dcounts) > _FUSE_MAX:
        for t, n in zip(dense, dcounts):
            _launch_flat(t, n)
        return
    for i in range(0, len(dense), _SLOTS):
        _launch_multi(dense[i : i + _SLOTS], dcounts[i : i + _SLOTS])


def foreach_round_(self):
    if not isinstance(self, (list, tuple)):
        raise TypeError("argument 'self' must be tuple of Tensors")
    if not self:
        raise RuntimeError("Tensor list must have at least one tensor.")
    if len(self) == 1:
        t = self[0]
        if not isinstance(t, torch.Tensor):
            raise TypeError("argument 'self' must be tuple of Tensors")
        _one(t)
        return None

    # Single pass over the list: keep order, record each numel once, and note
    # whether every surviving tensor shares one dtype (the common case, which
    # then needs no grouping dictionary at all).
    items = []
    counts = []
    dt0 = None
    mixed = False
    for t in self:
        if not isinstance(t, torch.Tensor):
            raise TypeError("argument 'self' must be tuple of Tensors")
        dt = t.dtype
        if dt in _INTS:
            continue
        if dt not in _FLOATS:
            raise NotImplementedError(
                '"foreach_unary_op_cuda_" not implemented for '
                f"'{str(dt).replace('torch.', '')}'"
            )
        n = t.numel()
        if not n:
            continue
        if dt0 is None:
            dt0 = dt
        elif dt is not dt0:
            mixed = True
        items.append(t)
        counts.append(n)
    if not items:
        return None
    if not mixed:
        _dispatch(items, counts)
        return None

    groups = {}
    for i, t in enumerate(items):
        g = groups.get(t.dtype)
        if g is None:
            groups[t.dtype] = ([t], [counts[i]])
        else:
            g[0].append(t)
            g[1].append(counts[i])
    for ts, cs in groups.values():
        _dispatch(ts, cs)
    return None
