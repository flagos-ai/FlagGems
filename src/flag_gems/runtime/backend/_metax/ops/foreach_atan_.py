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

"""In-place ``aten::_foreach_atan_``: ``atan`` applied to every tensor of a list.

Architecture (v4): one Triton launch for the whole list when the elements are
mutually compatible, otherwise one launch per element.

* A tensor whose memory span is exactly ``numel`` elements (contiguous, or any
  non-overlapping-and-dense view such as a transpose) is traversed in memory
  order by a flat unit-stride kernel.  ``atan`` is element-wise, so visiting the
  same address set in memory order is element-wise correct and keeps accesses
  coalesced, while the store lands in the tensor's own storage (in place, so
  storage/stride identity is preserved).
* Fusion: a contiguous, same-dtype group with one common ``numel`` is served by
  a single launch whose grid covers ``len(group) * tiles`` programs.  The
  program index is decoded into (tensor, tile) and the tile is processed inside
  an explicit per-slot branch, so every load/store goes through a *distinct
  typed pointer argument* and keeps Triton's 16-byte pointer divisibility hint
  (and the vector loads that follow from it) for every slot.  Selecting the
  pointer with ``tl.where`` instead measured ~17% off the fp32 byte rate and
  ~2x off the fp16/bfloat16 element rate.  Lists longer than the slot count are
  chunked.
* A gappy view (``x[:, ::2]``) is traversed by a merged-stride index map with
  the same element mask on load and store.
* Launch geometry: ``BLOCK = 4096`` elements per program at ``num_warps = 4``
  (64-wide warps) gives sixteen elements per thread -- half the programs of a
  2048-element tile at the same 128-bit access width.  At 1024 only the 4- and
  8-byte dtypes reach 128 bits while the 2-byte dtypes are stuck at 64-bit,
  which measured 9-16% slower on every float16/bfloat16 workload.  The value is
  a launch parameter, not a hardware assumption, and can differ on another
  device family.
* Per-call host cost dominates the small workloads: a minimal launch measures
  ~24 us end to end while an 18-argument launch costs ~17 us more, i.e. roughly
  1 us per kernel argument regardless of its type.  So every shape-dependent
  scalar that can be folded is passed as ``tl.constexpr`` instead of as a
  runtime argument (``N``, ``TILES``), which also folds ``pid // tiles`` into a
  shift and removes the element mask whenever ``numel`` divides the tile, and
  the host wrapper validates the list in a single pass.
* Tiny groups (``len(group) * numel <= 2**20``) are launch-bound, so they take
  a variant of the fused kernel whose per-tensor addresses travel as int64
  scalars: the launcher then processes plain integers instead of tensor
  arguments, which measured ~2.5 us per call, and the kernel keeps a single
  body.  That cast loses the per-argument divisibility hint, which only matters
  once device time dominates.
* Nothing is allocated, no copy, no staging: read + write of the same bytes.
"""

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

_BLOCK = 4096
_MAX_DIMS = 8
_SLOTS = 16
_INT32_MAX = 0x7FFFFFFF

# In-place atan of an integral/bool tensor cannot be represented: ATen refuses
# with the same cast error its slow path would hit.
_ALLOWED = frozenset((torch.float16, torch.bfloat16, torch.float32, torch.float64))
_DTYPE_NAME = {
    torch.float64: "Double",
    torch.float32: "Float",
    torch.float16: "Half",
    torch.bfloat16: "BFloat16",
    torch.int64: "Long",
    torch.int32: "Int",
    torch.int16: "Short",
    torch.int8: "Char",
    torch.uint8: "Byte",
    torch.bool: "Bool",
}

_OVERLAP_MSG = (
    "unsupported operation: more than one element of the written-to tensor "
    "refers to a single memory location. Please clone() the tensor before "
    "performing the operation."
)

_TL_DTYPE = {
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
    torch.float32: tl.float32,
    torch.float64: tl.float64,
}

# Below this many elements in the whole group the kernel is launch-bound, so the
# int64-pointer variant (cheaper launcher argument processing) is used instead
# of the typed-pointer one, whose per-argument divisibility hints only matter
# once the group is big enough for device time to dominate.
_TINY_ELEMS = 1 << 20


@triton.jit
def _atan(x):
    """atan with ATen's per-dtype precision: fp64 in double, half in fp32."""
    if x.dtype == tl.float64:
        return libdevice.atan(x)
    return libdevice.atan(x.to(tl.float32)).to(x.dtype)


@triton.jit
def _tile(ptr, off, m):
    """One in-place tile reached through a single typed pointer argument."""
    v = tl.load(ptr + off, mask=m, other=0.0)
    tl.store(ptr + off, _atan(v), mask=m)


@triton.jit
def _flat_kernel(x_ptr, N: tl.constexpr, I64: tl.constexpr, BLOCK: tl.constexpr):
    """Single tensor in memory order; ``N`` is folded so the mask can vanish."""
    pid = tl.program_id(0)
    if I64:
        base = pid.to(tl.int64) * BLOCK
    else:
        base = pid * BLOCK
    off = base + tl.arange(0, BLOCK)
    _tile(x_ptr, off, off < N)


@triton.jit
def _multi_kernel(
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
    tiles,
    K: tl.constexpr = 16,
    BLOCK: tl.constexpr = 4096,
):
    """All ``K`` tensors of one contiguous, equal-length group in one launch.

    ``pid // tiles`` is the tensor index, the remainder the tile inside it.
    Each slot has its own branch, so the address arithmetic of every access
    starts from a distinct typed pointer argument.
    """
    pid = tl.program_id(0)
    t = pid // tiles
    off = (pid - t * tiles) * BLOCK + tl.arange(0, BLOCK)
    m = off < n
    if t == 0:
        _tile(p0, off, m)
    if K > 1:
        if t == 1:
            _tile(p1, off, m)
    if K > 2:
        if t == 2:
            _tile(p2, off, m)
    if K > 3:
        if t == 3:
            _tile(p3, off, m)
    if K > 4:
        if t == 4:
            _tile(p4, off, m)
    if K > 5:
        if t == 5:
            _tile(p5, off, m)
    if K > 6:
        if t == 6:
            _tile(p6, off, m)
    if K > 7:
        if t == 7:
            _tile(p7, off, m)
    if K > 8:
        if t == 8:
            _tile(p8, off, m)
    if K > 9:
        if t == 9:
            _tile(p9, off, m)
    if K > 10:
        if t == 10:
            _tile(p10, off, m)
    if K > 11:
        if t == 11:
            _tile(p11, off, m)
    if K > 12:
        if t == 12:
            _tile(p12, off, m)
    if K > 13:
        if t == 13:
            _tile(p13, off, m)
    if K > 14:
        if t == 14:
            _tile(p14, off, m)
    if K > 15:
        if t == 15:
            _tile(p15, off, m)


@triton.jit
def _multi_ptr_kernel(
    q0,
    q1,
    q2,
    q3,
    q4,
    q5,
    q6,
    q7,
    q8,
    q9,
    q10,
    q11,
    q12,
    q13,
    q14,
    q15,
    N: tl.constexpr,
    TILES: tl.constexpr,
    DT: tl.constexpr,
    K: tl.constexpr = 16,
    BLOCK: tl.constexpr = 4096,
):
    """Launch-cost-first variant of :func:`_multi_kernel` for tiny groups.

    The per-tensor addresses travel as int64 scalars and are selected
    arithmetically, so the launcher processes plain integers instead of tensor
    arguments and the kernel keeps a single body.  ``N`` and ``TILES`` are
    constexpr, so this launch carries only the sixteen addresses and the tile
    mask folds away whenever the group is tile-aligned.  The int-to-pointer cast
    drops the per-argument 16-byte divisibility hint, which is irrelevant for a
    group small enough to be launch-bound; large groups use the typed-pointer
    kernel.
    """
    pid = tl.program_id(0)
    t = pid // TILES
    off = (pid - t * TILES) * BLOCK + tl.arange(0, BLOCK)
    m = off < N
    q = q0
    if K > 1:
        q = tl.where(t == 1, q1, q)
    if K > 2:
        q = tl.where(t == 2, q2, q)
    if K > 3:
        q = tl.where(t == 3, q3, q)
    if K > 4:
        q = tl.where(t == 4, q4, q)
    if K > 5:
        q = tl.where(t == 5, q5, q)
    if K > 6:
        q = tl.where(t == 6, q6, q)
    if K > 7:
        q = tl.where(t == 7, q7, q)
    if K > 8:
        q = tl.where(t == 8, q8, q)
    if K > 9:
        q = tl.where(t == 9, q9, q)
    if K > 10:
        q = tl.where(t == 10, q10, q)
    if K > 11:
        q = tl.where(t == 11, q11, q)
    if K > 12:
        q = tl.where(t == 12, q12, q)
    if K > 13:
        q = tl.where(t == 13, q13, q)
    if K > 14:
        q = tl.where(t == 14, q14, q)
    if K > 15:
        q = tl.where(t == 15, q15, q)
    _tile(q.to(tl.pointer_type(DT)), off, m)


@triton.jit
def _strided_kernel(
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
    t0,
    t1,
    t2,
    t3,
    t4,
    t5,
    t6,
    t7,
    NDIM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Gappy view: dims are supplied inner-first and unrolled at compile time."""
    pid = tl.program_id(0)
    idx = (pid * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)
    m = idx < n
    rem = idx
    addr = tl.zeros((BLOCK,), tl.int64)
    if NDIM > 0:
        addr += (rem % s0) * t0
        rem = rem // s0
    if NDIM > 1:
        addr += (rem % s1) * t1
        rem = rem // s1
    if NDIM > 2:
        addr += (rem % s2) * t2
        rem = rem // s2
    if NDIM > 3:
        addr += (rem % s3) * t3
        rem = rem // s3
    if NDIM > 4:
        addr += (rem % s4) * t4
        rem = rem // s4
    if NDIM > 5:
        addr += (rem % s5) * t5
        rem = rem // s5
    if NDIM > 6:
        addr += (rem % s6) * t6
        rem = rem // s6
    if NDIM > 7:
        addr += (rem % s7) * t7
    v = tl.load(x_ptr + addr, mask=m, other=0.0)
    tl.store(x_ptr + addr, _atan(v), mask=m)


def _flat_ok(t):
    """True when ``[data_ptr, data_ptr + numel)`` covers exactly the tensor."""
    if t.is_contiguous():
        return True
    if t.numel() <= 1:
        return True
    expected = 1
    for stride, size in sorted((s, z) for s, z in zip(t.stride(), t.shape) if z != 1):
        if stride != expected:
            return False
        expected *= size
    return True


def _merged_dims(t):
    """(size, stride) pairs inner-first after dropping size-1 dims and merging."""
    merged = []
    for size, stride in reversed(
        [(z, s) for z, s in zip(t.shape, t.stride()) if z != 1]
    ):
        if merged and stride == merged[-1][1] * merged[-1][0]:
            inner_size, inner_stride = merged[-1]
            merged[-1] = (inner_size * size, inner_stride)
        else:
            merged.append((size, stride))
    return merged


def _launch_one(t, n, contig):
    """One tensor, one launch.  ``contig`` is the already known fast path."""
    if n == 0:
        return
    if contig or _flat_ok(t):
        _flat_kernel[((n + _BLOCK - 1) // _BLOCK,)](t, n, n > _INT32_MAX, _BLOCK)
        return
    if torch._debug_has_internal_overlap(t) == 1:
        raise RuntimeError(_OVERLAP_MSG)
    dims = _merged_dims(t)
    ndim = len(dims)
    if ndim > _MAX_DIMS:
        raise RuntimeError("foreach_atan_: too many merged dims")
    sizes = [d[0] for d in dims] + [1] * (_MAX_DIMS - ndim)
    strides = [d[1] for d in dims] + [0] * (_MAX_DIMS - ndim)
    _strided_kernel[((n + _BLOCK - 1) // _BLOCK,)](t, n, *sizes, *strides, ndim, _BLOCK)


def _pad_to_slots(seq):
    """Repeat a short group up to the slot count (padding slots are never used)."""
    k = len(seq)
    if k == _SLOTS:
        return list(seq)
    reps = (_SLOTS + k - 1) // k
    return (list(seq) * reps)[:_SLOTS]


def _launch_fused_tiny(group, k, n, tiles, dt, base=0):
    """Fused launch whose addresses travel as int64 scalars, in 16-slot chunks."""
    while base < k:
        used = k - base
        if used > _SLOTS:
            used = _SLOTS
        addrs = [group[base + i].data_ptr() for i in range(used)]
        if used < _SLOTS:
            addrs.extend([0] * (_SLOTS - used))
        _multi_ptr_kernel[(used * tiles,)](*addrs, n, tiles, dt, used)
        base += used


def foreach_atan_(self):
    if not isinstance(self, (list, tuple)):
        raise TypeError("argument 'self' must be tuple of Tensors")
    k = len(self)
    if k == 0:
        raise RuntimeError("Tensor list must have at least one tensor.")

    first = None
    dtype = None
    n = 0
    same = True
    idx = 0
    for t in self:
        if not isinstance(t, torch.Tensor):
            raise TypeError("argument 'self' must be tuple of Tensors")
        dt = t.dtype
        if dt not in _ALLOWED:
            raise RuntimeError(
                "result type Float can't be cast to the desired output type "
                + _DTYPE_NAME.get(dt, str(dt).replace("torch.", ""))
            )
        if idx == 0:
            first = t
            dtype = dt
            n = t.numel()
            same = t.is_contiguous()
        elif same and (dt is not dtype or t.numel() != n or not t.is_contiguous()):
            same = False
        idx += 1

    if n == 0:
        return None
    if k == 1:
        _launch_one(first, n, same)
        return None
    if same and n <= _INT32_MAX:
        tiles = (n + _BLOCK - 1) // _BLOCK
        if k * n <= _TINY_ELEMS:
            _launch_fused_tiny(self, k, n, tiles, _TL_DTYPE[dtype])
        elif k <= _SLOTS:
            _multi_kernel[(k * tiles,)](*_pad_to_slots(self), n, tiles, k)
        else:
            for base in range(0, k, _SLOTS):
                chunk = self[base : base + _SLOTS]
                used = len(chunk)
                _multi_kernel[(used * tiles,)](*_pad_to_slots(chunk), n, tiles, used)
        return None

    for t in self:
        _launch_one(t, t.numel(), t.is_contiguous())
    return None
