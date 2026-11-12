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

"""In-place ``aten::_foreach_sin_`` implemented with Triton kernels.

``_foreach_sin_(self)`` applies ``sin`` to every tensor of the list, in place,
preserving each element's storage, dtype and stride, and returns ``None``.

Design:
* every list element that is flat-addressable (directly contiguous, or
  contiguous after a zero-copy stride-sorted permutation) is handled by a single
  fused launch per dtype -- with per-slot pointers and element counts passed as
  kernel arguments, so no host->device metadata buffer is ever built -- but only
  while the per-tensor work is launch-bound; larger tensors, whose GPU work
  already hides the launch latency, keep one launch per tensor because the
  runtime pointer select inside the fused kernel costs access width there;
* a list element that is a genuinely gappy view (e.g. ``x[:, ::2]``) is handled
  by its own strided launch so that no copying fallback can lose the in-place
  write;
* aliasing list entries (same storage repeated) fall back to strictly ordered
  per-tensor launches so the mutation order matches ATen;
* non floating dtypes are rejected before anything is mutated, matching ATen;
* half/bfloat16 are evaluated in float32 and rounded back on store.
"""

import torch
import triton
import triton.language as tl

_BLOCK = 1024
_NUM_WARPS = 4
_SLOTS = 16
_INT32_MAX = 2147483647
# Fusion only pays while the per-tensor work is launch-bound: a host Triton
# launch costs ~11-15us here, so a fused launch wins at 4096 elements/tensor
# (0.246ms -> 0.077ms for 16 tensors) but loses once each tensor already runs
# tens of microseconds of GPU work, where the runtime pointer select also
# costs access width (1.036ms -> 1.717ms for 16 x 4096x4096 float16).
_FUSE_MAX = 1 << 16

_FLOAT_DTYPES = frozenset((torch.float16, torch.bfloat16, torch.float32, torch.float64))

_OUT_NAME = {
    torch.float16: "Half",
    torch.bfloat16: "BFloat16",
    torch.float32: "Float",
    torch.float64: "Double",
    torch.int8: "Char",
    torch.int16: "Short",
    torch.int32: "Int",
    torch.int64: "Long",
    torch.uint8: "Byte",
    torch.bool: "Bool",
}


# ---------------------------------------------------------------------------
# kernels
# ---------------------------------------------------------------------------


@triton.jit
def _sin_flat_kernel(
    x_ptr,
    n,
    BLOCK: tl.constexpr,
    HIGH: tl.constexpr,
    WIDE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    if WIDE:
        offs = offs.to(tl.int64)
    mask = offs < n
    v = tl.load(x_ptr + offs, mask=mask)
    if HIGH:
        y = tl.sin(v)
    else:
        y = tl.sin(v.to(tl.float32)).to(v.dtype)
    tl.store(x_ptr + offs, y, mask=mask)


@triton.jit
def _sin_multi_kernel(
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
    BLOCK: tl.constexpr,
    HIGH: tl.constexpr,
):
    """One launch for the whole list: axis 0 selects the tensor."""
    t = tl.program_id(0)
    base = p0
    n = n0
    if t == 1:
        base = p1
        n = n1
    if t == 2:
        base = p2
        n = n2
    if t == 3:
        base = p3
        n = n3
    if t == 4:
        base = p4
        n = n4
    if t == 5:
        base = p5
        n = n5
    if t == 6:
        base = p6
        n = n6
    if t == 7:
        base = p7
        n = n7
    if t == 8:
        base = p8
        n = n8
    if t == 9:
        base = p9
        n = n9
    if t == 10:
        base = p10
        n = n10
    if t == 11:
        base = p11
        n = n11
    if t == 12:
        base = p12
        n = n12
    if t == 13:
        base = p13
        n = n13
    if t == 14:
        base = p14
        n = n14
    if t == 15:
        base = p15
        n = n15

    pid = tl.program_id(1)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    v = tl.load(base + offs, mask=mask)
    if HIGH:
        y = tl.sin(v)
    else:
        y = tl.sin(v.to(tl.float32)).to(v.dtype)
    tl.store(base + offs, y, mask=mask)


@triton.jit
def _sin_strided_kernel(
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
    HIGH: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = (pid * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)
    mask = offs < n
    rem = offs
    off = tl.zeros([BLOCK], dtype=tl.int64)
    if NDIM > 7:
        q = rem % s7
        rem = rem // s7
        off += q * t7
    if NDIM > 6:
        q = rem % s6
        rem = rem // s6
        off += q * t6
    if NDIM > 5:
        q = rem % s5
        rem = rem // s5
        off += q * t5
    if NDIM > 4:
        q = rem % s4
        rem = rem // s4
        off += q * t4
    if NDIM > 3:
        q = rem % s3
        rem = rem // s3
        off += q * t3
    if NDIM > 2:
        q = rem % s2
        rem = rem // s2
        off += q * t2
    if NDIM > 1:
        q = rem % s1
        rem = rem // s1
        off += q * t1
    q = rem % s0
    off += q * t0

    v = tl.load(x_ptr + off, mask=mask)
    if HIGH:
        y = tl.sin(v)
    else:
        y = tl.sin(v.to(tl.float32)).to(v.dtype)
    tl.store(x_ptr + off, y, mask=mask)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _launch_flat(t, n):
    grid = ((n + _BLOCK - 1) // _BLOCK,)
    _sin_flat_kernel[grid](
        t,
        n,
        BLOCK=_BLOCK,
        HIGH=(t.dtype is torch.float64),
        WIDE=(n > _INT32_MAX),
        num_warps=_NUM_WARPS,
    )


def _launch_group(items, dtype):
    """One fused launch for every flat-addressable tensor of one dtype."""
    nt = len(items)
    if nt > _SLOTS:
        for i in range(0, nt, _SLOTS):
            _launch_group(items[i : i + _SLOTS], dtype)
        return
    max_n = 0
    for _, n in items:
        if n > max_n:
            max_n = n
    if nt == 1 or max_n > _FUSE_MAX or max_n > _INT32_MAX:
        for t, n in items:
            _launch_flat(t, n)
        return
    ptrs = []
    ns = []
    for t, n in items:
        ptrs.append(t)
        ns.append(n)
    pad = _SLOTS - nt
    tail = items[-1]
    grid = (nt, (max_n + _BLOCK - 1) // _BLOCK)
    _sin_multi_kernel[grid](
        *(ptrs + [tail[0]] * pad + ns + [tail[1]] * pad),
        BLOCK=_BLOCK,
        HIGH=(dtype is torch.float64),
        num_warps=_NUM_WARPS,
    )


def _canon_dims(shape, strides):
    """Drop unit dims and merge adjacent dims that are contiguous in memory."""
    dims = []
    for s, st in zip(shape, strides):
        s = int(s)
        if s == 1:
            continue
        st = int(st)
        if dims and dims[-1][1] == s * st:
            dims[-1][0] *= s
        else:
            dims.append([s, st])
    return dims


def _launch_strided(t, n, dims):
    nd = len(dims)
    sizes = [d[0] for d in dims]
    strides = [d[1] for d in dims]
    pad = 8 - nd
    grid = ((n + _BLOCK - 1) // _BLOCK,)
    _sin_strided_kernel[grid](
        t,
        n,
        *(sizes + [1] * pad),
        *(strides + [0] * pad),
        NDIM=nd,
        BLOCK=_BLOCK,
        HIGH=(t.dtype is torch.float64),
        num_warps=_NUM_WARPS,
    )


def _flat_view(t):
    """A flat zero-copy view over the same storage, or ``None``."""
    if t.is_contiguous():
        return t
    perm = t.permute(sorted(range(t.dim()), key=lambda d: abs(t.stride(d))))
    if perm.is_contiguous():
        # Zero-copy view over the same storage: flattening it and writing sin
        # element-wise still mutates the original tensor.
        return perm.view(-1)
    return None


def _launch_gappy(t):
    """Strided launch for a view that cannot be flattened without copying."""
    perm = t.permute(sorted(range(t.dim()), key=lambda d: abs(t.stride(d))))
    dims = _canon_dims(perm.shape, perm.stride())
    if len(dims) > 8:
        # Extremely rare: peel the outermost dimension into zero-copy views.
        for i in range(perm.shape[0]):
            sub = perm[i]
            flat = _flat_view(sub)
            if flat is None:
                _launch_gappy(sub)
            else:
                _launch_flat(flat, sub.numel())
        return
    _launch_strided(t, t.numel(), dims)


def _launch_in_order(tensors):
    """Strictly ordered per-tensor launches (used when storage aliases)."""
    for t in tensors:
        n = t.numel()
        if n == 0:
            continue
        flat = _flat_view(t)
        if flat is None:
            _launch_gappy(t)
        else:
            _launch_flat(flat, n)


def _as_tensor_list(args, kwargs):
    if kwargs:
        for key in ("self", "tensors", "input", "inputs", "tensor_list"):
            if key in kwargs:
                args = tuple(args) + (kwargs[key],)
                break
    if len(args) == 0:
        raise TypeError("argument 'self' must be tuple of Tensors")
    if len(args) == 1:
        a = args[0]
        if isinstance(a, torch.Tensor):
            return [a]
        if isinstance(a, (list, tuple)):
            return list(a)
        raise TypeError("argument 'self' must be tuple of Tensors")
    out = []
    for a in args:
        if isinstance(a, (list, tuple)):
            out.extend(a)
        elif isinstance(a, torch.Tensor):
            out.append(a)
        else:
            raise TypeError("argument 'self' must be tuple of Tensors")
    return out


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def foreach_sin_(*args, **kwargs):
    tensors = _as_tensor_list(args, kwargs)
    if len(tensors) == 0:
        raise RuntimeError("Tensor list must have at least one tensor.")

    # Single pass: validate (before any mutation, as ATen does) and bucket.
    groups = {}
    gappy = []
    for t in tensors:
        if not isinstance(t, torch.Tensor):
            raise TypeError("argument 'self' must be tuple of Tensors")
        dt = t.dtype
        if dt not in _FLOAT_DTYPES:
            raise RuntimeError(
                "result type Float can't be cast to the desired output type "
                + _OUT_NAME.get(dt, str(dt).replace("torch.", ""))
            )
        n = t.numel()
        if n == 0:
            continue
        flat = _flat_view(t)
        if flat is None:
            gappy.append(t)
            continue
        bucket = groups.get(dt)
        if bucket is None:
            groups[dt] = [(flat, n)]
        else:
            bucket.append((flat, n))

    launch_groups = list(groups.values())

    # Repeated storage inside one list would make the fused launch order
    # undefined; process such lists strictly in order instead.
    if len(tensors) > 1:
        seen = set()
        for t in tensors:
            p = t.data_ptr()
            if p in seen:
                _launch_in_order(tensors)
                return None
            seen.add(p)

    for items in launch_groups:
        _launch_group(items, items[0][0].dtype)
    for t in gappy:
        _launch_gappy(t)
    return None
