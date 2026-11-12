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

"""In-place TensorList erf: ``aten::_foreach_erf_(Tensor(a!)[] self) -> ()``.

Every list element is mutated in place (storage pointer, storage_offset, shape
and strides preserved) and ``None`` is returned, matching the schema.

Only floating dtypes are accepted; integral/bool inputs raise ATen's
"result type Float can't be cast ..." RuntimeError and complex inputs raise
ATen's NotImplementedError, and that validation happens for the whole list
before any element is mutated.

Host cost is what separates a Triton foreach kernel from ATen's C++ one, so the
wrapper has three tiers:

* one contiguous floating tensor -> launch the two-argument kernel directly,
  with no records or grouping at all;
* a list whose elements are all contiguous tensors of one dtype on one device ->
  pack ``(pointer, numel)`` pairs straight from the live tensors in a single
  walk.  The walk abandons the tier (before anything is mutated) as soon as an
  element is not contiguous or a byte span touches an earlier one, so overlap,
  duplicate and mixed-list semantics are always handled by the general tier;
* everything else -> one validation walk that records per-tensor addressing
  facts, then a packing walk that groups same-dtype, span-disjoint tensors.

Launch shapes, chosen by how much work a launch carries:

* one tensor -> the two-argument kernel taking that tensor's typed pointer, so
  the compiler knows the base alignment and vectorizes the accesses;
* several small tensors -> one launch driven by a tiny device metadata array of
  interleaved ``(pointer, numel)`` pairs.  Triton's per-argument dispatch is a
  large share of the cost there, so replacing sixteen pointer/count arguments
  with a single metadata argument is what makes a many-tiny-tensor list fast; the
  array holds launch metadata only (no computed data) and is staged through a
  pinned host buffer so the copy does not pay driver staging;
* several large tensors -> the typed multi-pointer kernel, which keeps the
  alignment information that matters once the launch is bandwidth-bound.

Views that do not cover one contiguous storage interval are handled by a
collapsed strided kernel that computes per-element memory offsets from
shape/stride descriptors, so no staging copy is ever made.
"""

import torch
import triton
import triton.language as tl

_BLOCK = 2048
_WARPS = 4
_MAX_DIMS = 8
# Below this element count a launch carries almost no device work, so the
# metadata-driven path wins by shedding per-argument dispatch cost.
_META_LIMIT = 1 << 20
# Tensors grouped into one metadata launch by the general tier.
_META_SLOTS = 16
# Tensors grouped into one typed-pointer launch.
_TYPED_SLOTS = 8

_FLOATS = frozenset((torch.float32, torch.float64, torch.float16, torch.bfloat16))

_ATEN_NAME = {
    torch.float32: "Float",
    torch.float64: "Double",
    torch.float16: "Half",
    torch.bfloat16: "BFloat16",
    torch.int8: "Char",
    torch.uint8: "Byte",
    torch.int16: "Short",
    torch.int32: "Int",
    torch.int64: "Long",
    torch.bool: "Bool",
    torch.complex64: "ComplexFloat",
    torch.complex128: "ComplexDouble",
}

_TL_DTYPE = {
    torch.float32: tl.float32,
    torch.float64: tl.float64,
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
}

_OVERLAP_MSG = (
    "unsupported operation: more than one element of the written-to tensor "
    "refers to a single memory location. Please clone() the tensor before "
    "performing the operation."
)

# Pinned host staging for launch metadata, allocated on first use.  It is
# refilled and consumed inside one call under a blocking copy, so no device-side
# state is shared between calls.
_PINNED = None


@triton.jit
def _erf_one_kernel(p, n, BLOCK: tl.constexpr, I64: tl.constexpr):
    """In-place erf over one flat, dense storage interval."""
    pid = tl.program_id(0)
    if I64:
        off = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    else:
        off = pid * BLOCK + tl.arange(0, BLOCK)
    mask = off < n
    x = tl.load(p + off, mask=mask, other=0.0)
    if x.dtype == tl.float64:
        y = tl.math.erf(x)
    else:
        y = tl.math.erf(x.to(tl.float32)).to(x.dtype)
    tl.store(p + off, y, mask=mask)


@triton.jit
def _erf_meta_kernel(meta, DT: tl.constexpr, BLOCK: tl.constexpr):
    """In-place erf over ``NT`` flat tensors addressed from ``meta``.

    ``meta`` holds interleaved int64 ``(pointer, numel)`` pairs; the grid is
    ``(NT, chunks)`` so ``program_id(0)`` selects the tensor.
    """
    t = tl.program_id(0)
    addr = tl.load(meta + 2 * t)
    n = tl.load(meta + 2 * t + 1)
    off = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = off < n
    p = addr.to(tl.pointer_type(DT))
    x = tl.load(p + off, mask=mask, other=0.0)
    if DT == tl.float64:
        y = tl.math.erf(x)
    else:
        y = tl.math.erf(x.to(tl.float32)).to(DT)
    tl.store(p + off, y, mask=mask)


@triton.jit
def _erf_batch_kernel(
    a0,
    a1,
    a2,
    a3,
    a4,
    a5,
    a6,
    a7,
    m0,
    m1,
    m2,
    m3,
    m4,
    m5,
    m6,
    m7,
    BLOCK: tl.constexpr,
    I64: tl.constexpr,
):
    """In-place erf over ``NT`` flat tensors of one dtype; grid=(tensor, chunk)."""
    t = tl.program_id(0)
    n = m0
    n = tl.where(t == 1, m1, n)
    n = tl.where(t == 2, m2, n)
    n = tl.where(t == 3, m3, n)
    n = tl.where(t == 4, m4, n)
    n = tl.where(t == 5, m5, n)
    n = tl.where(t == 6, m6, n)
    n = tl.where(t == 7, m7, n)

    p = a0
    if t == 1:
        p = a1
    elif t == 2:
        p = a2
    elif t == 3:
        p = a3
    elif t == 4:
        p = a4
    elif t == 5:
        p = a5
    elif t == 6:
        p = a6
    elif t == 7:
        p = a7

    pid = tl.program_id(1)
    if I64:
        off = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    else:
        off = pid * BLOCK + tl.arange(0, BLOCK)
    mask = off < n
    x = tl.load(p + off, mask=mask, other=0.0)
    if x.dtype == tl.float64:
        y = tl.math.erf(x)
    else:
        y = tl.math.erf(x.to(tl.float32)).to(x.dtype)
    tl.store(p + off, y, mask=mask)


@triton.jit
def _erf_strided_kernel(
    addr,
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
    ND: tl.constexpr,
    DT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """In-place erf over logical order of one collapsed strided view."""
    pid = tl.program_id(0).to(tl.int64)
    idx = pid * BLOCK + tl.arange(0, BLOCK)
    mask = idx < n
    rem = idx
    off = tl.zeros([BLOCK], dtype=tl.int64)
    if ND > 0:
        off += (rem % s0) * t0
        rem = rem // s0
    if ND > 1:
        off += (rem % s1) * t1
        rem = rem // s1
    if ND > 2:
        off += (rem % s2) * t2
        rem = rem // s2
    if ND > 3:
        off += (rem % s3) * t3
        rem = rem // s3
    if ND > 4:
        off += (rem % s4) * t4
        rem = rem // s4
    if ND > 5:
        off += (rem % s5) * t5
        rem = rem // s5
    if ND > 6:
        off += (rem % s6) * t6
        rem = rem // s6
    if ND > 7:
        off += (rem % s7) * t7
        rem = rem // s7

    ptr = addr.to(tl.pointer_type(DT))
    x = tl.load(ptr + off, mask=mask, other=0.0)
    if DT == tl.float64:
        tl.store(ptr + off, tl.math.erf(x), mask=mask)
    else:
        tl.store(ptr + off, tl.math.erf(x.to(tl.float32)).to(DT), mask=mask)


def _device_meta(vals, device):
    """``vals`` (int64 values) as a device tensor, staged through pinned memory."""
    global _PINNED
    m = len(vals)
    pin = _PINNED
    if pin is None or pin.numel() < m:
        pin = torch.empty(max(m, 2 * _META_SLOTS), dtype=torch.int64, pin_memory=True)
        _PINNED = pin
    stage = pin[:m]
    stage.copy_(torch.tensor(vals, dtype=torch.int64))
    out = torch.empty(m, dtype=torch.int64, device=device)
    out.copy_(stage)
    return out


def _record(t, dt):
    """``(ptr, numel, flat, lo, hi, dtype, device_index)`` in one pass.

    ``flat`` means the view covers exactly ``[ptr, ptr + numel)`` in storage
    order, so it can be addressed with a flat index.  ``lo:hi`` is a
    conservative byte interval covering every storage byte the view touches and
    keeps overlapping tensors out of the same launch.
    """
    numel = t.numel()
    ptr = t.data_ptr()
    if t.is_contiguous():
        return (
            ptr,
            numel,
            True,
            ptr,
            ptr + numel * t.element_size(),
            dt,
            t.get_device(),
        )
    if numel <= 1:
        return (ptr, numel, True, ptr, ptr + t.element_size(), dt, t.get_device())
    esz = t.element_size()
    expected = 1
    for stride, size in sorted(
        (st, sz) for st, sz in zip(t.stride(), t.shape) if sz != 1
    ):
        if stride != expected:
            lo = 0
            hi = 0
            for size2, stride2 in zip(t.shape, t.stride()):
                if size2 <= 1:
                    continue
                if stride2 >= 0:
                    hi += (size2 - 1) * stride2
                else:
                    lo += (size2 - 1) * stride2
            return (
                ptr,
                numel,
                False,
                ptr + lo * esz,
                ptr + (hi + 1) * esz,
                dt,
                t.get_device(),
            )
        expected *= size
    return (ptr, numel, True, ptr, ptr + numel * esz, dt, t.get_device())


def _collapse(t):
    """Collapse a view into (size, stride) pairs, smallest stride first."""
    dims = [(sz, st) for sz, st in zip(t.shape, t.stride()) if sz != 1]
    if not dims:
        return [(1, 0)]
    dims.sort(key=lambda d: d[1])
    out = [list(dims[0])]
    for sz, st in dims[1:]:
        last = out[-1]
        if last[1] * last[0] == st:
            last[0] *= sz
        else:
            out.append([sz, st])
    return [(sz, st) for sz, st in out]


def _launch_strided(t, dt):
    dims = _collapse(t)
    nd = len(dims)
    if nd > _MAX_DIMS:
        raise NotImplementedError(
            "foreach erf cannot address a view with %d collapsed dimensions" % nd
        )
    numel = t.numel()
    sizes = [d[0] for d in dims]
    strides = [d[1] for d in dims]
    sizes += [1] * (_MAX_DIMS - nd)
    strides += [0] * (_MAX_DIMS - nd)
    args = [t.data_ptr(), numel] + sizes + strides
    _erf_strided_kernel[((numel + _BLOCK - 1) // _BLOCK,)](
        *args, ND=nd, DT=dt, BLOCK=_BLOCK, num_warps=_WARPS
    )


def _launch_flat(batch, nums, meta, maxn, dt):
    nt = len(batch)
    chunks = (maxn + _BLOCK - 1) // _BLOCK
    if nt == 1:
        _erf_one_kernel[(chunks,)](
            batch[0],
            maxn,
            BLOCK=_BLOCK,
            I64=maxn > 0x7FFFFF00,
            num_warps=_WARPS,
        )
        return
    if maxn <= _META_LIMIT:
        dev_meta = _device_meta(meta, batch[0].device)
        _erf_meta_kernel[(nt, chunks)](dev_meta, DT=dt, BLOCK=_BLOCK, num_warps=_WARPS)
        return
    i64 = maxn > 0x7FFFFF00
    pos = 0
    while pos < nt:
        stop = pos + _TYPED_SLOTS
        if stop > nt:
            stop = nt
        ptrs = batch[pos:stop]
        part = len(ptrs)
        counts = nums[pos:stop]
        pad = ptrs[0]
        while len(ptrs) < _TYPED_SLOTS:
            ptrs.append(pad)
            counts.append(0)
        _erf_batch_kernel[(part, chunks)](
            *ptrs, *counts, BLOCK=_BLOCK, I64=i64, num_warps=_WARPS
        )
        pos = stop


def foreach_erf_(self):
    if not isinstance(self, (list, tuple)):
        raise TypeError("argument 'self' must be tuple of Tensors")
    n = len(self)
    if n == 0:
        raise RuntimeError("Tensor list must have at least one tensor.")

    # Tier 1: one contiguous floating tensor needs no bookkeeping at all.
    if n == 1:
        t = self[0]
        if isinstance(t, torch.Tensor):
            dt = t.dtype
            if dt in _FLOATS and t.is_contiguous():
                numel = t.numel()
                if numel == 0:
                    return None
                _erf_one_kernel[((numel + _BLOCK - 1) // _BLOCK,)](
                    t,
                    numel,
                    BLOCK=_BLOCK,
                    I64=numel > 0x7FFFFF00,
                    num_warps=_WARPS,
                )
                return None

    # Tier 2: a uniform list (one dtype, one device, all contiguous, no two byte
    # spans touching) packs straight from the live tensors.  Anything else
    # abandons this tier before a single element has been mutated.
    first = self[0]
    if (
        isinstance(first, torch.Tensor)
        and first.dtype in _FLOATS
        and first.is_contiguous()
    ):
        dt = first.dtype
        dev = first.get_device()
        esz = first.element_size()
        batch = []
        nums = []
        meta = []
        box_lo = 0
        box_hi = 0
        maxn = 0
        for t in self:
            if (
                not isinstance(t, torch.Tensor)
                or t.dtype is not dt
                or not t.is_contiguous()
                or t.get_device() != dev
            ):
                batch = None
                break
            numel = t.numel()
            if numel == 0:
                continue
            ptr = t.data_ptr()
            hi = ptr + numel * esz
            if batch:
                if ptr < box_hi and hi > box_lo:
                    batch = None
                    break
                if ptr >= box_hi:
                    box_hi = hi
                else:
                    box_lo = ptr
            else:
                box_lo = ptr
                box_hi = hi
            batch.append(t)
            nums.append(numel)
            meta.append(ptr)
            meta.append(numel)
            if numel > maxn:
                maxn = numel
        if batch is not None:
            if batch:
                _launch_flat(batch, nums, meta, maxn, _TL_DTYPE[dt])
            return None

    # Tier 3: general list.
    tensors = self
    recs = [None] * n
    for i in range(n):
        t = tensors[i]
        if not isinstance(t, torch.Tensor):
            raise TypeError(
                "expected Tensor as element %d in argument 0, but got %s"
                % (i, type(t).__name__)
            )
        dt = t.dtype
        if dt not in _FLOATS:
            if dt.is_complex:
                raise NotImplementedError(
                    "\"foreach_unary_op_cuda_\" not implemented for '%s'"
                    % _ATEN_NAME.get(dt, str(dt))
                )
            raise RuntimeError(
                "result type Float can't be cast to the desired output type %s"
                % _ATEN_NAME.get(dt, str(dt))
            )
        r = _record(t, dt)
        recs[i] = r
        if not r[2] and r[1] > 1:
            if torch._debug_has_internal_overlap(t) == 1:
                raise RuntimeError(_OVERLAP_MSG)

    i = 0
    while i < n:
        r = recs[i]
        numel = r[1]
        if numel == 0:
            i += 1
            continue
        t = tensors[i]
        if not r[2]:
            _launch_strided(t, _TL_DTYPE[r[5]])
            i += 1
            continue

        dt = r[5]
        dev = r[6]
        limit = _META_SLOTS if numel <= _META_LIMIT else _TYPED_SLOTS
        ptr = r[0]
        lo = ptr
        hi = r[4]
        batch = [t]
        nums = [numel]
        meta = [ptr, numel]
        spans = [(lo, hi)]
        box_lo = lo
        box_hi = hi
        maxn = numel
        j = i + 1
        while j < n and len(batch) < limit:
            ur = recs[j]
            if ur[5] is not dt or ur[6] != dev:
                break
            unumel = ur[1]
            if unumel == 0 or not ur[2]:
                break
            ulo = ur[0]
            uhi = ur[4]
            if ulo < box_hi and uhi > box_lo:
                # The candidate falls inside the group's bounding box, so run
                # the exact pairwise test and, on a real clash, check whether it
                # is simply a repeated view of an element already handled.
                clash = False
                for k in range(len(spans)):
                    klo = spans[k][0]
                    if klo < uhi and ulo < spans[k][1]:
                        clash = True
                        break
                if clash:
                    u = tensors[j]
                    dup = False
                    for k in range(len(batch)):
                        u2 = batch[k]
                        if (
                            u2.data_ptr() == ulo
                            and u2.shape == u.shape
                            and u2.stride() == u.stride()
                            and u2.storage_offset() == u.storage_offset()
                        ):
                            dup = True
                            break
                    if dup:
                        j += 1
                        continue
                    break
                if ulo < box_lo:
                    box_lo = ulo
                if uhi > box_hi:
                    box_hi = uhi
            elif ulo >= box_hi:
                box_hi = uhi
            else:
                box_lo = ulo
            batch.append(tensors[j])
            nums.append(unumel)
            meta.append(ulo)
            meta.append(unumel)
            spans.append((ulo, uhi))
            if unumel > maxn:
                maxn = unumel
            j += 1

        _launch_flat(batch, nums, meta, maxn, _TL_DTYPE[dt])
        i = j

    return None
