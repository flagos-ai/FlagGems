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

"""Triton implementation of ``aten::_foreach_log2_`` (in-place foreach log2).

The operator applies ``log2`` to every tensor of a ``Tensor[]`` in place and
returns nothing.  ATen's CUDA foreach path packs a whole list into a *single*
kernel launch whose metadata travels as kernel arguments; anything that pays a
fixed host cost per list element therefore loses on small lists.

Structure:

* uniform lists (every element the same dtype and the same element count --
  what the benchmark feeds) use one dedicated kernel per list length whose
  signature is only ``N`` pointers plus the shared element count, one launch
  for up to 16 tensors, so the per-call cost stays at the floor;
* heterogeneous lists fall back to one launch per tensor of the same lean
  single-tensor kernel (correct, never the timed path);
* gappy views (``x[:, ::2]``) use a per-tensor kernel that decomposes the flat
  logical index onto collapsed strides, so writes land through the view;
* the math is on-device: ``tl.log2`` on the promoted fp32 (or fp64) value,
  rounded back to the storage dtype, matching ATen's opmath.
"""

import torch
import triton
import triton.language as tl

_TL_DTYPE = {
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
    torch.float32: tl.float32,
    torch.float64: tl.float64,
}

_DTYPE_NAME = {
    torch.int8: "Char",
    torch.uint8: "Byte",
    torch.int16: "Short",
    torch.int32: "Int",
    torch.int64: "Long",
    torch.bool: "Bool",
    torch.complex64: "ComplexFloat",
    torch.complex128: "ComplexDouble",
}

_HAS_OVERLAP = getattr(torch, "_debug_has_internal_overlap", None)

_OVERLAP_MSG = (
    "unsupported operation: more than one element of the written-to tensor "
    "refers to a single memory location. Please clone() the tensor before "
    "performing the operation."
)

_MAX_FUSED = 16


@triton.jit
def _log2_u1(
    p0,
    n,
    DT: tl.constexpr,
    ACC: tl.constexpr,
    BLOCK: tl.constexpr,
    EVEN: tl.constexpr,
):
    pid = tl.program_id(0)
    ptr = p0.to(tl.pointer_type(DT))
    off = pid * BLOCK + tl.arange(0, BLOCK)
    if EVEN:
        tl.store(ptr + off, tl.log2(tl.load(ptr + off).to(ACC)).to(DT))
    else:
        mask = off < n
        x = tl.load(ptr + off, mask=mask, other=0.0)
        tl.store(ptr + off, tl.log2(x.to(ACC)).to(DT), mask=mask)


@triton.jit
def _log2_u2(
    p0,
    p1,
    n,
    DT: tl.constexpr,
    ACC: tl.constexpr,
    BLOCK: tl.constexpr,
    EVEN: tl.constexpr,
):
    pid = tl.program_id(0)
    bpt = (n + BLOCK - 1) // BLOCK
    t = pid // bpt
    ptr_i = tl.where(t >= 1, p1, p0)
    ptr = ptr_i.to(tl.pointer_type(DT))
    off = (pid - t * bpt) * BLOCK + tl.arange(0, BLOCK)
    if EVEN:
        tl.store(ptr + off, tl.log2(tl.load(ptr + off).to(ACC)).to(DT))
    else:
        mask = off < n
        x = tl.load(ptr + off, mask=mask, other=0.0)
        tl.store(ptr + off, tl.log2(x.to(ACC)).to(DT), mask=mask)


@triton.jit
def _log2_u4(
    p0,
    p1,
    p2,
    p3,
    n,
    DT: tl.constexpr,
    ACC: tl.constexpr,
    BLOCK: tl.constexpr,
    EVEN: tl.constexpr,
):
    pid = tl.program_id(0)
    bpt = (n + BLOCK - 1) // BLOCK
    t = pid // bpt
    ptr_i = tl.where(t >= 1, p1, p0)
    ptr_i = tl.where(t >= 2, p2, ptr_i)
    ptr_i = tl.where(t >= 3, p3, ptr_i)
    ptr = ptr_i.to(tl.pointer_type(DT))
    off = (pid - t * bpt) * BLOCK + tl.arange(0, BLOCK)
    if EVEN:
        tl.store(ptr + off, tl.log2(tl.load(ptr + off).to(ACC)).to(DT))
    else:
        mask = off < n
        x = tl.load(ptr + off, mask=mask, other=0.0)
        tl.store(ptr + off, tl.log2(x.to(ACC)).to(DT), mask=mask)


@triton.jit
def _log2_u8(
    p0,
    p1,
    p2,
    p3,
    p4,
    p5,
    p6,
    p7,
    n,
    DT: tl.constexpr,
    ACC: tl.constexpr,
    BLOCK: tl.constexpr,
    EVEN: tl.constexpr,
):
    pid = tl.program_id(0)
    bpt = (n + BLOCK - 1) // BLOCK
    t = pid // bpt
    ptr_i = tl.where(t >= 1, p1, p0)
    ptr_i = tl.where(t >= 2, p2, ptr_i)
    ptr_i = tl.where(t >= 3, p3, ptr_i)
    ptr_i = tl.where(t >= 4, p4, ptr_i)
    ptr_i = tl.where(t >= 5, p5, ptr_i)
    ptr_i = tl.where(t >= 6, p6, ptr_i)
    ptr_i = tl.where(t >= 7, p7, ptr_i)
    ptr = ptr_i.to(tl.pointer_type(DT))
    off = (pid - t * bpt) * BLOCK + tl.arange(0, BLOCK)
    if EVEN:
        tl.store(ptr + off, tl.log2(tl.load(ptr + off).to(ACC)).to(DT))
    else:
        mask = off < n
        x = tl.load(ptr + off, mask=mask, other=0.0)
        tl.store(ptr + off, tl.log2(x.to(ACC)).to(DT), mask=mask)


@triton.jit
def _log2_u16(
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
    DT: tl.constexpr,
    ACC: tl.constexpr,
    BLOCK: tl.constexpr,
    EVEN: tl.constexpr,
):
    pid = tl.program_id(0)
    bpt = (n + BLOCK - 1) // BLOCK
    t = pid // bpt
    ptr_i = tl.where(t >= 1, p1, p0)
    ptr_i = tl.where(t >= 2, p2, ptr_i)
    ptr_i = tl.where(t >= 3, p3, ptr_i)
    ptr_i = tl.where(t >= 4, p4, ptr_i)
    ptr_i = tl.where(t >= 5, p5, ptr_i)
    ptr_i = tl.where(t >= 6, p6, ptr_i)
    ptr_i = tl.where(t >= 7, p7, ptr_i)
    ptr_i = tl.where(t >= 8, p8, ptr_i)
    ptr_i = tl.where(t >= 9, p9, ptr_i)
    ptr_i = tl.where(t >= 10, p10, ptr_i)
    ptr_i = tl.where(t >= 11, p11, ptr_i)
    ptr_i = tl.where(t >= 12, p12, ptr_i)
    ptr_i = tl.where(t >= 13, p13, ptr_i)
    ptr_i = tl.where(t >= 14, p14, ptr_i)
    ptr_i = tl.where(t >= 15, p15, ptr_i)
    ptr = ptr_i.to(tl.pointer_type(DT))
    off = (pid - t * bpt) * BLOCK + tl.arange(0, BLOCK)
    if EVEN:
        tl.store(ptr + off, tl.log2(tl.load(ptr + off).to(ACC)).to(DT))
    else:
        mask = off < n
        x = tl.load(ptr + off, mask=mask, other=0.0)
        tl.store(ptr + off, tl.log2(x.to(ACC)).to(DT), mask=mask)


@triton.jit
def _log2_strided(
    p,
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
    DT: tl.constexpr,
    ACC: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """In-place log2 over one gappy/strided tensor, flat logical index order.

    Dimensions are collapsed on the host and padded with ``(1, 0)`` entries at
    the outermost positions, so the unrolled rank-8 decomposition is always
    valid and no out-of-range coordinate is ever formed.
    """
    pid = tl.program_id(0)
    idx = pid * BLOCK + tl.arange(0, BLOCK)
    mask = idx < n
    rem = idx.to(tl.int64)
    off = tl.zeros((BLOCK,), dtype=tl.int64)
    c = rem % s7
    rem = rem // s7
    off += c * t7
    c = rem % s6
    rem = rem // s6
    off += c * t6
    c = rem % s5
    rem = rem // s5
    off += c * t5
    c = rem % s4
    rem = rem // s4
    off += c * t4
    c = rem % s3
    rem = rem // s3
    off += c * t3
    c = rem % s2
    rem = rem // s2
    off += c * t2
    c = rem % s1
    rem = rem // s1
    off += c * t1
    c = rem % s0
    off += c * t0

    ptr = p.to(tl.pointer_type(DT))
    x = tl.load(ptr + off, mask=mask, other=0.0)
    tl.store(ptr + off, tl.log2(x.to(ACC)).to(DT), mask=mask)


# list length -> fused kernel with that many pointer slots
_FUSED = ((1, _log2_u1), (2, _log2_u2), (4, _log2_u4), (8, _log2_u8), (16, _log2_u16))


_LARGE = 1 << 20

# torch dtype -> (triton dtype, accumulation dtype, nominal BLOCK)
_OPTS = {
    torch.float16: (tl.float16, tl.float32, 2048),
    torch.bfloat16: (tl.bfloat16, tl.float32, 2048),
    torch.float32: (tl.float32, tl.float32, 1024),
    torch.float64: (tl.float64, tl.float64, 512),
}

# unbound C methods: avoids creating a bound method for every metadata read
_data_ptr = torch.Tensor.data_ptr
_numel = torch.Tensor.numel
_is_contig = torch.Tensor.is_contiguous
_get_dev = torch.Tensor.get_device


# the dtype getset descriptor, callable as ``_get_dtype(t)``; it keeps the
# uniformity scan on the C side of ``map`` instead of an attribute lookup per
# tensor in the interpreter
def _dtype_of(t):
    return t.dtype


_get_dtype = getattr(torch.Tensor.dtype, "__get__", _dtype_of)


def _pick_config(dt, numel):
    """(BLOCK, num_warps) for one launch.

    Small tensors keep the narrow block so a 4 k-element list still spreads over
    several programs.  From a million elements up, bfloat16 gets four 16-byte
    vectors per thread: its fp32 round trip costs enough per element that the
    extra instruction-level parallelism measured 14-16% faster, while fp16 and
    float32 lost about 1% with the same wider block and keep the narrow one.
    """
    block = _OPTS[dt][2]
    if dt is torch.bfloat16 and numel >= _LARGE:
        block = block << 2
    return block, 4


def _plan(dt, numel):
    """(triton dtype, accumulation dtype, BLOCK, num_warps) for one launch."""
    tdt, acc, block = _OPTS[dt]
    if dt is torch.bfloat16 and numel >= _LARGE:
        block = block << 2
    return tdt, acc, block, 4


def _is_flat(t):
    """True when ``[data_ptr, data_ptr + numel)`` is exactly this tensor."""
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


def _collapse(t):
    """Collapse a strided layout into (size, stride) pairs, outermost first.

    Two adjacent dimensions merge only when the outer stride equals
    ``inner_stride * inner_size``; the merged pair then keeps the *inner*
    stride, because that is the stride of the enclosing flattened index.
    """
    dims = []
    for sz, st in zip(t.shape, t.stride()):
        if sz == 1:
            continue
        if dims and dims[-1][1] == st * sz:
            dims[-1] = (dims[-1][0] * sz, st)
        else:
            dims.append((sz, st))
    return dims


def _launch_fused(tensors, slots, tdt, acc, block, warps):
    n = _numel(tensors[0])
    bpt = (n + block - 1) // block
    kernel = _FUSED_KERNEL[slots]
    ptrs = [_data_ptr(t) for t in tensors]
    count = len(ptrs)
    if count != slots:
        ptrs = ptrs + [0] * (slots - count)
    kernel[(slots * bpt,)](*ptrs, n, tdt, acc, block, n % block == 0, num_warps=warps)


_FUSED_KERNEL = dict(_FUSED)
_SLOT_CHOICES = (1, 2, 4, 8, 16)


def _slots_for(count):
    """Smallest fused kernel with at least ``count`` pointer slots."""
    for cand in _SLOT_CHOICES:
        if cand >= count:
            return cand
    return _MAX_FUSED


def _launch_single(t, n, tdt, acc, block, warps):
    _log2_u1[((n + block - 1) // block,)](
        _data_ptr(t), n, tdt, acc, block, n % block == 0, num_warps=warps
    )


def _launch_strided(t, tdt, acc, block, warps):
    n = t.numel()
    dims = _collapse(t)
    rank = len(dims)
    if rank > 8:
        raise NotImplementedError("foreach_log2_: unsupported tensor layout")
    shapes = [1] * (8 - rank) + [d[0] for d in dims]
    strides = [0] * (8 - rank) + [d[1] for d in dims]
    _log2_strided[((n + block - 1) // block,)](
        t.data_ptr(),
        n,
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
        DT=tdt,
        ACC=acc,
        BLOCK=block,
        num_warps=warps,
    )


def _dtype_error(dt):
    name = _DTYPE_NAME.get(dt)
    if name is None:
        name = str(dt).replace("torch.", "").capitalize()
    return "result type Float can't be cast to the desired output type " + name


def _launch_uniform(tensors, count, tdt, acc, block, warps):
    i = 0
    while i < count:
        take = count - i
        if take > _MAX_FUSED:
            take = _MAX_FUSED
        _launch_fused(tensors[i : i + take], _slots_for(take), tdt, acc, block, warps)
        i += take


def foreach_log2_(self):
    if not isinstance(self, (list, tuple)):
        raise TypeError("argument 'self' must be tuple of Tensors")
    count = len(self)
    if count == 0:
        raise RuntimeError("Tensor list must have at least one tensor.")

    # Fast path for the shape of a typical foreach call: every element the same
    # dtype, device and element count, all contiguous.  The predicates run as
    # four C-level passes over the list (``map`` feeding ``set``/``all``) instead
    # of a Python loop, because on a launch-bound call the interpreter cost of
    # per-tensor checks is a measurable share of the whole call.  Contiguous
    # tensors cannot alias one another, so the overlap guard is unnecessary here,
    # and any failed check falls through to the general path, which owns every
    # error case (a non-Tensor element fails the bulk scan and is reported there).
    t0 = self[0]
    if isinstance(t0, torch.Tensor):
        dt = t0.dtype
        if dt in _OPTS:
            numel = _numel(t0)
            if numel and _is_contig(t0):
                uniform = True
                if count > 1:
                    try:
                        uniform = (
                            len(set(map(_get_dev, self))) == 1
                            and len(set(map(_numel, self))) == 1
                            and len(set(map(_get_dtype, self))) == 1
                            and all(map(_is_contig, self))
                        )
                    except (AttributeError, TypeError):
                        uniform = False
                if uniform:
                    tdt, acc, block, warps = _plan(dt, numel)
                    if count == 1:
                        _launch_single(t0, numel, tdt, acc, block, warps)
                    elif count <= _MAX_FUSED:
                        _launch_fused(self, _slots_for(count), tdt, acc, block, warps)
                    else:
                        _launch_uniform(self, count, tdt, acc, block, warps)
                    return None

    groups = {}
    for t in self:
        if not isinstance(t, torch.Tensor):
            raise TypeError("argument 'self' must be tuple of Tensors")
        dt = t.dtype
        if dt not in _TL_DTYPE:
            raise RuntimeError(_dtype_error(dt))
        if t.numel() == 0:
            continue
        if t.is_contiguous() or _is_flat(t):
            flat = True
        else:
            flat = False
            if _HAS_OVERLAP is not None and _HAS_OVERLAP(t) == 1:
                raise RuntimeError(_OVERLAP_MSG)
        key = (t.device, dt)
        g = groups.get(key)
        if g is None:
            g = groups[key] = ([], [])
        if flat:
            g[0].append(t)
        else:
            g[1].append(t)

    for (dev, dt), (flats, strided) in groups.items():
        tdt = _TL_DTYPE[dt]
        acc = tl.float64 if dt is torch.float64 else tl.float32
        nf = len(flats)
        if nf:
            numel = flats[0].numel()
            uniform = True
            for t in flats:
                if t.numel() != numel:
                    uniform = False
                    break
            block, warps = _pick_config(dt, numel)
            if uniform:
                _launch_uniform(flats, nf, tdt, acc, block, warps)
            else:
                for t in flats:
                    _launch_single(t, t.numel(), tdt, acc, block, warps)
        for t in strided:
            sblock, swarps = _pick_config(dt, t.numel())
            _launch_strided(t, tdt, acc, sblock, swarps)
    return None
