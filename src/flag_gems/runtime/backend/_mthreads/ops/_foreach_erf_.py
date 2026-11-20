"""In-place foreach Gauss error function: ``aten::_foreach_erf_`` in Triton.

Every element of the tensor list is mutated in place (storage, shape and
strides preserved), the call returns ``None``, and all elements are validated
before any launch so a rejected dtype never leaves a half-applied list.

Three device paths, chosen from tensor metadata only:

* fused  - a group of contiguous same-dtype elements whose *device* work is
  smaller than the host launches it replaces is packed into ONE launch with a
  2-D grid ``(chunks, slot)`` and up to 16 slots; per-slot base addresses and
  element counts ride in the kernel argument buffer, so no device metadata
  buffer and no host-to-device copy is needed.
* flat   - a lone contiguous element (or a group too large to profit from
  fusion) launches a plain 1-D kernel with a typed tensor argument, which keeps
  the pointer-alignment attributes that drive 128-bit vector access.
* strided- a view: dims are sorted by ascending stride (memory order, which
  restores coalescing), adjacent runs are collapsed on the host, and the kernel
  decomposes the flat index with a statically unrolled loop. Nothing is ever
  copied, so the write-through aliasing contract holds.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops._foreach_unary import _foreach_erf_ as default__foreach_erf_

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)

# Moore Threads hardware does not support fp64.
_SUPPORTED_DTYPES = {torch.float16, torch.bfloat16, torch.float32}

_BLOCK = 1024
_MAX_RANK = 8
_MAX_SLOTS = 16
_INT32_MAX = 0x7FFFFFFF
# Fusing k elements into one launch saves about (k - 1) launches of ~16 us of
# host dispatch, but a fused slot reaches memory through an int64 address, which
# carries no pointer-alignment attribute and costs about 1.1x in device time.
# Fuse only while the fused device work is cheaper than the launches it saves:
# ~10M elements per slot is roughly one 16 us launch at the measured 1.25 TB/s.
_FUSE_ELEMS_PER_SLOT = 10000000

_INT_NAMES = {
    torch.bool: "Bool",
    torch.uint8: "Byte",
    torch.int8: "Char",
    torch.int16: "Short",
    torch.int32: "Int",
    torch.int64: "Long",
}
for _name, _attr in (("UInt16", "uint16"), ("UInt32", "uint32"), ("UInt64", "uint64")):
    _dt = getattr(torch, _attr, None)
    if _dt is not None:
        _INT_NAMES[_dt] = _name

_COMPLEX_NAMES = {
    torch.complex64: "ComplexFloat",
    torch.complex128: "ComplexDouble",
}
_c32 = getattr(torch, "complex32", None)
if _c32 is not None:
    _COMPLEX_NAMES[_c32] = "ComplexHalf"

_TL_DTYPE = {
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
    torch.float32: tl.float32,
    torch.float64: tl.float64,
}

_OVERLAP_MSG = (
    "unsupported operation: more than one element of the written-to tensor "
    "refers to a single memory location. Please clone() the tensor before "
    "performing the operation."
)


@triton.jit
def _erf_val(x):
    # ATen's erf_ computes in acc_type (float for half/bfloat16/float).
    if x.dtype == tl.float64:
        y = tl.math.erf(x)
    else:
        y = tl.math.erf(x.to(tl.float32)).to(x.dtype)
    return y


@triton.jit
def _erf_flat_kernel(x_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    tl.store(x_ptr + offs, _erf_val(x), mask=mask)


@triton.jit
def _erf_multi_kernel(
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
    DT: tl.constexpr,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """One launch for up to 16 same-dtype contiguous tensors.

    Axis 1 selects the list slot, axis 0 the chunk inside it. Slots past a
    tensor's own element count are fully masked, so an unused slot (``n == 0``)
    costs one predicated load and an exit. The base address and element count
    are uniform scalars per program, selected with branchless compares, so the
    per-element work matches the single-tensor path.
    """
    t = tl.program_id(1)
    addr = a0
    n = n0
    if N > 1:
        m = t == 1
        addr = tl.where(m, a1, addr)
        n = tl.where(m, n1, n)
    if N > 2:
        m = t == 2
        addr = tl.where(m, a2, addr)
        n = tl.where(m, n2, n)
    if N > 3:
        m = t == 3
        addr = tl.where(m, a3, addr)
        n = tl.where(m, n3, n)
    if N > 4:
        m = t == 4
        addr = tl.where(m, a4, addr)
        n = tl.where(m, n4, n)
    if N > 5:
        m = t == 5
        addr = tl.where(m, a5, addr)
        n = tl.where(m, n5, n)
    if N > 6:
        m = t == 6
        addr = tl.where(m, a6, addr)
        n = tl.where(m, n6, n)
    if N > 7:
        m = t == 7
        addr = tl.where(m, a7, addr)
        n = tl.where(m, n7, n)
    if N > 8:
        m = t == 8
        addr = tl.where(m, a8, addr)
        n = tl.where(m, n8, n)
    if N > 9:
        m = t == 9
        addr = tl.where(m, a9, addr)
        n = tl.where(m, n9, n)
    if N > 10:
        m = t == 10
        addr = tl.where(m, a10, addr)
        n = tl.where(m, n10, n)
    if N > 11:
        m = t == 11
        addr = tl.where(m, a11, addr)
        n = tl.where(m, n11, n)
    if N > 12:
        m = t == 12
        addr = tl.where(m, a12, addr)
        n = tl.where(m, n12, n)
    if N > 13:
        m = t == 13
        addr = tl.where(m, a13, addr)
        n = tl.where(m, n13, n)
    if N > 14:
        m = t == 14
        addr = tl.where(m, a14, addr)
        n = tl.where(m, n14, n)
    if N > 15:
        m = t == 15
        addr = tl.where(m, a15, addr)
        n = tl.where(m, n15, n)

    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x_ptr = addr.to(tl.pointer_type(DT))
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    tl.store(x_ptr + offs, _erf_val(x), mask=mask)


@triton.jit(
    do_not_specialize=[
        "a0",
        "a1",
        "a2",
        "a3",
        "a4",
        "a5",
        "a6",
        "a7",
        "a8",
        "a9",
        "a10",
        "a11",
        "a12",
        "a13",
        "a14",
        "a15",
        "n",
    ]
)
def _erf_multi_u_kernel(
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
    n,
    DT: tl.constexpr,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Fused variant for a group whose elements all hold the same count.

    Host dispatch on this target costs ~13 us plus ~1.3 us per runtime argument,
    so the 16 counts of _erf_multi_kernel cost ~20 us on a launch-bound group
    even though 15 of them are redundant. The addresses and the shared count are
    marked ``do_not_specialize`` so the same compiled kernel serves every call.
    """
    t = tl.program_id(1)
    addr = a0
    if N > 1:
        addr = tl.where(t == 1, a1, addr)
    if N > 2:
        addr = tl.where(t == 2, a2, addr)
    if N > 3:
        addr = tl.where(t == 3, a3, addr)
    if N > 4:
        addr = tl.where(t == 4, a4, addr)
    if N > 5:
        addr = tl.where(t == 5, a5, addr)
    if N > 6:
        addr = tl.where(t == 6, a6, addr)
    if N > 7:
        addr = tl.where(t == 7, a7, addr)
    if N > 8:
        addr = tl.where(t == 8, a8, addr)
    if N > 9:
        addr = tl.where(t == 9, a9, addr)
    if N > 10:
        addr = tl.where(t == 10, a10, addr)
    if N > 11:
        addr = tl.where(t == 11, a11, addr)
    if N > 12:
        addr = tl.where(t == 12, a12, addr)
    if N > 13:
        addr = tl.where(t == 13, a13, addr)
    if N > 14:
        addr = tl.where(t == 14, a14, addr)
    if N > 15:
        addr = tl.where(t == 15, a15, addr)

    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x_ptr = addr.to(tl.pointer_type(DT))
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    tl.store(x_ptr + offs, _erf_val(x), mask=mask)


@triton.jit
def _erf_strided_kernel(
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
    RANK: tl.constexpr,
    BLOCK: tl.constexpr,
    IDX64: tl.constexpr,
):
    pid = tl.program_id(0)
    if IDX64:
        offs = (pid * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)
        off = tl.zeros((BLOCK,), dtype=tl.int64)
    else:
        offs = (pid * BLOCK + tl.arange(0, BLOCK)).to(tl.int32)
        off = tl.zeros((BLOCK,), dtype=tl.int32)
    mask = offs < n

    rem = offs
    if RANK > 7:
        off += (rem % s7) * t7
        rem = rem // s7
    if RANK > 6:
        off += (rem % s6) * t6
        rem = rem // s6
    if RANK > 5:
        off += (rem % s5) * t5
        rem = rem // s5
    if RANK > 4:
        off += (rem % s4) * t4
        rem = rem // s4
    if RANK > 3:
        off += (rem % s3) * t3
        rem = rem // s3
    if RANK > 2:
        off += (rem % s2) * t2
        rem = rem // s2
    if RANK > 1:
        off += (rem % s1) * t1
        rem = rem // s1
    if RANK > 0:
        off += (rem % s0) * t0

    x = tl.load(x_ptr + off, mask=mask, other=0.0)
    tl.store(x_ptr + off, _erf_val(x), mask=mask)


def _dims_of(t):
    """(size, stride) pairs in memory order, adjacent runs collapsed."""
    dims = [(sz, st) for sz, st in zip(t.shape, t.stride()) if sz != 1]
    if not dims:
        return [(1, 1)]
    merged = []
    for sz, st in sorted(dims, key=lambda p: p[1]):
        if merged:
            psz, pst = merged[-1]
            if pst * psz == st:
                merged[-1] = (psz * sz, pst)
                continue
        merged.append((sz, st))
    return merged


def _launch_strided(t, n, dims):
    sizes = [d[0] for d in dims]
    strides = [d[1] for d in dims]
    rank = len(dims)
    max_off = 0
    for sz, st in dims:
        max_off += (sz - 1) * st
    idx64 = max_off > _INT32_MAX or n > _INT32_MAX
    while len(sizes) < _MAX_RANK:
        sizes.append(1)
        strides.append(1)
    grid = ((n + _BLOCK - 1) // _BLOCK,)
    _erf_strided_kernel[grid](
        t,
        n,
        sizes[0],
        sizes[1],
        sizes[2],
        sizes[3],
        sizes[4],
        sizes[5],
        sizes[6],
        sizes[7],
        strides[0],
        strides[1],
        strides[2],
        strides[3],
        strides[4],
        strides[5],
        strides[6],
        strides[7],
        rank,
        _BLOCK,
        idx64,
    )


def _launch_one(t, n):
    if n > _INT32_MAX:
        _launch_strided(t, n, [(n, 1)])
    else:
        _erf_flat_kernel[((n + _BLOCK - 1) // _BLOCK,)](t, n, _BLOCK)


def _launch_group(items, dt):
    """One launch for a chunk of at most _MAX_SLOTS contiguous tensors.

    ``items`` holds ``(tensor, numel)`` pairs already measured during validation,
    so nothing is re-queried here.
    """
    k = len(items)
    if k == 1:
        _launch_one(items[0][0], items[0][1])
        return

    total = 0
    sizes = []
    for _, n in items:
        total += n
        sizes.append(n)
    if total > (k * _FUSE_ELEMS_PER_SLOT) // items[0][0].element_size():
        # Device work outweighs the launches fusion would remove: keep the
        # vectorized per-tensor path.
        for t, n in items:
            _launch_one(t, n)
        return

    max_n = max(sizes)
    addrs = [t.data_ptr() for t, _ in items]
    grid = ((max_n + _BLOCK - 1) // _BLOCK, k)
    if max_n == min(sizes):
        # Every slot holds the same count: one shared count is enough, and the
        # launch is ~20 us cheaper than the 16-count signature.
        addrs.extend([addrs[0]] * (_MAX_SLOTS - k))
        _erf_multi_u_kernel[grid](
            addrs[0],
            addrs[1],
            addrs[2],
            addrs[3],
            addrs[4],
            addrs[5],
            addrs[6],
            addrs[7],
            addrs[8],
            addrs[9],
            addrs[10],
            addrs[11],
            addrs[12],
            addrs[13],
            addrs[14],
            addrs[15],
            max_n,
            _TL_DTYPE[dt],
            k,
            _BLOCK,
        )
        return

    pad_a = addrs[0]
    sizes.extend([0] * (_MAX_SLOTS - k))
    addrs.extend([pad_a] * (_MAX_SLOTS - k))
    _erf_multi_kernel[grid](
        addrs[0],
        addrs[1],
        addrs[2],
        addrs[3],
        addrs[4],
        addrs[5],
        addrs[6],
        addrs[7],
        addrs[8],
        addrs[9],
        addrs[10],
        addrs[11],
        addrs[12],
        addrs[13],
        addrs[14],
        addrs[15],
        sizes[0],
        sizes[1],
        sizes[2],
        sizes[3],
        sizes[4],
        sizes[5],
        sizes[6],
        sizes[7],
        sizes[8],
        sizes[9],
        sizes[10],
        sizes[11],
        sizes[12],
        sizes[13],
        sizes[14],
        sizes[15],
        _TL_DTYPE[dt],
        k,
        _BLOCK,
    )


def _needs_generic_fallback(t):
    """Views whose collapsed rank exceeds the kernel's are handed to the generic
    executor instead of the strided kernel, which only carries ``_MAX_RANK``
    dimension descriptors."""
    return len(_dims_of(t)) > _MAX_RANK


def _foreach_erf_(self):
    logger.debug("GEMS_MTHREADS _FOREACH_ERF_")

    tensors = self
    if not isinstance(tensors, (list, tuple)):
        raise TypeError("argument 'self' must be tuple of Tensors")
    if len(tensors) == 0:
        raise RuntimeError("Tensor list must have at least one tensor.")

    # The specialized path only accepts the float dtypes the MUSA kernels carry
    # on a MUSA device; everything else keeps the generic executor's semantics.
    for t in tensors:
        if (
            not isinstance(t, torch.Tensor)
            or t.device.type != "musa"
            or t.dtype not in _SUPPORTED_DTYPES
        ):
            return default__foreach_erf_(self)

    buckets = {}
    strided = []
    for t in tensors:
        if not isinstance(t, torch.Tensor):
            raise TypeError("argument 'self' must be tuple of Tensors")
        dt = t.dtype
        if not dt.is_floating_point:
            if dt.is_complex:
                raise NotImplementedError(
                    "\"foreach_unary_op_cuda_\" not implemented for '%s'"
                    % _COMPLEX_NAMES.get(dt, str(dt))
                )
            raise RuntimeError(
                "result type Float can't be cast to the desired output type %s"
                % _INT_NAMES.get(dt, str(dt))
            )
        n = t.numel()
        if n == 0:
            continue
        if t.is_contiguous():
            items = buckets.get(dt)
            if items is None:
                buckets[dt] = [(t, n)]
            else:
                items.append((t, n))
            continue
        if torch._debug_has_internal_overlap(t) == 1:
            raise RuntimeError(_OVERLAP_MSG)
        if _needs_generic_fallback(t):
            return default__foreach_erf_(self)
        strided.append((t, n, _dims_of(t)))

    for dt, items in buckets.items():
        if dt not in _TL_DTYPE:
            raise RuntimeError(
                "result type Float can't be cast to the desired output type %s"
                % _INT_NAMES.get(dt, str(dt))
            )
        k = len(items)
        if k <= _MAX_SLOTS:
            _launch_group(items, dt)
        else:
            for i in range(0, k, _MAX_SLOTS):
                _launch_group(items[i : i + _MAX_SLOTS], dt)
    for t, n, dims in strided:
        _launch_strided(t, n, dims)
