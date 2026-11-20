"""In-place multi-tensor erfc (aten::_foreach_erfc_) for the MUSA target.

Architecture
------------
Tensors are bucketed by ``(dtype, numel, contiguity)``.  A bucket of same-dtype
contiguous tensors with the *same* element count is served by a single Triton
launch: the entry pointers travel as ordinary pointer arguments and the entry is
selected inside the kernel from ``program_id(0)``, so one launch replaces up to
16 per-entry launches without a host->device metadata copy.  Singleton buckets
use a lean three-argument kernel.  Non-contiguous entries use a generic kernel
that walks coalesced strides (rank <= 8) with integer division, which keeps the
compiled kernel independent of the concrete shape.

Element math is the backend libdevice erfc with fp32 opmath for 16-bit storage
types (ATen's ``opmath_t``) and native fp64 for float64 storage.
"""

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

_BLOCK = 1024
_WARPS = 4
_RMAX = 8
_KMAX = 16

_TORCH_NAME = {
    torch.int64: "Long",
    torch.int32: "Int",
    torch.int16: "Short",
    torch.int8: "Char",
    torch.uint8: "Byte",
    torch.bool: "Bool",
}
_COMPLEX_NAME = {
    torch.complex32: "ComplexHalf",
    torch.complex64: "ComplexFloat",
    torch.complex128: "ComplexDouble",
}
_OVERLAP_MSG = (
    "unsupported operation: more than one element of the written-to tensor "
    "refers to a single memory location. Please clone() the tensor before "
    "performing the operation."
)


@triton.jit
def _erfc_val(x):
    if x.dtype == tl.float64:
        return libdevice.erfc(x)
    else:
        return libdevice.erfc(x.to(tl.float32)).to(x.dtype)


@triton.jit(do_not_specialize=["n"])
def _erfc_flat_kernel(p, n, EVEN: tl.constexpr, BLOCK: tl.constexpr):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    if EVEN:
        x = tl.load(p + off)
        tl.store(p + off, _erfc_val(x))
    else:
        m = off < n
        x = tl.load(p + off, mask=m)
        tl.store(p + off, _erfc_val(x), mask=m)


@triton.jit(do_not_specialize=["n"])
def _erfc_fused_kernel(
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
    EVEN: tl.constexpr,
    BLOCK: tl.constexpr,
):
    t = tl.program_id(0)
    p = a0
    if t == 1:
        p = a1
    if t == 2:
        p = a2
    if t == 3:
        p = a3
    if t == 4:
        p = a4
    if t == 5:
        p = a5
    if t == 6:
        p = a6
    if t == 7:
        p = a7
    if t == 8:
        p = a8
    if t == 9:
        p = a9
    if t == 10:
        p = a10
    if t == 11:
        p = a11
    if t == 12:
        p = a12
    if t == 13:
        p = a13
    if t == 14:
        p = a14
    if t == 15:
        p = a15
    off = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    if EVEN:
        x = tl.load(p + off)
        tl.store(p + off, _erfc_val(x))
    else:
        m = off < n
        x = tl.load(p + off, mask=m)
        tl.store(p + off, _erfc_val(x), mask=m)


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
        "t0",
        "t1",
        "t2",
        "t3",
        "t4",
        "t5",
        "t6",
        "t7",
    ]
)
def _erfc_strided_kernel(
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
    R: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    m = i < n
    off = tl.zeros([BLOCK], dtype=tl.int64)
    # Arrays are ordered innermost-first and front-padded to 8 entries, so the
    # outermost real dimension always sits at index 7 and the flat logical
    # index is decomposed from the innermost side with modulo.
    if R > 7:
        c = i % s0
        off += c * t0
        i = i // s0
    if R > 6:
        c = i % s1
        off += c * t1
        i = i // s1
    if R > 5:
        c = i % s2
        off += c * t2
        i = i // s2
    if R > 4:
        c = i % s3
        off += c * t3
        i = i // s3
    if R > 3:
        c = i % s4
        off += c * t4
        i = i // s4
    if R > 2:
        c = i % s5
        off += c * t5
        i = i // s5
    if R > 1:
        c = i % s6
        off += c * t6
        i = i // s6
    off += i * t7
    x = tl.load(p + off, mask=m)
    tl.store(p + off, _erfc_val(x), mask=m)


def _coalesce(t):
    """Coalesced (shape, stride) in outermost-first order plus the rank.

    Size-1 dimensions are dropped, and adjacent dimensions are merged when the
    outer stride equals the inner stride times the inner size.  The merged
    extent takes the *inner* stride, because it enumerates memory in
    increasing order.
    """
    shape = []
    stride = []
    for sz, st in zip(t.shape, t.stride()):
        if sz == 1:
            continue
        if shape and stride[-1] == st * sz:
            shape[-1] *= sz
            stride[-1] = st
        else:
            shape.append(sz)
            stride.append(st)
    if not shape:
        return (1,), (1,), 1
    return tuple(shape), tuple(stride), len(shape)


def _flat_ok(t):
    """True when ``data_ptr() .. data_ptr() + numel`` covers exactly the view.

    Non-overlapping-and-dense views (transposes, permutes, channels-last and
    dense slices) can then be traversed in memory order: erfc is elementwise,
    so each element receives the same value whichever order it is visited in,
    and the flat kernel costs exactly what the contiguous path costs.
    """
    if t.is_contiguous():
        return True
    if t.numel() <= 1:
        return True
    expected = 1
    for st, sz in sorted(zip(t.stride(), t.shape)):
        if sz == 1:
            continue
        if st != expected:
            return False
        expected *= sz
    return True


def _launch_flat(t, n):
    _erfc_flat_kernel[((n + _BLOCK - 1) // _BLOCK,)](
        t, n, EVEN=(n % _BLOCK == 0), BLOCK=_BLOCK, num_warps=_WARPS
    )


def _launch_strided(t):
    n = t.numel()
    shape, stride, rank = _coalesce(t)
    while rank > _RMAX:
        for idx in range(t.shape[0]):
            _launch_strided(t.narrow(0, idx, 1).squeeze(0))
        return
    shapes = [1] * (_RMAX - rank) + list(shape)[::-1]
    strides = [0] * (_RMAX - rank) + list(stride)[::-1]
    _erfc_strided_kernel[((n + _BLOCK - 1) // _BLOCK,)](
        t,
        n,
        *shapes,
        *strides,
        R=rank,
        BLOCK=_BLOCK,
        num_warps=_WARPS,
    )


def _launch_bucket(ts, n):
    m = len(ts)
    if m == 1:
        _launch_flat(ts[0], n)
        return
    even = n % _BLOCK == 0
    chunks = (n + _BLOCK - 1) // _BLOCK
    if m < _KMAX:
        ptrs = list(ts) + [ts[0]] * (_KMAX - m)
    else:
        ptrs = list(ts)
    _erfc_fused_kernel[(m, chunks)](*ptrs, n, EVEN=even, BLOCK=_BLOCK, num_warps=_WARPS)


def _bad_dtype(dt):
    if dt.is_complex:
        raise NotImplementedError(
            '"foreach_unary_op_cuda_" not implemented for \''
            + _COMPLEX_NAME.get(dt, str(dt).replace("torch.", "").capitalize())
            + "'"
        )
    raise RuntimeError(
        "result type Float can't be cast to the desired output type "
        + _TORCH_NAME.get(dt, str(dt).replace("torch.", "").capitalize())
    )


def _foreach_erfc_(self):
    tensors = self
    if not isinstance(tensors, (list, tuple)):
        raise TypeError(
            "_foreach_erfc_(): argument 'self' (position 1) must be tuple of "
            "Tensors, not " + type(tensors).__name__
        )
    if len(tensors) == 0:
        raise RuntimeError("Tensor list must have at least one tensor.")

    buckets = {}
    strided = []
    seen = None
    for t in tensors:
        if not isinstance(t, torch.Tensor):
            raise TypeError(
                "_foreach_erfc_(): argument 'self' must be tuple of Tensors"
            )
        dt = t.dtype
        if not dt.is_floating_point:
            _bad_dtype(dt)
        n = t.numel()
        if n == 0:
            continue
        if _flat_ok(t):
            key = (dt, n)
            ptr = t.data_ptr()
            if seen is None:
                seen = {ptr}
            else:
                if ptr in seen:
                    continue
                seen.add(ptr)
            bucket = buckets.get(key)
            if bucket is None:
                buckets[key] = [t]
            else:
                bucket.append(t)
        else:
            if n > 1 and 0 in t.stride():
                raise RuntimeError(_OVERLAP_MSG)
            strided.append(t)

    for (dt, n), ts in buckets.items():
        _launch_bucket(ts, n)
    for t in strided:
        _launch_strided(t)
    return None
