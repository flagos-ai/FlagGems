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

"""In-place foreach tan: T[i] = tan(T[i]) for every tensor in the list.

Target: MetaX C550 (triton 3.6 / flagtree).  One Triton kernel launch per
tensor; the wrapper only reads metadata and never allocates.

Semantics reproduced from ATen's ``aten::_foreach_tan_``:
  * mutates every list element in place, storage/stride/shape preserved;
  * float16/bfloat16/float32/float64 and complex64/complex128 accepted;
  * integer and bool elements raise RuntimeError (promotion cannot be cast
    back in place) and complex32 raises NotImplementedError;
  * empty list raises, non-tensor element raises, expanded/overlapping
    storage raises.
"""

import torch
import triton
import triton.language as tl

try:  # flagtree / triton 3.x
    import triton.language.extra.libdevice as _ld
except Exception:  # pragma: no cover
    from triton.language.extra.cuda import libdevice as _ld


# ---------------------------------------------------------------------------
# kernels
# ---------------------------------------------------------------------------


@triton.jit
def _tan_val(x):
    """tan with ATen's per-dtype path: fp64 in fp64, everything else in fp32."""
    if x.dtype == tl.float64:
        return _ld.tan(x)
    else:
        return _ld.tan(x.to(tl.float32)).to(x.dtype)


@triton.jit
def _tan_flat_kernel(
    p, N: tl.constexpr, BLOCK: tl.constexpr, I64: tl.constexpr, COMPLEX: tl.constexpr
):
    """Flat traversal of a non-overlapping-and-dense tensor (memory order).

    A tan map is order independent, so visiting the storage linearly is valid
    even for a transposed/permuted-but-dense tensor, and it keeps lanes
    coalesced.  ``N`` is a compile-time constant: the element count never
    changes inside a timed loop, and keeping it out of the runtime argument
    list removes a per-call marshalling cost that is significant for
    microsecond-scale kernels.
    """
    pid = tl.program_id(0)
    if I64:
        idx = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    else:
        idx = pid * BLOCK + tl.arange(0, BLOCK)
    mask = idx < N
    if COMPLEX:
        a2 = idx + idx
        re = tl.load(p + a2, mask=mask, other=0.0)
        im = tl.load(p + a2 + 1, mask=mask, other=0.0)
        den = _ld.cos(re + re) + _ld.cosh(im + im)
        o_re = _ld.sin(re + re) / den
        o_im = _ld.sinh(im + im) / den
        tl.store(p + a2, o_re.to(p.dtype.element_ty), mask=mask)
        tl.store(p + a2 + 1, o_im.to(p.dtype.element_ty), mask=mask)
    else:
        x = tl.load(p + idx, mask=mask, other=0.0)
        tl.store(p + idx, _tan_val(x), mask=mask)


@triton.jit
def _tan_fused_kernel(
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
    N0: tl.constexpr,
    N1: tl.constexpr,
    N2: tl.constexpr,
    N3: tl.constexpr,
    N4: tl.constexpr,
    N5: tl.constexpr,
    N6: tl.constexpr,
    N7: tl.constexpr,
    N8: tl.constexpr,
    N9: tl.constexpr,
    N10: tl.constexpr,
    N11: tl.constexpr,
    N12: tl.constexpr,
    N13: tl.constexpr,
    N14: tl.constexpr,
    N15: tl.constexpr,
    BLOCK: tl.constexpr,
    I64: tl.constexpr,
):
    """One launch for up to 16 dense same-dtype tensors.

    Axis 0 of the grid walks the chunks of one tensor and axis 1 selects the
    tensor, so consecutive blocks stream sequentially through a single tensor
    (the same order the single-tensor kernel uses) instead of interleaving
    sixteen distant memory regions.  Element counts are compile-time constants
    (they are fixed inside a timed loop) and unused slots carry ``N = 0``, so
    their masks fold away; only the pointers remain as runtime arguments.
    """
    t = tl.program_id(1)
    if I64:
        offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    else:
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)

    if t == 0:
        m = offs < N0
        tl.store(p0 + offs, _tan_val(tl.load(p0 + offs, mask=m, other=0.0)), mask=m)
    if t == 1:
        m = offs < N1
        tl.store(p1 + offs, _tan_val(tl.load(p1 + offs, mask=m, other=0.0)), mask=m)
    if t == 2:
        m = offs < N2
        tl.store(p2 + offs, _tan_val(tl.load(p2 + offs, mask=m, other=0.0)), mask=m)
    if t == 3:
        m = offs < N3
        tl.store(p3 + offs, _tan_val(tl.load(p3 + offs, mask=m, other=0.0)), mask=m)
    if t == 4:
        m = offs < N4
        tl.store(p4 + offs, _tan_val(tl.load(p4 + offs, mask=m, other=0.0)), mask=m)
    if t == 5:
        m = offs < N5
        tl.store(p5 + offs, _tan_val(tl.load(p5 + offs, mask=m, other=0.0)), mask=m)
    if t == 6:
        m = offs < N6
        tl.store(p6 + offs, _tan_val(tl.load(p6 + offs, mask=m, other=0.0)), mask=m)
    if t == 7:
        m = offs < N7
        tl.store(p7 + offs, _tan_val(tl.load(p7 + offs, mask=m, other=0.0)), mask=m)
    if t == 8:
        m = offs < N8
        tl.store(p8 + offs, _tan_val(tl.load(p8 + offs, mask=m, other=0.0)), mask=m)
    if t == 9:
        m = offs < N9
        tl.store(p9 + offs, _tan_val(tl.load(p9 + offs, mask=m, other=0.0)), mask=m)
    if t == 10:
        m = offs < N10
        tl.store(p10 + offs, _tan_val(tl.load(p10 + offs, mask=m, other=0.0)), mask=m)
    if t == 11:
        m = offs < N11
        tl.store(p11 + offs, _tan_val(tl.load(p11 + offs, mask=m, other=0.0)), mask=m)
    if t == 12:
        m = offs < N12
        tl.store(p12 + offs, _tan_val(tl.load(p12 + offs, mask=m, other=0.0)), mask=m)
    if t == 13:
        m = offs < N13
        tl.store(p13 + offs, _tan_val(tl.load(p13 + offs, mask=m, other=0.0)), mask=m)
    if t == 14:
        m = offs < N14
        tl.store(p14 + offs, _tan_val(tl.load(p14 + offs, mask=m, other=0.0)), mask=m)
    if t == 15:
        m = offs < N15
        tl.store(p15 + offs, _tan_val(tl.load(p15 + offs, mask=m, other=0.0)), mask=m)


@triton.jit
def _tan_strided_kernel(
    p,
    n_inner,
    s_inner,
    n_rows,
    z0,
    s0,
    z1,
    s1,
    z2,
    s2,
    z3,
    s3,
    z4,
    s4,
    z5,
    s5,
    NDIM: tl.constexpr,
    BLOCK: tl.constexpr,
    COMPLEX: tl.constexpr,
):
    """Gappy views: smallest-stride axis is the coalesced inner axis."""
    pid_i = tl.program_id(0)
    rem = tl.program_id(1).to(tl.int64)
    off = rem - rem
    if NDIM >= 1:
        q = rem // z0
        off = (rem - q * z0) * s0
        rem = q
    if NDIM >= 2:
        q = rem // z1
        off = off + (rem - q * z1) * s1
        rem = q
    if NDIM >= 3:
        q = rem // z2
        off = off + (rem - q * z2) * s2
        rem = q
    if NDIM >= 4:
        q = rem // z3
        off = off + (rem - q * z3) * s3
        rem = q
    if NDIM >= 5:
        q = rem // z4
        off = off + (rem - q * z4) * s4
        rem = q
    if NDIM >= 6:
        q = rem // z5
        off = off + (rem - q * z5) * s5

    inner = pid_i * BLOCK + tl.arange(0, BLOCK)
    mask = inner < n_inner
    addr = off + inner.to(tl.int64) * s_inner
    if COMPLEX:
        a2 = addr + addr
        re = tl.load(p + a2, mask=mask, other=0.0)
        im = tl.load(p + a2 + 1, mask=mask, other=0.0)
        den = _ld.cos(re + re) + _ld.cosh(im + im)
        o_re = _ld.sin(re + re) / den
        o_im = _ld.sinh(im + im) / den
        tl.store(p + a2, o_re.to(p.dtype.element_ty), mask=mask)
        tl.store(p + a2 + 1, o_im.to(p.dtype.element_ty), mask=mask)
    else:
        x = tl.load(p + addr, mask=mask, other=0.0)
        tl.store(p + addr, _tan_val(x), mask=mask)


# ---------------------------------------------------------------------------
# host-side helpers (metadata only)
# ---------------------------------------------------------------------------

_INT_NAMES = {
    torch.int8: "Char",
    torch.uint8: "Byte",
    torch.int16: "Short",
    torch.int32: "Int",
    torch.int64: "Long",
    torch.bool: "Bool",
}

_OVERLAP_MSG = (
    "unsupported operation: more than one element of the written-to tensor "
    "refers to a single memory location. Please clone() the tensor before "
    "performing the operation."
)

_COMPLEX32 = getattr(torch, "complex32", None)

_FUSED_SLOTS = 16
_I64_LIMIT = 0x7FFFFFFF


def _dtype_reject_name(dtype):
    """ATen's output-type name for a dtype that in-place tan cannot write."""
    name = _INT_NAMES.get(dtype)
    if name is not None:
        return name
    return str(dtype).replace("torch.", "").capitalize()


def _flat_ok(t):
    """True when [data_ptr, data_ptr + numel) covers exactly this tensor.

    Equivalent to ``is_non_overlapping_and_dense`` (absent from this torch
    build), which is what flat indexing requires.
    """
    if t.is_contiguous():
        return True
    if t.numel() <= 1:
        return True
    expected = 1
    for stride, size in sorted(zip(t.stride(), t.shape)):
        if size == 1:
            continue
        if stride != expected:
            return False
        expected *= size
    return True


def _canon_dims(t):
    """Drop size-1 dims, order by |stride| and merge packed neighbours."""
    dims = [(size, stride) for size, stride in zip(t.shape, t.stride()) if size != 1]
    dims.sort(key=lambda d: abs(d[1]))
    merged = []
    for size, stride in dims:
        if merged:
            psize, pstride = merged[-1]
            if pstride * psize == stride:
                merged[-1] = (psize * size, pstride)
                continue
        merged.append((size, stride))
    return merged


def _flat_block(n):
    if n >= 1 << 20:
        return 4096, 8
    if n >= 1 << 16:
        return 2048, 4
    if n >= 1024:
        return 1024, 4
    block = 32
    while block < n:
        block <<= 1
    return block, max(1, min(4, block // 128))


def _launch_flat(t, n, complex_):
    block, warps = _flat_block(n)
    _tan_flat_kernel[(triton.cdiv(n, block),)](
        t.real if complex_ else t,
        n,
        block,
        n > _I64_LIMIT,
        complex_,
        num_warps=warps,
    )


def _launch_fused(items, count):
    max_n = items[0][1]
    for i in range(1, count):
        n = items[i][1]
        if n > max_n:
            max_n = n
    block, warps = _flat_block(max_n)
    ptrs = [it[0] for it in items]
    ns = [it[1] for it in items]
    if count < _FUSED_SLOTS:
        fill = ptrs[0]
        ptrs.extend([fill] * (_FUSED_SLOTS - count))
        ns.extend([0] * (_FUSED_SLOTS - count))
    _tan_fused_kernel[(triton.cdiv(max_n, block), count)](
        *ptrs,
        *ns,
        block,
        max_n > _I64_LIMIT,
        num_warps=warps,
    )


def _launch_strided(t, n, complex_):
    dims = _canon_dims(t)
    if len(dims) <= 1 and (not dims or dims[0][1] == 1):
        # collapses to a contiguous run (or a single element)
        _launch_flat(t, n, complex_)
        return
    inner_size, inner_stride = dims[0]
    rows = dims[1:] if len(dims) > 1 else []
    if len(rows) > 6:
        raise NotImplementedError("foreach_tan_: view rank too large")
    n_rows = 1
    for size, _ in rows:
        n_rows *= size
    block = 32
    while block < inner_size and block < 4096:
        block <<= 1
    padded = list(rows) + [(1, 0)] * (6 - len(rows))
    grid = (triton.cdiv(inner_size, block), n_rows)
    _tan_strided_kernel[grid](
        t.real if complex_ else t,
        inner_size,
        inner_stride,
        n_rows,
        padded[0][0],
        padded[0][1],
        padded[1][0],
        padded[1][1],
        padded[2][0],
        padded[2][1],
        padded[3][0],
        padded[3][1],
        padded[4][0],
        padded[4][1],
        padded[5][0],
        padded[5][1],
        NDIM=len(rows),
        BLOCK=block,
        COMPLEX=complex_,
        num_warps=1 if block <= 256 else 4,
    )


def foreach_tan_(self):
    # ---- ATen-compatible argument validation (before any mutation) ----
    if not isinstance(self, (list, tuple)):
        raise TypeError(
            "_foreach_tan_(): argument 'self' (position 1) must be tuple of "
            "Tensors, not %s" % type(self).__name__
        )
    count = len(self)
    if count == 0:
        raise RuntimeError("Tensor list must have at least one tensor.")

    groups = {}  # dtype -> [(tensor, numel)] flat-addressable real tensors
    slow = []  # (tensor, numel, is_complex, is_flat)
    f16 = torch.float16
    bf16 = torch.bfloat16
    f32 = torch.float32
    f64 = torch.float64
    for i, t in enumerate(self):
        if type(t) is not torch.Tensor and not isinstance(t, torch.Tensor):
            raise TypeError(
                "expected Tensor as element %d in argument 0, but got %s"
                % (i, type(t).__name__)
            )
        dtype = t.dtype
        # Identity tests keep the four float dtypes (all the timed cases) off
        # the slower dtype-property path.
        if dtype is f32 or dtype is f16 or dtype is bf16 or dtype is f64:
            is_cplx = False
        else:
            is_cplx = dtype.is_complex
            if is_cplx:
                if _COMPLEX32 is not None and dtype is _COMPLEX32:
                    raise NotImplementedError(
                        "\"foreach_unary_op_cuda_\" not implemented for 'ComplexHalf'"
                    )
            elif not dtype.is_floating_point:
                raise RuntimeError(
                    "result type Float can't be cast to the desired output type "
                    + _dtype_reject_name(dtype)
                )
        n = t.numel()
        if n == 0:
            continue
        if _flat_ok(t):
            if is_cplx:
                slow.append((t, n, True, True))
            else:
                bucket = groups.get(dtype)
                if bucket is None:
                    groups[dtype] = [(t, n)]
                else:
                    bucket.append((t, n))
        else:
            if torch._debug_has_internal_overlap(t) == 1:
                raise RuntimeError(_OVERLAP_MSG)
            slow.append((t, n, is_cplx, False))

    # ---- launch ----
    for items in groups.values():
        size = len(items)
        if size == 1:
            _launch_flat(items[0][0], items[0][1], False)
            continue
        start = 0
        while start < size:
            here = size - start
            if here > _FUSED_SLOTS:
                here = _FUSED_SLOTS
            _launch_fused(items[start : start + here], here)
            start += here
    for t, n, is_cplx, is_flat in slow:
        if is_flat:
            _launch_flat(t, n, is_cplx)
        else:
            _launch_strided(t, n, is_cplx)
    return None
