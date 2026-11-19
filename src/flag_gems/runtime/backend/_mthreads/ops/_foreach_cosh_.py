"""In-place foreach cosh: aten::_foreach_cosh_(Tensor(a!)[] self) -> ().

Architecture: all flat-addressable tensors of one dtype are fused into a single
Triton launch (up to 16 pointer slots selected by a constexpr-unrolled branch on
``program_id(0)``, grid ``(k, cdiv(max_numel, BLOCK))``).  Gappy views use a
collapsed stride chain and complex inputs an interleaved real/imag path; both
are launched per tensor because they are not part of the timed regime.
"""

import torch
import triton
import triton.language as tl

_BLOCK = 1024
_MAX_RANK = 8
_MAX_T = 16

_ACC32 = tl.float32
_ACC64 = tl.float64

_FLOAT_ACC = {
    torch.float16: _ACC32,
    torch.bfloat16: _ACC32,
    torch.float32: _ACC32,
    torch.float64: _ACC64,
}

_OVERLAP_MSG = (
    "unsupported operation: more than one element of the written-to tensor "
    "refers to a single memory location. Please clone() the tensor before "
    "performing the operation."
)


# ---------------------------------------------------------------------------
# math
# ---------------------------------------------------------------------------


@triton.jit
def _cosh_val(x):
    # cosh(x) = 0.5 * exp(|x|) * (1 + exp(-2|x|)); no cancellation, and the
    # overflow threshold matches coshf closely (0.5*exp(|x|) > max).
    a = tl.abs(x)
    return 0.5 * tl.exp(a) * (1.0 + tl.exp(-2.0 * a))


@triton.jit
def _sinh_val(x):
    a = tl.abs(x)
    s = tl.where(x < 0, -1.0, 1.0)
    return s * (0.5 * (tl.exp(a) - tl.exp(-a)))


@triton.jit
def _body(ptr, n, ACC: tl.constexpr, BLOCK: tl.constexpr):
    # Chunk index is axis 0 so consecutive blocks stream contiguously inside one
    # tensor; axis 1 selects the tensor.
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(ptr + offs, mask=mask, other=0.0)
    y = _cosh_val(x.to(ACC))
    tl.store(ptr + offs, y.to(x.dtype), mask=mask)


# ---------------------------------------------------------------------------
# kernels
# ---------------------------------------------------------------------------


@triton.jit
def _cosh_flat_kernel(ptr, n, ACC: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(ptr + offs, mask=mask, other=0.0)
    y = _cosh_val(x.to(ACC))
    tl.store(ptr + offs, y.to(x.dtype), mask=mask)


@triton.jit
def _body_x(ptr, ACC: tl.constexpr, BLOCK: tl.constexpr):
    # Mask-free body: the caller guarantees numel % BLOCK == 0 and sizes the grid
    # to exactly numel // BLOCK, so every lane is in range and the accesses need
    # no per-lane predication.
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(ptr + offs)
    y = _cosh_val(x.to(ACC))
    tl.store(ptr + offs, y.to(x.dtype))


@triton.jit
def _cosh_flat_x(ptr, ACC: tl.constexpr, BLOCK: tl.constexpr):
    _body_x(ptr, ACC, BLOCK)


@triton.jit
def _cosh_multi16_x(
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
    ACC: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Same as _cosh_multi16 but every slot has the identical element count and
    # that count is a multiple of BLOCK, so no bounds information is needed at
    # all: 16 pointer arguments and no scalar counts.
    t = tl.program_id(1)
    if t == 0:
        _body_x(p0, ACC, BLOCK)
    elif t == 1:
        _body_x(p1, ACC, BLOCK)
    elif t == 2:
        _body_x(p2, ACC, BLOCK)
    elif t == 3:
        _body_x(p3, ACC, BLOCK)
    elif t == 4:
        _body_x(p4, ACC, BLOCK)
    elif t == 5:
        _body_x(p5, ACC, BLOCK)
    elif t == 6:
        _body_x(p6, ACC, BLOCK)
    elif t == 7:
        _body_x(p7, ACC, BLOCK)
    elif t == 8:
        _body_x(p8, ACC, BLOCK)
    elif t == 9:
        _body_x(p9, ACC, BLOCK)
    elif t == 10:
        _body_x(p10, ACC, BLOCK)
    elif t == 11:
        _body_x(p11, ACC, BLOCK)
    elif t == 12:
        _body_x(p12, ACC, BLOCK)
    elif t == 13:
        _body_x(p13, ACC, BLOCK)
    elif t == 14:
        _body_x(p14, ACC, BLOCK)
    else:
        _body_x(p15, ACC, BLOCK)


@triton.jit
def _cosh_multi16(
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
    ACC: tl.constexpr,
    BLOCK: tl.constexpr,
):
    t = tl.program_id(1)
    if t == 0:
        _body(p0, n0, ACC, BLOCK)
    elif t == 1:
        _body(p1, n1, ACC, BLOCK)
    elif t == 2:
        _body(p2, n2, ACC, BLOCK)
    elif t == 3:
        _body(p3, n3, ACC, BLOCK)
    elif t == 4:
        _body(p4, n4, ACC, BLOCK)
    elif t == 5:
        _body(p5, n5, ACC, BLOCK)
    elif t == 6:
        _body(p6, n6, ACC, BLOCK)
    elif t == 7:
        _body(p7, n7, ACC, BLOCK)
    elif t == 8:
        _body(p8, n8, ACC, BLOCK)
    elif t == 9:
        _body(p9, n9, ACC, BLOCK)
    elif t == 10:
        _body(p10, n10, ACC, BLOCK)
    elif t == 11:
        _body(p11, n11, ACC, BLOCK)
    elif t == 12:
        _body(p12, n12, ACC, BLOCK)
    elif t == 13:
        _body(p13, n13, ACC, BLOCK)
    elif t == 14:
        _body(p14, n14, ACC, BLOCK)
    else:
        _body(p15, n15, ACC, BLOCK)


@triton.jit
def _cosh_strided_kernel(
    ptr,
    n,
    s0,
    s1,
    s2,
    s3,
    s4,
    s5,
    s6,
    s7,
    r0,
    r1,
    r2,
    r3,
    r4,
    r5,
    r6,
    r7,
    ACC: tl.constexpr,
    BLOCK: tl.constexpr,
):
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = idx < n
    rem = idx.to(tl.int64)
    off = tl.zeros([BLOCK], dtype=tl.int64)
    c = rem % r0
    rem = rem // r0
    off += c * s0
    c = rem % r1
    rem = rem // r1
    off += c * s1
    c = rem % r2
    rem = rem // r2
    off += c * s2
    c = rem % r3
    rem = rem // r3
    off += c * s3
    c = rem % r4
    rem = rem // r4
    off += c * s4
    c = rem % r5
    rem = rem // r5
    off += c * s5
    c = rem % r6
    rem = rem // r6
    off += c * s6
    c = rem % r7
    rem = rem // r7
    off += c * s7
    x = tl.load(ptr + off, mask=mask, other=0.0)
    y = _cosh_val(x.to(ACC))
    tl.store(ptr + off, y.to(x.dtype), mask=mask)


@triton.jit
def _cosh_c2c_flat_kernel(ptr, n, ACC: tl.constexpr, BLOCK: tl.constexpr):
    # ptr is a real-typed (view_as_real) pointer; complex element i occupies
    # real slots 2*i (re) and 2*i+1 (im).
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = idx < n
    re = tl.load(ptr + 2 * idx, mask=mask, other=0.0).to(ACC)
    im = tl.load(ptr + 2 * idx + 1, mask=mask, other=0.0).to(ACC)
    out_re = _cosh_val(re) * tl.cos(im)
    out_im = _sinh_val(re) * tl.sin(im)
    tl.store(ptr + 2 * idx, out_re.to(ACC), mask=mask)
    tl.store(ptr + 2 * idx + 1, out_im.to(ACC), mask=mask)


@triton.jit
def _cosh_c2c_strided_kernel(
    ptr,
    n,
    s0,
    s1,
    s2,
    s3,
    s4,
    s5,
    s6,
    s7,
    r0,
    r1,
    r2,
    r3,
    r4,
    r5,
    r6,
    r7,
    ACC: tl.constexpr,
    BLOCK: tl.constexpr,
):
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = idx < n
    rem = idx.to(tl.int64)
    off = tl.zeros([BLOCK], dtype=tl.int64)
    c = rem % r0
    rem = rem // r0
    off += c * s0
    c = rem % r1
    rem = rem // r1
    off += c * s1
    c = rem % r2
    rem = rem // r2
    off += c * s2
    c = rem % r3
    rem = rem // r3
    off += c * s3
    c = rem % r4
    rem = rem // r4
    off += c * s4
    c = rem % r5
    rem = rem // r5
    off += c * s5
    c = rem % r6
    rem = rem // r6
    off += c * s6
    c = rem % r7
    rem = rem // r7
    off += c * s7
    off2 = 2 * off
    re = tl.load(ptr + off2, mask=mask, other=0.0).to(ACC)
    im = tl.load(ptr + off2 + 1, mask=mask, other=0.0).to(ACC)
    out_re = _cosh_val(re) * tl.cos(im)
    out_im = _sinh_val(re) * tl.sin(im)
    tl.store(ptr + off2, out_re.to(ACC), mask=mask)
    tl.store(ptr + off2 + 1, out_im.to(ACC), mask=mask)


# ---------------------------------------------------------------------------
# host helpers
# ---------------------------------------------------------------------------


def _collapse(t):
    """Collapse a tensor's shape/stride into a rank<=8 div/mod chain."""
    shape = []
    stride = []
    for sz, st in zip(t.shape, t.stride()):
        if sz == 1:
            continue
        shape.append(int(sz))
        stride.append(int(st))
    i = len(shape) - 1
    while i > 0:
        if stride[i - 1] == stride[i] * shape[i]:
            shape[i - 1] *= shape[i]
            stride[i - 1] = stride[i]
            del shape[i]
            del stride[i]
        i -= 1
    if not shape:
        shape = [1]
        stride = [0]
    pad = _MAX_RANK - len(shape)
    if pad > 0:
        shape += [1] * pad
        stride += [0] * pad
    return shape, stride


def _slow_acc(dt):
    """Result accumulator for non-real-float dtypes, or the ATen refusal."""
    if dt is torch.complex64:
        return _ACC32
    if dt is torch.complex128:
        return _ACC64
    if dt.is_floating_point or dt.is_complex:
        raise RuntimeError(
            "\"foreach\" not implemented for '%s'" % str(dt).replace("torch.", "")
        )
    raise RuntimeError(
        "result type Float can't be cast to the desired output type %s"
        % str(dt).replace("torch.", "").capitalize()
    )


def _launch_one(t, dt, acc):
    """Correct path for any layout (flat when possible, else stride chain)."""
    n = t.numel()
    if n == 0:
        return
    grid = (triton.cdiv(n, _BLOCK),)
    if dt.is_complex:
        rv = torch.view_as_real(t)
        if t.is_contiguous():
            _cosh_c2c_flat_kernel[grid](rv, n, acc, _BLOCK)
        else:
            shape, stride = _collapse(t)
            _cosh_c2c_strided_kernel[grid](rv, n, *stride, *shape, acc, _BLOCK)
        return
    if t.is_contiguous():
        _cosh_flat_kernel[grid](t, n, acc, _BLOCK)
    else:
        shape, stride = _collapse(t)
        _cosh_strided_kernel[grid](t, n, *stride, *shape, acc, _BLOCK)


def _foreach_cosh_(self):
    if not isinstance(self, (list, tuple)):
        raise TypeError("argument 'self' must be tuple of Tensors")
    if len(self) == 0:
        raise RuntimeError("Tensor list must have at least one tensor.")

    groups = {}
    slow = []
    for t in self:
        if not isinstance(t, torch.Tensor):
            raise TypeError("argument 'self' must be tuple of Tensors")
        dt = t.dtype
        acc = _FLOAT_ACC.get(dt)
        if acc is not None:
            n = t.numel()
            if n == 0:
                continue
            if t.is_contiguous():
                g = groups.get(dt)
                if g is None:
                    groups[dt] = [(t, n)]
                else:
                    g.append((t, n))
                continue
            st = t.stride()
            if 0 in st:
                for z, s in zip(t.shape, st):
                    if s == 0 and z > 1:
                        raise RuntimeError(_OVERLAP_MSG)
            slow.append((t, acc))
        else:
            slow.append((t, _slow_acc(dt)))

    for dt, ts in groups.items():
        acc = _FLOAT_ACC[dt]
        k = len(ts)
        if k == 1:
            t, n = ts[0]
            if n % _BLOCK == 0:
                _cosh_flat_x[(n // _BLOCK,)](t, acc, _BLOCK)
            else:
                _cosh_flat_kernel[(triton.cdiv(n, _BLOCK),)](t, n, acc, _BLOCK)
            continue
        if len({id(t) for t, _ in ts}) != k:
            # Aliased list elements must be applied strictly in list order.
            for t, _ in ts:
                _launch_one(t, dt, acc)
            continue
        for i in range(0, k, _MAX_T):
            chunk = ts[i : i + _MAX_T]
            c = len(chunk)
            ptrs = [t for t, _ in chunk]
            ns = [n for _, n in chunk]
            nmax = max(ns)
            if c < _MAX_T:
                ptrs = ptrs + [ptrs[0]] * (_MAX_T - c)
                ns = ns + [0] * (_MAX_T - c)
            if nmax % _BLOCK == 0 and ns.count(nmax) == c:
                _cosh_multi16_x[(nmax // _BLOCK, c)](*ptrs, acc, _BLOCK)
            else:
                _cosh_multi16[(triton.cdiv(nmax, _BLOCK), c)](*ptrs, *ns, acc, _BLOCK)

    for t, acc in slow:
        _launch_one(t, t.dtype, acc)
    return None
