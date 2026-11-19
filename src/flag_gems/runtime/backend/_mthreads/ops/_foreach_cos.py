import torch
import triton
import triton.language as tl

_MAX_DIMS = 8
_BLOCK = 1024
_NUM_WARPS = 4
_GROUP = 16
_SMALL_BLOCK = 512
_SMALL_WARPS = 2
_SMALL_MAX_ELEMS = 1 << 20
_WIDE_BLOCK = 2048
_WIDE_WARPS = 4
_SHARED_MAX_ELEMS = 1 << 22


@triton.jit
def _cos_block(
    in_ptr,
    out_ptr,
    size,
    local_block,
    BLOCK: tl.constexpr,
    COMPUTE_F32: tl.constexpr,
    EXACT: tl.constexpr,
):
    off = local_block * BLOCK + tl.arange(0, BLOCK)
    if EXACT:
        mask = None
    else:
        mask = off < size
    x = tl.load(in_ptr + off, mask=mask)
    if COMPUTE_F32:
        y = tl.cos(x.to(tl.float32))
        tl.store(out_ptr + off, y.to(out_ptr.dtype.element_ty), mask=mask)
    else:
        y = tl.cos(x)
        tl.store(out_ptr + off, y, mask=mask)


@triton.jit
def _foreach_cos_fused(
    i0,
    i1,
    i2,
    i3,
    i4,
    i5,
    i6,
    i7,
    i8,
    i9,
    i10,
    i11,
    i12,
    i13,
    i14,
    i15,
    o0,
    o1,
    o2,
    o3,
    o4,
    o5,
    o6,
    o7,
    o8,
    o9,
    o10,
    o11,
    o12,
    o13,
    o14,
    o15,
    s0,
    s1,
    s2,
    s3,
    s4,
    s5,
    s6,
    s7,
    s8,
    s9,
    s10,
    s11,
    s12,
    s13,
    s14,
    s15,
    BLOCK: tl.constexpr,
    COMPUTE_F32: tl.constexpr,
    EXACT: tl.constexpr,
):
    bid = tl.program_id(0)
    nb = (s0 + (BLOCK - 1)) // BLOCK
    if bid < nb:
        _cos_block(i0, o0, s0, bid, BLOCK, COMPUTE_F32, EXACT)
        return
    bid -= nb
    nb = (s1 + (BLOCK - 1)) // BLOCK
    if bid < nb:
        _cos_block(i1, o1, s1, bid, BLOCK, COMPUTE_F32, EXACT)
        return
    bid -= nb
    nb = (s2 + (BLOCK - 1)) // BLOCK
    if bid < nb:
        _cos_block(i2, o2, s2, bid, BLOCK, COMPUTE_F32, EXACT)
        return
    bid -= nb
    nb = (s3 + (BLOCK - 1)) // BLOCK
    if bid < nb:
        _cos_block(i3, o3, s3, bid, BLOCK, COMPUTE_F32, EXACT)
        return
    bid -= nb
    nb = (s4 + (BLOCK - 1)) // BLOCK
    if bid < nb:
        _cos_block(i4, o4, s4, bid, BLOCK, COMPUTE_F32, EXACT)
        return
    bid -= nb
    nb = (s5 + (BLOCK - 1)) // BLOCK
    if bid < nb:
        _cos_block(i5, o5, s5, bid, BLOCK, COMPUTE_F32, EXACT)
        return
    bid -= nb
    nb = (s6 + (BLOCK - 1)) // BLOCK
    if bid < nb:
        _cos_block(i6, o6, s6, bid, BLOCK, COMPUTE_F32, EXACT)
        return
    bid -= nb
    nb = (s7 + (BLOCK - 1)) // BLOCK
    if bid < nb:
        _cos_block(i7, o7, s7, bid, BLOCK, COMPUTE_F32, EXACT)
        return
    bid -= nb
    nb = (s8 + (BLOCK - 1)) // BLOCK
    if bid < nb:
        _cos_block(i8, o8, s8, bid, BLOCK, COMPUTE_F32, EXACT)
        return
    bid -= nb
    nb = (s9 + (BLOCK - 1)) // BLOCK
    if bid < nb:
        _cos_block(i9, o9, s9, bid, BLOCK, COMPUTE_F32, EXACT)
        return
    bid -= nb
    nb = (s10 + (BLOCK - 1)) // BLOCK
    if bid < nb:
        _cos_block(i10, o10, s10, bid, BLOCK, COMPUTE_F32, EXACT)
        return
    bid -= nb
    nb = (s11 + (BLOCK - 1)) // BLOCK
    if bid < nb:
        _cos_block(i11, o11, s11, bid, BLOCK, COMPUTE_F32, EXACT)
        return
    bid -= nb
    nb = (s12 + (BLOCK - 1)) // BLOCK
    if bid < nb:
        _cos_block(i12, o12, s12, bid, BLOCK, COMPUTE_F32, EXACT)
        return
    bid -= nb
    nb = (s13 + (BLOCK - 1)) // BLOCK
    if bid < nb:
        _cos_block(i13, o13, s13, bid, BLOCK, COMPUTE_F32, EXACT)
        return
    bid -= nb
    nb = (s14 + (BLOCK - 1)) // BLOCK
    if bid < nb:
        _cos_block(i14, o14, s14, bid, BLOCK, COMPUTE_F32, EXACT)
        return
    bid -= nb
    nb = (s15 + (BLOCK - 1)) // BLOCK
    if bid < nb:
        _cos_block(i15, o15, s15, bid, BLOCK, COMPUTE_F32, EXACT)
        return


@triton.jit
def _cos_flat_kernel(
    in_ptr,
    out_ptr,
    n_elem,
    BLOCK: tl.constexpr,
    COMPUTE_F32: tl.constexpr,
    COMPLEX: tl.constexpr,
    EXACT: tl.constexpr,
):
    pid = tl.program_id(0)
    idx = pid * BLOCK + tl.arange(0, BLOCK)
    if EXACT:
        mask = None
    else:
        mask = idx < n_elem
    if COMPLEX:
        off = 2 * idx
        a = tl.load(in_ptr + off, mask=mask, other=0.0)
        b = tl.load(in_ptr + off + 1, mask=mask, other=0.0)
        ca = tl.cos(a)
        sa = tl.sin(a)
        eb = tl.exp(b)
        em = tl.exp(-b)
        cb = 0.5 * (eb + em)
        sb = 0.5 * (eb - em)
        tl.store(out_ptr + off, ca * cb, mask=mask)
        tl.store(out_ptr + off + 1, -(sa * sb), mask=mask)
    else:
        x = tl.load(in_ptr + idx, mask=mask)
        if COMPUTE_F32:
            y = tl.cos(x.to(tl.float32))
            tl.store(out_ptr + idx, y.to(out_ptr.dtype.element_ty), mask=mask)
        else:
            y = tl.cos(x)
            tl.store(out_ptr + idx, y, mask=mask)


@triton.jit
def _cos_strided_kernel(
    in_ptr,
    out_ptr,
    n_elem,
    d0,
    d1,
    d2,
    d3,
    d4,
    d5,
    d6,
    d7,
    i0,
    i1,
    i2,
    i3,
    i4,
    i5,
    i6,
    i7,
    o0,
    o1,
    o2,
    o3,
    o4,
    o5,
    o6,
    o7,
    BLOCK: tl.constexpr,
    RANK: tl.constexpr,
    COMPUTE_F32: tl.constexpr,
    COMPLEX: tl.constexpr,
):
    pid = tl.program_id(0)
    idx = pid * BLOCK + tl.arange(0, BLOCK)
    mask = idx < n_elem
    rem = idx
    off_in = idx * 0
    off_out = idx * 0
    if RANK > 0:
        c = rem % d0
        rem = rem // d0
        off_in += c * i0
        off_out += c * o0
    if RANK > 1:
        c = rem % d1
        rem = rem // d1
        off_in += c * i1
        off_out += c * o1
    if RANK > 2:
        c = rem % d2
        rem = rem // d2
        off_in += c * i2
        off_out += c * o2
    if RANK > 3:
        c = rem % d3
        rem = rem // d3
        off_in += c * i3
        off_out += c * o3
    if RANK > 4:
        c = rem % d4
        rem = rem // d4
        off_in += c * i4
        off_out += c * o4
    if RANK > 5:
        c = rem % d5
        rem = rem // d5
        off_in += c * i5
        off_out += c * o5
    if RANK > 6:
        c = rem % d6
        rem = rem // d6
        off_in += c * i6
        off_out += c * o6
    if RANK > 7:
        c = rem % d7
        rem = rem // d7
        off_in += c * i7
        off_out += c * o7

    if COMPLEX:
        a = tl.load(in_ptr + off_in, mask=mask, other=0.0)
        b = tl.load(in_ptr + off_in + 1, mask=mask, other=0.0)
        ca = tl.cos(a)
        sa = tl.sin(a)
        eb = tl.exp(b)
        em = tl.exp(-b)
        cb = 0.5 * (eb + em)
        sb = 0.5 * (eb - em)
        tl.store(out_ptr + off_out, ca * cb, mask=mask)
        tl.store(out_ptr + off_out + 1, -(sa * sb), mask=mask)
    else:
        x = tl.load(in_ptr + off_in, mask=mask)
        if COMPUTE_F32:
            y = tl.cos(x.to(tl.float32))
            tl.store(out_ptr + off_out, y.to(out_ptr.dtype.element_ty), mask=mask)
        else:
            y = tl.cos(x)
            tl.store(out_ptr + off_out, y, mask=mask)


def _out_dtype(dt):
    if dt.is_complex or dt.is_floating_point:
        return dt
    return torch.get_default_dtype()


def _needs_f32_compute(dt, is_complex):
    if is_complex:
        return False
    return not dt.is_floating_point or dt in (torch.float16, torch.bfloat16)


def _alloc_out(t):
    dt = _out_dtype(t.dtype)
    if t.is_contiguous():
        return torch.empty(t.shape, dtype=dt, device=t.device)
    return torch.empty_like(t, dtype=dt, memory_format=torch.preserve_format)


def _alloc_outs(tensors):
    """Allocate one output per input.

    For a uniform, contiguous, small list all outputs are zero-copy views of
    a single freshly allocated buffer: this replaces N allocator calls with
    three cheap calls and removes the dominant host cost of many-tiny-tensor
    launches.  Every other case allocates per tensor so layout semantics stay
    identical to ATen.
    """
    n = len(tensors)
    if n > 1:
        t0 = tensors[0]
        shape0 = t0.shape
        total = 0
        uniform = True
        for t in tensors:
            if t.dtype != t0.dtype or not t.is_contiguous() or t.shape != shape0:
                uniform = False
                break
            total += t.numel()
        if uniform and 0 < total <= _SHARED_MAX_ELEMS:
            flat = torch.empty(total, dtype=_out_dtype(t0.dtype), device=t0.device)
            if len(shape0) == 0:
                return list(flat.unbind(0))
            return list(flat.view((n,) + tuple(shape0)).unbind(0))
    return [_alloc_out(t) for t in tensors]


def _fusable(tensors):
    dt = tensors[0].dtype
    for t in tensors:
        if t.dtype != dt or t.is_complex() or not t.is_contiguous():
            return False
    return True


def _geometry(total, compute_f32):
    """Size/dtype adaptive launch geometry.

    Small launches are latency bound, so use more, smaller CTAs (still 8
    elements, i.e. 16 bytes, per thread). Large f32 launches measured
    fastest with a 2x wider tile; large f16/bf16 launches are insensitive
    to the tile.
    """
    if total < _SMALL_MAX_ELEMS:
        return _SMALL_BLOCK, _SMALL_WARPS
    if not compute_f32:
        return _WIDE_BLOCK, _WIDE_WARPS
    return _BLOCK, _NUM_WARPS


def _launch_fused(tensors, outs):
    compute_f32 = _needs_f32_compute(tensors[0].dtype, False)
    dummy_in = tensors[0]
    dummy_out = outs[0]
    n = len(tensors)
    for start in range(0, n, _GROUP):
        chunk = tensors[start : start + _GROUP]
        ochunk = outs[start : start + _GROUP]
        pad = _GROUP - len(chunk)
        ins = list(chunk) + [dummy_in] * pad
        ous = list(ochunk) + [dummy_out] * pad
        sizes = [t.numel() for t in chunk] + [0] * pad
        block, warps = _geometry(sum(sizes), compute_f32)
        total_blocks = 0
        for s in sizes:
            total_blocks += (s + block - 1) // block
        if total_blocks == 0:
            continue
        exact = True
        for s in sizes:
            if s % block != 0:
                exact = False
                break
        _foreach_cos_fused[(total_blocks,)](
            *ins,
            *ous,
            *sizes,
            BLOCK=block,
            COMPUTE_F32=compute_f32,
            EXACT=exact,
            num_warps=warps,
        )


def _merge_dims(shape, in_stride, out_stride, scale):
    dims = [
        (sz, is_ * scale, os_ * scale)
        for sz, is_, os_ in zip(shape, in_stride, out_stride)
        if sz != 1
    ]
    merged = []
    for sz, is_, os_ in reversed(dims):
        if (
            merged
            and merged[-1][0] * merged[-1][1] == is_
            and merged[-1][0] * merged[-1][2] == os_
        ):
            merged[-1] = (merged[-1][0] * sz, merged[-1][1], merged[-1][2])
        else:
            merged.append((sz, is_, os_))
    merged.reverse()
    return merged


def _launch_strided(t, out, in_v, out_v, n, compute_f32, is_complex):
    scale = 2 if is_complex else 1
    dims = _merge_dims(t.shape, t.stride(), out.stride(), scale)
    if len(dims) > _MAX_DIMS:
        for i in range(t.shape[0]):
            sub = t.select(0, i)
            _launch_strided(
                sub,
                out.select(0, i),
                in_v.select(0, i),
                out_v.select(0, i),
                sub.numel(),
                compute_f32,
                is_complex,
            )
        return
    sizes = [1] * _MAX_DIMS
    istr = [0] * _MAX_DIMS
    ostr = [0] * _MAX_DIMS
    for k, (sz, is_, os_) in enumerate(dims):
        sizes[k] = sz
        istr[k] = is_
        ostr[k] = os_
    grid = (triton.cdiv(n, _BLOCK),)
    _cos_strided_kernel[grid](
        in_v,
        out_v,
        n,
        *sizes,
        *istr,
        *ostr,
        BLOCK=_BLOCK,
        RANK=len(dims),
        COMPUTE_F32=compute_f32,
        COMPLEX=is_complex,
        num_warps=_NUM_WARPS,
    )


def _foreach_cos(self, *args):
    if isinstance(self, torch.Tensor):
        tensors = [self]
    else:
        tensors = list(self)
    tensors.extend(args)
    if len(tensors) == 0:
        raise RuntimeError("Tensor list must have at least one tensor.")

    outs = _alloc_outs(tensors)

    if len(tensors) > 1 and _fusable(tensors):
        _launch_fused(tensors, outs)
        return outs

    for t, out in zip(tensors, outs):
        n = t.numel()
        if n == 0:
            continue
        is_complex = t.is_complex()
        compute_f32 = _needs_f32_compute(t.dtype, is_complex)
        in_v = torch.view_as_real(t) if is_complex else t
        out_v = torch.view_as_real(out) if is_complex else out
        if t.is_contiguous():
            block, warps = _geometry(n, compute_f32)
            grid = (triton.cdiv(n, block),)
            _cos_flat_kernel[grid](
                in_v,
                out_v,
                n,
                BLOCK=block,
                COMPUTE_F32=compute_f32,
                COMPLEX=is_complex,
                EXACT=(not is_complex) and (n % block == 0),
                num_warps=warps,
            )
        else:
            _launch_strided(t, out, in_v, out_v, n, compute_f32, is_complex)
    return outs
