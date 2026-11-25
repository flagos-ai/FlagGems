"""In-place foreach scalar division implemented with Triton kernels.

``run(self, scalar)`` mirrors ``aten::_foreach_div_.Scalar``: every tensor in
``self`` is divided in place by ``scalar`` and ``None`` is returned.  No
framework arithmetic is used; the mutation is produced by the Triton kernels.

Small lists of identically shaped contiguous tensors (the common foreach case)
are folded into a single fused launch, and every launch goes through a cached
compiled kernel invoked through the raw launcher, so the per-call host cost is
one driver stream query plus one C launcher call.
"""

import math
import struct

import torch
import triton
import triton.language as tl

try:  # low-overhead launch path
    from triton.runtime import driver as _driver_mod
except Exception:  # pragma: no cover
    _driver_mod = None

_CACHE = {}
_MISS = object()
_DRV = [None]
# this Triton build forwards one placeholder object per constexpr parameter
_CONST = (0, 0, 0, 0)


def _driver():
    drv = _DRV[0]
    if drv is None:
        drv = _driver_mod.active
        _DRV[0] = drv
    return drv


@triton.jit
def _flat_body(p, offs, s, n, EVEN: tl.constexpr):
    if EVEN:
        x = tl.load(p + offs)
        tl.store(p + offs, x / s)
    else:
        m = offs < n
        x = tl.load(p + offs, mask=m)
        tl.store(p + offs, x / s, mask=m)


@triton.jit(do_not_specialize=["n"])
def _div_flat_kernel(
    x_ptr,
    n,
    s_hi,
    s_lo,
    BLOCK: tl.constexpr,
    IS64: tl.constexpr,
    I64: tl.constexpr,
    EVEN: tl.constexpr,
):
    pid = tl.program_id(0)
    if I64:
        offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    else:
        offs = pid * BLOCK + tl.arange(0, BLOCK)
    if IS64:
        s = s_hi.to(tl.float64) + s_lo.to(tl.float64)
    else:
        s = s_hi
    _flat_body(x_ptr, offs, s, n, EVEN)


@triton.jit(do_not_specialize=["n", "ppt"])
def _div_fused_kernel(
    x0,
    x1,
    x2,
    x3,
    n,
    ppt,
    s_hi,
    s_lo,
    BLOCK: tl.constexpr,
    IS64: tl.constexpr,
    I64: tl.constexpr,
    EVEN: tl.constexpr,
):
    pid = tl.program_id(0)
    tid = pid // ppt
    blk = pid - tid * ppt
    if I64:
        offs = blk.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    else:
        offs = blk * BLOCK + tl.arange(0, BLOCK)
    if IS64:
        s = s_hi.to(tl.float64) + s_lo.to(tl.float64)
    else:
        s = s_hi
    if tid == 0:
        _flat_body(x0, offs, s, n, EVEN)
    elif tid == 1:
        _flat_body(x1, offs, s, n, EVEN)
    elif tid == 2:
        _flat_body(x2, offs, s, n, EVEN)
    else:
        _flat_body(x3, offs, s, n, EVEN)


@triton.jit(do_not_specialize=["base", "n_in", "s_in", "s1", "s2"])
def _div_strided_kernel(
    x_ptr,
    base,
    n_in,
    s_in,
    s1,
    s2,
    s_hi,
    s_lo,
    BLOCK: tl.constexpr,
    IS64: tl.constexpr,
    I64: tl.constexpr,
    EVEN: tl.constexpr,
):
    pid = tl.program_id(0)
    cols = pid * BLOCK + tl.arange(0, BLOCK)
    if I64:
        off = (
            base
            + cols.to(tl.int64) * s_in
            + tl.program_id(1).to(tl.int64) * s1
            + tl.program_id(2).to(tl.int64) * s2
        )
    else:
        off = base + cols * s_in + tl.program_id(1) * s1 + tl.program_id(2) * s2
    if IS64:
        s = s_hi.to(tl.float64) + s_lo.to(tl.float64)
    else:
        s = s_hi
    if EVEN:
        x = tl.load(x_ptr + off)
        tl.store(x_ptr + off, x / s)
    else:
        mask = cols < n_in
        x = tl.load(x_ptr + off, mask=mask)
        tl.store(x_ptr + off, x / s, mask=mask)


def _collapse(shape, strides):
    """TensorIterator-style dim drop + merge, sorted by descending |stride|."""
    dims = [(int(sz), int(st)) for sz, st in zip(shape, strides) if sz != 1]
    dims.sort(key=lambda d: -abs(d[1]))
    merged = []
    for sz, st in dims:
        if merged and merged[-1][1] == st * sz:
            merged[-1][0] *= sz
            merged[-1][1] = st
        else:
            merged.append([sz, st])
    return merged


def _scalar_residual(value):
    """fp32 rounding residual of an fp64 scalar, used to rebuild it in fp64."""
    if not math.isfinite(value):
        return 0.0
    try:
        hi = struct.unpack("f", struct.pack("f", value))[0]
    except OverflowError:
        return 0.0
    return value - hi


def _block_for(extent):
    if extent >= 1024:
        return 1024
    block = 1
    while block < extent:
        block *= 2
    return block


def _fast_launcher(ck):
    """Return (raw_launcher, function, packed_metadata) or None."""
    try:
        launcher = ck.run
        return (launcher.launch, ck.function, ck.packed_metadata)
    except Exception:
        return None


def _flat_kwargs(block, is64, i64, even, warps):
    return {
        "BLOCK": block,
        "IS64": is64,
        "I64": i64,
        "EVEN": even,
        "num_warps": warps,
    }


def _stream_of(t):
    drv = _driver()
    idx = t.device.index
    if idx is None:
        idx = drv.get_current_device()
    return drv.get_current_stream(idx)


def _launch_flat_group(group, ptrs, n, s_hi, s_lo, is64, stream):
    """One launch for one, or up to four, contiguous tensors of equal size."""
    if n <= 65536:
        block = 64
        warps = 1
    else:
        block = 1024
        warps = 4
    even = (n % block) == 0
    i64 = n >= 2**31
    aligned = True
    for p in ptrs:
        if p & 15:
            aligned = False
            break
    k = len(group)
    dtype = group[0].dtype
    key = (k, block, warps, i64, is64, even, aligned, dtype)
    fast = _CACHE.get(key, _MISS)
    if k == 1:
        grid = (triton.cdiv(n, block), 1, 1)
        if fast is _MISS:
            kw = _flat_kwargs(block, is64, i64, even, warps)
            try:
                ck = _div_flat_kernel.warmup(
                    group[0], n, s_hi, s_lo, grid=(1, 1, 1), **kw
                )
            except Exception:
                ck = None
            fast = None if ck is None else _fast_launcher(ck)
            _CACHE[key] = fast
        args = (ptrs[0], n, s_hi, s_lo) + _CONST
        if fast is not None:
            launch, kfn, kmeta = fast
            launch(grid[0], 1, 1, stream, kfn, kmeta, None, None, None, *args)
        else:
            _div_flat_kernel[grid](
                group[0], n, s_hi, s_lo, **_flat_kwargs(block, is64, i64, even, warps)
            )
        return
    ppt = triton.cdiv(n, block)
    grid = (k * ppt, 1, 1)
    if fast is _MISS:
        padded = list(group) + [group[-1]] * (4 - k)
        kw = _flat_kwargs(block, is64, i64, even, warps)
        try:
            ck = _div_fused_kernel.warmup(
                padded[0],
                padded[1],
                padded[2],
                padded[3],
                n,
                ppt,
                s_hi,
                s_lo,
                grid=(1, 1, 1),
                **kw,
            )
        except Exception:
            ck = None
        fast = None if ck is None else _fast_launcher(ck)
        _CACHE[key] = fast
    p0, p1 = ptrs[0], ptrs[1]
    p2 = ptrs[2] if k > 2 else p1
    p3 = ptrs[3] if k > 3 else p2
    args = (p0, p1, p2, p3, n, ppt, s_hi, s_lo) + _CONST
    if fast is not None:
        launch, kfn, kmeta = fast
        launch(grid[0], 1, 1, stream, kfn, kmeta, None, None, None, *args)
    else:
        g0 = group[0]
        g1 = group[1]
        g2 = group[2] if k > 2 else g1
        g3 = group[3] if k > 3 else g2
        _div_fused_kernel[grid](
            g0,
            g1,
            g2,
            g3,
            n,
            ppt,
            s_hi,
            s_lo,
            **_flat_kwargs(block, is64, i64, even, warps),
        )


def _launch_strided(t, ptr, merged, s_hi, s_lo, is64, stream):
    n_in, s_in = merged[-1]
    outer = merged[:-1]

    grid_dims = []
    for dim in reversed(outer):
        if len(grid_dims) >= 2 or dim[0] > 65535:
            break
        grid_dims.append(dim)
    loop_dims = outer[: len(outer) - len(grid_dims)]

    max_off = 0
    for sz, st in merged:
        max_off += (sz - 1) * abs(st)
    i64 = max_off >= 2**31

    gy = grid_dims[0] if len(grid_dims) > 0 else None
    gz = grid_dims[1] if len(grid_dims) > 1 else None

    block = _block_for(n_in)
    even = (n_in % block) == 0
    num_warps = 4 if block >= 512 else 1
    gx = triton.cdiv(n_in, block)
    gy_n = gy[0] if gy is not None else 1
    gz_n = gz[0] if gz is not None else 1
    grid = (gx, gy_n, gz_n)
    s1 = gy[1] if gy is not None else 0
    s2 = gz[1] if gz is not None else 0
    kw = {
        "BLOCK": block,
        "IS64": is64,
        "I64": i64,
        "EVEN": even,
        "num_warps": num_warps,
    }
    key = ("s", i64, is64, even, block, num_warps, (ptr & 15) == 0, t.dtype)
    fast = _CACHE.get(key, _MISS)
    if fast is _MISS:
        try:
            ck = _div_strided_kernel.warmup(
                t, 0, n_in, s_in, s1, s2, s_hi, s_lo, grid=(1, 1, 1), **kw
            )
        except Exception:
            ck = None
        fast = None if ck is None else _fast_launcher(ck)
        _CACHE[key] = fast

    combos = [()]
    for sz, st in loop_dims:
        combos = [c + (i,) for c in combos for i in range(sz)]
    for combo in combos:
        base = 0
        for (sz, st), idx in zip(loop_dims, combo):
            base += idx * st
        if fast is not None:
            launch, kfn, kmeta = fast
            launch(
                gx,
                gy_n,
                gz_n,
                stream,
                kfn,
                kmeta,
                None,
                None,
                None,
                ptr,
                base,
                n_in,
                s_in,
                s1,
                s2,
                s_hi,
                s_lo,
                *_CONST,
            )
        else:
            _div_strided_kernel[grid](t, base, n_in, s_in, s1, s2, s_hi, s_lo, **kw)


def _foreach_div__Scalar(self, scalar):
    s_d = float(scalar)
    tensors = self
    if torch.is_tensor(tensors):
        tensors = (tensors,)
    length = len(tensors)
    if length == 0:
        return None
    stream = None
    residual = None
    i = 0
    while i < length:
        t = tensors[i]
        i += 1
        if not torch.is_tensor(t):
            continue
        n = t.numel()
        if n == 0:
            continue
        is64 = t.dtype is torch.float64
        if is64:
            if residual is None:
                residual = _scalar_residual(s_d)
            lo = residual
        else:
            lo = 0.0
        if stream is None:
            stream = _stream_of(t)
        if not t.is_contiguous():
            _launch_strided(
                t, t.data_ptr(), _collapse(t.shape, t.stride()), s_d, lo, is64, stream
            )
            continue
        ptrs = [t.data_ptr()]
        group = [t]
        span = n * t.element_size()
        dtype = t.dtype
        dev = t.device
        while i < length and len(ptrs) < 4:
            u = tensors[i]
            if not (
                torch.is_tensor(u)
                and u.dtype is dtype
                and u.numel() == n
                and u.device == dev
                and u.is_contiguous()
            ):
                break
            pu = u.data_ptr()
            clash = False
            for pk in ptrs:
                if -span < pu - pk < span:
                    clash = True
                    break
            if clash:
                break
            group.append(u)
            ptrs.append(pu)
            i += 1
        _launch_flat_group(group, ptrs, n, s_d, lo, is64, stream)
    return None
