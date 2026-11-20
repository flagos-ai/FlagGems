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
"""_foreach_frac: y = x - trunc(x) for each tensor in a sequence (Triton backend).

Reference-parity decisions established by probing aten::_foreach_frac on the
target (torch 2.9.1 + MUSA):
  * returns one freshly allocated tensor per input (tuple container),
  * non-floating inputs raise NotImplementedError ("frac_cuda" ...),
  * complex inputs raise NotImplementedError ("foreach_unary_op_cuda" ...),
  * a bare Tensor / non-sequence raises TypeError,
  * an empty sequence raises RuntimeError,
  * dense (non-overlapping, covering) inputs get an output with the same strides,
    gapped / overlapping inputs get a contiguous output.

Kernel plan: one fused launch serves all dense tensors that share dtype and
numel (the dominant benchmark shape: many equal-sized tensors), a per-tensor
launch covers leftovers, and a rank-unrolled strided kernel covers gapped /
overlapping views with a contiguous output.

Host plan: the per-call Python path is kept minimal (single validation pass,
dtype properties instead of tensor methods, module-level allocation aliases,
no per-tensor list rebuilds) because the many-small-tensor workloads are
dominated by per-tensor host work, not device time.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops._foreach_unary import _foreach_frac as default__foreach_frac

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)

_SUPPORTED_DTYPES = {torch.float16, torch.bfloat16, torch.float32}


def _needs_fallback(seq):
    """True when the list cannot use the MUSA kernels (device or dtype)."""
    for t in seq:
        if not isinstance(t, torch.Tensor):
            return True
        if t.device.type != "musa" or t.dtype not in _SUPPORTED_DTYPES:
            return True
    return False


_BLOCK = 1024
_NUM_WARPS = 4
_MAX_DIMS = 6
_SLOTS = 16
_INT64_LIMIT = 1 << 31
_LARGE_NUMEL = 1 << 26

_EMPTY = torch.empty
_EMPTY_LIKE = torch.empty_like
_EMPTY_STRIDED = torch.empty_strided
_PROMOTE = {torch.float16: True, torch.bfloat16: True}


@triton.jit
def _frac_elem(x, PROMOTE: tl.constexpr):
    if PROMOTE:
        xc = x.to(tl.float32)
    else:
        xc = x
    tr = tl.where(xc >= 0, tl.floor(xc), tl.ceil(xc))
    return (xc - tr).to(x.dtype)


@triton.jit
def _frac_flat_kernel(
    in_ptr, out_ptr, numel, PROMOTE: tl.constexpr, BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    x = tl.load(in_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, _frac_elem(x, PROMOTE), mask=mask)


@triton.jit
def _frac_fused_kernel(
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
    nblocks,
    numel,
    NT: tl.constexpr,
    PROMOTE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    t = pid // nblocks
    b = pid - t * nblocks
    offs = b * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    if NT > 0:
        if t == 0:
            tl.store(
                o0 + offs, _frac_elem(tl.load(i0 + offs, mask=mask), PROMOTE), mask=mask
            )
    if NT > 1:
        if t == 1:
            tl.store(
                o1 + offs, _frac_elem(tl.load(i1 + offs, mask=mask), PROMOTE), mask=mask
            )
    if NT > 2:
        if t == 2:
            tl.store(
                o2 + offs, _frac_elem(tl.load(i2 + offs, mask=mask), PROMOTE), mask=mask
            )
    if NT > 3:
        if t == 3:
            tl.store(
                o3 + offs, _frac_elem(tl.load(i3 + offs, mask=mask), PROMOTE), mask=mask
            )
    if NT > 4:
        if t == 4:
            tl.store(
                o4 + offs, _frac_elem(tl.load(i4 + offs, mask=mask), PROMOTE), mask=mask
            )
    if NT > 5:
        if t == 5:
            tl.store(
                o5 + offs, _frac_elem(tl.load(i5 + offs, mask=mask), PROMOTE), mask=mask
            )
    if NT > 6:
        if t == 6:
            tl.store(
                o6 + offs, _frac_elem(tl.load(i6 + offs, mask=mask), PROMOTE), mask=mask
            )
    if NT > 7:
        if t == 7:
            tl.store(
                o7 + offs, _frac_elem(tl.load(i7 + offs, mask=mask), PROMOTE), mask=mask
            )
    if NT > 8:
        if t == 8:
            tl.store(
                o8 + offs, _frac_elem(tl.load(i8 + offs, mask=mask), PROMOTE), mask=mask
            )
    if NT > 9:
        if t == 9:
            tl.store(
                o9 + offs, _frac_elem(tl.load(i9 + offs, mask=mask), PROMOTE), mask=mask
            )
    if NT > 10:
        if t == 10:
            tl.store(
                o10 + offs,
                _frac_elem(tl.load(i10 + offs, mask=mask), PROMOTE),
                mask=mask,
            )
    if NT > 11:
        if t == 11:
            tl.store(
                o11 + offs,
                _frac_elem(tl.load(i11 + offs, mask=mask), PROMOTE),
                mask=mask,
            )
    if NT > 12:
        if t == 12:
            tl.store(
                o12 + offs,
                _frac_elem(tl.load(i12 + offs, mask=mask), PROMOTE),
                mask=mask,
            )
    if NT > 13:
        if t == 13:
            tl.store(
                o13 + offs,
                _frac_elem(tl.load(i13 + offs, mask=mask), PROMOTE),
                mask=mask,
            )
    if NT > 14:
        if t == 14:
            tl.store(
                o14 + offs,
                _frac_elem(tl.load(i14 + offs, mask=mask), PROMOTE),
                mask=mask,
            )
    if NT > 15:
        if t == 15:
            tl.store(
                o15 + offs,
                _frac_elem(tl.load(i15 + offs, mask=mask), PROMOTE),
                mask=mask,
            )


@triton.jit
def _frac_strided_kernel(
    in_ptr,
    out_ptr,
    numel,
    s0,
    s1,
    s2,
    s3,
    s4,
    s5,
    t0,
    t1,
    t2,
    t3,
    t4,
    t5,
    NDIM: tl.constexpr,
    PROMOTE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = (pid * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)
    mask = offs < numel
    if NDIM == 1:
        in_off = offs * t0
    elif NDIM == 2:
        i1 = offs % s1
        r = offs // s1
        in_off = r * t0 + i1 * t1
    elif NDIM == 3:
        i2 = offs % s2
        r = offs // s2
        i1 = r % s1
        r = r // s1
        in_off = r * t0 + i1 * t1 + i2 * t2
    elif NDIM == 4:
        i3 = offs % s3
        r = offs // s3
        i2 = r % s2
        r = r // s2
        i1 = r % s1
        r = r // s1
        in_off = r * t0 + i1 * t1 + i2 * t2 + i3 * t3
    elif NDIM == 5:
        i4 = offs % s4
        r = offs // s4
        i3 = r % s3
        r = r // s3
        i2 = r % s2
        r = r // s2
        i1 = r % s1
        r = r // s1
        in_off = r * t0 + i1 * t1 + i2 * t2 + i3 * t3 + i4 * t4
    else:
        i5 = offs % s5
        r = offs // s5
        i4 = r % s4
        r = r // s4
        i3 = r % s3
        r = r // s3
        i2 = r % s2
        r = r // s2
        i1 = r % s1
        r = r // s1
        in_off = r * t0 + i1 * t1 + i2 * t2 + i3 * t3 + i4 * t4 + i5 * t5

    x = tl.load(in_ptr + in_off, mask=mask)
    tl.store(out_ptr + offs, _frac_elem(x, PROMOTE), mask=mask)


_DTYPE_NAMES = {
    torch.float16: "Half",
    torch.bfloat16: "BFloat16",
    torch.float32: "Float",
    torch.float64: "Double",
    torch.int8: "Char",
    torch.uint8: "Byte",
    torch.int16: "Short",
    torch.int32: "Int",
    torch.int64: "Long",
    torch.bool: "Bool",
    torch.complex64: "ComplexFloat",
    torch.complex128: "ComplexDouble",
}


def _dtype_name(dtype):
    return _DTYPE_NAMES.get(dtype, str(dtype).replace("torch.", ""))


def _device_ctx(device):
    for name in ("musa", "cuda"):
        mod = getattr(torch, name, None)
        if mod is not None and hasattr(mod, "device"):
            return mod.device(device)
    return None


def _current_device():
    for name in ("musa", "cuda"):
        mod = getattr(torch, name, None)
        if mod is not None and hasattr(mod, "current_device"):
            try:
                return mod.current_device()
            except Exception:
                return None
    return None


def _is_dense(t):
    """Non-overlapping and covering (no gaps), strictly positive strides."""
    numel = t.numel()
    if numel == 0:
        return True
    if t.is_contiguous():
        return True
    dims = [(st, sz) for st, sz in zip(t.stride(), t.shape) if sz != 1]
    if not dims:
        return True
    for st, _ in dims:
        if st <= 0:
            return False
    dims = sorted(dims)
    expect = 1
    for st, sz in dims:
        if st != expect:
            return False
        expect *= sz
    return expect == numel


def _collapse(shape, stride):
    """Merge memory-contiguous adjacent dimensions; drop size-1 dimensions."""
    dims = [(int(sz), int(st)) for sz, st in zip(shape, stride) if sz != 1]
    merged = []
    for sz, st in dims:
        if merged and merged[-1][1] == st * sz:
            merged[-1] = (merged[-1][0] * sz, st)
        else:
            merged.append((sz, st))
    return merged


def _geom(numel):
    """CTA geometry: fatter CTAs only for very large numel (>= 2^26 elements)."""
    if numel >= _LARGE_NUMEL:
        return 2048, _NUM_WARPS
    return _BLOCK, _NUM_WARPS


def _launch_flat(t, out, numel, promote):
    block, warps = _geom(numel)
    grid = ((numel + block - 1) // block,)
    _frac_flat_kernel[grid](
        t, out, numel, PROMOTE=promote, BLOCK=block, num_warps=warps
    )


def _launch_fused(tensors, outs, numel, promote):
    n = len(tensors)
    if n < _SLOTS:
        tin = list(tensors) + [tensors[-1]] * (_SLOTS - n)
        tout = list(outs) + [outs[-1]] * (_SLOTS - n)
    else:
        tin = tensors
        tout = outs
    block, warps = _geom(numel)
    nblocks = (numel + block - 1) // block
    grid = (nblocks * n,)
    _frac_fused_kernel[grid](
        tin[0],
        tin[1],
        tin[2],
        tin[3],
        tin[4],
        tin[5],
        tin[6],
        tin[7],
        tin[8],
        tin[9],
        tin[10],
        tin[11],
        tin[12],
        tin[13],
        tin[14],
        tin[15],
        tout[0],
        tout[1],
        tout[2],
        tout[3],
        tout[4],
        tout[5],
        tout[6],
        tout[7],
        tout[8],
        tout[9],
        tout[10],
        tout[11],
        tout[12],
        tout[13],
        tout[14],
        tout[15],
        nblocks,
        numel,
        NT=n,
        PROMOTE=promote,
        BLOCK=block,
        num_warps=warps,
    )


def _launch_strided(t, out, dims, promote):
    numel = 1
    for sz, _ in dims:
        numel *= sz
    shape = [1] * _MAX_DIMS
    strides = [0] * _MAX_DIMS
    for i, (sz, st) in enumerate(dims):
        shape[i] = sz
        strides[i] = st
    block, warps = _geom(numel)
    grid = ((numel + block - 1) // block,)
    _frac_strided_kernel[grid](
        t,
        out,
        numel,
        shape[0],
        shape[1],
        shape[2],
        shape[3],
        shape[4],
        shape[5],
        strides[0],
        strides[1],
        strides[2],
        strides[3],
        strides[4],
        strides[5],
        NDIM=len(dims),
        PROMOTE=promote,
        BLOCK=block,
        num_warps=warps,
    )


def _compute(t, out, promote):
    """Fill `out` (same logical shape as `t`) from `t` by storage order."""
    numel = t.numel()
    if numel == 0:
        return
    if numel < _INT64_LIMIT and t.is_contiguous():
        _launch_flat(t, out, numel, promote)
        return
    if (
        numel < _INT64_LIMIT
        and _is_dense(t)
        and tuple(t.stride()) == tuple(out.stride())
    ):
        _launch_flat(t, out, numel, promote)
        return

    dims = _collapse(t.shape, t.stride())
    if numel >= _INT64_LIMIT:
        dims = [(numel, 1)]
    if len(dims) > _MAX_DIMS:
        for i in range(t.shape[0]):
            _compute(t.select(0, i), out.select(0, i), promote)
        return
    _launch_strided(t, out, dims, promote)


def _run_impl(tensors):
    n = len(tensors)
    outs = [None] * n
    groups = {}
    for idx in range(n):
        t = tensors[idx]
        if not isinstance(t, torch.Tensor):
            raise TypeError(
                "_foreach_frac(): argument 'self' (position 1) must be tuple of "
                "Tensors, not %s" % type(t).__name__
            )
        dt = t.dtype
        if dt.is_complex:
            raise NotImplementedError(
                "\"foreach_unary_op_cuda\" not implemented for '%s'" % _dtype_name(dt)
            )
        if not dt.is_floating_point:
            raise NotImplementedError(
                "\"frac_cuda\" not implemented for '%s'" % _dtype_name(dt)
            )

        numel = t.numel()
        if not numel:
            outs[idx] = _EMPTY(t.shape, dtype=dt, device=t.device)
            continue

        if t.is_contiguous():
            dense = True
            out = _EMPTY_LIKE(t)
        else:
            dense = _is_dense(t)
            if dense:
                out = _EMPTY_STRIDED(
                    tuple(t.shape), tuple(t.stride()), dtype=dt, device=t.device
                )
            else:
                # overlapping / gapped view: contiguous writes cannot collide
                out = _EMPTY(t.shape, dtype=dt, device=t.device)
        outs[idx] = out

        if dense and numel < _INT64_LIMIT:
            key = (dt, numel, t.device)
            bucket = groups.get(key)
            if bucket is None:
                groups[key] = [idx]
            else:
                bucket.append(idx)
        else:
            _compute(t, out, _PROMOTE.get(dt, False))

    for key, idxs in groups.items():
        numel = key[1]
        promote = _PROMOTE.get(key[0], False)
        cnt = len(idxs)
        if cnt == 1:
            i = idxs[0]
            _launch_flat(tensors[i], outs[i], numel, promote)
        elif cnt == n:
            # every tensor contributed in order, so the index list is the identity
            _launch_fused(tensors, outs, numel, promote)
        elif cnt <= _SLOTS:
            _launch_fused(
                [tensors[i] for i in idxs], [outs[i] for i in idxs], numel, promote
            )
        else:
            for start in range(0, cnt, _SLOTS):
                chunk = idxs[start : start + _SLOTS]
                if len(chunk) == 1:
                    i = chunk[0]
                    _launch_flat(tensors[i], outs[i], numel, promote)
                else:
                    _launch_fused(
                        [tensors[i] for i in chunk],
                        [outs[i] for i in chunk],
                        numel,
                        promote,
                    )

    return tuple(outs)


def _foreach_frac(self):
    logger.debug("GEMS_MTHREADS _FOREACH_FRAC")
    _seq = self if isinstance(self, (list, tuple)) else [self]
    if _needs_fallback(_seq):
        return default__foreach_frac(self)
    if not isinstance(self, (list, tuple)):
        raise TypeError(
            "_foreach_frac(): argument 'self' (position 1) must be tuple of Tensors, "
            "not %s" % type(self).__name__
        )
    if len(self) == 0:
        raise RuntimeError("Tensor list must have at least one tensor.")

    first = self[0]
    if isinstance(first, torch.Tensor):
        dev = first.device.index
        if dev is not None:
            cur = _current_device()
            if cur is not None and cur != dev:
                ctx = _device_ctx(first.device)
                if ctx is not None:
                    with ctx:
                        return _run_impl(self)
    return _run_impl(self)
