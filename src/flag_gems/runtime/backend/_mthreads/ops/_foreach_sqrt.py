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
"""Triton implementation of aten::_foreach_sqrt.

Each input tensor is processed by its own Triton kernel launch:
  * contiguous inputs use a flat 1-D kernel (fully coalesced),
  * non-contiguous inputs use a collapsed-rank strided kernel that maps the
    flat logical index onto the input's strides while writing a freshly
    allocated contiguous output,
  * complex inputs use the same collapsed-rank mapping over a zero-copy float
    view of the components and a C99 complex-sqrt formula.
Output dtype mirrors ATen: floating dtypes are preserved, integral/bool
inputs are promoted to the default floating dtype.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops._foreach_unary import _foreach_sqrt as default__foreach_sqrt

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)
_BLOCK = 1024
_NUM_WARPS = 4
_GROUP_SLOTS = 16
# Large homogeneous lists also use one grouped launch: the per-tensor work is
# still contiguous and equal, so 16 back-to-back kernels only add 15
# drain/ramp gaps.  The cap keeps flat offsets inside int32.
_GROUP_MAX_NUMEL = 1 << 30


@triton.jit
def _sqrt_apply(x, MODE: tl.constexpr):
    # tl.sqrt maps to the same IEEE round-to-nearest hardware/libdevice sqrt
    # that ATen uses on this target (verified bitwise for fp32 and fp64).
    if MODE == 1:  # half / bfloat16: ATen computes in float (opmath) then rounds
        y = tl.sqrt(x.to(tl.float32)).to(x.dtype)
    elif MODE == 0:  # float32
        y = tl.sqrt(x)
    elif MODE == 2:  # float64
        y = tl.sqrt(x)
    else:  # integral / bool -> promoted float32
        y = tl.sqrt(x.to(tl.float32))
    return y


@triton.jit
def _sqrt_flat_kernel(in_ptr, out_ptr, n, BLOCK: tl.constexpr, MODE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(in_ptr + offs, mask=mask, other=0)
    y = _sqrt_apply(x, MODE)
    tl.store(out_ptr + offs, y, mask=mask)


@triton.jit
def _sqrt_copy_flat(in_ptr, out_ptr, offs, mask, MODE: tl.constexpr):
    x = tl.load(in_ptr + offs, mask=mask, other=0)
    tl.store(out_ptr + offs, _sqrt_apply(x, MODE), mask=mask)


@triton.jit
def _sqrt_group_kernel(
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
    n,
    BLOCK: tl.constexpr,
    MODE: tl.constexpr,
):
    """One launch for a homogeneous list of contiguous tensors.

    grid = (cdiv(n, BLOCK), k) with k <= 16, so program_id(1) is always a live
    slot; unused pointer arguments are padded with slot 0's pointers and never
    dereferenced.
    """
    slot = tl.program_id(1)
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    if slot == 0:
        _sqrt_copy_flat(i0, o0, offs, mask, MODE)
    elif slot == 1:
        _sqrt_copy_flat(i1, o1, offs, mask, MODE)
    elif slot == 2:
        _sqrt_copy_flat(i2, o2, offs, mask, MODE)
    elif slot == 3:
        _sqrt_copy_flat(i3, o3, offs, mask, MODE)
    elif slot == 4:
        _sqrt_copy_flat(i4, o4, offs, mask, MODE)
    elif slot == 5:
        _sqrt_copy_flat(i5, o5, offs, mask, MODE)
    elif slot == 6:
        _sqrt_copy_flat(i6, o6, offs, mask, MODE)
    elif slot == 7:
        _sqrt_copy_flat(i7, o7, offs, mask, MODE)
    elif slot == 8:
        _sqrt_copy_flat(i8, o8, offs, mask, MODE)
    elif slot == 9:
        _sqrt_copy_flat(i9, o9, offs, mask, MODE)
    elif slot == 10:
        _sqrt_copy_flat(i10, o10, offs, mask, MODE)
    elif slot == 11:
        _sqrt_copy_flat(i11, o11, offs, mask, MODE)
    elif slot == 12:
        _sqrt_copy_flat(i12, o12, offs, mask, MODE)
    elif slot == 13:
        _sqrt_copy_flat(i13, o13, offs, mask, MODE)
    elif slot == 14:
        _sqrt_copy_flat(i14, o14, offs, mask, MODE)
    else:
        _sqrt_copy_flat(i15, o15, offs, mask, MODE)


@triton.jit
def _sqrt_strided_kernel(
    in_ptr,
    out_ptr,
    n,
    e1,
    e2,
    e3,
    e4,
    s0,
    s1,
    s2,
    s3,
    s4,
    RANK: tl.constexpr,
    MODE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = offs < n
    idx = offs
    off = tl.zeros([BLOCK], dtype=tl.int64)
    if RANK == 5:
        off += (idx % e4) * s4
        idx = idx // e4
    if RANK >= 4:
        off += (idx % e3) * s3
        idx = idx // e3
    if RANK >= 3:
        off += (idx % e2) * s2
        idx = idx // e2
    if RANK >= 2:
        off += (idx % e1) * s1
        idx = idx // e1
    off += idx * s0
    x = tl.load(in_ptr + off, mask=mask, other=0)
    y = _sqrt_apply(x, MODE)
    tl.store(out_ptr + offs, y, mask=mask)


@triton.jit
def _sqrt_complex_kernel(
    in_ptr,
    out_ptr,
    n,
    e1,
    e2,
    e3,
    e4,
    s0,
    s1,
    s2,
    s3,
    s4,
    RANK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # in_ptr points at the real component of a complex tensor; the imaginary
    # component of the same element is the very next float.  out_ptr points at
    # the (contiguous float view of the) freshly allocated complex output, so
    # flat complex index i stores to float offsets 2*i and 2*i+1.
    pid = tl.program_id(0)
    offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = offs < n
    idx = offs
    off = tl.zeros([BLOCK], dtype=tl.int64)
    if RANK == 5:
        off += (idx % e4) * s4
        idx = idx // e4
    if RANK >= 4:
        off += (idx % e3) * s3
        idx = idx // e3
    if RANK >= 3:
        off += (idx % e2) * s2
        idx = idx // e2
    if RANK >= 2:
        off += (idx % e1) * s1
        idx = idx // e1
    off += idx * s0

    a = tl.load(in_ptr + off, mask=mask, other=0.0)
    b = tl.load(in_ptr + off + 1, mask=mask, other=0.0)
    # C99 / std::complex formula: t = sqrt((|z| + |a|) / 2)
    mag = tl.sqrt(a * a + b * b)
    t = tl.sqrt(0.5 * (mag + tl.abs(a)))
    den = 2.0 * t
    pos = a >= 0.0
    re = tl.where(pos, t, tl.where(den > 0.0, tl.abs(b) / den, 0.0))
    im = tl.where(pos, b / den, tl.where(b < 0.0, -t, t))
    im = tl.where(den > 0.0, im, 0.0)

    tl.store(out_ptr + 2 * offs, re, mask=mask)
    tl.store(out_ptr + 2 * offs + 1, im, mask=mask)


_FLOAT_MODES = {
    torch.float16: 1,
    torch.bfloat16: 1,
    torch.float32: 0,
    torch.float64: 2,
}

# complex64 / complex128 -> component float dtype
_COMPLEX_PARTS = {
    torch.complex64: torch.float32,
    torch.complex128: torch.float64,
}


def _collapse(shape, strides):
    """Drop extent-1 dims and merge adjacent dims that are contiguous."""
    dims = [(int(s), int(st)) for s, st in zip(shape, strides) if int(s) != 1]
    if not dims:
        return (), ()
    out_s = [dims[0][0]]
    out_st = [dims[0][1]]
    for s, st in dims[1:]:
        if out_st[-1] == st * s:
            # merge: the inner dimension's stride becomes the merged stride
            out_s[-1] *= s
            out_st[-1] = st
        else:
            out_s.append(s)
            out_st.append(st)
    return tuple(out_s), tuple(out_st)


def _run_complex(t, n, part):
    out = torch.empty(t.shape, dtype=t.dtype, device=t.device)
    if n == 0:
        return out
    real = t.real  # zero-copy float view; the imag component is the next float
    # zero-copy float view of the freshly allocated contiguous output
    out_f = out.view(part) if out.dim() else out.unsqueeze(0).view(part)
    shape, strides = _collapse(real.shape, real.stride())
    rank = len(shape)
    if rank == 0:
        shape, strides, rank = (1,), (1,), 1
    if rank > 5:
        raise NotImplementedError("unsupported tensor rank for strided path")
    sizes = list(shape) + [1] * (5 - rank)
    strd = list(strides) + [1] * (5 - rank)
    grid = ((n + _BLOCK - 1) // _BLOCK,)
    _sqrt_complex_kernel[grid](
        real,
        out_f,
        n,
        sizes[1],
        sizes[2],
        sizes[3],
        sizes[4],
        strd[0],
        strd[1],
        strd[2],
        strd[3],
        strd[4],
        RANK=rank,
        BLOCK=_BLOCK,
        num_warps=_NUM_WARPS,
    )
    return out


def _run_one(t):
    dt = t.dtype
    n = t.numel()

    if dt.is_complex:
        part = _COMPLEX_PARTS.get(dt)
        if part is None:
            raise NotImplementedError("unsupported complex dtype: %s" % (dt,))
        return _run_complex(t, n, part)

    if dt.is_floating_point:
        out_dtype = dt
        mode = _FLOAT_MODES[dt]
    else:
        out_dtype = torch.get_default_dtype()
        mode = 3

    out = torch.empty(t.shape, dtype=out_dtype, device=t.device)
    if n == 0:
        return out

    if t.is_contiguous():
        grid = ((n + _BLOCK - 1) // _BLOCK,)
        _sqrt_flat_kernel[grid](
            t, out, n, BLOCK=_BLOCK, MODE=mode, num_warps=_NUM_WARPS
        )
        return out

    shape, strides = _collapse(t.shape, t.stride())
    rank = len(shape)
    if rank == 1 and strides[0] == 1:
        grid = ((n + _BLOCK - 1) // _BLOCK,)
        _sqrt_flat_kernel[grid](
            t, out, n, BLOCK=_BLOCK, MODE=mode, num_warps=_NUM_WARPS
        )
        return out

    if rank == 0:
        # numel 1 with a degenerate stride pattern
        shape, strides, rank = (1,), (1,), 1
    if rank > 5:
        raise NotImplementedError("unsupported tensor rank for strided path")

    sizes = list(shape) + [1] * (5 - rank)
    strd = list(strides) + [1] * (5 - rank)
    grid = ((n + _BLOCK - 1) // _BLOCK,)
    _sqrt_strided_kernel[grid](
        t,
        out,
        n,
        sizes[1],
        sizes[2],
        sizes[3],
        sizes[4],
        strd[0],
        strd[1],
        strd[2],
        strd[3],
        strd[4],
        RANK=rank,
        MODE=mode,
        BLOCK=_BLOCK,
        num_warps=_NUM_WARPS,
    )
    return out


def _group_run(tensors):
    """One launch for a homogeneous list of small contiguous tensors.

    Returns None (caller falls back to one launch per tensor) unless every
    tensor has the same dtype, shape, device and contiguous layout.
    """
    k = len(tensors)
    if k > _GROUP_SLOTS:
        return None
    first = tensors[0]
    dt = first.dtype
    if dt.is_complex:
        return None
    n = first.numel()
    if n == 0 or n > _GROUP_MAX_NUMEL:
        return None
    if not first.is_contiguous():
        return None
    shape = first.shape
    device = first.device
    for t in tensors[1:]:
        if (
            t.dtype is not dt
            or t.shape != shape
            or t.device != device
            or not t.is_contiguous()
        ):
            return None

    mode = _FLOAT_MODES.get(dt)
    out_dtype = dt
    if mode is None:
        out_dtype = torch.get_default_dtype()
        mode = 3

    outs = [torch.empty(shape, dtype=out_dtype, device=device) for _ in range(k)]
    ins = list(tensors)
    pad_out = outs[0]
    while len(ins) < _GROUP_SLOTS:
        ins.append(first)
        outs.append(pad_out)
    grid = ((n + _BLOCK - 1) // _BLOCK, k)
    _sqrt_group_kernel[grid](
        *ins,
        *outs,
        n,
        BLOCK=_BLOCK,
        MODE=mode,
        num_warps=_NUM_WARPS,
    )
    return outs[:k]


def _foreach_sqrt(self, *args, **kwargs):
    logger.debug("GEMS_MTHREADS _FOREACH_SQRT")
    tensors = self
    if isinstance(tensors, torch.Tensor):
        tensors = [tensors]
    elif not isinstance(tensors, (list, tuple)):
        if tensors is None and "tensors" in kwargs:
            tensors = kwargs["tensors"]
        else:
            tensors = list(tensors)
    if len(tensors) == 0:
        raise RuntimeError("Tensor list must have at least one tensor.")
    if any(
        not isinstance(t, torch.Tensor) or t.device.type != "musa" or t.dtype.is_complex
        for t in tensors
    ):
        return default__foreach_sqrt(tensors)
    if len(tensors) > 1:
        grouped = _group_run(tensors)
        if grouped is not None:
            return grouped
    return [_run_one(t) for t in tensors]
