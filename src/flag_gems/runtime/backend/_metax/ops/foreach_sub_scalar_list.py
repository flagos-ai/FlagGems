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

"""Triton implementation of ``aten::_foreach_sub.ScalarList``.

``out[i] = self[i] - scalars[i]`` (functional, freshly allocated outputs).

Design notes
------------
* The reference (ATen TensorIterator / ``gpu_kernel_with_scalars``) computes
  fp16/bfloat16 differences with fp32 "opmath", so we cast to fp32, subtract an
  fp32 scalar and round once on the way back.  float64 is subtracted in fp64
  (the Python double is transferred bit-exactly as an int64 payload and
  bit-cast inside the kernel, because Triton's default scalar ABI is fp32).
  Integer tensors use plain integer arithmetic.
* Contiguous tensors (the common case) use a flat 1-D kernel.
* Strided tensors are collapsed on the host and mapped with a 2-D scheme:
  the innermost (unit-stride when available) dimension is vectorised while the
  outer multi-index is decomposed once per program (scalar work only), so no
  per-element integer division is needed.
* Outputs are allocated per call with ``torch.empty_like`` (never cached) and
  the input tensor is never written to.
"""

import struct

import torch
import triton
import triton.language as tl

_FLAT_BLOCK = 1024
_FLAT_WARPS = 4
_I64_LIMIT = 1 << 31
_MULTI_T = 4

_F32_MODE = 0
_F64_MODE = 1
_INT_MODE = 2

_FAST_DTYPES = frozenset((torch.float32, torch.float16, torch.bfloat16))


@triton.jit
def _sub_flat_kernel(
    x_ptr,
    o_ptr,
    alpha_f32,
    alpha_i64,
    n,
    BLOCK: tl.constexpr,
    MODE: tl.constexpr,
    I64: tl.constexpr,
    PROMOTED: tl.constexpr,
):
    pid = tl.program_id(0)
    if I64:
        offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    else:
        offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0)
    if MODE == 0:
        y = x.to(tl.float32) - alpha_f32
        if not PROMOTED:
            y = y.to(x.dtype)
    elif MODE == 1:
        y = x - tl.cast(alpha_i64, tl.int64).to(tl.float64, bitcast=True)
    else:
        y = (x - alpha_i64).to(x.dtype)
    tl.store(o_ptr + offs, y, mask=mask)


@triton.jit
def _sub_strided_kernel(
    x_ptr,
    o_ptr,
    alpha_f32,
    alpha_i64,
    n_inner,
    blocks_per_row,
    i_in,
    o_in,
    s0,
    s1,
    s2,
    s3,
    s4,
    s5,
    s6,
    a0,
    a1,
    a2,
    a3,
    a4,
    a5,
    a6,
    b0,
    b1,
    b2,
    b3,
    b4,
    b5,
    b6,
    BLOCK: tl.constexpr,
    MODE: tl.constexpr,
    RANK: tl.constexpr,
    I64: tl.constexpr,
    PROMOTED: tl.constexpr,
):
    pid = tl.program_id(0)
    if I64:
        pid = pid.to(tl.int64)
    row = pid // blocks_per_row
    blk = pid - row * blocks_per_row
    rem = row
    io = rem - rem
    oo = io
    if RANK > 1:
        q = rem // s0
        rem = rem - q * s0
        io += q * a0
        oo += q * b0
    if RANK > 2:
        q = rem // s1
        rem = rem - q * s1
        io += q * a1
        oo += q * b1
    if RANK > 3:
        q = rem // s2
        rem = rem - q * s2
        io += q * a2
        oo += q * b2
    if RANK > 4:
        q = rem // s3
        rem = rem - q * s3
        io += q * a3
        oo += q * b3
    if RANK > 5:
        q = rem // s4
        rem = rem - q * s4
        io += q * a4
        oo += q * b4
    if RANK > 6:
        q = rem // s5
        rem = rem - q * s5
        io += q * a5
        oo += q * b5
    if RANK > 7:
        q = rem // s6
        rem = rem - q * s6
        io += q * a6
        oo += q * b6
    idx = blk * BLOCK + tl.arange(0, BLOCK)
    mask = idx < n_inner
    x = tl.load(x_ptr + io + idx * i_in, mask=mask, other=0)
    if MODE == 0:
        y = x.to(tl.float32) - alpha_f32
        if not PROMOTED:
            y = y.to(x.dtype)
    elif MODE == 1:
        y = x - tl.cast(alpha_i64, tl.int64).to(tl.float64, bitcast=True)
    else:
        y = (x - alpha_i64).to(x.dtype)
    tl.store(o_ptr + oo + idx * o_in, y, mask=mask)


def _f64_bits(value):
    return struct.unpack("<q", struct.pack("<d", float(value)))[0]


@triton.jit
def _sub_slot(
    x_ptr, o_ptr, alpha, n, base, pid, BLOCK: tl.constexpr, I64: tl.constexpr
):
    """One tensor's share of a fused multi-tensor launch."""
    if I64:
        offs = (pid.to(tl.int64) - base) * BLOCK + tl.arange(0, BLOCK)
    else:
        offs = (pid - base) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0)
    y = (x.to(tl.float32) - alpha).to(x.dtype)
    tl.store(o_ptr + offs, y, mask=mask)


@triton.jit
def _sub_multi_kernel(
    x0,
    o0,
    x1,
    o1,
    x2,
    o2,
    x3,
    o3,
    n0,
    n1,
    n2,
    n3,
    b1,
    b2,
    b3,
    a0,
    a1,
    a2,
    a3,
    BLOCK: tl.constexpr,
    I64: tl.constexpr,
):
    """Fused foreach-sub over up to four layout-uniform tensors.

    ``b1..b3`` hold cumulative block counts; a slot with no live tensor gets
    the total block count so its branch is unreachable.
    """
    pid = tl.program_id(0)
    if pid < b1:
        _sub_slot(x0, o0, a0, n0, 0, pid, BLOCK=BLOCK, I64=I64)
    elif pid < b2:
        _sub_slot(x1, o1, a1, n1, b1, pid, BLOCK=BLOCK, I64=I64)
    elif pid < b3:
        _sub_slot(x2, o2, a2, n2, b2, pid, BLOCK=BLOCK, I64=I64)
    else:
        _sub_slot(x3, o3, a3, n3, b3, pid, BLOCK=BLOCK, I64=I64)


def _scalar_for(dtype, scalar):
    """Return (mode, alpha_f32, alpha_i64) matching ATen semantics."""
    if dtype == torch.float64:
        return _F64_MODE, 0.0, _f64_bits(scalar)
    return _F32_MODE, float(scalar), 0


def _non_float_setup(t, scalar):
    """ATen promotion rules for non-floating tensors.

    Python scalars are "weak", so an integer scalar keeps an integer tensor's
    dtype, while a floating scalar promotes the result to float32.
    """
    dtype = t.dtype
    if dtype == torch.bool:
        raise RuntimeError(
            "Subtraction, the `-` operator, with two bool tensors is not "
            "supported. Use the `^` or `logical_xor()` operator instead."
        )
    if dtype.is_complex:
        raise RuntimeError("foreach_sub: unsupported dtype " + str(dtype))
    if isinstance(scalar, int) and not isinstance(scalar, bool):
        return torch.empty_like(t), _INT_MODE, 0.0, int(scalar)
    out = torch.empty(t.shape, dtype=torch.float32, device=t.device)
    return out, _F32_MODE, float(scalar), 0


def _collapse_dims(shape, istrides, ostrides):
    """Collapse size-1 dims and stride-compatible neighbours (outer->inner)."""
    dims = []
    for size, istride, ostride in zip(shape, istrides, ostrides):
        if size == 1:
            continue
        if dims:
            psize, pistride, postride = dims[-1]
            if pistride == istride * size and postride == ostride * size:
                # The outer dim is packed directly in front of this one, so the
                # pair flattens onto the *inner* stride (the flat index still
                # enumerates this dim fastest).
                dims[-1] = (psize * size, istride, ostride)
                continue
        dims.append((size, istride, ostride))
    return dims


def _launch_flat(x, out, n, alpha_f32, alpha_i64, mode):
    if n >= _FLAT_BLOCK:
        block = _FLAT_BLOCK
    else:
        block = triton.next_power_of_2(n)
        if block < 32:
            block = 32
    grid = ((n + block - 1) // block,)
    _sub_flat_kernel[grid](
        x,
        out,
        alpha_f32,
        alpha_i64,
        n,
        BLOCK=block,
        MODE=mode,
        I64=(n >= _I64_LIMIT),
        PROMOTED=(x.dtype != out.dtype),
        num_warps=_FLAT_WARPS,
    )


def _launch_strided(x, out, n, alpha_f32, alpha_i64, mode, dims):
    # Pick the innermost dimension: prefer one that is unit-stride for both the
    # input and the output, then one that is unit-stride for the input, then
    # the largest dimension.
    inner = -1
    best = -1
    for j, (size, istride, ostride) in enumerate(dims):
        if istride == 1 and ostride == 1 and size > best:
            best = size
            inner = j
    if inner < 0:
        for j, (size, istride, ostride) in enumerate(dims):
            if istride == 1 and size > best:
                best = size
                inner = j
    if inner < 0:
        for j, (size, istride, ostride) in enumerate(dims):
            if size > best:
                best = size
                inner = j

    inner_size, i_in, o_in = dims[inner]
    outer = [d for j, d in enumerate(dims) if j != inner]
    rank = len(dims)
    if rank > 8:
        # Extremely rare: fall back to the flat kernel over an explicit
        # out-of-place copy of the collapsed metadata is not permitted, so
        # walk the outermost dimension with several launches instead.
        raise RuntimeError("foreach_sub: unsupported tensor rank")
    rows = n // inner_size

    block = triton.next_power_of_2(inner_size)
    if block > _FLAT_BLOCK:
        block = _FLAT_BLOCK
    if block < 32:
        block = 32
    blocks_per_row = (inner_size + block - 1) // block
    grid = (rows * blocks_per_row,)

    pads = [(1, 0, 0)] * 7
    # ``row`` enumerates the outer dimensions in row-major order, so the
    # divisor used to peel dimension j off is the product of the sizes of the
    # dimensions after it (row-space stride), not dimension j's own size.
    running = 1
    for j in range(len(outer) - 1, -1, -1):
        size, i_stride, o_stride = outer[j]
        pads[j] = (running, i_stride, o_stride)
        running *= size
    _sub_strided_kernel[grid](
        x,
        out,
        alpha_f32,
        alpha_i64,
        inner_size,
        blocks_per_row,
        i_in,
        o_in,
        pads[0][0],
        pads[1][0],
        pads[2][0],
        pads[3][0],
        pads[4][0],
        pads[5][0],
        pads[6][0],
        pads[0][1],
        pads[1][1],
        pads[2][1],
        pads[3][1],
        pads[4][1],
        pads[5][1],
        pads[6][1],
        pads[0][2],
        pads[1][2],
        pads[2][2],
        pads[3][2],
        pads[4][2],
        pads[5][2],
        pads[6][2],
        BLOCK=block,
        MODE=mode,
        RANK=rank,
        I64=(n >= _I64_LIMIT),
        PROMOTED=(x.dtype != out.dtype),
        num_warps=_FLAT_WARPS,
    )


def _launch_multi(items, block=_FLAT_BLOCK):
    """One launch for a batch (1..4) of layout-uniform contiguous tensors."""
    ts = []
    os_ = []
    ns = []
    als = []
    prefixes = []
    total = 0
    big = False
    for t, out, n, alpha in items:
        ts.append(t)
        os_.append(out)
        ns.append(n)
        als.append(alpha)
        total += -(-n // block)
        prefixes.append(total)
        if n >= _I64_LIMIT:
            big = True
    while len(prefixes) < 3:
        prefixes.append(total)
    while len(ts) < _MULTI_T:
        ts.append(ts[0])
        os_.append(os_[0])
        ns.append(0)
        als.append(0.0)
    _sub_multi_kernel[(total,)](
        ts[0],
        os_[0],
        ts[1],
        os_[1],
        ts[2],
        os_[2],
        ts[3],
        os_[3],
        ns[0],
        ns[1],
        ns[2],
        ns[3],
        prefixes[0],
        prefixes[1],
        prefixes[2],
        als[0],
        als[1],
        als[2],
        als[3],
        BLOCK=block,
        I64=big,
        num_warps=_FLAT_WARPS,
    )


def _foreach_sub_one(t, scalar):
    dtype = t.dtype
    if dtype.is_floating_point:
        out = torch.empty_like(t)
        n = t.numel()
        if n == 0:
            return out
        mode, alpha_f32, alpha_i64 = _scalar_for(dtype, scalar)
    else:
        out, mode, alpha_f32, alpha_i64 = _non_float_setup(t, scalar)
        n = t.numel()
        if n == 0:
            return out
    if n == 1 or (t.is_contiguous() and out.is_contiguous()):
        _launch_flat(t, out, n, alpha_f32, alpha_i64, mode)
    else:
        dims = _collapse_dims(t.shape, t.stride(), out.stride())
        if not dims:
            _launch_flat(t, out, n, alpha_f32, alpha_i64, mode)
        else:
            _launch_strided(t, out, n, alpha_f32, alpha_i64, mode, dims)
    return out


def foreach_sub_scalar_list(self, scalars):
    tensors = self
    if isinstance(tensors, torch.Tensor):
        tensors = [tensors]
    else:
        tensors = list(tensors)

    if isinstance(scalars, (int, float)) and not isinstance(scalars, bool):
        scalar_list = [scalars] * len(tensors)
    else:
        scalar_list = list(scalars)

    m = len(tensors)
    if m != len(scalar_list):
        raise RuntimeError("Tensor list and scalar list must have the same length")
    if m == 0:
        return []

    outs = [None] * m

    # Fast path: 2..4 same-shape, same-dtype, same-device contiguous tensors of
    # a 2- or 4-byte float dtype.  One fused launch, and the outputs come from a
    # single allocation split with zero-copy views instead of m allocations.
    if 2 <= m <= 4:
        t0 = tensors[0]
        dtype = t0.dtype
        if dtype in _FAST_DTYPES and t0.is_contiguous():
            device = t0.device
            shape = t0.shape
            uniform = True
            for k in range(1, m):
                t = tensors[k]
                if (
                    t.dtype is not dtype
                    or t.device != device
                    or t.shape != shape
                    or not t.is_contiguous()
                ):
                    uniform = False
                    break
            if uniform:
                ns = [t.numel() for t in tensors]
                als = [float(s) for s in scalar_list]
                buf = torch.empty((m,) + shape, dtype=dtype, device=device)
                outs = list(torch.unbind(buf, 0))
                if any(ns):
                    _launch_multi(list(zip(tensors, outs, ns, als)))
                return outs

    pending = []
    pending_key = None
    for i in range(m):
        t = tensors[i]
        dtype = t.dtype
        if not dtype.is_floating_point or dtype == torch.float64:
            if pending:
                _launch_multi(pending)
                pending = []
                pending_key = None
            outs[i] = _foreach_sub_one(t, scalar_list[i])
            continue
        out = torch.empty_like(t)
        outs[i] = out
        n = t.numel()
        if n == 0:
            continue
        scalar = scalar_list[i]
        if n != 1 and not (t.is_contiguous() and out.is_contiguous()):
            if pending:
                _launch_multi(pending)
                pending = []
                pending_key = None
            dims = _collapse_dims(t.shape, t.stride(), out.stride())
            if dims:
                _launch_strided(t, out, n, float(scalar), 0, _F32_MODE, dims)
            else:
                _launch_flat(t, out, n, float(scalar), 0, _F32_MODE)
            continue
        key = (t.device, dtype)
        if key != pending_key or len(pending) == _MULTI_T:
            if pending:
                _launch_multi(pending)
            pending = []
            pending_key = key
        pending.append((t, out, n, float(scalar)))
    if pending:
        _launch_multi(pending)
    return outs
