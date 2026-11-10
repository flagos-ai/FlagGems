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

"""Triton implementation of aten::_foreach_clamp_min_ (TensorList, ScalarList/Scalar).

In-place elementwise ``self[i] = max(self[i], scalars[i])`` for every tensor in the
input list, returning ``None``.  Reference-exact elementwise selection is
``r = s if (x < s) else x`` which propagates NaN inputs and preserves signed zeros.

Dispatch:
  * lists of 2..4 same-dtype, equally sized, contiguous, non-aliasing tensors whose
    whole call is small enough to be launch bound are served by ONE fused launch:
    a 2D grid (blocks, slot) with a uniform slot branch, a single shared length and
    real per-slot pointer arguments (so alignment / vectorisation is preserved);
  * everything else uses one flat kernel per contiguous tensor, which reaches the
    device memory-bandwidth ceiling for large tensors;
  * non-contiguous tensors use a collapsed-dimension 2D kernel.

The host path is deliberately lean: each tensor's metadata is read exactly once per
call and, for the launch-bound fused path, the already-compiled kernel is re-launched
through the compiled-kernel runner instead of re-entering JIT dispatch.  The runner
is only reused when the canonical specialisation is guaranteed (numel a non-one
multiple of 16 and every data pointer 16-byte aligned); otherwise the normal
``kernel[grid](...)`` launch is used.
"""

import struct

import torch
import triton
import triton.language as tl

_BLOCK = 1024
_NUM_WARPS = 4
_MAX_OUTER = 8
_MAX_FUSED = 4
_FUSE_MAX_BLOCKS = 16384

_MODE_FLOAT = 0
_MODE_INT = 1
_MODE_F64 = 2

_MODE_BY_DTYPE = {
    torch.float16: _MODE_FLOAT,
    torch.bfloat16: _MODE_FLOAT,
    torch.float32: _MODE_FLOAT,
    torch.float64: _MODE_F64,
    torch.int8: _MODE_INT,
    torch.uint8: _MODE_INT,
    torch.int16: _MODE_INT,
    torch.int32: _MODE_INT,
    torch.int64: _MODE_INT,
    torch.bool: _MODE_INT,
}

# (blocks, slots) -> compiled-kernel runner for the canonical aligned case.
_FUSED_RUNNERS = {}


@triton.jit
def _body_f32(x_ptr, n, s_f32, lb, BLOCK: tl.constexpr):
    idx = lb.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = idx < n
    x = tl.load(x_ptr + idx, mask=mask)
    s = s_f32.to(x.dtype)
    tl.store(x_ptr + idx, tl.where(x < s, s, x), mask=mask)


@triton.jit
def _body_bits(x_ptr, n, s_bits, lb, BLOCK: tl.constexpr, MODE: tl.constexpr):
    idx = lb.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = idx < n
    x = tl.load(x_ptr + idx, mask=mask)
    if MODE == 1:
        s = s_bits.to(x.dtype)
    else:
        s = s_bits.to(tl.float64, bitcast=True)
    tl.store(x_ptr + idx, tl.where(x < s, s, x), mask=mask)


@triton.jit
def _cm_flat_kernel(x_ptr, n, s_f32, s_i64, BLOCK: tl.constexpr, MODE: tl.constexpr):
    pid = tl.program_id(0)
    idx = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = idx < n
    x = tl.load(x_ptr + idx, mask=mask)
    if MODE == 0:
        s = s_f32.to(x.dtype)
    elif MODE == 1:
        s = s_i64.to(x.dtype)
    else:
        s = s_i64.to(tl.float64, bitcast=True)
    tl.store(x_ptr + idx, tl.where(x < s, s, x), mask=mask)


@triton.jit
def _cm_fused_f32_kernel(x0, x1, x2, x3, n, s0, s1, s2, s3, BLOCK: tl.constexpr):
    lb = tl.program_id(0)
    slot = tl.program_id(1)
    if slot == 0:
        _body_f32(x0, n, s0, lb, BLOCK)
    elif slot == 1:
        _body_f32(x1, n, s1, lb, BLOCK)
    elif slot == 2:
        _body_f32(x2, n, s2, lb, BLOCK)
    else:
        _body_f32(x3, n, s3, lb, BLOCK)


@triton.jit
def _cm_fused_bits_kernel(
    x0,
    x1,
    x2,
    x3,
    n,
    s0,
    s1,
    s2,
    s3,
    BLOCK: tl.constexpr,
    MODE: tl.constexpr,
):
    lb = tl.program_id(0)
    slot = tl.program_id(1)
    if slot == 0:
        _body_bits(x0, n, s0, lb, BLOCK, MODE)
    elif slot == 1:
        _body_bits(x1, n, s1, lb, BLOCK, MODE)
    elif slot == 2:
        _body_bits(x2, n, s2, lb, BLOCK, MODE)
    else:
        _body_bits(x3, n, s3, lb, BLOCK, MODE)


@triton.jit
def _cm_strided_kernel(
    x_ptr,
    s_f32,
    s_i64,
    sz0,
    sz1,
    sz2,
    sz3,
    sz4,
    sz5,
    sz6,
    sz7,
    st0,
    st1,
    st2,
    st3,
    st4,
    st5,
    st6,
    st7,
    n_inner,
    i_stride,
    BLOCK: tl.constexpr,
    MODE: tl.constexpr,
    CONTIG_INNER: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    nb = (n_inner + (BLOCK - 1)) // BLOCK
    o = pid // nb
    b = pid - o * nb

    rem = o
    r0 = rem % sz0
    rem = rem // sz0
    r1 = rem % sz1
    rem = rem // sz1
    r2 = rem % sz2
    rem = rem // sz2
    r3 = rem % sz3
    rem = rem // sz3
    r4 = rem % sz4
    rem = rem // sz4
    r5 = rem % sz5
    rem = rem // sz5
    r6 = rem % sz6
    rem = rem // sz6
    r7 = rem % sz7
    off = (
        r0 * st0
        + r1 * st1
        + r2 * st2
        + r3 * st3
        + r4 * st4
        + r5 * st5
        + r6 * st6
        + r7 * st7
    )

    idx = b * BLOCK + tl.arange(0, BLOCK)
    mask = idx < n_inner
    if CONTIG_INNER:
        ptrs = x_ptr + off + idx
    else:
        ptrs = x_ptr + off + idx * i_stride
    x = tl.load(ptrs, mask=mask)
    if MODE == 0:
        s = s_f32.to(x.dtype)
    elif MODE == 1:
        s = s_i64.to(x.dtype)
    else:
        s = s_i64.to(tl.float64, bitcast=True)
    r = tl.where(x < s, s, x)
    tl.store(ptrs, r, mask=mask)


def _collapse_dims(shape, stride):
    """Drop size-1 dims and merge adjacent dims that are dense in memory."""
    dims = []
    for sz, st in zip(shape, stride):
        sz = int(sz)
        st = int(st)
        if sz == 1:
            continue
        if dims and dims[-1][1] == st * sz:
            psz, pst = dims[-1]
            dims[-1] = (psz * sz, pst)
        else:
            dims.append((sz, st))
    if not dims:
        dims = [(1, 1)]
    return dims


def _scalar_pair(value, mode):
    """Return (s_f32, s_i64) kernel scalar arguments for the given element mode."""
    if isinstance(value, torch.Tensor):
        value = value.item()
    if mode == _MODE_F64:
        return 0.0, struct.unpack("<q", struct.pack("<d", float(value)))[0]
    if mode == _MODE_INT:
        if isinstance(value, bool):
            return 0.0, 1 if value else 0
        return 0.0, int(value)
    return float(value), 0


def _launch_single(t, s_f32, s_i64, mode):
    numel = t.numel()
    if t.is_contiguous():
        _cm_flat_kernel[((numel + _BLOCK - 1) // _BLOCK,)](
            t, numel, s_f32, s_i64, _BLOCK, mode, num_warps=_NUM_WARPS
        )
        return

    dims = _collapse_dims(t.shape, t.stride())
    inner_idx = -1
    for i, (sz, st) in enumerate(dims):
        if st == 1 and sz > 1:
            inner_idx = i
            break
    if inner_idx < 0:
        inner_idx = len(dims) - 1
    inner_sz, inner_st = dims[inner_idx]
    outer = dims[:inner_idx] + dims[inner_idx + 1 :]
    if len(outer) > _MAX_OUTER:
        raise RuntimeError(
            "foreach_clamp_min: unsupported tensor rank after collapsing"
        )
    n_inner = inner_sz
    n_outer = numel // n_inner

    sz = [1] * _MAX_OUTER
    st = [0] * _MAX_OUTER
    for i, (dsz, dst) in enumerate(outer):
        sz[i] = dsz
        st[i] = dst

    n_blocks = (n_inner + _BLOCK - 1) // _BLOCK
    _cm_strided_kernel[(n_outer * n_blocks,)](
        t,
        s_f32,
        s_i64,
        sz[0],
        sz[1],
        sz[2],
        sz[3],
        sz[4],
        sz[5],
        sz[6],
        sz[7],
        st[0],
        st[1],
        st[2],
        st[3],
        st[4],
        st[5],
        st[6],
        st[7],
        n_inner,
        inner_st,
        _BLOCK,
        mode,
        inner_st == 1,
        num_warps=_NUM_WARPS,
    )


def _launch_fused(ts, numel, svals, mode, blocks, fast):
    """One launch for 2..4 equally sized same-dtype contiguous tensors."""
    m = len(ts)
    if m < _MAX_FUSED:
        ts = ts + [ts[-1]] * (_MAX_FUSED - m)
        svals = svals + [svals[-1]] * (_MAX_FUSED - m)
    args = (ts[0], ts[1], ts[2], ts[3], numel, svals[0], svals[1], svals[2], svals[3])
    if mode != _MODE_FLOAT:
        _cm_fused_bits_kernel[(blocks, m)](*args, _BLOCK, mode, num_warps=_NUM_WARPS)
        return
    if fast:
        key = (blocks, m)
        runner = _FUSED_RUNNERS.get(key)
        if runner is not None:
            runner(*args, _BLOCK)
            return
        compiled = _cm_fused_f32_kernel[(blocks, m)](
            *args, _BLOCK, num_warps=_NUM_WARPS
        )
        if compiled is not None:
            try:
                _FUSED_RUNNERS[key] = compiled[(blocks, m, 1)]
            except Exception:
                pass
        return
    _cm_fused_f32_kernel[(blocks, m)](*args, _BLOCK, num_warps=_NUM_WARPS)


def foreach_clamp_min_scalar_list_(self, scalars):
    tensors = list(self)
    if not tensors:
        return None

    if isinstance(scalars, torch.Tensor):
        if scalars.dim() == 0:
            scalar_list = None
            scalar_value = scalars
        else:
            scalar_list = list(scalars)
            scalar_value = None
    elif isinstance(scalars, (int, float, bool)):
        scalar_list = None
        scalar_value = scalars
    else:
        try:
            scalar_list = list(scalars)
            scalar_value = None
        except TypeError:
            scalar_list = None
            scalar_value = scalars

    if scalar_list is not None and len(scalar_list) != len(tensors):
        raise RuntimeError("Tensor list must have same size as scalar list.")

    dev = tensors[0].get_device()
    if dev >= 0 and dev != torch.cuda.current_device():
        with torch.cuda.device(dev):
            _dispatch(tensors, scalar_list, scalar_value)
    else:
        _dispatch(tensors, scalar_list, scalar_value)
    return None


def _dispatch(tensors, scalar_list, scalar_value):
    ntl = len(tensors)
    if ntl > 1:
        dt0 = tensors[0].dtype
        numel0 = tensors[0].numel()
        blocks = (numel0 + _BLOCK - 1) // _BLOCK
        fusable = numel0 > 0 and blocks <= _FUSE_MAX_BLOCKS
        aligned = (numel0 & 15) == 0 and numel0 != 1
        if fusable:
            ptrs = []
            for t in tensors:
                if t.dtype is not dt0 or t.numel() != numel0 or not t.is_contiguous():
                    fusable = False
                    break
                p = t.data_ptr()
                if p in ptrs:
                    fusable = False
                    break
                ptrs.append(p)
                if p & 15:
                    aligned = False
        if fusable:
            mode = _MODE_BY_DTYPE.get(dt0, _MODE_FLOAT)
            svals = []
            if scalar_list is None:
                v = scalar_value
                if mode == _MODE_FLOAT:
                    if not isinstance(v, float):
                        v = _scalar_pair(v, mode)[0]
                    svals = [v] * ntl
                else:
                    svals = [_scalar_pair(v, mode)[1]] * ntl
            elif mode == _MODE_FLOAT:
                for v in scalar_list:
                    svals.append(
                        v if isinstance(v, float) else _scalar_pair(v, mode)[0]
                    )
            else:
                for v in scalar_list:
                    svals.append(_scalar_pair(v, mode)[1])

            for start in range(0, ntl, _MAX_FUSED):
                m = ntl - start
                if m > _MAX_FUSED:
                    m = _MAX_FUSED
                chunk = tensors[start : start + m]
                if m == 1:
                    sv = svals[start]
                    if mode == _MODE_FLOAT:
                        _launch_single(chunk[0], sv, 0, mode)
                    else:
                        _launch_single(chunk[0], 0.0, sv, mode)
                else:
                    _launch_fused(
                        chunk, numel0, svals[start : start + m], mode, blocks, aligned
                    )
            return

    for i, t in enumerate(tensors):
        if t.numel() == 0:
            continue
        mode = _MODE_BY_DTYPE.get(t.dtype, _MODE_FLOAT)
        value = scalar_value if scalar_list is None else scalar_list[i]
        s_f32, s_i64 = _scalar_pair(value, mode)
        _launch_single(t, s_f32, s_i64, mode)
