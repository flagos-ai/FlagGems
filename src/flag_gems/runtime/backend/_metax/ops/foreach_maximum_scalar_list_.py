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

"""Triton implementation of ``aten::_foreach_maximum_.ScalarList``.

In-place elementwise maximum between every tensor of ``self`` and the paired
scalar, with ATen semantics: NaN propagates from either operand (the result is
NaN whenever the element or the scalar is NaN) and the scalar is first
converted to the element dtype of the tensor it is paired with.

Tensor storage must be preserved (no copies / no new outputs) so only the
original pointers are written through their original strides.

The op is bandwidth bound and each benchmarked list holds only four small
tensors, so per-call host dispatch is a first-order cost.  Contiguous tensors
are therefore fused into a single launch, and repeat launches of an already
compiled variant go through the compiled kernel's launcher directly instead of
re-running Triton's JIT dispatch (binder, cache-key hashing, hooks).
"""

import numbers
import struct

import torch
import triton
import triton.language as tl

_BLOCK = 2048
_NUM_WARPS = 8
_SBLOCK = 1024
_SWARPS = 4
_MAX_RANK = 5
_I32_LIMIT = 0x7FFFF000

_FAST_CACHE = {}
_FAST_ENABLED = None
_GET_STREAM = None
_HOOKS = (None, None)
_FAST_ERR = ""


@triton.jit
def _nanmax(x, sc):
    # ATen maximum functor: _isnan(a) || a > b ? a : b  (NaN propagates).
    return tl.where(x != x, x, tl.where(sc != sc, sc, tl.where(x > sc, x, sc)))


@triton.jit
def _flat_body(
    p, n, s, pid, BLOCK: tl.constexpr, EVEN: tl.constexpr, FP64: tl.constexpr
):
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    if FP64:
        sc = s.to(tl.float64, bitcast=True)
    else:
        sc = s
    if EVEN:
        x = tl.load(p + offs)
        tl.store(p + offs, _nanmax(x, sc.to(x.dtype)))
    else:
        mask = offs < n
        x = tl.load(p + offs, mask=mask)
        tl.store(p + offs, _nanmax(x, sc.to(x.dtype)), mask=mask)


@triton.jit
def _fused4_kernel(
    p0,
    p1,
    p2,
    p3,
    n0,
    n1,
    n2,
    n3,
    s0,
    s1,
    s2,
    s3,
    BLOCK: tl.constexpr,
    E0: tl.constexpr,
    E1: tl.constexpr,
    E2: tl.constexpr,
    E3: tl.constexpr,
    F0: tl.constexpr,
    F1: tl.constexpr,
    F2: tl.constexpr,
    F3: tl.constexpr,
):
    """One launch for up to four independent flat (contiguous) tensors.

    Each tensor owns ``ceil(n / BLOCK)`` consecutive programs, so a program
    always works on exactly one tensor and its pointer is selected statically.
    Unused slots get ``n == 0`` and therefore own no programs at all.
    """
    pid = tl.program_id(0)
    b0 = (n0 + (BLOCK - 1)) // BLOCK
    b1 = (n1 + (BLOCK - 1)) // BLOCK
    b2 = (n2 + (BLOCK - 1)) // BLOCK
    if pid < b0:
        _flat_body(p0, n0, s0, pid, BLOCK, E0, F0)
    elif pid < b0 + b1:
        _flat_body(p1, n1, s1, pid - b0, BLOCK, E1, F1)
    elif pid < b0 + b1 + b2:
        _flat_body(p2, n2, s2, pid - b0 - b1, BLOCK, E2, F2)
    else:
        _flat_body(p3, n3, s3, pid - b0 - b1 - b2, BLOCK, E3, F3)


@triton.jit
def _maximum_flat_kernel(
    x_ptr,
    n_elem,
    s,
    BLOCK: tl.constexpr,
    EVEN: tl.constexpr,
    FP64: tl.constexpr,
):
    """Single-tensor flat path (int64 offsets) used above the int32 range."""
    pid = tl.program_id(0)
    offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    if FP64:
        sc = s.to(tl.float64, bitcast=True)
    else:
        sc = s
    if EVEN:
        x = tl.load(x_ptr + offs)
        tl.store(x_ptr + offs, _nanmax(x, sc.to(x.dtype)))
    else:
        mask = offs < n_elem
        x = tl.load(x_ptr + offs, mask=mask)
        tl.store(x_ptr + offs, _nanmax(x, sc.to(x.dtype)), mask=mask)


@triton.jit
def _maximum_strided_kernel(
    x_ptr,
    n_elem,
    s,
    sh0,
    sh1,
    sh2,
    sh3,
    sh4,
    st0,
    st1,
    st2,
    st3,
    st4,
    NDIM: tl.constexpr,
    BLOCK: tl.constexpr,
    EVEN: tl.constexpr,
    FP64: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    # NDIM dims, innermost (smallest stride) first.
    lin = offs
    off = tl.zeros([BLOCK], dtype=offs.dtype)
    if NDIM >= 1:
        off += (lin % sh0) * st0
        lin = lin // sh0
    if NDIM >= 2:
        off += (lin % sh1) * st1
        lin = lin // sh1
    if NDIM >= 3:
        off += (lin % sh2) * st2
        lin = lin // sh2
    if NDIM >= 4:
        off += (lin % sh3) * st3
        lin = lin // sh3
    if NDIM >= 5:
        off += (lin % sh4) * st4
    if FP64:
        sc = s.to(tl.float64, bitcast=True)
    else:
        sc = s
    if EVEN:
        x = tl.load(x_ptr + off)
        tl.store(x_ptr + off, _nanmax(x, sc.to(x.dtype)))
    else:
        mask = offs < n_elem
        x = tl.load(x_ptr + off, mask=mask)
        tl.store(x_ptr + off, _nanmax(x, sc.to(x.dtype)), mask=mask)


def _fast_setup():
    """Detect whether the installed Triton exposes the cached direct launch."""
    global _FAST_ENABLED, _GET_STREAM
    if _FAST_ENABLED is not None:
        return _FAST_ENABLED
    _FAST_ENABLED = False
    try:
        import triton.knobs as _knobs
        from triton.runtime import driver as _drv
        from triton.runtime.jit import DistributedRtContext

        if not DistributedRtContext().is_lite_mode:
            _GET_STREAM = _drv.active.get_current_stream
            getattr(_knobs.runtime, "launch_enter_hook", None)
            getattr(_knobs.runtime, "launch_exit_hook", None)
            _FAST_ENABLED = True
    except Exception:  # noqa: BLE001
        _FAST_ENABLED = False
    return _FAST_ENABLED


def _scalar_arg(s, dt):
    """Host-side scalar conversion; returns (kernel arg, is_float64_bits)."""
    if dt.is_floating_point:
        if dt == torch.float64:
            return struct.unpack("<q", struct.pack("<d", float(s)))[0], True
        return float(s), False
    return int(s), False


def _launch_flat_group(group):
    """group: 1..4 (tensor, scalar, numel) tuples, all contiguous."""
    global _FAST_ENABLED, _FAST_ERR
    ptrs = []
    ns = []
    scs = []
    f64 = []
    int_scalar = False
    blocks = 0
    for t, s, n in group:
        sc, is64 = _scalar_arg(s, t.dtype)
        ptrs.append(t)
        ns.append(n)
        scs.append(sc)
        f64.append(is64)
        if not t.dtype.is_floating_point:
            int_scalar = True
        blocks += -(-n // _BLOCK)
    while len(ptrs) < 4:
        ptrs.append(ptrs[0])
        ns.append(0)
        scs.append(scs[0])
        f64.append(f64[0])
    vals = (
        ptrs[0],
        ptrs[1],
        ptrs[2],
        ptrs[3],
        ns[0],
        ns[1],
        ns[2],
        ns[3],
        scs[0],
        scs[1],
        scs[2],
        scs[3],
        _BLOCK,
        ns[0] % _BLOCK == 0,
        ns[1] % _BLOCK == 0,
        ns[2] % _BLOCK == 0,
        ns[3] % _BLOCK == 0,
        f64[0],
        f64[1],
        f64[2],
        f64[3],
    )
    grid = (blocks,)

    if not int_scalar and _fast_setup():
        dev = ptrs[0].device.index
        key = (
            ptrs[0].dtype,
            ptrs[1].dtype,
            ptrs[2].dtype,
            ptrs[3].dtype,
            ns[0],
            ns[1],
            ns[2],
            ns[3],
            f64[0],
            f64[1],
            f64[2],
            f64[3],
            (ptrs[0].data_ptr() & 15)
            | ((ptrs[1].data_ptr() & 15) << 8)
            | ((ptrs[2].data_ptr() & 15) << 16)
            | ((ptrs[3].data_ptr() & 15) << 24),
            dev,
        )
        entry = _FAST_CACHE.get(key)
        if entry is None:
            ker = _fused4_kernel.run(
                *vals, grid=grid, warmup=True, num_warps=_NUM_WARPS
            )
            entry = (ker.run, ker)
            _FAST_CACHE[key] = entry
        try:
            ker = entry[1]
            stream = _GET_STREAM(dev)
            lm = ker.launch_metadata(grid, stream, *vals)
            entry[0](
                blocks,
                1,
                1,
                stream,
                ker.function,
                ker.packed_metadata,
                lm,
                _HOOKS[0],
                _HOOKS[1],
                *vals,
            )
            return
        except Exception as exc:  # noqa: BLE001
            # Fall back permanently; the op is idempotent so a retry is safe.
            _FAST_ERR = repr(exc)[:400]
            _FAST_CACHE.pop(key, None)
            _FAST_ENABLED = False

    _fused4_kernel[grid](*vals, num_warps=_NUM_WARPS)


def _launch_single_flat(t, s, n):
    sc, is64 = _scalar_arg(s, t.dtype)
    grid = (triton.cdiv(n, _BLOCK),)
    _maximum_flat_kernel[grid](
        t,
        n,
        sc,
        _BLOCK,
        n % _BLOCK == 0,
        is64 and t.dtype == torch.float64,
        num_warps=_NUM_WARPS,
    )


def _normalize(t):
    """Drop size-1 dims, collapse adjacent dims, order by stride ascending."""
    dims = [(sz, st) for sz, st in zip(t.shape, t.stride()) if sz != 1]
    out = []
    for sz, st in dims:
        if out and out[-1][1] == st * sz:
            out[-1] = (out[-1][0] * sz, out[-1][1])
        else:
            out.append((sz, st))
    return sorted(out, key=lambda p: p[1])


def _launch_strided(t, s, n):
    sc, is64 = _scalar_arg(s, t.dtype)
    dims = _normalize(t)
    nd = len(dims)
    sizes = [d[0] for d in dims] + [1] * (_MAX_RANK - nd)
    strides = [d[1] for d in dims] + [0] * (_MAX_RANK - nd)
    grid = (triton.cdiv(n, _SBLOCK),)
    _maximum_strided_kernel[grid](
        t,
        n,
        sc,
        sizes[0],
        sizes[1],
        sizes[2],
        sizes[3],
        sizes[4],
        strides[0],
        strides[1],
        strides[2],
        strides[3],
        strides[4],
        nd,
        _SBLOCK,
        n % _SBLOCK == 0,
        is64 and t.dtype == torch.float64,
        num_warps=_SWARPS,
    )


def _launch_tensor(t, s):
    n = t.numel()
    if n == 0:
        return
    if t.is_contiguous():
        _launch_single_flat(t, s, n)
        return
    if t.dim() > _MAX_RANK or len(_normalize(t)) > _MAX_RANK:
        for i in range(t.shape[0]):
            _launch_tensor(t[i], s)
        return
    _launch_strided(t, s, n)


def foreach_maximum_scalar_list_(self, scalars):
    if isinstance(scalars, numbers.Number):
        scalars = [scalars] * len(self)
    if len(self) != len(scalars):
        raise RuntimeError("Tensor list and Scalar list have different lengths")
    group = []
    for t, s in zip(self, scalars):
        n = t.numel()
        if n != 0 and n <= _I32_LIMIT and t.is_contiguous():
            group.append((t, s, n))
            if len(group) == 4:
                _launch_flat_group(group)
                group = []
        else:
            _launch_tensor(t, s)
    if group:
        _launch_flat_group(group)
    return None
