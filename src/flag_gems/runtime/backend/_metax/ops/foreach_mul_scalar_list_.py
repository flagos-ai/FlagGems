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

"""Triton implementation of ``aten::_foreach_mul_.ScalarList`` (in place).

For every i:  self[i] *= scalars[i]   (element-wise, in place, rounded to
self[i].dtype).  The call returns ``None`` and mutates storage.

Design
------
* Small lists (2..4 entries) of same dtype / same numel / contiguous /
  provably non-aliasing tensors are folded into ONE launch: program ids are
  partitioned into per-entry tile ranges, so a single launch covers the whole
  list and per-launch CPU overhead is paid once instead of N times.
* Anything else falls back to one launch per entry (flat 1D kernel when the
  tensor is contiguous, 2D strip kernel over host-collapsed dims otherwise).
* The unit / smallest-stride collapsed dim drives the vectorised axis, so
  transposed views still read and write coalesced.
* Scalar typing mirrors ATen's MulFunctor: fp16/bf16 go through fp32 "opmath"
  (native Triton promotion), fp64 receives its scalar exactly as a hi/lo fp32
  pair recombined in fp64 inside the kernel.
* No allocation, no data movement, no framework compute; returns ``None``.
"""

import math
import struct

import torch
import triton
import triton.language as tl

_f32 = struct.Struct("<f")

_MAX_GRID_Y = 65535
# Tensor payload (bytes) up to which a compatible list is folded into a single
# launch.  Kept well below the multi-GiB tensors, which already stream at the
# DRAM roofline and gain nothing from removing three launch gaps.
_FUSE_BYTES = 1 << 29

# Steady-state launch handles for the fused kernel, keyed by every property
# that participates in Triton's specialisation of that launch.
_LAUNCHERS = {}
_DIRECT = True


@triton.jit
def _mul_flat(x_ptr, s, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(x_ptr + offs, x * s, mask=mask)


@triton.jit
def _mul_flat_f64(x_ptr, s_hi, s_lo, n, BLOCK: tl.constexpr):
    sv = s_hi.to(tl.float64) + s_lo.to(tl.float64)
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(x_ptr + offs, x * sv, mask=mask)


@triton.jit
def _mul_k(x0, x1, x2, x3, s0, s1, s2, s3, n, K: tl.constexpr, BLOCK: tl.constexpr):
    """One launch for up to 4 entries that share dtype and numel."""
    pid = tl.program_id(0)
    T = (n + BLOCK - 1) // BLOCK
    if pid < T:
        o0 = pid * BLOCK + tl.arange(0, BLOCK)
        m0 = o0 < n
        v0 = tl.load(x0 + o0, mask=m0)
        tl.store(x0 + o0, v0 * s0, mask=m0)
    else:
        if K > 1:
            if pid < 2 * T:
                o1 = (pid - T) * BLOCK + tl.arange(0, BLOCK)
                m1 = o1 < n
                v1 = tl.load(x1 + o1, mask=m1)
                tl.store(x1 + o1, v1 * s1, mask=m1)
            else:
                if K > 2:
                    if pid < 3 * T:
                        o2 = (pid - 2 * T) * BLOCK + tl.arange(0, BLOCK)
                        m2 = o2 < n
                        v2 = tl.load(x2 + o2, mask=m2)
                        tl.store(x2 + o2, v2 * s2, mask=m2)
                    else:
                        o3 = (pid - 3 * T) * BLOCK + tl.arange(0, BLOCK)
                        m3 = o3 < n
                        v3 = tl.load(x3 + o3, mask=m3)
                        tl.store(x3 + o3, v3 * s3, mask=m3)


@triton.jit
def _mul_strided(
    x_ptr,
    s,
    last_len,
    stride_last,
    o0,
    o1,
    o2,
    st0,
    st1,
    st2,
    pid1_start,
    BLOCK: tl.constexpr,
):
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1) + pid1_start
    i2 = pid1 % o2
    r = pid1 // o2
    i1 = r % o1
    i0 = r // o1
    base = i0.to(tl.int64) * st0 + i1.to(tl.int64) * st1 + i2.to(tl.int64) * st2
    offs = pid0 * BLOCK + tl.arange(0, BLOCK)
    mask = offs < last_len
    idx = base + offs.to(tl.int64) * stride_last
    x = tl.load(x_ptr + idx, mask=mask)
    tl.store(x_ptr + idx, x * s, mask=mask)


@triton.jit
def _mul_strided_f64(
    x_ptr,
    s_hi,
    s_lo,
    last_len,
    stride_last,
    o0,
    o1,
    o2,
    st0,
    st1,
    st2,
    pid1_start,
    BLOCK: tl.constexpr,
):
    sv = s_hi.to(tl.float64) + s_lo.to(tl.float64)
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1) + pid1_start
    i2 = pid1 % o2
    r = pid1 // o2
    i1 = r % o1
    i0 = r // o1
    base = i0.to(tl.int64) * st0 + i1.to(tl.int64) * st1 + i2.to(tl.int64) * st2
    offs = pid0 * BLOCK + tl.arange(0, BLOCK)
    mask = offs < last_len
    idx = base + offs.to(tl.int64) * stride_last
    x = tl.load(x_ptr + idx, mask=mask)
    tl.store(x_ptr + idx, x * sv, mask=mask)


def _cfg(n):
    b = 1 << max(5, (n - 1).bit_length())
    if b > 1024:
        b = 1024
    if b <= 32:
        nw = 1
    elif b <= 128:
        nw = 2
    else:
        nw = 4
    return b, nw


def _collapse(shape, strides):
    """Merge adjacent dims that form a single contiguous run."""
    dims = [(sz, st) for sz, st in zip(shape, strides) if sz != 1]
    if not dims:
        return [(1, 0)]
    out = [dims[0]]
    for sz, st in dims[1:]:
        psz, pst = out[-1]
        if pst == sz * st:
            out[-1] = (psz * sz, st)
        else:
            out.append((sz, st))
    return out


def _launch_flat(t, sv, lo, n, is64):
    BLOCK, nw = _cfg(n)
    grid = ((n + BLOCK - 1) // BLOCK,)
    if is64:
        _mul_flat_f64[grid](t, sv, lo, n, BLOCK, num_warps=nw)
    else:
        _mul_flat[grid](t, sv, n, BLOCK, num_warps=nw)


def _launch_strided(t, sv, lo, is64, dims):
    sizes = [d[0] for d in dims]
    strides = [d[1] for d in dims]
    # innermost handled axis: prefer unit stride, then the smallest stride
    best = 0
    best_key = None
    for d in range(len(strides)):
        st = strides[d]
        key = (0 if st == 1 else (1 if st > 0 else 2), st)
        if best_key is None or key < best_key:
            best_key = key
            best = d
    last_len = sizes.pop(best)
    stride_last = strides.pop(best)
    o = [1, 1, 1]
    st = [0, 0, 0]
    k = len(sizes)
    for j in range(k):
        src = k - 1 - j
        o[j] = sizes[src]
        st[j] = strides[src]
    outer = o[0] * o[1] * o[2]
    BLOCK, nw = _cfg(last_len)
    gx = (last_len + BLOCK - 1) // BLOCK
    if is64:
        kern = _mul_strided_f64
        args = (t, sv, lo, last_len, stride_last, o[0], o[1], o[2], st[0], st[1], st[2])
    else:
        kern = _mul_strided
        args = (t, sv, last_len, stride_last, o[0], o[1], o[2], st[0], st[1], st[2])
    if outer <= _MAX_GRID_Y:
        kern[(gx, outer)](*args, 0, BLOCK, num_warps=nw)
    else:
        for start in range(0, outer, _MAX_GRID_Y):
            rows = min(_MAX_GRID_Y, outer - start)
            kern[(gx, rows)](*args, start, BLOCK, num_warps=nw)


def _apply(t, s):
    n = t.numel()
    if n == 0:
        return
    dt = t.dtype
    is64 = dt == torch.float64
    if is64:
        v = float(s)
        try:
            hi = _f32.unpack(_f32.pack(v))[0]
        except OverflowError:
            hi = v
        lo = 0.0 if (math.isinf(hi) or math.isnan(hi)) else v - hi
    else:
        if dt.is_floating_point:
            hi = float(s)
        else:
            if isinstance(s, float):
                raise RuntimeError(
                    "result type Float can't be cast to the desired output "
                    "type " + str(dt)
                )
            hi = int(s)
        lo = 0.0
    if t.is_contiguous():
        _launch_flat(t, hi, lo, n, is64)
        return
    dims = _collapse(t.shape, t.stride())
    for sz, st in dims:
        if st == 0 and sz > 1:
            raise RuntimeError(
                "unsupported operation: more than one element of the "
                "written-to tensor refers to a single memory location. "
                "Please clone() the tensor before calling this function."
            )
    if len(dims) > 4:
        # rare: peel the outermost dim until the remainder fits the kernel
        for i in range(t.shape[0]):
            _apply(t[i], s)
        return
    _launch_strided(t, hi, lo, is64, dims)


def _fused_launch(t0, t1, t2, t3, s0, s1, s2, s3, n, K, BLOCK, nw):
    """Single launch of ``_mul_k`` with a cached steady-state launch handle.

    The first call for a given specialisation goes through the ordinary JIT
    dispatch (which also returns the compiled kernel); the resulting
    ``CompiledKernel`` runner is cached, so later calls skip Triton's per-call
    argument binding, specialisation and cache-key machinery.  Anything
    unexpected disables the shortcut permanently and falls back to the normal
    dispatch, so numerics never depend on it.
    """
    global _DIRECT
    g0 = K * ((n + BLOCK - 1) // BLOCK)
    if _DIRECT:
        key = (
            K,
            BLOCK,
            nw,
            n,
            t0.dtype,
            t0.device.index,
            t0.data_ptr() & 15,
            t1.data_ptr() & 15,
            t2.data_ptr() & 15,
            t3.data_ptr() & 15,
        )
        ler = _LAUNCHERS.get(key)
        if ler is not None:
            try:
                ler(t0, t1, t2, t3, s0, s1, s2, s3, n, K, BLOCK)
                return
            except Exception:  # noqa: BLE001
                _DIRECT = False
        else:
            ck = _mul_k[(g0,)](
                t0, t1, t2, t3, s0, s1, s2, s3, n, K, BLOCK, num_warps=nw
            )
            try:
                _LAUNCHERS[key] = ck[(g0, 1, 1)]
            except Exception:  # noqa: BLE001
                _DIRECT = False
            return
    _mul_k[(g0,)](t0, t1, t2, t3, s0, s1, s2, s3, n, K, BLOCK, num_warps=nw)


def _try_fused(tensors, sl, K):
    """One launch for 2..4 mutually compatible entries.  Returns True on use."""
    t0 = tensors[0]
    dt = t0.dtype
    n = t0.numel()
    if n == 0 or dt == torch.float64 or not dt.is_floating_point:
        return False
    if n * dt.itemsize > _FUSE_BYTES or not t0.is_contiguous():
        return False
    ptrs = [t0.data_ptr()]
    for i in range(1, K):
        t = tensors[i]
        if t.dtype != dt or t.numel() != n or not t.is_contiguous():
            return False
        ptrs.append(t.data_ptr())
    # sequential semantics require disjoint storage when fusing
    nb = n * dt.itemsize
    ps = sorted(ptrs)
    for i in range(K - 1):
        if ps[i + 1] - ps[i] < nb:
            return False
    BLOCK, nw = _cfg(n)
    pad = tensors[K - 1]
    t1 = tensors[1]
    t2 = tensors[2] if K > 2 else pad
    t3 = tensors[3] if K > 3 else pad
    _fused_launch(
        t0,
        t1,
        t2,
        t3,
        float(sl[0]),
        float(sl[1]),
        float(sl[2]) if K > 2 else 1.0,
        float(sl[3]) if K > 3 else 1.0,
        n,
        K,
        BLOCK,
        nw,
    )
    return True


def foreach_mul_scalar_list_(self=None, scalars=None, *args, **kwargs):
    tensors = self
    if tensors is None and "self" in kwargs:
        tensors = kwargs["self"]
    if scalars is None and "scalars" in kwargs:
        scalars = kwargs["scalars"]
    if scalars is None and args:
        scalars = args[0]

    if torch.is_tensor(tensors):
        tensors = [tensors]
    K = len(tensors)
    if K == 0:
        return None

    if isinstance(scalars, (list, tuple)):
        if len(scalars) != K:
            raise RuntimeError(
                "Expected len(tensors) == len(scalars) but got len(tensors) = "
                + str(K)
                + " and len(scalars) = "
                + str(len(scalars))
            )
        if 2 <= K <= 4 and _try_fused(tensors, scalars, K):
            return None
        for i in range(K):
            _apply(tensors[i], scalars[i])
    else:
        for i in range(K):
            _apply(tensors[i], scalars)
    return None
