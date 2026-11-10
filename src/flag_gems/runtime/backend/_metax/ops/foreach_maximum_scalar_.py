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

import torch
import triton
import triton.language as tl

try:
    from triton import knobs as _knobs
except Exception:  # pragma: no cover
    _knobs = None

# ---------------------------------------------------------------------------
# foreach_maximum_(TensorList self, Scalar other)  ->  None
# In-place elementwise maximum between every tensor of the list and a scalar,
# NaN propagating (torch.maximum semantics), no temporary/copy produced.
#
# Fast path: one launch for a uniform contiguous list (same dtype, same numel)
# through the precompiled kernel handle, which skips the per-call JIT binder /
# cache-key / validation work that dominates short lists.
# ---------------------------------------------------------------------------

_BLOCK = 1024
_NUM_WARPS = 4
_MAXD = 5  # max collapsed dims handled by the strided kernel (1 inner + 4 outer)


@triton.jit
def _kmax_c(
    x_ptr,
    s,
    s_ptr,
    n_elem,
    IS_FLOAT: tl.constexpr,
    S_IS_PTR: tl.constexpr,
    I64: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    if I64:
        offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
        m = offs < n_elem.to(tl.int64)
    else:
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        m = offs < n_elem
    x = tl.load(x_ptr + offs, mask=m)
    if S_IS_PTR:
        sv = tl.load(s_ptr)
    else:
        sv = s
    s2 = sv.to(x.dtype)
    if IS_FLOAT:
        r = tl.maximum(x, s2, propagate_nan=tl.PropagateNan.ALL)
    else:
        r = tl.maximum(x, s2)
    tl.store(x_ptr + offs, r, mask=m)


@triton.jit
def _kmax_s(
    x_ptr,
    s,
    s_ptr,
    n_inner,
    inner_blocks,
    ST0,
    S1,
    S2,
    S3,
    S4,
    ST1,
    ST2,
    ST3,
    ST4,
    NDIM: tl.constexpr,
    IS_FLOAT: tl.constexpr,
    S_IS_PTR: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    row = pid // inner_blocks
    ib = pid - row * inner_blocks
    rem = row.to(tl.int64)
    base = tl.zeros([1], dtype=tl.int64)
    if NDIM >= 2:
        i = rem % S1
        rem = rem // S1
        base = base + i * ST1
    if NDIM >= 3:
        i = rem % S2
        rem = rem // S2
        base = base + i * ST2
    if NDIM >= 4:
        i = rem % S3
        rem = rem // S3
        base = base + i * ST3
    if NDIM >= 5:
        i = rem % S4
        rem = rem // S4
        base = base + i * ST4
    off_in = ib * BLOCK + tl.arange(0, BLOCK)
    m = off_in < n_inner
    offs = base + off_in.to(tl.int64) * ST0
    x = tl.load(x_ptr + offs, mask=m)
    if S_IS_PTR:
        sv = tl.load(s_ptr)
    else:
        sv = s
    s2 = sv.to(x.dtype)
    if IS_FLOAT:
        r = tl.maximum(x, s2, propagate_nan=tl.PropagateNan.ALL)
    else:
        r = tl.maximum(x, s2)
    tl.store(x_ptr + offs, r, mask=m)


@triton.jit
def _kmax_f4(
    t0,
    t1,
    t2,
    t3,
    n_elem,
    s,
    G: tl.constexpr,
    IS_FLOAT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    ib = tl.program_id(0)
    tid = tl.program_id(1)
    x_ptr = t0
    if G > 1:
        x_ptr = tl.where(tid == 1, t1, x_ptr)
    if G > 2:
        x_ptr = tl.where(tid == 2, t2, x_ptr)
    if G > 3:
        x_ptr = tl.where(tid == 3, t3, x_ptr)
    offs = ib * BLOCK + tl.arange(0, BLOCK)
    m = offs < n_elem
    x = tl.load(x_ptr + offs, mask=m)
    s2 = s.to(x.dtype)
    if IS_FLOAT:
        r = tl.maximum(x, s2, propagate_nan=tl.PropagateNan.ALL)
    else:
        r = tl.maximum(x, s2)
    tl.store(x_ptr + offs, r, mask=m)


@triton.jit
def _kmax_f8(
    t0,
    t1,
    t2,
    t3,
    t4,
    t5,
    t6,
    t7,
    n_elem,
    s,
    G: tl.constexpr,
    IS_FLOAT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    ib = tl.program_id(0)
    tid = tl.program_id(1)
    x_ptr = t0
    if G > 1:
        x_ptr = tl.where(tid == 1, t1, x_ptr)
    if G > 2:
        x_ptr = tl.where(tid == 2, t2, x_ptr)
    if G > 3:
        x_ptr = tl.where(tid == 3, t3, x_ptr)
    if G > 4:
        x_ptr = tl.where(tid == 4, t4, x_ptr)
    if G > 5:
        x_ptr = tl.where(tid == 5, t5, x_ptr)
    if G > 6:
        x_ptr = tl.where(tid == 6, t6, x_ptr)
    if G > 7:
        x_ptr = tl.where(tid == 7, t7, x_ptr)
    offs = ib * BLOCK + tl.arange(0, BLOCK)
    m = offs < n_elem
    x = tl.load(x_ptr + offs, mask=m)
    s2 = s.to(x.dtype)
    if IS_FLOAT:
        r = tl.maximum(x, s2, propagate_nan=tl.PropagateNan.ALL)
    else:
        r = tl.maximum(x, s2)
    tl.store(x_ptr + offs, r, mask=m)


_DT = {
    torch.float64: (True, True),
    torch.float32: (True, False),
    torch.float16: (True, False),
    torch.bfloat16: (True, False),
}
_DEF = (False, False)
_FAST_DT = {torch.float16, torch.bfloat16, torch.float32}

# key -> (run, function, packed_metadata, launch_metadata, gx, gy, gz)
# False marks a key whose fast setup failed (fall back to kernel[grid](...)).
_CK = {}


def _fast_launch(ts, n, s, g, dt, dev_idx):
    key = (dev_idx, dt, n, g, type(s) is float)
    info = _CK.get(key)
    if info is None:
        t0 = ts[0]
        a1 = ts[1] if g > 1 else t0
        a2 = ts[2] if g > 2 else t0
        a3 = ts[3] if g > 3 else t0
        nb = (n + _BLOCK - 1) // _BLOCK
        try:
            ck = _kmax_f4.warmup(
                t0,
                a1,
                a2,
                a3,
                n,
                s,
                g,
                True,
                BLOCK=_BLOCK,
                grid=(nb, g),
                num_warps=_NUM_WARPS,
            )
            if ck is None:
                info = False
            else:
                info = (
                    ck.run,
                    ck.function,
                    ck.packed_metadata,
                    ck.launch_metadata,
                    nb,
                    1,
                    1,
                )
        except Exception:
            info = False
        _CK[key] = info
    if info is False:
        return False
    run_fn, fn, pmeta, lmd, gx, gy, gz = info
    t0 = ts[0]
    args = (
        t0,
        ts[1] if g > 1 else t0,
        ts[2] if g > 2 else t0,
        ts[3] if g > 3 else t0,
        n,
        s,
        g,
        True,
        _BLOCK,
    )
    stream = torch.cuda.current_stream(dev_idx).cuda_stream
    if _knobs is None:
        lm = eh = xh = None
    else:
        rk = _knobs.runtime
        lm = lmd((gx, gy, gz), stream, *args)
        eh = rk.launch_enter_hook
        xh = rk.launch_exit_hook
    run_fn(gx, gy, gz, stream, fn, pmeta, lm, eh, xh, *args)
    return True


def _emit_flat(t, sv, sp, n, is_float, is64):
    grid = ((n + _BLOCK - 1) // _BLOCK,)
    _kmax_c[grid](
        t,
        sv,
        sp,
        n,
        is_float,
        is64,
        n >= 2147483648,
        BLOCK=_BLOCK,
        num_warps=_NUM_WARPS,
    )


def _emit_strided(t, sv, sp, is_float, is64, n):
    shape = t.shape
    strides = t.stride()
    dims = [(sz, st) for sz, st in zip(shape, strides) if sz != 1]
    if not dims:
        _emit_flat(t, sv, sp, n, is_float, is64)
        return
    merged = []
    for sz, st in dims:
        if merged and merged[-1][1] == st * sz:
            merged[-1] = (merged[-1][0] * sz, merged[-1][1])
        else:
            merged.append((sz, st))
    if len(merged) > _MAXD:
        n0 = int(shape[0])
        for i in range(n0):
            sub = t.select(0, i)
            _emit_strided(sub, sv, sp, is_float, is64, sub.numel())
        return
    merged.reverse()  # innermost dimension first
    inner_sz, inner_st = merged[0]
    outer = 1
    for sz, _ in merged[1:]:
        outer *= sz
    inner_blocks = (inner_sz + _BLOCK - 1) // _BLOCK
    sizes = [1] * _MAXD
    strds = [1] * _MAXD
    for k in range(1, len(merged)):
        sizes[k] = int(merged[k][0])
        strds[k] = int(merged[k][1])
    grid = (outer * inner_blocks,)
    _kmax_s[grid](
        t,
        sv,
        sp,
        inner_sz,
        inner_blocks,
        int(inner_st),
        sizes[1],
        sizes[2],
        sizes[3],
        sizes[4],
        strds[1],
        strds[2],
        strds[3],
        strds[4],
        len(merged),
        is_float,
        is64,
        BLOCK=_BLOCK,
        num_warps=_NUM_WARPS,
    )


def _apply_scalar(tensors, scalar):
    n_t = len(tensors)
    if n_t == 0:
        return
    t0 = tensors[0]
    n0 = t0.numel()
    if (
        n0 > 0
        and n_t <= 8
        and t0.is_contiguous()
        and t0.dtype in _FAST_DT
        and type(scalar) in (float, int)
    ):
        dt = t0.dtype
        ok = True
        for i in range(1, n_t):
            ti = tensors[i]
            if ti.dtype is not dt or ti.numel() != n0 or not ti.is_contiguous():
                ok = False
                break
        if ok:
            grid = ((n0 + _BLOCK - 1) // _BLOCK, n_t)
            if n_t <= 4:
                aligned = (t0.data_ptr() & 15) == 0
                if aligned:
                    for i in range(1, n_t):
                        if tensors[i].data_ptr() & 15:
                            aligned = False
                            break
                if aligned and _fast_launch(
                    tensors, n0, scalar, n_t, dt, t0.device.index
                ):
                    return
                if n_t == 1:
                    _kmax_f4[grid](
                        t0,
                        t0,
                        t0,
                        t0,
                        n0,
                        scalar,
                        1,
                        True,
                        BLOCK=_BLOCK,
                        num_warps=_NUM_WARPS,
                    )
                elif n_t == 2:
                    _kmax_f4[grid](
                        t0,
                        tensors[1],
                        t0,
                        t0,
                        n0,
                        scalar,
                        2,
                        True,
                        BLOCK=_BLOCK,
                        num_warps=_NUM_WARPS,
                    )
                elif n_t == 3:
                    _kmax_f4[grid](
                        t0,
                        tensors[1],
                        tensors[2],
                        t0,
                        n0,
                        scalar,
                        3,
                        True,
                        BLOCK=_BLOCK,
                        num_warps=_NUM_WARPS,
                    )
                else:
                    _kmax_f4[grid](
                        t0,
                        tensors[1],
                        tensors[2],
                        tensors[3],
                        n0,
                        scalar,
                        4,
                        True,
                        BLOCK=_BLOCK,
                        num_warps=_NUM_WARPS,
                    )
            else:
                _kmax_f8[grid](
                    t0,
                    tensors[1],
                    tensors[2],
                    tensors[3],
                    tensors[4],
                    tensors[5],
                    tensors[6],
                    tensors[7],
                    n0,
                    scalar,
                    8,
                    True,
                    BLOCK=_BLOCK,
                    num_warps=_NUM_WARPS,
                )
            return
    s64 = None
    for idx in range(n_t):
        t = tensors[idx]
        n = t.numel()
        if n == 0:
            continue
        is_float, is64 = _DT.get(t.dtype, _DEF)
        if is64:
            if s64 is None:
                s64 = torch.tensor(float(scalar), dtype=torch.float64, device=t.device)
            sv = 0.0
            sp = s64
        else:
            sv = scalar
            sp = t
        if t.is_contiguous():
            _emit_flat(t, sv, sp, n, is_float, is64)
        else:
            _emit_strided(t, sv, sp, is_float, is64, n)


def foreach_maximum_scalar_(self, scalar=None, **kwargs):
    if scalar is None and "other" in kwargs:
        scalar = kwargs["other"]
    tensors = self if isinstance(self, (list, tuple)) else [self]
    if isinstance(scalar, (list, tuple)):
        if len(scalar) != len(tensors):
            raise RuntimeError("Tensor list length mismatch")
        for t, o in zip(tensors, scalar):
            if isinstance(o, torch.Tensor):
                o = o.item()
            _apply_scalar([t], o)
        return None
    if isinstance(scalar, torch.Tensor):
        scalar = scalar.item()
    if len(tensors) == 0:
        return None
    _apply_scalar(tensors, scalar)
    return None
