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

"""Triton implementation of aten::_foreach_div_.Scalar (in-place scalar division).

All tensors of the list are divided by the same scalar in place and None is
returned.  Host work is a single metadata pass (no copies, no allocation): a
storage-span overlap check, then either one fused multi-tensor launch (small
lists, where the per-call launch cost dominates) or one pipelined launch per
tensor (larger tensors, where enqueue cost hides behind device work).

Launches reuse a cached CompiledKernel launcher keyed by the full specialization
identity (element dtype, pointers, sizes, scalar, scalar bits, device), so the
selected kernel is exactly the one a JIT launch would pick while the per-call
argument binding, specialization and cache-key computation are skipped.

Kernels: storage-dense (contiguous or permuted-dense) uses offset = k,
uniform-step views use offset = k*step, and the remaining views use a merged
up-to-8-dim mixed-radix decomposition.  fp16/bf16 divide in fp32 (opmath) and
cast back; fp32 uses true division; fp64 divides by the exact double bit
pattern.  No output is allocated and no input is copied.
"""

import logging
import struct

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

_NUM_WARPS = 4
_SLOTS = 4
_FUSE_BYTES = 8 << 20
_INT32_MAX = 2147483647
_FLOATS = (torch.float16, torch.bfloat16, torch.float32, torch.float64)
_MAX_RUNNERS = 256
# tag 0 = fused multi-tensor kernel, tag 1 = per-tensor dense kernel.
_RUNNERS = {}
# 32 bytes of element data per thread with 4 warps = two 128-bit accesses in
# flight per thread, which is the measured optimum for every element width.
_ITEMSIZE = {torch.float16: 2, torch.bfloat16: 2, torch.float32: 4, torch.float64: 8}
_BLK = {
    torch.float16: 2048,
    torch.bfloat16: 2048,
    torch.float32: 1024,
    torch.float64: 512,
}


def _sbits_of(s):
    return struct.unpack("<q", struct.pack("<d", s))[0]


@triton.jit
def _div_slot(
    ptr,
    blk,
    n,
    sval,
    BLOCK: tl.constexpr,
    EVEN: tl.constexpr,
    F64: tl.constexpr,
    I64: tl.constexpr,
):
    if I64:
        offs = blk.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    else:
        offs = blk * BLOCK + tl.arange(0, BLOCK)
    if EVEN:
        x = tl.load(ptr + offs, cache_modifier=".cg")
        if F64:
            y = x / sval
        else:
            y = x.to(tl.float32) / sval
        tl.store(ptr + offs, y.to(x.dtype), cache_modifier=".cs")
    else:
        mask = offs < n
        x = tl.load(ptr + offs, mask=mask, other=0.0, cache_modifier=".cg")
        if F64:
            y = x / sval
        else:
            y = x.to(tl.float32) / sval
        tl.store(ptr + offs, y.to(x.dtype), mask=mask, cache_modifier=".cs")


@triton.jit
def _div_multi4(
    p0,
    p1,
    p2,
    p3,
    n0,
    n1,
    n2,
    n3,
    s,
    sbits,
    BLOCK: tl.constexpr,
    EVEN: tl.constexpr,
    F64: tl.constexpr,
    I64: tl.constexpr,
):
    pid = tl.program_id(0)
    if F64:
        sval = sbits.to(tl.float64, bitcast=True)
    else:
        sval = s
    c0 = (n0 + (BLOCK - 1)) // BLOCK
    if pid < c0:
        _div_slot(p0, pid, n0, sval, BLOCK, EVEN, F64, I64)
    else:
        c1 = c0 + (n1 + (BLOCK - 1)) // BLOCK
        if pid < c1:
            _div_slot(p1, pid - c0, n1, sval, BLOCK, EVEN, F64, I64)
        else:
            c2 = c1 + (n2 + (BLOCK - 1)) // BLOCK
            if pid < c2:
                _div_slot(p2, pid - c1, n2, sval, BLOCK, EVEN, F64, I64)
            else:
                _div_slot(p3, pid - c2, n3, sval, BLOCK, EVEN, F64, I64)


@triton.jit
def _div_dense_kernel(
    ptr,
    n,
    s,
    sbits,
    BLOCK: tl.constexpr,
    EVEN: tl.constexpr,
    F64: tl.constexpr,
    I64: tl.constexpr,
):
    if F64:
        sval = sbits.to(tl.float64, bitcast=True)
    else:
        sval = s
    _div_slot(ptr, tl.program_id(0), n, sval, BLOCK, EVEN, F64, I64)


@triton.jit
def _div_step_kernel(
    ptr,
    n,
    s,
    sbits,
    step,
    BLOCK: tl.constexpr,
    EVEN: tl.constexpr,
    F64: tl.constexpr,
):
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    addr = offs * step.to(tl.int64)
    if F64:
        sval = sbits.to(tl.float64, bitcast=True)
    else:
        sval = s
    if EVEN:
        x = tl.load(ptr + addr)
        if F64:
            y = x / sval
        else:
            y = x.to(tl.float32) / sval
        tl.store(ptr + addr, y.to(x.dtype))
    else:
        mask = offs < n
        x = tl.load(ptr + addr, mask=mask, other=0.0)
        if F64:
            y = x / sval
        else:
            y = x.to(tl.float32) / sval
        tl.store(ptr + addr, y.to(x.dtype), mask=mask)


@triton.jit
def _div_nd_kernel(
    ptr,
    n,
    s,
    sbits,
    sh0,
    sh1,
    sh2,
    sh3,
    sh4,
    sh5,
    sh6,
    sh7,
    st0,
    st1,
    st2,
    st3,
    st4,
    st5,
    st6,
    st7,
    ND: tl.constexpr,
    BLOCK: tl.constexpr,
    F64: tl.constexpr,
):
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    rem = offs
    addr = (rem % sh0) * st0
    rem = rem // sh0
    if ND > 1:
        addr += (rem % sh1) * st1
        rem = rem // sh1
    if ND > 2:
        addr += (rem % sh2) * st2
        rem = rem // sh2
    if ND > 3:
        addr += (rem % sh3) * st3
        rem = rem // sh3
    if ND > 4:
        addr += (rem % sh4) * st4
        rem = rem // sh4
    if ND > 5:
        addr += (rem % sh5) * st5
        rem = rem // sh5
    if ND > 6:
        addr += (rem % sh6) * st6
        rem = rem // sh6
    if ND > 7:
        addr += (rem % sh7) * st7
    if F64:
        sval = sbits.to(tl.float64, bitcast=True)
    else:
        sval = s
    mask = offs < n
    x = tl.load(ptr + addr, mask=mask, other=0.0)
    if F64:
        y = x / sval
    else:
        y = x.to(tl.float32) / sval
    tl.store(ptr + addr, y.to(x.dtype), mask=mask)


def _plan(t):
    """Return (mode, step, dims) for a non-contiguous tensor; metadata only."""
    dims = [(sz, st) for sz, st in zip(t.shape, t.stride()) if sz > 1]
    if not dims:
        return 0, 1, None
    dims = sorted(dims, key=lambda d: d[1])
    c = dims[0][1]
    expected = c
    ok = True
    for sz, st in dims:
        if st != expected:
            ok = False
            break
        expected *= sz
    if ok:
        return (0, 1, None) if c == 1 else (1, c, None)
    merged = []
    for sz, st in zip(t.shape, t.stride()):
        if sz == 1:
            continue
        if merged and merged[-1][1] == st * sz:
            psz, pst = merged[-1]
            merged[-1] = (psz * sz, pst)
        else:
            merged.append((sz, st))
    merged = sorted(merged, key=lambda d: d[1])
    return 2, 0, merged


def _scalar_to_float(scalar):
    if isinstance(scalar, torch.Tensor):
        if scalar.numel() != 1:
            raise RuntimeError("_foreach_div_ expects a Number scalar")
        return float(scalar.reshape(()).item())
    return float(scalar)


def _stream():
    return torch.cuda.current_stream().cuda_stream


def _launch_fused(ts, ptrs, nums, s, sbits, dt, f64, dev):
    blk = _BLK[dt]
    total = 0
    even = True
    big = False
    for n in nums:
        total += (n + blk - 1) // blk
        if n % blk:
            even = False
        if n > _INT32_MAX:
            big = True
    key = (0, dt, tuple(ptrs), tuple(nums), s, sbits, f64, dev)
    ent = _RUNNERS.get(key)
    if ent is not None:
        ent[0](
            ent[1],
            1,
            1,
            _stream(),
            ent[2],
            ent[3],
            None,
            None,
            None,
            *ts,
            *nums,
            s,
            sbits,
            blk,
            even,
            f64,
            big,
        )
        return
    ck = _div_multi4[(total,)](
        *ts,
        *nums,
        s,
        sbits,
        BLOCK=blk,
        EVEN=even,
        F64=f64,
        I64=big,
        num_warps=_NUM_WARPS,
    )
    if len(_RUNNERS) >= _MAX_RUNNERS:
        _RUNNERS.clear()
    _RUNNERS[key] = (ck.run, total, ck.function, ck.packed_metadata)


def _launch_dense(t, numel, s, dev):
    dt = t.dtype
    f64 = dt is torch.float64
    sbits = _sbits_of(s) if f64 else 0
    blk = _BLK[dt]
    even = (numel % blk) == 0
    i64 = numel > _INT32_MAX
    key = (1, dt, t.data_ptr(), numel, s, sbits, f64, dev)
    ent = _RUNNERS.get(key)
    if ent is not None:
        ent[0](
            ent[1],
            1,
            1,
            _stream(),
            ent[2],
            ent[3],
            None,
            None,
            None,
            t,
            numel,
            s,
            sbits,
            blk,
            even,
            f64,
            i64,
        )
        return
    grid = (numel + blk - 1) // blk
    ck = _div_dense_kernel[(grid,)](
        t,
        numel,
        s,
        sbits,
        BLOCK=blk,
        EVEN=even,
        F64=f64,
        I64=i64,
        num_warps=_NUM_WARPS,
    )
    if len(_RUNNERS) >= _MAX_RUNNERS:
        _RUNNERS.clear()
    _RUNNERS[key] = (ck.run, grid, ck.function, ck.packed_metadata)


def _launch_strided(t, mode, step, dims, s):
    if t.dim() > 8:
        # Rare: reduce rank with zero-copy views until the kernels apply.
        for i in range(t.shape[0]):
            sub = t.narrow(0, i, 1).squeeze(0)
            smode, sstep, sdim = _plan(sub)
            _launch_strided(sub, smode, sstep, sdim, s)
        return
    numel = t.numel()
    dt = t.dtype
    f64 = dt is torch.float64
    sbits = _sbits_of(s) if f64 else 0
    blk = _BLK[dt]
    grid = ((numel + blk - 1) // blk,)
    if mode == 1:
        _div_step_kernel[grid](
            t,
            numel,
            s,
            sbits,
            step,
            BLOCK=blk,
            EVEN=(numel % blk == 0),
            F64=f64,
            num_warps=_NUM_WARPS,
        )
    else:
        shapes = [1] * 8
        strides = [0] * 8
        for i, (sz, st) in enumerate(dims):
            shapes[i] = sz
            strides[i] = st
        _div_nd_kernel[grid](
            t,
            numel,
            s,
            sbits,
            shapes[0],
            shapes[1],
            shapes[2],
            shapes[3],
            shapes[4],
            shapes[5],
            shapes[6],
            shapes[7],
            strides[0],
            strides[1],
            strides[2],
            strides[3],
            strides[4],
            strides[5],
            strides[6],
            strides[7],
            ND=min(len(dims), 8),
            BLOCK=blk,
            F64=f64,
            num_warps=_NUM_WARPS,
        )


def _dispatch(t, numel, contig, s, dev):
    if contig:
        _launch_dense(t, numel, s, dev)
    else:
        mode, step, dims = _plan(t)
        if mode == 0:
            _launch_dense(t, numel, s, dev)
        else:
            _launch_strided(t, mode, step, dims, s)


def _foreach_div__Scalar(self=None, scalar=None, **kwargs):
    logger.debug("GEMS_HYGON _FOREACH_DIV__SCALAR")
    if scalar is None:
        for key in ("scalar", "other", "value"):
            if key in kwargs:
                scalar = kwargs[key]
                break
    if isinstance(self, torch.Tensor):
        self = (self,)
    if not self:
        return None
    s = _scalar_to_float(scalar)

    tensors = []
    nums = []
    ptrs = []
    devs = []
    contigs = []
    spans = []
    dense = True
    dtype0 = None
    mixed_dtype = False
    dev0 = None
    mixed_dev = False
    total_bytes = 0
    for t in self:
        n = t.numel()
        if n == 0:
            continue
        dt = t.dtype
        if dtype0 is None:
            dtype0 = dt
        elif dt is not dtype0:
            mixed_dtype = True
        es = _ITEMSIZE.get(dt)
        if es is None:
            raise RuntimeError(
                "result type Float can't be cast to the desired output type " + str(dt)
            )
        dev = t.get_device()
        if dev0 is None:
            dev0 = dev
        elif dev != dev0:
            mixed_dev = True
        tensors.append(t)
        nums.append(n)
        devs.append(dev)
        c = t.is_contiguous()
        contigs.append(c)
        if c:
            dp = t.data_ptr()
            ptrs.append(dp)
            nb = n * es
            total_bytes += nb
            spans.append((dp, dp + nb))
        else:
            dense = False
            ptrs.append(0)
            st = t.untyped_storage()
            dp = st.data_ptr()
            spans.append((dp, dp + st.nbytes()))

    k = len(tensors)
    if k == 0:
        return None

    overlap = False
    if k > 1:
        prev = -1
        for start, end in sorted(spans):
            if start < prev:
                overlap = True
                break
            prev = end

    f64 = dtype0 is torch.float64
    if (
        not overlap
        and dense
        and not mixed_dtype
        and not mixed_dev
        and total_bytes <= _FUSE_BYTES
    ):
        sbits = _sbits_of(s) if f64 else 0
        pad = _SLOTS
        for i in range(0, k, pad):
            chunk = tensors[i : i + pad]
            cn = nums[i : i + pad]
            cp = ptrs[i : i + pad]
            m = len(chunk)
            if m < pad:
                chunk = chunk + [chunk[0]] * (pad - m)
                cp = cp + [cp[0]] * (pad - m)
                cn = cn + [0] * (pad - m)
            _launch_fused(chunk, cp, cn, s, sbits, dtype0, f64, dev0)
        return None

    cur = None
    if dev0 is not None:
        cur = torch.cuda.current_device()
    for j in range(k):
        t = tensors[j]
        dev = devs[j]
        if dev is None or dev == cur:
            _dispatch(t, nums[j], contigs[j], s, dev)
        else:
            with torch.cuda.device(dev):
                _dispatch(t, nums[j], contigs[j], s, dev)
    return None
