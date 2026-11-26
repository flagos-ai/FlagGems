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
"""Triton implementation of aten::_foreach_sigmoid.

Applies sigmoid elementwise and independently to every tensor of the input
sequence, returning one newly allocated tensor per input element (no aliasing).

Three execution paths:
  * flat path: when the input occupies a dense block of memory with
    non-negative strides (contiguous, transposed, permuted, channels_last), the
    output is allocated with exactly the input strides, so the logical-index ->
    memory-offset bijection is identical for both tensors and a purely flat,
    fully coalesced 1-D kernel over the memory range is correct.
  * strided path: for gapped / overlapping / broadcast inputs the output is
    contiguous (matching ATen's TensorIterator behaviour) and a rank-unrolled
    C-order decomposition (innermost dimension first) gathers the input.
  * complex path: the same two strategies on the real-valued view, using the
    closed form sigmoid(a+bi) = (1+u+iv)/(1+2u+e^-2a), u=e^-a cos b, v=e^-a sin b.

Semantics confirmed against the target reference: floating dtypes keep their
dtype, integral/bool inputs are promoted to the default float dtype, complex
dtypes stay complex, the input container type is preserved, and an empty list
raises the same RuntimeError.
"""

import logging

import torch
import triton
import triton.language as tl
from triton.runtime import driver as _driver

from flag_gems.ops._foreach_unary import _foreach_sigmoid as default__foreach_sigmoid

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)


# Above this element count the runtime is dominated by the kernel rather than the
# launch, so the flat kernel is allowed to autotune block size, warp count and
# streaming cache hints per (dtype, size class) on the first call of that class.
_TUNE_MIN_NUMEL = 1 << 22
_ALIGN_UNIT = 1 << 14

_DT_ID = {
    torch.float16: 0,
    torch.bfloat16: 1,
    torch.float32: 2,
}

_TUNE_CONFIGS = [
    triton.Config({"BLOCK": block, "EVICT": evict, "CESTORE": cstore}, num_warps=warps)
    for block, warps in ((1024, 4), (2048, 4), (2048, 8), (4096, 8), (8192, 8))
    for evict, cstore in ((0, 0), (1, 1))
]


@triton.autotune(configs=_TUNE_CONFIGS, key=["DTID", "BUCKET"])
@triton.jit(do_not_specialize=["n", "EXACT"])
def _sigmoid_flat_tuned_kernel(
    x_ptr,
    o_ptr,
    n,
    EXACT,
    DTID: tl.constexpr,
    BUCKET: tl.constexpr,
    FP64: tl.constexpr,
    INT64: tl.constexpr,
    BLOCK: tl.constexpr,
    EVICT: tl.constexpr,
    CESTORE: tl.constexpr,
):
    pid = tl.program_id(0)
    if INT64:
        offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    else:
        offs = pid * BLOCK + tl.arange(0, BLOCK)
    if EXACT != 0:
        offs = tl.max_contiguous(tl.multiple_of(offs, BLOCK), BLOCK)
        if EVICT != 0:
            x = tl.load(x_ptr + offs, eviction_policy="evict_first")
        else:
            x = tl.load(x_ptr + offs)
        if FP64:
            y = 1.0 / (1.0 + tl.exp(-x))
        else:
            y = 1.0 / (1.0 + tl.exp(-x.to(tl.float32)))
        if CESTORE == 1:
            tl.store(o_ptr + offs, y.to(o_ptr.dtype.element_ty), cache_modifier=".cs")
        elif CESTORE == 2:
            tl.store(o_ptr + offs, y.to(o_ptr.dtype.element_ty), cache_modifier=".wt")
        else:
            tl.store(o_ptr + offs, y.to(o_ptr.dtype.element_ty))
    else:
        mask = offs < n
        x = tl.load(x_ptr + offs, mask=mask, other=0)
        if FP64:
            y = 1.0 / (1.0 + tl.exp(-x))
        else:
            y = 1.0 / (1.0 + tl.exp(-x.to(tl.float32)))
        tl.store(o_ptr + offs, y.to(o_ptr.dtype.element_ty), mask=mask)


@triton.jit(do_not_specialize=["n"])
def _sigmoid_flat_kernel(
    x_ptr,
    o_ptr,
    n,
    BLOCK: tl.constexpr,
    FP64: tl.constexpr,
    INT64: tl.constexpr,
    EVEN: tl.constexpr,
):
    pid = tl.program_id(0)
    if EVEN:
        # numel is an exact multiple of BLOCK: no mask, so the access is
        # statically contiguous and Triton emits one wide vector per thread.
        if INT64:
            offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
        else:
            offs = pid * BLOCK + tl.arange(0, BLOCK)
        offs = tl.max_contiguous(tl.multiple_of(offs, BLOCK), BLOCK)
        x = tl.load(x_ptr + offs)
        if FP64:
            y = 1.0 / (1.0 + tl.exp(-x))
        else:
            y = 1.0 / (1.0 + tl.exp(-x.to(tl.float32)))
        tl.store(o_ptr + offs, y.to(o_ptr.dtype.element_ty))
    else:
        if INT64:
            offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
            mask = offs < n.to(tl.int64)
        else:
            offs = pid * BLOCK + tl.arange(0, BLOCK)
            mask = offs < n
        x = tl.load(x_ptr + offs, mask=mask, other=0)
        if FP64:
            y = 1.0 / (1.0 + tl.exp(-x))
        else:
            y = 1.0 / (1.0 + tl.exp(-x.to(tl.float32)))
        tl.store(o_ptr + offs, y.to(o_ptr.dtype.element_ty), mask=mask)


@triton.jit(
    do_not_specialize=[
        "n",
        "n0",
        "n1",
        "n2",
        "n3",
        "n4",
        "n5",
        "n6",
        "n7",
        "s0",
        "s1",
        "s2",
        "s3",
        "s4",
        "s5",
        "s6",
        "s7",
    ]
)
def _sigmoid_strided_kernel(
    x_ptr,
    o_ptr,
    n,
    n0,
    n1,
    n2,
    n3,
    n4,
    n5,
    n6,
    n7,
    s0,
    s1,
    s2,
    s3,
    s4,
    s5,
    s6,
    s7,
    NDIM: tl.constexpr,
    BLOCK: tl.constexpr,
    FP64: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = offs < n.to(tl.int64)
    idx = offs
    off = tl.zeros((BLOCK,), dtype=tl.int64)
    # n0/s0 is the innermost (fastest varying) dimension.
    if NDIM >= 1:
        off += (idx % n0) * s0
        idx = idx // n0
    if NDIM >= 2:
        off += (idx % n1) * s1
        idx = idx // n1
    if NDIM >= 3:
        off += (idx % n2) * s2
        idx = idx // n2
    if NDIM >= 4:
        off += (idx % n3) * s3
        idx = idx // n3
    if NDIM >= 5:
        off += (idx % n4) * s4
        idx = idx // n4
    if NDIM >= 6:
        off += (idx % n5) * s5
        idx = idx // n5
    if NDIM >= 7:
        off += (idx % n6) * s6
        idx = idx // n6
    if NDIM >= 8:
        off += (idx % n7) * s7
        idx = idx // n7
    x = tl.load(x_ptr + off, mask=mask, other=0)
    if FP64:
        y = 1.0 / (1.0 + tl.exp(-x))
    else:
        y = 1.0 / (1.0 + tl.exp(-x.to(tl.float32)))
    tl.store(o_ptr + offs, y.to(o_ptr.dtype.element_ty), mask=mask)


@triton.jit(do_not_specialize=["n"])
def _sigmoid_complex_flat_kernel(
    xr_ptr,
    or_ptr,
    n,
    BLOCK: tl.constexpr,
    FP64: tl.constexpr,
    INT64: tl.constexpr,
):
    pid = tl.program_id(0)
    if INT64:
        i = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
        mask = i < n.to(tl.int64)
    else:
        i = pid * BLOCK + tl.arange(0, BLOCK)
        mask = i < n
    p = i + i
    a = tl.load(xr_ptr + p, mask=mask, other=0)
    b = tl.load(xr_ptr + p + 1, mask=mask, other=0)
    if not FP64:
        a = a.to(tl.float32)
        b = b.to(tl.float32)
    ea = tl.exp(-a)
    u = ea * tl.cos(b)
    v = ea * tl.sin(b)
    d = 1.0 + 2.0 * u + ea * ea
    re = (1.0 + u) / d
    im = v / d
    tl.store(or_ptr + p, re.to(or_ptr.dtype.element_ty), mask=mask)
    tl.store(or_ptr + p + 1, im.to(or_ptr.dtype.element_ty), mask=mask)


@triton.jit(
    do_not_specialize=[
        "n",
        "n0",
        "n1",
        "n2",
        "n3",
        "n4",
        "n5",
        "n6",
        "n7",
        "s0",
        "s1",
        "s2",
        "s3",
        "s4",
        "s5",
        "s6",
        "s7",
    ]
)
def _sigmoid_complex_strided_kernel(
    xr_ptr,
    or_ptr,
    n,
    n0,
    n1,
    n2,
    n3,
    n4,
    n5,
    n6,
    n7,
    s0,
    s1,
    s2,
    s3,
    s4,
    s5,
    s6,
    s7,
    NDIM: tl.constexpr,
    BLOCK: tl.constexpr,
    FP64: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = offs < n.to(tl.int64)
    idx = offs
    off = tl.zeros((BLOCK,), dtype=tl.int64)
    if NDIM >= 1:
        off += (idx % n0) * s0
        idx = idx // n0
    if NDIM >= 2:
        off += (idx % n1) * s1
        idx = idx // n1
    if NDIM >= 3:
        off += (idx % n2) * s2
        idx = idx // n2
    if NDIM >= 4:
        off += (idx % n3) * s3
        idx = idx // n3
    if NDIM >= 5:
        off += (idx % n4) * s4
        idx = idx // n4
    if NDIM >= 6:
        off += (idx % n5) * s5
        idx = idx // n5
    if NDIM >= 7:
        off += (idx % n6) * s6
        idx = idx // n6
    if NDIM >= 8:
        off += (idx % n7) * s7
        idx = idx // n7
    p = off + off
    a = tl.load(xr_ptr + p, mask=mask, other=0)
    b = tl.load(xr_ptr + p + 1, mask=mask, other=0)
    if not FP64:
        a = a.to(tl.float32)
        b = b.to(tl.float32)
    ea = tl.exp(-a)
    u = ea * tl.cos(b)
    v = ea * tl.sin(b)
    d = 1.0 + 2.0 * u + ea * ea
    re = (1.0 + u) / d
    im = v / d
    q = offs + offs
    tl.store(or_ptr + q, re.to(or_ptr.dtype.element_ty), mask=mask)
    tl.store(or_ptr + q + 1, im.to(or_ptr.dtype.element_ty), mask=mask)


@triton.jit(do_not_specialize=["n"])
def _sigmoid_multi_kernel(
    p0,
    p1,
    p2,
    p3,
    p4,
    p5,
    p6,
    p7,
    p8,
    p9,
    p10,
    p11,
    p12,
    p13,
    p14,
    p15,
    q0,
    q1,
    q2,
    q3,
    q4,
    q5,
    q6,
    q7,
    q8,
    q9,
    q10,
    q11,
    q12,
    q13,
    q14,
    q15,
    n,
    BLOCK: tl.constexpr,
    FP64: tl.constexpr,
    EVEN: tl.constexpr,
):
    """One launch for a group of same-dtype, same-numel, memory-dense tensors.

    program_id(0) selects the tensor of the group, program_id(1) the block.
    Unused slots receive the first tensor's pointers and are never selected
    because the grid's first dimension equals the live group size.
    """
    t = tl.program_id(0)
    blk = tl.program_id(1)
    xp = p0
    op = q0
    if t == 1:
        xp = p1
        op = q1
    if t == 2:
        xp = p2
        op = q2
    if t == 3:
        xp = p3
        op = q3
    if t == 4:
        xp = p4
        op = q4
    if t == 5:
        xp = p5
        op = q5
    if t == 6:
        xp = p6
        op = q6
    if t == 7:
        xp = p7
        op = q7
    if t == 8:
        xp = p8
        op = q8
    if t == 9:
        xp = p9
        op = q9
    if t == 10:
        xp = p10
        op = q10
    if t == 11:
        xp = p11
        op = q11
    if t == 12:
        xp = p12
        op = q12
    if t == 13:
        xp = p13
        op = q13
    if t == 14:
        xp = p14
        op = q14
    if t == 15:
        xp = p15
        op = q15
    if EVEN:
        offs = blk * BLOCK + tl.arange(0, BLOCK)
        offs = tl.max_contiguous(tl.multiple_of(offs, BLOCK), BLOCK)
        x = tl.load(xp + offs)
        if FP64:
            y = 1.0 / (1.0 + tl.exp(-x))
        else:
            y = 1.0 / (1.0 + tl.exp(-x.to(tl.float32)))
        tl.store(op + offs, y.to(op.dtype.element_ty))
    else:
        offs = blk * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        x = tl.load(xp + offs, mask=mask, other=0)
        if FP64:
            y = 1.0 / (1.0 + tl.exp(-x))
        else:
            y = 1.0 / (1.0 + tl.exp(-x.to(tl.float32)))
        tl.store(op + offs, y.to(op.dtype.element_ty), mask=mask)


def _block_for(n):
    if n <= 4096:
        return 256
    if n <= 65536:
        return 512
    if n <= 1048576:
        return 1024
    return 2048


def _is_dense(shape, strides):
    """True when the tensor covers a dense block of memory (strides >= 0).

    Equivalent to ATen's is_non_overlapping_and_dense for non-negative strides:
    when the dimensions with size > 1 are ordered by increasing stride, the
    stride sequence must be 1, d0, d0*d1, ... (i.e. a dense permutation).
    """
    dims = [(s, d) for d, s in zip(shape, strides) if d > 1]
    if not dims:
        return True
    used = [False] * len(dims)
    expect = 1
    for _ in dims:
        hit = -1
        for i, (s, _d) in enumerate(dims):
            if not used[i] and s == expect:
                hit = i
                break
        if hit < 0:
            return False
        used[hit] = True
        expect *= dims[hit][1]
    return True


def _run_complex(t, numel):
    """sigmoid(a+bi) = (1+u+iv) / (1+2u+e^-2a), u=e^-a cos b, v=e^-a sin b."""
    shape = t.shape
    strides = t.stride()
    fp64 = t.dtype == torch.complex128
    xr = torch.view_as_real(t)
    block = _block_for(numel)
    warps = 8 if block >= 2048 else 4
    grid = ((numel + block - 1) // block,)
    if _is_dense(shape, strides):
        out = torch.empty_strided(shape, strides, dtype=t.dtype, device=t.device)
        if out.stride() == strides:
            orr = torch.view_as_real(out)
            _sigmoid_complex_flat_kernel[grid](
                xr,
                orr,
                numel,
                BLOCK=block,
                FP64=fp64,
                INT64=numel * 2 > 0x7FFFFFFF,
                num_warps=warps,
            )
            return out
    out = torch.empty(shape, dtype=t.dtype, device=t.device)
    orr = torch.view_as_real(out)
    ndim = t.dim()
    pad_shape = [int(v) for v in shape[::-1]] + [1] * (8 - ndim)
    pad_strd = [int(s) for s in strides[::-1]] + [0] * (8 - ndim)
    _sigmoid_complex_strided_kernel[grid](
        xr,
        orr,
        numel,
        *pad_shape,
        *pad_strd,
        NDIM=ndim,
        BLOCK=block,
        FP64=fp64,
        num_warps=warps,
    )
    return out


def _run_one(t):
    numel = t.numel()
    dtype = t.dtype
    if dtype.is_floating_point or dtype.is_complex:
        out_dtype = dtype
    else:
        out_dtype = torch.get_default_dtype()

    if numel == 0:
        if t.is_contiguous():
            return torch.empty(t.shape, dtype=out_dtype, device=t.device)
        return torch.empty_like(t, dtype=out_dtype, memory_format=torch.preserve_format)

    if dtype.is_complex:
        return _run_complex(t, numel)

    shape = t.shape
    strides = t.stride()
    if _is_dense(shape, strides):
        out = torch.empty_strided(shape, strides, dtype=out_dtype, device=t.device)
        use_flat = out.stride() == strides
    else:
        out = torch.empty(shape, dtype=out_dtype, device=t.device)
        use_flat = False

    block = _block_for(numel)
    warps = 8 if block >= 2048 else 4
    grid = ((numel + block - 1) // block,)
    fp64 = dtype == torch.float64
    int64 = numel > 0x7FFFFFFF

    if use_flat:
        if numel > _TUNE_MIN_NUMEL:
            _sigmoid_flat_tuned_kernel[
                lambda meta: ((numel + meta["BLOCK"] - 1) // meta["BLOCK"],)
            ](
                t,
                out,
                numel,
                1 if numel % _ALIGN_UNIT == 0 else 0,
                DTID=_DT_ID.get(dtype, 3),
                BUCKET=1 if numel <= (1 << 24) else 2,
                FP64=fp64,
                INT64=int64,
            )
        else:
            _sigmoid_flat_kernel[grid](
                t,
                out,
                numel,
                BLOCK=block,
                FP64=fp64,
                INT64=int64,
                EVEN=(numel % block == 0),
                num_warps=warps,
            )
    else:
        ndim = t.dim()
        # The kernels decompose the flat C-order index starting from the
        # innermost (fastest varying) dimension, so shapes/strides are reversed.
        pad_shape = [int(v) for v in shape[::-1]] + [1] * (8 - ndim)
        pad_strd = [int(s) for s in strides[::-1]] + [0] * (8 - ndim)
        _sigmoid_strided_kernel[grid](
            t,
            out,
            numel,
            *pad_shape,
            *pad_strd,
            NDIM=ndim,
            BLOCK=block,
            FP64=fp64,
            num_warps=warps,
        )
    return out


# Grouping several tensors into one launch pays off only while the per-tensor
# GPU work is comparable to a launch (~15 us on this target), i.e. roughly up to
# a couple of million elements. Above that the grouped kernel is measurably less
# bandwidth-efficient than one launch per tensor, so large tensors stay separate.
_FUSE_MAX_NUMEL = 1 << 21


def _flat_ok(t):
    """Fusable by the grouped launch: small, non-empty, memory-dense, non-complex."""
    numel = t.numel()
    if numel == 0 or numel > _FUSE_MAX_NUMEL or t.dtype.is_complex:
        return False
    return _is_dense(t.shape, t.stride())


def _run_group(group, numel):
    """One launch for up to 16 same-dtype/same-numel dense tensors."""
    dtype = group[0].dtype
    if dtype.is_floating_point:
        out_dtype = dtype
    else:
        out_dtype = torch.get_default_dtype()
    outs = []
    for u in group:
        o = torch.empty_strided(u.shape, u.stride(), dtype=out_dtype, device=u.device)
        if o.stride() != u.stride():
            return None
        outs.append(o)
    block = _block_for(numel)
    warps = 8 if block >= 2048 else 4
    pad = 16 - len(group)
    ins = list(group) + [group[0]] * pad
    ous = outs + [outs[0]] * pad
    even = numel % block == 0
    fp64 = dtype == torch.float64
    grid = (len(group), (numel + block - 1) // block)
    key = (
        dtype,
        block,
        even,
        tuple(t.data_ptr() & 15 for t in ins),
        tuple(t.data_ptr() & 15 for t in ous),
    )
    _dispatch_multi(grid, ins + ous + [numel], block, fp64, even, warps, key)
    return outs


_FAST = {}


def _dispatch_multi(grid, args, block, fp64, even, warps, key):
    """Launch the grouped kernel, replaying the compiled variant once it exists.

    Triton's Python-level dispatch costs more than this tiny grouped launch on
    the target, so a cached ``CompiledKernel`` is replayed directly, as the
    other MThreads foreach kernels do.  The key carries everything the compile
    specialisation depends on (dtype, BLOCK/EVEN and pointer alignment); a stale
    handle falls back to the ordinary launch path.
    """
    entry = _FAST.get(key)
    if entry is not None:
        compiled, stream = entry
        try:
            compiled.run(
                grid[0],
                grid[1],
                1,
                stream,
                compiled.function,
                compiled.packed_metadata,
                None,
                None,
                None,
                *args,
                block,
                fp64,
                even,
            )
            return
        except Exception:
            _FAST.pop(key, None)

    compiled = _sigmoid_multi_kernel[grid](
        *args, BLOCK=block, FP64=fp64, EVEN=even, num_warps=warps
    )
    try:
        drv = _driver.active
        _FAST[key] = (compiled, drv.get_current_stream(drv.get_current_device()))
    except Exception:
        _FAST.pop(key, None)


def _foreach_sigmoid(self=None, *args, **kwargs):
    logger.debug("GEMS_MTHREADS _FOREACH_SIGMOID")
    _seq = self if isinstance(self, (list, tuple)) else [self]
    if any(not isinstance(t, torch.Tensor) or t.device.type != "musa" for t in _seq):
        return default__foreach_sigmoid(self)
    seq = self
    if seq is None:
        if args:
            seq = args[0]
        else:
            for key in ("self", "tensors", "tensor_list", "input", "tensors_list"):
                if key in kwargs:
                    seq = kwargs[key]
                    break
    if not isinstance(seq, (list, tuple)):
        seq = [seq]
    n = len(seq)
    if n == 0:
        raise RuntimeError("Tensor list must have at least one tensor.")
    res = [None] * n
    i = 0
    while i < n:
        t = seq[i]
        if not _flat_ok(t):
            res[i] = _run_one(t)
            i += 1
            continue
        dtype = t.dtype
        numel = t.numel()
        j = i + 1
        while j < n and (j - i) < 16:
            u = seq[j]
            if u.dtype != dtype or u.numel() != numel or not _flat_ok(u):
                break
            j += 1
        if j - i > 1:
            outs = _run_group(seq[i:j], numel)
            if outs is not None:
                res[i:j] = outs
                i = j
                continue
        res[i] = _run_one(t)
        i += 1
    if isinstance(seq, tuple):
        return tuple(res)
    return res
