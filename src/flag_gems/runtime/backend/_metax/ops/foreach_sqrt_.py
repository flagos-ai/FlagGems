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

"""Triton implementation of ``aten::_foreach_sqrt_`` (in-place foreach sqrt).

Applies ``sqrt`` to every tensor of a TensorList, writing the result back into
each tensor's own storage (storage pointer, shape, strides, dtype and device are
preserved) and returning ``None``.

Design
------
* One Triton launch covers a whole call whenever possible: the flat kernel takes
  up to 16 (pointer, count) slots and derives the tensor of a program from
  cumulative block boundaries, so a 16-tensor list costs one launch instead of
  sixteen (a launch is ~15 us on this target, the kernels are microseconds).
* Non-overlapping-and-dense tensors - contiguous, transposed-dense, or any
  permutation-dense view - are traversed with flat indices, because sqrt is
  elementwise and the element set is unchanged by the permutation.
* Gappy views (``x[:, ::2]``) are addressed by explicit stride decomposition;
  the mutation itself is the observable output, so no materialising copy and no
  ``copy_`` write-back is used.
* fp16/bf16 are computed with fp32 opmath and rounded back, matching ATen's
  float opmath for half types.

Semantics reproduced from ATen: floating dtypes are accepted, integral and bool
dtypes are rejected (their promoted float result cannot be cast back in place),
complex dtypes are handled through the interleaved real/imag layout, tensors
whose elements alias one location are rejected, and the call returns ``None``.
"""

import torch
import triton
import triton.language as tl

try:  # vendor launch wrapper used by JITFunction.run on this build
    from triton.runtime.jit import DistributedRtContext as _DistributedRtContext
except Exception:  # pragma: no cover - defensive
    _DistributedRtContext = None

_MAX_T = 16
_BLOCK = 512
_EVEN_BLOCK = 2048
_NUM_WARPS = 4
_ND = 8
# Fusion trades one extra argument per tensor (~1.2 us) against one launch per
# tensor (~15 us).  It only pays while the fused GPU work is smaller than the
# launch cost, i.e. for small groups; above this element budget the per-launch
# host cost is hidden behind real kernel time and per-tensor launches keep the
# kernel small (16 live pointer parameters cost occupancy on big tensors).
_FUSE_MAX_ELEMS = 1 << 20
# The mask-free vectorized kernel keeps a *tensor* pointer argument: Triton
# carries the 16-byte divisibility of a pointer argument into the addresses it
# derives, and an int scalar cast to a pointer loses it (measured 11-13% slower
# for fp16/bf16).  The masked fused kernel cannot vectorize anyway, so there the
# address is passed as an int64 scalar, which the launcher converts much more
# cheaply than a tensor object (~1.3 us vs ~1.6 us per argument).
_DUMMY_PTR = 1 << 40
_PTR_I64_MIN = 1 << 31
_DT = {
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
    torch.float32: tl.float32,
    torch.float64: tl.float64,
}


# ---------------------------------------------------------------------------
# element-wise math
# ---------------------------------------------------------------------------


@triton.jit
def _sqrt_val(x):
    """sqrt in the input dtype; half types are computed in fp32 opmath."""
    if x.dtype == tl.float16:
        return tl.sqrt(x.to(tl.float32)).to(x.dtype)
    else:
        if x.dtype == tl.bfloat16:
            return tl.sqrt(x.to(tl.float32)).to(x.dtype)
        else:
            return tl.sqrt(x)


@triton.jit
def _hypot_val(a, b):
    """Stable sqrt(a*a + b*b)."""
    aa = tl.abs(a)
    bb = tl.abs(b)
    mx = tl.maximum(aa, bb)
    mn = tl.minimum(aa, bb)
    r = mn / mx
    out = mx * tl.sqrt(1.0 + r * r)
    return tl.where(mx == 0, mx, out)


@triton.jit
def _csqrt_pair(re, im):
    """sqrt of a complex value from its real/imag components."""
    r = _hypot_val(re, im)
    o_re = tl.sqrt(0.5 * (r + re))
    o_im = tl.sqrt(tl.maximum(0.5 * (r - re), 0.0))
    o_im = tl.where(im < 0, -o_im, o_im)
    return o_re, o_im


# ---------------------------------------------------------------------------
# kernels
# ---------------------------------------------------------------------------


@triton.jit
def _sqrt_block(p, base, n, BLOCK: tl.constexpr):
    i = base + tl.arange(0, BLOCK)
    m = i < n
    tl.store(p + i, _sqrt_val(tl.load(p + i, mask=m, other=0)), mask=m)


@triton.jit
def _sqrt_flat1_kernel(p, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    _sqrt_block(p, pid * BLOCK, n, BLOCK)


@triton.jit
def _sqrt_even_kernel(p, BLOCK: tl.constexpr):
    """Whole-block, unmasked variant: no runtime mask, so each thread's
    contiguous slice lowers to full-width vector loads/stores (16 B) instead of
    one element per memory instruction.  Used only when numel % BLOCK == 0."""
    pid = tl.program_id(0)
    i = pid * BLOCK + tl.arange(0, BLOCK)
    i = tl.max_contiguous(tl.multiple_of(i, BLOCK), BLOCK)
    tl.store(p + i, _sqrt_val(tl.load(p + i)))


@triton.jit
def _sqrt_meta_kernel(meta, N: tl.constexpr, BLOCK: tl.constexpr, DT: tl.constexpr):
    """Fused variant for a uniform group whose address table lives on device.

    ``meta`` holds the tensor addresses as int64, one per tensor, in group
    order.  With a uniform element count each tensor owns the same number of
    blocks ``b``, so a program's tensor is simply ``pid // b`` and its chunk is
    ``pid % b`` - no branch chain and a single argument to the launcher.
    """
    pid = tl.program_id(0)
    b = (N + BLOCK - 1) // BLOCK
    p = tl.load(meta + pid // b).to(tl.pointer_type(DT))
    _sqrt_block(p, (pid % b) * BLOCK, N, BLOCK)


@triton.jit
def _sqrt_flat_uniform_kernel(
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
    N: tl.constexpr,
    NT: tl.constexpr,
    BLOCK: tl.constexpr,
    DT: tl.constexpr,
):
    """Fused variant for a group whose tensors all have the same element count.

    ``N`` being a constexpr removes one kernel argument per tensor and makes the
    per-tensor block count a compile-time constant; the addresses arrive as
    int64 scalars, which the launcher converts more cheaply than tensor objects.
    """
    pid = tl.program_id(0)
    b = (N + BLOCK - 1) // BLOCK
    if pid < b:
        _sqrt_block(q0.to(tl.pointer_type(DT)), pid * BLOCK, N, BLOCK)
    if NT > 1:
        if (pid >= b) & (pid < 2 * b):
            _sqrt_block(q1.to(tl.pointer_type(DT)), (pid - b) * BLOCK, N, BLOCK)
    if NT > 2:
        if (pid >= 2 * b) & (pid < 3 * b):
            _sqrt_block(q2.to(tl.pointer_type(DT)), (pid - 2 * b) * BLOCK, N, BLOCK)
    if NT > 3:
        if (pid >= 3 * b) & (pid < 4 * b):
            _sqrt_block(q3.to(tl.pointer_type(DT)), (pid - 3 * b) * BLOCK, N, BLOCK)
    if NT > 4:
        if (pid >= 4 * b) & (pid < 5 * b):
            _sqrt_block(q4.to(tl.pointer_type(DT)), (pid - 4 * b) * BLOCK, N, BLOCK)
    if NT > 5:
        if (pid >= 5 * b) & (pid < 6 * b):
            _sqrt_block(q5.to(tl.pointer_type(DT)), (pid - 5 * b) * BLOCK, N, BLOCK)
    if NT > 6:
        if (pid >= 6 * b) & (pid < 7 * b):
            _sqrt_block(q6.to(tl.pointer_type(DT)), (pid - 6 * b) * BLOCK, N, BLOCK)
    if NT > 7:
        if (pid >= 7 * b) & (pid < 8 * b):
            _sqrt_block(q7.to(tl.pointer_type(DT)), (pid - 7 * b) * BLOCK, N, BLOCK)
    if NT > 8:
        if (pid >= 8 * b) & (pid < 9 * b):
            _sqrt_block(q8.to(tl.pointer_type(DT)), (pid - 8 * b) * BLOCK, N, BLOCK)
    if NT > 9:
        if (pid >= 9 * b) & (pid < 10 * b):
            _sqrt_block(q9.to(tl.pointer_type(DT)), (pid - 9 * b) * BLOCK, N, BLOCK)
    if NT > 10:
        if (pid >= 10 * b) & (pid < 11 * b):
            _sqrt_block(q10.to(tl.pointer_type(DT)), (pid - 10 * b) * BLOCK, N, BLOCK)
    if NT > 11:
        if (pid >= 11 * b) & (pid < 12 * b):
            _sqrt_block(q11.to(tl.pointer_type(DT)), (pid - 11 * b) * BLOCK, N, BLOCK)
    if NT > 12:
        if (pid >= 12 * b) & (pid < 13 * b):
            _sqrt_block(q12.to(tl.pointer_type(DT)), (pid - 12 * b) * BLOCK, N, BLOCK)
    if NT > 13:
        if (pid >= 13 * b) & (pid < 14 * b):
            _sqrt_block(q13.to(tl.pointer_type(DT)), (pid - 13 * b) * BLOCK, N, BLOCK)
    if NT > 14:
        if (pid >= 14 * b) & (pid < 15 * b):
            _sqrt_block(q14.to(tl.pointer_type(DT)), (pid - 14 * b) * BLOCK, N, BLOCK)
    if NT > 15:
        if (pid >= 15 * b) & (pid < 16 * b):
            _sqrt_block(q15.to(tl.pointer_type(DT)), (pid - 15 * b) * BLOCK, N, BLOCK)


@triton.jit
def _sqrt_flat_kernel(
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
    n0,
    n1,
    n2,
    n3,
    n4,
    n5,
    n6,
    n7,
    n8,
    n9,
    n10,
    n11,
    n12,
    n13,
    n14,
    n15,
    NT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    b0 = tl.cdiv(n0, BLOCK)
    b1 = b0 + tl.cdiv(n1, BLOCK)
    b2 = b1 + tl.cdiv(n2, BLOCK)
    b3 = b2 + tl.cdiv(n3, BLOCK)
    b4 = b3 + tl.cdiv(n4, BLOCK)
    b5 = b4 + tl.cdiv(n5, BLOCK)
    b6 = b5 + tl.cdiv(n6, BLOCK)
    b7 = b6 + tl.cdiv(n7, BLOCK)
    b8 = b7 + tl.cdiv(n8, BLOCK)
    b9 = b8 + tl.cdiv(n9, BLOCK)
    b10 = b9 + tl.cdiv(n10, BLOCK)
    b11 = b10 + tl.cdiv(n11, BLOCK)
    b12 = b11 + tl.cdiv(n12, BLOCK)
    b13 = b12 + tl.cdiv(n13, BLOCK)
    b14 = b13 + tl.cdiv(n14, BLOCK)
    b15 = b14 + tl.cdiv(n15, BLOCK)
    if pid < b0:
        _sqrt_block(p0, pid * BLOCK, n0, BLOCK)
    if NT > 1:
        if (pid >= b0) & (pid < b1):
            _sqrt_block(p1, (pid - b0) * BLOCK, n1, BLOCK)
    if NT > 2:
        if (pid >= b1) & (pid < b2):
            _sqrt_block(p2, (pid - b1) * BLOCK, n2, BLOCK)
    if NT > 3:
        if (pid >= b2) & (pid < b3):
            _sqrt_block(p3, (pid - b2) * BLOCK, n3, BLOCK)
    if NT > 4:
        if (pid >= b3) & (pid < b4):
            _sqrt_block(p4, (pid - b3) * BLOCK, n4, BLOCK)
    if NT > 5:
        if (pid >= b4) & (pid < b5):
            _sqrt_block(p5, (pid - b4) * BLOCK, n5, BLOCK)
    if NT > 6:
        if (pid >= b5) & (pid < b6):
            _sqrt_block(p6, (pid - b5) * BLOCK, n6, BLOCK)
    if NT > 7:
        if (pid >= b6) & (pid < b7):
            _sqrt_block(p7, (pid - b6) * BLOCK, n7, BLOCK)
    if NT > 8:
        if (pid >= b7) & (pid < b8):
            _sqrt_block(p8, (pid - b7) * BLOCK, n8, BLOCK)
    if NT > 9:
        if (pid >= b8) & (pid < b9):
            _sqrt_block(p9, (pid - b8) * BLOCK, n9, BLOCK)
    if NT > 10:
        if (pid >= b9) & (pid < b10):
            _sqrt_block(p10, (pid - b9) * BLOCK, n10, BLOCK)
    if NT > 11:
        if (pid >= b10) & (pid < b11):
            _sqrt_block(p11, (pid - b10) * BLOCK, n11, BLOCK)
    if NT > 12:
        if (pid >= b11) & (pid < b12):
            _sqrt_block(p12, (pid - b11) * BLOCK, n12, BLOCK)
    if NT > 13:
        if (pid >= b12) & (pid < b13):
            _sqrt_block(p13, (pid - b12) * BLOCK, n13, BLOCK)
    if NT > 14:
        if (pid >= b13) & (pid < b14):
            _sqrt_block(p14, (pid - b13) * BLOCK, n14, BLOCK)
    if NT > 15:
        if (pid >= b14) & (pid < b15):
            _sqrt_block(p15, (pid - b14) * BLOCK, n15, BLOCK)


@triton.jit
def _csqrt_flat_kernel(p, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    idx = pid * BLOCK + tl.arange(0, BLOCK)
    m = idx < n
    j = idx.to(tl.int64) * 2
    re = tl.load(p + j, mask=m, other=0.0)
    im = tl.load(p + j + 1, mask=m, other=0.0)
    o_re, o_im = _csqrt_pair(re, im)
    tl.store(p + j, o_re, mask=m)
    tl.store(p + j + 1, o_im, mask=m)


@triton.jit
def _decompose(
    idx,
    d0,
    d1,
    d2,
    d3,
    d4,
    d5,
    d6,
    d7,
    s0,
    s1,
    s2,
    s3,
    s4,
    s5,
    s6,
    s7,
    ND: tl.constexpr,
):
    """Logical flat index -> storage offset for a collapsed (innermost-first) view."""
    rem = idx
    off = idx.to(tl.int64) * 0
    i = rem % d0
    rem = rem // d0
    off = off + i.to(tl.int64) * s0
    if ND > 1:
        i = rem % d1
        rem = rem // d1
        off = off + i.to(tl.int64) * s1
        if ND > 2:
            i = rem % d2
            rem = rem // d2
            off = off + i.to(tl.int64) * s2
            if ND > 3:
                i = rem % d3
                rem = rem // d3
                off = off + i.to(tl.int64) * s3
                if ND > 4:
                    i = rem % d4
                    rem = rem // d4
                    off = off + i.to(tl.int64) * s4
                    if ND > 5:
                        i = rem % d5
                        rem = rem // d5
                        off = off + i.to(tl.int64) * s5
                        if ND > 6:
                            i = rem % d6
                            rem = rem // d6
                            off = off + i.to(tl.int64) * s6
                            if ND > 7:
                                i = rem % d7
                                rem = rem // d7
                                off = off + i.to(tl.int64) * s7
    return off


@triton.jit
def _sqrt_strided_kernel(
    p,
    n,
    d0,
    d1,
    d2,
    d3,
    d4,
    d5,
    d6,
    d7,
    s0,
    s1,
    s2,
    s3,
    s4,
    s5,
    s6,
    s7,
    ND: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    idx = pid * BLOCK + tl.arange(0, BLOCK)
    m = idx < n
    off = _decompose(
        idx, d0, d1, d2, d3, d4, d5, d6, d7, s0, s1, s2, s3, s4, s5, s6, s7, ND
    )
    x = tl.load(p + off, mask=m, other=0)
    tl.store(p + off, _sqrt_val(x), mask=m)


@triton.jit
def _csqrt_strided_kernel(
    p,
    n,
    d0,
    d1,
    d2,
    d3,
    d4,
    d5,
    d6,
    d7,
    s0,
    s1,
    s2,
    s3,
    s4,
    s5,
    s6,
    s7,
    ND: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    idx = pid * BLOCK + tl.arange(0, BLOCK)
    m = idx < n
    off = _decompose(
        idx, d0, d1, d2, d3, d4, d5, d6, d7, s0, s1, s2, s3, s4, s5, s6, s7, ND
    )
    j = off * 2
    re = tl.load(p + j, mask=m, other=0.0)
    im = tl.load(p + j + 1, mask=m, other=0.0)
    o_re, o_im = _csqrt_pair(re, im)
    tl.store(p + j, o_re, mask=m)
    tl.store(p + j + 1, o_im, mask=m)


# ---------------------------------------------------------------------------
# host-side layout classification
# ---------------------------------------------------------------------------


def _is_flat(t):
    """True when [data_ptr, data_ptr + numel) covers exactly this tensor.

    Such a tensor (contiguous, transposed-dense, sliced-but-dense, ...) can be
    traversed with flat indices: the element *set* is unchanged even when the
    memory order is a permutation, and sqrt is elementwise.
    """
    if t.is_contiguous():
        return True
    if t.numel() <= 1:
        return True
    expected = 1
    for s, sz in sorted(zip(t.stride(), t.shape)):
        if sz == 1:
            continue
        if s != expected:
            return False
        expected *= sz
    return True


def _collapse(t):
    """Drop size-1 dims and merge adjacent dims; return innermost-first lists."""
    sh = []
    st = []
    for sz, s in zip(t.shape, t.stride()):
        if sz != 1:
            sh.append(sz)
            st.append(s)
    if not sh:
        return [1], [0]
    osh = [sh[-1]]
    ost = [st[-1]]
    for k in range(len(sh) - 2, -1, -1):
        if st[k] == ost[-1] * osh[-1]:
            osh[-1] *= sh[k]
        else:
            osh.append(sh[k])
            ost.append(st[k])
    return osh, ost


# ---------------------------------------------------------------------------
# launches
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# cached compiled-kernel launch
#
# JITFunction.run re-binds the arguments, recomputes the specialization key,
# looks the kernel up in a cache and walks the vendor pre/post-launch machinery
# on every call, which costs ~15 us - more than the kernel itself for the small
# workloads.  The compiled kernel for a fixed signature can be obtained once and
# launched directly; this is only done when it is provably equivalent:
#   * the vendor lite/distributed mode is off (JITFunction.run would then inject
#     two extra launch parameters that a direct launch would omit),
#   * every pointer is 16-byte aligned, matching the alignment the warmup
#     specialization assumed, and
#   * no int runtime argument is left in the signature (their value-dependent
#     specializations would make a cached binary wrong).
# Any failure falls back to the ordinary kernel[grid](...).
# ---------------------------------------------------------------------------

_FAST_CACHE = {}
_FAST_STATE = []


def _fast_ok():
    if not _FAST_STATE:
        ok = False
        try:
            if _DistributedRtContext is not None:
                ok = not _DistributedRtContext().is_lite_mode
        except Exception:
            ok = False
        _FAST_STATE.append(ok)
        return ok
    return _FAST_STATE[0]


def _even_ck(dtype, dev):
    key = (0, dtype, dev.index)
    ck = _FAST_CACHE.get(key)
    if ck is None:
        ck = False
        if _fast_ok():
            try:
                proto = torch.empty(1, dtype=dtype, device=dev)
                ck = _sqrt_even_kernel.warmup(proto, _EVEN_BLOCK, grid=(1,))
            except Exception:
                ck = False
        _FAST_CACHE[key] = ck
    return ck


def _uniform_ck(dtype, dev, n, k):
    key = (1, dtype, dev.index, n, k)
    ck = _FAST_CACHE.get(key)
    if ck is None:
        ck = False
        if _fast_ok():
            try:
                ck = _sqrt_flat_uniform_kernel.warmup(
                    *([_DUMMY_PTR] * _MAX_T), n, k, _BLOCK, _DT[dtype], grid=(1,)
                )
            except Exception:
                ck = False
        _FAST_CACHE[key] = ck
    return ck


def _launch_flat_one(t, n):
    if n % _EVEN_BLOCK == 0:
        blocks = n // _EVEN_BLOCK
        if not t.data_ptr() & 15:
            ck = _even_ck(t.dtype, t.device)
            if ck:
                ck[(blocks, 1, 1)](t, _EVEN_BLOCK)
                return
        _sqrt_even_kernel[(blocks,)](t, _EVEN_BLOCK, num_warps=_NUM_WARPS)
    else:
        _sqrt_flat1_kernel[((n + _BLOCK - 1) // _BLOCK,)](
            t, n, _BLOCK, num_warps=_NUM_WARPS
        )


_META_CACHE = {}


def _meta_buf(ptrs, n, k, dt, dev):
    """Device int64 address table, re-uploaded whenever anything differs.

    Only the upload is memoised; the current addresses and counts are always
    recomputed and compared, so a stale table can never be used.
    """
    key = (dt, dev.index)
    ent = _META_CACHE.get(key)
    if ent is not None and ent[0] == ptrs and ent[1] == n:
        return ent[2]
    buf = torch.tensor(ptrs, dtype=torch.int64, device=dev)
    _META_CACHE[key] = (list(ptrs), n, buf)
    return buf


def _meta_ck(dtype, dev, n):
    key = (2, dtype, dev.index, n)
    ck = _FAST_CACHE.get(key)
    if ck is None:
        ck = False
        if _fast_ok():
            try:
                proto = torch.empty(1, dtype=torch.int64, device=dev)
                ck = _sqrt_meta_kernel.warmup(proto, n, _BLOCK, _DT[dtype], grid=(1,))
            except Exception:
                ck = False
        _FAST_CACHE[key] = ck
    return ck


def _launch_fused(items):
    """One launch for up to _MAX_T flat-addressable tensors of one dtype.

    ``items`` holds ``(tensor, numel)`` pairs so counts are not recomputed.
    """
    cnt = len(items)
    for start in range(0, cnt, _MAX_T):
        chunk = items[start : start + _MAX_T]
        k = len(chunk)
        n = chunk[0][1]
        uniform = True
        for _, v in chunk:
            if v != n:
                uniform = False
                break
        if uniform:
            ptrs = [t.data_ptr() for t, _ in chunk]
            blocks = k * ((n + _BLOCK - 1) // _BLOCK)
            dt = chunk[0][0].dtype
            dev = chunk[0][0].device
            buf = None
            try:
                buf = _meta_buf(ptrs, n, k, dt, dev)
            except Exception:
                buf = None
            if buf is not None and not buf.data_ptr() & 15:
                ck = _meta_ck(dt, dev, n)
                if ck:
                    ck[(blocks, 1, 1)](buf, n, _BLOCK, _DT[dt])
                else:
                    _sqrt_meta_kernel[(blocks,)](
                        buf, n, _BLOCK, _DT[dt], num_warps=_NUM_WARPS
                    )
            else:
                fast = True
                for p in ptrs:
                    if p < _PTR_I64_MIN or p & 15:
                        fast = False
                        break
                if k < _MAX_T:
                    ptrs.extend([ptrs[0]] * (_MAX_T - k))
                ck = False
                if fast:
                    ck = _uniform_ck(dt, dev, n, k)
                if ck:
                    ck[(blocks, 1, 1)](*ptrs, n, k, _BLOCK, _DT[dt])
                else:
                    _sqrt_flat_uniform_kernel[(blocks,)](
                        *ptrs, n, k, _BLOCK, _DT[dt], num_warps=_NUM_WARPS
                    )
        else:
            tensors = [t for t, _ in chunk]
            ns = [v for _, v in chunk]
            pad = _MAX_T - k
            if pad:
                tensors.extend([tensors[0]] * pad)
                ns.extend([0] * pad)
            blocks = 0
            for v in ns[:k]:
                blocks += (v + _BLOCK - 1) // _BLOCK
            _sqrt_flat_kernel[(blocks,)](*tensors, *ns, k, _BLOCK, num_warps=_NUM_WARPS)


def _launch_cflat(t, n):
    rv = torch.view_as_real(t)
    _csqrt_flat_kernel[((n + _BLOCK - 1) // _BLOCK,)](
        rv, n, _BLOCK, num_warps=_NUM_WARPS
    )


def _launch_strided(t):
    n = t.numel()
    dims, strides = _collapse(t)
    if len(dims) > _ND:
        for i in range(t.shape[0]):
            _launch_strided(t.select(0, i))
        return
    nd = len(dims)
    dims = dims + [1] * (_ND - nd)
    strides = strides + [0] * (_ND - nd)
    grid = ((n + _BLOCK - 1) // _BLOCK,)
    if t.dtype.is_complex:
        _csqrt_strided_kernel[grid](
            torch.view_as_real(t),
            n,
            *dims,
            *strides,
            nd,
            _BLOCK,
            num_warps=_NUM_WARPS,
        )
    else:
        _sqrt_strided_kernel[grid](
            t,
            n,
            *dims,
            *strides,
            nd,
            _BLOCK,
            num_warps=_NUM_WARPS,
        )


def _dispatch(items):
    real_flat = []
    rest = []
    for t, flat, n in items:
        if flat and not t.dtype.is_complex:
            real_flat.append((t, n))
        else:
            rest.append((t, flat, n))
    if real_flat:
        if len(real_flat) == 1:
            t, n = real_flat[0]
            _launch_flat_one(t, n)
        else:
            total = 0
            for _, n in real_flat:
                total += n
            if total <= _FUSE_MAX_ELEMS:
                _launch_fused(real_flat)
            else:
                for t, n in real_flat:
                    _launch_flat_one(t, n)
    for t, flat, n in rest:
        if flat:
            _launch_cflat(t, n)
        else:
            _launch_strided(t)


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def _reject_dtype(dt):
    raise RuntimeError(
        "result type Float can't be cast to the desired output type "
        + str(dt).replace("torch.", "").capitalize()
    )


def _overlap_error():
    return RuntimeError(
        "unsupported operation: more than one element of the written-to "
        "tensor refers to a single memory location. Please clone() the "
        "tensor before performing the operation."
    )


_CUR_DEV = -1


def _switch_needed(index):
    """Cheap device check: only touch the runtime when the device changes."""
    global _CUR_DEV
    if index == _CUR_DEV:
        return False
    cur = torch.cuda.current_device()
    _CUR_DEV = cur
    return cur != index


def foreach_sqrt_(self):
    tensors = self
    if not isinstance(tensors, (list, tuple)):
        raise TypeError("argument 'self' must be tuple of Tensors")
    count = len(tensors)
    if count == 0:
        raise RuntimeError("Tensor list must have at least one tensor.")
    first = tensors[0]
    if not isinstance(first, torch.Tensor):
        raise TypeError("argument 'self' must be tuple of Tensors")

    if count == 1:
        dt = first.dtype
        if not dt.is_complex and not dt.is_floating_point:
            _reject_dtype(dt)
        n = first.numel()
        if n:
            flat = _is_flat(first)
            if not flat and torch._debug_has_internal_overlap(first) == 1:
                raise _overlap_error()
            if _switch_needed(first.device.index):
                with torch.cuda.device(first.device):
                    if flat:
                        if dt.is_complex:
                            _launch_cflat(first, n)
                        else:
                            _launch_flat_one(first, n)
                    else:
                        _launch_strided(first)
            else:
                if flat:
                    if dt.is_complex:
                        _launch_cflat(first, n)
                    else:
                        _launch_flat_one(first, n)
                else:
                    _launch_strided(first)
        return None

    groups = {}
    for t in tensors:
        if not isinstance(t, torch.Tensor):
            raise TypeError("argument 'self' must be tuple of Tensors")
        dt = t.dtype
        if not dt.is_complex and not dt.is_floating_point:
            _reject_dtype(dt)
        n = t.numel()
        if n == 0:
            continue
        flat = _is_flat(t)
        if not flat:
            if torch._debug_has_internal_overlap(t) == 1:
                raise _overlap_error()
        groups.setdefault((t.device, dt), []).append((t, flat, n))

    if not groups:
        return None

    dev = next(iter(groups))[0]
    if _switch_needed(dev.index):
        with torch.cuda.device(dev):
            for items in groups.values():
                _dispatch(items)
    else:
        for items in groups.values():
            _dispatch(items)
    return None
