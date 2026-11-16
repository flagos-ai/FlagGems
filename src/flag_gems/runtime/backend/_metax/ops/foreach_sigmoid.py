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

_MAX_DIMS = 8
_MAX_TENSORS = 16
_BLOCK = 1024
_BLOCK_16 = 4096
_BLOCK_16_BIG = 8192
_LARGE_ELEMS = 1 << 18
_NUM_WARPS = 4
_WIDE = 1 << 31
_BIG = 1 << 30
_FUSE_MAX_TILES = 1 << 15


@triton.jit
def _sig_tile(
    in_ptr,
    out_ptr,
    n_elem,
    pid,
    BLOCK: tl.constexpr,
    FP32_MATH: tl.constexpr,
    FAST_EXP: tl.constexpr,
    EVEN: tl.constexpr,
):
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    if EVEN:
        x = tl.load(in_ptr + offs)
    else:
        mask = offs < n_elem
        x = tl.load(in_ptr + offs, mask=mask, other=0)
    if FP32_MATH:
        xf = x.to(tl.float32)
        if FAST_EXP:
            y = 1.0 / (1.0 + tl.exp2(-xf * 1.4426950408889634))
        else:
            y = 1.0 / (1.0 + tl.exp(-xf))
    else:
        y = 1.0 / (1.0 + tl.exp(-x))
    if EVEN:
        tl.store(out_ptr + offs, y)
    else:
        tl.store(out_ptr + offs, y, mask=mask)


@triton.jit(do_not_specialize=["n_elem"])
def _sigmoid_flat_kernel(
    in_ptr,
    out_ptr,
    n_elem,
    BLOCK: tl.constexpr,
    FP32_MATH: tl.constexpr,
    FAST_EXP: tl.constexpr,
    EVEN: tl.constexpr,
):
    _sig_tile(
        in_ptr, out_ptr, n_elem, tl.program_id(0), BLOCK, FP32_MATH, FAST_EXP, EVEN
    )


@triton.jit(
    do_not_specialize=[
        "n0",
        "n1",
        "n2",
        "n3",
        "n4",
        "n5",
        "n6",
        "n7",
        "n8",
        "n9",
        "n10",
        "n11",
        "n12",
        "n13",
        "n14",
        "n15",
        "p1",
        "p2",
        "p3",
        "p4",
        "p5",
        "p6",
        "p7",
        "p8",
        "p9",
        "p10",
        "p11",
        "p12",
        "p13",
        "p14",
        "p15",
    ]
)
def _sigmoid_multi_kernel(
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
    BLOCK: tl.constexpr,
    FP32_MATH: tl.constexpr,
    FAST_EXP: tl.constexpr,
    EVEN: tl.constexpr,
):
    pid = tl.program_id(0)
    tid = (
        (pid >= p1).to(tl.int32)
        + (pid >= p2).to(tl.int32)
        + (pid >= p3).to(tl.int32)
        + (pid >= p4).to(tl.int32)
        + (pid >= p5).to(tl.int32)
        + (pid >= p6).to(tl.int32)
        + (pid >= p7).to(tl.int32)
        + (pid >= p8).to(tl.int32)
        + (pid >= p9).to(tl.int32)
        + (pid >= p10).to(tl.int32)
        + (pid >= p11).to(tl.int32)
        + (pid >= p12).to(tl.int32)
        + (pid >= p13).to(tl.int32)
        + (pid >= p14).to(tl.int32)
        + (pid >= p15).to(tl.int32)
    )
    if tid == 0:
        _sig_tile(i0, o0, n0, pid, BLOCK, FP32_MATH, FAST_EXP, EVEN)
    elif tid == 1:
        _sig_tile(i1, o1, n1, pid - p1, BLOCK, FP32_MATH, FAST_EXP, EVEN)
    elif tid == 2:
        _sig_tile(i2, o2, n2, pid - p2, BLOCK, FP32_MATH, FAST_EXP, EVEN)
    elif tid == 3:
        _sig_tile(i3, o3, n3, pid - p3, BLOCK, FP32_MATH, FAST_EXP, EVEN)
    elif tid == 4:
        _sig_tile(i4, o4, n4, pid - p4, BLOCK, FP32_MATH, FAST_EXP, EVEN)
    elif tid == 5:
        _sig_tile(i5, o5, n5, pid - p5, BLOCK, FP32_MATH, FAST_EXP, EVEN)
    elif tid == 6:
        _sig_tile(i6, o6, n6, pid - p6, BLOCK, FP32_MATH, FAST_EXP, EVEN)
    elif tid == 7:
        _sig_tile(i7, o7, n7, pid - p7, BLOCK, FP32_MATH, FAST_EXP, EVEN)
    elif tid == 8:
        _sig_tile(i8, o8, n8, pid - p8, BLOCK, FP32_MATH, FAST_EXP, EVEN)
    elif tid == 9:
        _sig_tile(i9, o9, n9, pid - p9, BLOCK, FP32_MATH, FAST_EXP, EVEN)
    elif tid == 10:
        _sig_tile(i10, o10, n10, pid - p10, BLOCK, FP32_MATH, FAST_EXP, EVEN)
    elif tid == 11:
        _sig_tile(i11, o11, n11, pid - p11, BLOCK, FP32_MATH, FAST_EXP, EVEN)
    elif tid == 12:
        _sig_tile(i12, o12, n12, pid - p12, BLOCK, FP32_MATH, FAST_EXP, EVEN)
    elif tid == 13:
        _sig_tile(i13, o13, n13, pid - p13, BLOCK, FP32_MATH, FAST_EXP, EVEN)
    elif tid == 14:
        _sig_tile(i14, o14, n14, pid - p14, BLOCK, FP32_MATH, FAST_EXP, EVEN)
    else:
        _sig_tile(i15, o15, n15, pid - p15, BLOCK, FP32_MATH, FAST_EXP, EVEN)


@triton.jit(do_not_specialize=["n_elem"])
def _sigmoid_cflat_kernel(
    in_ptr,
    out_ptr,
    n_elem,
    BLOCK: tl.constexpr,
    FP32_MATH: tl.constexpr,
    EVEN: tl.constexpr,
):
    # complex tensors are addressed in units of real scalars (2 per element)
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    two = offs * 2
    if EVEN:
        re = tl.load(in_ptr + two)
        im = tl.load(in_ptr + two + 1)
    else:
        mask = offs < n_elem
        re = tl.load(in_ptr + two, mask=mask, other=0)
        im = tl.load(in_ptr + two + 1, mask=mask, other=0)
    if FP32_MATH:
        re = re.to(tl.float32)
        im = im.to(tl.float32)
    u = tl.exp(-re)
    p = 1.0 + u * tl.cos(im)
    q = -u * tl.sin(im)
    d = p * p + q * q
    yre = p / d
    yim = -q / d
    if EVEN:
        tl.store(out_ptr + two, yre)
        tl.store(out_ptr + two + 1, yim)
    else:
        tl.store(out_ptr + two, yre, mask=mask)
        tl.store(out_ptr + two + 1, yim, mask=mask)


@triton.jit(
    do_not_specialize=[
        "n_elem",
        "d0",
        "d1",
        "d2",
        "d3",
        "d4",
        "d5",
        "d6",
        "d7",
        "s0",
        "s1",
        "s2",
        "s3",
        "s4",
        "s5",
        "s6",
        "s7",
        "o0",
        "o1",
        "o2",
        "o3",
        "o4",
        "o5",
        "o6",
        "o7",
    ]
)
def _sigmoid_strided_kernel(
    in_ptr,
    out_ptr,
    n_elem,
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
    o0,
    o1,
    o2,
    o3,
    o4,
    o5,
    o6,
    o7,
    NDIM: tl.constexpr,
    BLOCK: tl.constexpr,
    FP32_MATH: tl.constexpr,
    FAST_EXP: tl.constexpr,
    OUT_CONTIG: tl.constexpr,
    COMPLEX: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = (pid * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)
    idx = offs
    ioff = tl.zeros((BLOCK,), dtype=tl.int64)
    ooff = tl.zeros((BLOCK,), dtype=tl.int64)
    if NDIM > 0:
        ioff += (idx % d0) * s0
        if not OUT_CONTIG:
            ooff += (idx % d0) * o0
        idx = idx // d0
    if NDIM > 1:
        ioff += (idx % d1) * s1
        if not OUT_CONTIG:
            ooff += (idx % d1) * o1
        idx = idx // d1
    if NDIM > 2:
        ioff += (idx % d2) * s2
        if not OUT_CONTIG:
            ooff += (idx % d2) * o2
        idx = idx // d2
    if NDIM > 3:
        ioff += (idx % d3) * s3
        if not OUT_CONTIG:
            ooff += (idx % d3) * o3
        idx = idx // d3
    if NDIM > 4:
        ioff += (idx % d4) * s4
        if not OUT_CONTIG:
            ooff += (idx % d4) * o4
        idx = idx // d4
    if NDIM > 5:
        ioff += (idx % d5) * s5
        if not OUT_CONTIG:
            ooff += (idx % d5) * o5
        idx = idx // d5
    if NDIM > 6:
        ioff += (idx % d6) * s6
        if not OUT_CONTIG:
            ooff += (idx % d6) * o6
        idx = idx // d6
    if NDIM > 7:
        ioff += (idx % d7) * s7
        if not OUT_CONTIG:
            ooff += (idx % d7) * o7
        idx = idx // d7

    mask = offs < n_elem
    if COMPLEX:
        if OUT_CONTIG:
            ooff = offs
        re = tl.load(in_ptr + ioff * 2, mask=mask, other=0)
        im = tl.load(in_ptr + ioff * 2 + 1, mask=mask, other=0)
        if FP32_MATH:
            re = re.to(tl.float32)
            im = im.to(tl.float32)
        u = tl.exp(-re)
        p = 1.0 + u * tl.cos(im)
        q = -u * tl.sin(im)
        d = p * p + q * q
        yre = p / d
        yim = -q / d
        tl.store(out_ptr + ooff * 2, yre, mask=mask)
        tl.store(out_ptr + ooff * 2 + 1, yim, mask=mask)
    else:
        x = tl.load(in_ptr + ioff, mask=mask, other=0)
        if FP32_MATH:
            xf = x.to(tl.float32)
            if FAST_EXP:
                y = 1.0 / (1.0 + tl.exp2(-xf * 1.4426950408889634))
            else:
                y = 1.0 / (1.0 + tl.exp(-xf))
        else:
            y = 1.0 / (1.0 + tl.exp(-x))
        if OUT_CONTIG:
            tl.store(out_ptr + offs, y, mask=mask)
        else:
            tl.store(out_ptr + ooff, y, mask=mask)


def _collapse(shape, strides):
    """Merge adjacent dimensions that are contiguous in memory (outer->inner)."""
    merged = []
    for sz, st in zip(shape, strides):
        if sz == 1:
            continue
        if merged and merged[-1][1] == sz * st:
            psz, _ = merged[-1]
            merged[-1] = (psz * sz, st)
        else:
            merged.append((sz, st))
    return merged


def _plain(shape, strides):
    return [(sz, st) for sz, st in zip(shape, strides) if sz != 1]


def _is_dense(shape, strides):
    """True when the numel elements occupy a contiguous span (permutation-dense)."""
    order = sorted(range(len(shape)), key=lambda i: strides[i])
    expected = 1
    for i in order:
        sz = shape[i]
        if sz == 1:
            continue
        if strides[i] != expected:
            return False
        expected *= sz
    return True


def _run_strided(
    src, out, dims, n, fp32_math, fast_exp, out_contig, odims, is_complex, block
):
    nd = len(dims)
    if nd > _MAX_DIMS:
        raise RuntimeError("foreach_sigmoid: tensor rank too large: %d" % nd)
    dims = dims[::-1]  # innermost dimension first for cheap div/mod
    dv = [1] * _MAX_DIMS
    sv = [0] * _MAX_DIMS
    for k, (sz, st) in enumerate(dims):
        dv[k] = int(sz)
        sv[k] = int(st)
    ov = [0] * _MAX_DIMS
    if not out_contig:
        for k, (sz, st) in enumerate(odims[::-1]):
            ov[k] = int(st)
    grid = ((n + block - 1) // block,)
    _sigmoid_strided_kernel[grid](
        src,
        out,
        n,
        dv[0],
        dv[1],
        dv[2],
        dv[3],
        dv[4],
        dv[5],
        dv[6],
        dv[7],
        sv[0],
        sv[1],
        sv[2],
        sv[3],
        sv[4],
        sv[5],
        sv[6],
        sv[7],
        ov[0],
        ov[1],
        ov[2],
        ov[3],
        ov[4],
        ov[5],
        ov[6],
        ov[7],
        NDIM=nd,
        BLOCK=block,
        FP32_MATH=fp32_math,
        FAST_EXP=fast_exp,
        OUT_CONTIG=out_contig,
        COMPLEX=is_complex,
        num_warps=_NUM_WARPS,
    )


def _launch_single(src, dst, n, fp32_math, fast_exp, is_complex, block):
    if is_complex:
        grid = ((n + block - 1) // block,)
        _sigmoid_cflat_kernel[grid](
            src,
            dst,
            n,
            BLOCK=block,
            FP32_MATH=fp32_math,
            EVEN=(n % block == 0),
            num_warps=_NUM_WARPS,
        )
        return
    if n < _WIDE:
        grid = ((n + block - 1) // block,)
        _sigmoid_flat_kernel[grid](
            src,
            dst,
            n,
            BLOCK=block,
            FP32_MATH=fp32_math,
            FAST_EXP=fast_exp,
            EVEN=(n % block == 0),
            num_warps=_NUM_WARPS,
        )
    else:
        _run_strided(
            src, dst, [(n, 1)], n, fp32_math, fast_exp, True, None, False, block
        )


def _launch_multi(jobs, fp32_math, fast_exp, block):
    """Fuse up to _MAX_TENSORS contiguous 1-D jobs into a single launch.

    Fusion only pays while the saved per-launch fixed cost is significant
    relative to the group's work, so large groups keep per-tensor launches.
    """
    cnt = len(jobs)
    if cnt == 1:
        src, dst, n = jobs[0]
        _launch_single(src, dst, n, fp32_math, fast_exp, False, block)
        return
    acc = 0
    for job in jobs:
        acc += (job[2] + block - 1) // block
    if acc > _FUSE_MAX_TILES:
        for src, dst, n in jobs:
            _launch_single(src, dst, n, fp32_math, fast_exp, False, block)
        return
    ins = [jobs[0][0]] * _MAX_TENSORS
    outs = [jobs[0][1]] * _MAX_TENSORS
    ns = [0] * _MAX_TENSORS
    ps = [_BIG] * (_MAX_TENSORS - 1)
    acc = 0
    even = True
    for k in range(cnt):
        src, dst, n = jobs[k]
        if k:
            ps[k - 1] = acc
        ins[k] = src
        outs[k] = dst
        ns[k] = n
        acc += (n + block - 1) // block
        if n % block:
            even = False
    _sigmoid_multi_kernel[(acc,)](
        *ins,
        *outs,
        *ns,
        *ps,
        BLOCK=block,
        FP32_MATH=fp32_math,
        FAST_EXP=fast_exp,
        EVEN=even,
        num_warps=_NUM_WARPS,
    )


def foreach_sigmoid(self):
    tensors = self
    if not isinstance(tensors, (list, tuple)):
        raise TypeError(
            "_foreach_sigmoid(): argument 'self' (position 1) must be tuple of "
            "Tensors, not %s" % type(tensors).__name__
        )
    if len(tensors) == 0:
        raise RuntimeError("Tensor list must have at least one tensor.")

    outs = []
    groups = {}
    for pos, t in enumerate(tensors):
        if not isinstance(t, torch.Tensor):
            raise TypeError(
                "expected Tensor as element %d in argument 0, but got %s"
                % (pos, type(t).__name__)
            )
        dt = t.dtype
        is_complex = bool(dt.is_complex)
        if dt.is_floating_point or is_complex:
            out = torch.empty_like(t)
        else:
            out = torch.empty_like(t, dtype=torch.float32)
        outs.append(out)

        n = t.numel()
        if n == 0:
            continue

        fp32_math = dt is not torch.float64 and dt is not torch.complex128
        fast_exp = dt is torch.float16 or dt is torch.bfloat16
        # 16 bytes per thread is the fp32 optimum; 2-byte dtypes want more
        # bytes in flight per thread, and very large streams want the most.
        if fast_exp:
            block = _BLOCK_16_BIG if n >= _LARGE_ELEMS else _BLOCK_16
        else:
            block = _BLOCK
        if is_complex:
            src = torch.view_as_real(t)
            dst = torch.view_as_real(out)
            if src.is_contiguous() and dst.is_contiguous():
                _launch_single(src, dst, n, fp32_math, False, True, _BLOCK)
            elif out.stride() == t.stride() and _is_dense(t.shape, t.stride()):
                _launch_single(
                    src.as_strided((n,), (1,), src.storage_offset()),
                    dst.as_strided((n,), (1,), dst.storage_offset()),
                    n,
                    fp32_math,
                    False,
                    True,
                    _BLOCK,
                )
            else:
                out_contig = out.is_contiguous()
                if out_contig:
                    dims = _collapse(t.shape, t.stride())
                    odims = None
                else:
                    dims = _plain(t.shape, t.stride())
                    odims = _plain(out.shape, out.stride())
                _run_strided(
                    src, dst, dims, n, fp32_math, False, out_contig, odims, True, _BLOCK
                )
            continue

        src = t.view(torch.uint8) if dt is torch.bool else t
        dst = out
        if src.is_contiguous() and dst.is_contiguous():
            pass
        elif out.stride() == t.stride() and _is_dense(t.shape, t.stride()):
            src = src.as_strided((n,), (1,), src.storage_offset())
            dst = dst.as_strided((n,), (1,), dst.storage_offset())
        else:
            out_contig = out.is_contiguous()
            if out_contig:
                dims = _collapse(t.shape, t.stride())
                odims = None
            else:
                dims = _plain(t.shape, t.stride())
                odims = _plain(out.shape, out.stride())
            _run_strided(
                src, dst, dims, n, fp32_math, fast_exp, out_contig, odims, False, block
            )
            continue

        key = (src.device, src.dtype, block)
        grp = groups.get(key)
        if grp is None:
            grp = [fp32_math, fast_exp, block, []]
            groups[key] = grp
        jobs = grp[3]
        jobs.append((src, dst, n))
        if len(jobs) == _MAX_TENSORS:
            _launch_multi(jobs, fp32_math, fast_exp, block)
            del jobs[:]

    for grp in groups.values():
        jobs = grp[3]
        if jobs:
            _launch_multi(jobs, grp[0], grp[1], grp[2])

    return outs
