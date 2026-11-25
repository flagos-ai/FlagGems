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

"""Triton implementation of aten::_foreach_clamp_min.Scalar (out-of-place).

Semantics reproduced from the reference (measured on the target):
  out_i = NaN-propagating max(self_i, scalar)   # s cast to the promoted dtype
  * NaN in self or NaN scalar -> NaN
  * scalar is cast to the element dtype (fp16/bf16 round-to-nearest, ints trunc)
  * promoted output dtype: float tensor -> same dtype; int/bool tensor with a
    float scalar -> torch default dtype; bool + int scalar -> int64
  * outputs are freshly allocated, never aliasing self, laid out densely in the
    input's memory order (same stride pattern torch's TensorIterator produces)
  * an empty tensor list raises RuntimeError (same as the reference)

Launch strategy: runs of up to 4 consecutive compatible contiguous tensors are
fused into a single kernel launch (per-program tensor selection by cumulative
block count); noncontiguous tensors use a row/stride-decomposition kernel.
"""

import torch
import triton
import triton.language as tl

_FLAT_BLOCK = 1024
_FLAT_WARPS = 4
_MAX_OUTER = 8
_MAX_FUSE = 4


@triton.jit
def _kk_chunk(
    x_ptr, o_ptr, s: tl.constexpr, n, base, SV64: tl.constexpr, BLOCK: tl.constexpr
):
    offs = base + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask).to(o_ptr.dtype.element_ty)
    if SV64:
        sv = tl.full([1], s, tl.float64)
    else:
        sv = tl.full([1], s, tl.float64).to(tl.float32).to(o_ptr.dtype.element_ty)
    y = tl.where(x < sv, sv, x)
    y = tl.where(sv != sv, sv, y)
    tl.store(o_ptr + offs, y, mask=mask)


@triton.jit
def _kk_flat(x_ptr, o_ptr, s: tl.constexpr, n, SV64: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    _kk_chunk(x_ptr, o_ptr, s, n, pid.to(tl.int64) * BLOCK, SV64, BLOCK)


@triton.jit
def _kk_fused(
    x0,
    o0,
    x1,
    o1,
    x2,
    o2,
    x3,
    o3,
    s: tl.constexpr,
    n0,
    n1,
    n2,
    n3,
    c1,
    c2,
    c3,
    NT: tl.constexpr,
    SV64: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid < c1:
        _kk_chunk(x0, o0, s, n0, pid.to(tl.int64) * BLOCK, SV64, BLOCK)
    else:
        if NT > 1:
            if pid < c2:
                _kk_chunk(x1, o1, s, n1, (pid - c1).to(tl.int64) * BLOCK, SV64, BLOCK)
            else:
                if NT > 2:
                    if pid < c3:
                        _kk_chunk(
                            x2, o2, s, n2, (pid - c2).to(tl.int64) * BLOCK, SV64, BLOCK
                        )
                    else:
                        if NT > 3:
                            _kk_chunk(
                                x3,
                                o3,
                                s,
                                n3,
                                (pid - c3).to(tl.int64) * BLOCK,
                                SV64,
                                BLOCK,
                            )


@triton.jit
def _kk_rows(
    x_ptr,
    o_ptr,
    s: tl.constexpr,
    inner_len,
    x_inner_stride,
    n_iblk,
    z0,
    z1,
    z2,
    z3,
    z4,
    z5,
    z6,
    z7,
    t0,
    t1,
    t2,
    t3,
    t4,
    t5,
    t6,
    t7,
    NDIM: tl.constexpr,
    SV64: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    row = pid // n_iblk
    ib = pid - row * n_iblk
    rem = row
    in_off = tl.zeros([1], tl.int64)
    if NDIM > 0:
        idx = rem % z0
        rem = rem // z0
        in_off += idx.to(tl.int64) * t0
    if NDIM > 1:
        idx = rem % z1
        rem = rem // z1
        in_off += idx.to(tl.int64) * t1
    if NDIM > 2:
        idx = rem % z2
        rem = rem // z2
        in_off += idx.to(tl.int64) * t2
    if NDIM > 3:
        idx = rem % z3
        rem = rem // z3
        in_off += idx.to(tl.int64) * t3
    if NDIM > 4:
        idx = rem % z4
        rem = rem // z4
        in_off += idx.to(tl.int64) * t4
    if NDIM > 5:
        idx = rem % z5
        rem = rem // z5
        in_off += idx.to(tl.int64) * t5
    if NDIM > 6:
        idx = rem % z6
        rem = rem // z6
        in_off += idx.to(tl.int64) * t6
    if NDIM > 7:
        idx = rem % z7
        rem = rem // z7
        in_off += idx.to(tl.int64) * t7

    i = ib.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = i < inner_len
    x = tl.load(x_ptr + in_off + i * x_inner_stride, mask=mask).to(
        o_ptr.dtype.element_ty
    )
    if SV64:
        sv = tl.full([1], s, tl.float64)
    else:
        sv = tl.full([1], s, tl.float64).to(tl.float32).to(o_ptr.dtype.element_ty)
    y = tl.where(x < sv, sv, x)
    y = tl.where(sv != sv, sv, y)
    tl.store(o_ptr + row.to(tl.int64) * inner_len + i, y, mask=mask)


@triton.jit
def _kk_flat_dual(a_ptr, b_ptr, o_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    et = o_ptr.dtype.element_ty
    a = tl.load(a_ptr + offs, mask=mask).to(et)
    b = tl.load(b_ptr + offs, mask=mask).to(et)
    y = tl.where(a < b, b, a)
    y = tl.where(b != b, b, y)
    tl.store(o_ptr + offs, y, mask=mask)


def _out_dtype(dt, sval):
    """Result dtype of clamp_min(tensor, Scalar) for the reference."""
    if dt.is_floating_point or dt.is_complex:
        return dt
    if isinstance(sval, float) and not isinstance(sval, bool):
        return torch.get_default_dtype()
    if dt == torch.bool:
        return torch.int64
    return dt


def _ptr_of(t):
    # bool loads are not usable in Triton; reinterpret the same bytes as int8
    if t.dtype == torch.bool:
        return t.view(torch.int8)
    return t


def _strided_launch(t, odt, sval):
    sv64 = odt == torch.float64
    sizes = list(t.shape)
    strides = list(t.stride())
    nd = len(sizes)
    live = [d for d in range(nd) if sizes[d] > 1]
    inner = None
    for d in live:
        if strides[d] == 1:
            inner = d
            break
    if inner is None:
        inner = min(live, key=lambda d: strides[d])
    outer = sorted([d for d in live if d != inner], key=lambda d: strides[d])
    if len(outer) > _MAX_OUTER:
        raise NotImplementedError("rank too large for strided clamp_min")

    inner_len = sizes[inner]
    inner_stride = strides[inner]
    n_outer = 1
    for d in outer:
        n_outer *= sizes[d]

    blk = 32
    while blk < inner_len and blk < 1024:
        blk *= 2
    warps = max(1, min(4, blk // 128))
    n_iblk = (inner_len + blk - 1) // blk

    # dense output layout in the input's memory order (matches TensorIterator)
    ost = [0] * nd
    ost[inner] = 1
    acc = inner_len
    for d in outer:
        ost[d] = acc
        acc *= sizes[d]
    out = torch.empty_strided(t.shape, ost, dtype=odt, device=t.device)

    zs = [sizes[d] for d in outer] + [1] * (_MAX_OUTER - len(outer))
    ts = [strides[d] for d in outer] + [0] * (_MAX_OUTER - len(outer))
    _kk_rows[(n_outer * n_iblk,)](
        _ptr_of(t),
        out,
        sval,
        inner_len,
        inner_stride,
        n_iblk,
        zs[0],
        zs[1],
        zs[2],
        zs[3],
        zs[4],
        zs[5],
        zs[6],
        zs[7],
        ts[0],
        ts[1],
        ts[2],
        ts[3],
        ts[4],
        ts[5],
        ts[6],
        ts[7],
        NDIM=len(outer),
        SV64=sv64,
        BLOCK=blk,
        num_warps=warps,
    )
    return out


def _run_scalar(tensors, sval):
    n_t = len(tensors)
    if n_t == 0:
        raise RuntimeError("Tensor list must have at least one tensor.")
    outs = [None] * n_t
    i = 0
    while i < n_t:
        t = tensors[i]
        odt = _out_dtype(t.dtype, sval)
        if t.numel() == 0:
            outs[i] = torch.empty(t.shape, dtype=odt, device=t.device)
            i += 1
            continue
        if not t.is_contiguous():
            outs[i] = _strided_launch(t, odt, sval)
            i += 1
            continue

        # collect a run of compatible contiguous tensors -> one fused launch
        group = [t]
        j = i + 1
        while j < n_t and len(group) < _MAX_FUSE:
            u = tensors[j]
            if u.numel() == 0 or not u.is_contiguous() or u.device != t.device:
                break
            if _out_dtype(u.dtype, sval) != odt:
                break
            group.append(u)
            j += 1

        sv64 = odt == torch.float64
        if len(group) == 1:
            n = t.numel()
            out = torch.empty(t.shape, dtype=odt, device=t.device)
            _kk_flat[((n + _FLAT_BLOCK - 1) // _FLAT_BLOCK,)](
                _ptr_of(t),
                out,
                sval,
                n,
                SV64=sv64,
                BLOCK=_FLAT_BLOCK,
                num_warps=_FLAT_WARPS,
            )
            outs[i] = out
        else:
            xs = [_ptr_of(u) for u in group]
            gouts = [torch.empty(u.shape, dtype=odt, device=u.device) for u in group]
            ns = [u.numel() for u in group]
            blocks = [(n + _FLAT_BLOCK - 1) // _FLAT_BLOCK for n in ns]
            while len(xs) < _MAX_FUSE:
                xs.append(xs[0])
                gouts.append(gouts[0])
                ns.append(0)
                blocks.append(0)
            c1 = blocks[0]
            c2 = c1 + blocks[1]
            c3 = c2 + blocks[2]
            total = c3 + blocks[3]
            _kk_fused[(total,)](
                xs[0],
                gouts[0],
                xs[1],
                gouts[1],
                xs[2],
                gouts[2],
                xs[3],
                gouts[3],
                sval,
                ns[0],
                ns[1],
                ns[2],
                ns[3],
                c1,
                c2,
                c3,
                NT=len(group),
                SV64=sv64,
                BLOCK=_FLAT_BLOCK,
                num_warps=_FLAT_WARPS,
            )
            for k, o in enumerate(gouts[: len(group)]):
                outs[i + k] = o
        i = j if len(group) > 1 else i + 1
    return outs


def _run_dual(tensors, others):
    if len(tensors) != len(others):
        raise RuntimeError(
            "Tensor lists must have the same number of tensors, got "
            f"{len(tensors)} and {len(others)}"
        )
    outs = []
    for a, b in zip(tensors, others):
        odt = torch.promote_types(a.dtype, b.dtype)
        n = a.numel()
        if n == 0:
            outs.append(torch.empty(a.shape, dtype=odt, device=a.device))
            continue
        if a.shape != b.shape or not a.is_contiguous() or not b.is_contiguous():
            raise NotImplementedError("unsupported layout for tensor-list clamp_min")
        out = torch.empty(a.shape, dtype=odt, device=a.device)
        _kk_flat_dual[((n + _FLAT_BLOCK - 1) // _FLAT_BLOCK,)](
            _ptr_of(a), _ptr_of(b), out, n, BLOCK=_FLAT_BLOCK, num_warps=_FLAT_WARPS
        )
        outs.append(out)
    return outs


def foreach_clamp_min_scalar(self, scalar):
    tensors = self
    if isinstance(tensors, torch.Tensor):
        tensors = (tensors,)
    if isinstance(scalar, torch.Tensor):
        scalar = scalar.item()
    if isinstance(scalar, (list, tuple)):
        return _run_dual(tensors, scalar)
    return _run_scalar(tensors, scalar)
