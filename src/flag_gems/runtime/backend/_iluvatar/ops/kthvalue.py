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

import collections

import torch
import triton
import triton.language as tl

# dtype codes
_DT_F32 = 0
_DT_F16 = 1
_DT_BF16 = 2
_DT_I32 = 3
_DT_I64 = 4
_DT_F64 = 5
_DT_I16 = 6
_DT_I8 = 7
_DT_U32 = 8
_DT_U64 = 9
_DT_U16 = 10
_DT_U8 = 11

_SORT_MAX_N = 256
_HIST_BLOCK = 16384


def _dtype_code(dt):
    if dt == torch.float32:
        return _DT_F32
    if dt == torch.float16:
        return _DT_F16
    if dt == torch.bfloat16:
        return _DT_BF16
    if dt == torch.int32:
        return _DT_I32
    if dt == torch.int64:
        return _DT_I64
    if dt == torch.float64:
        return _DT_F64
    if dt == torch.int16:
        return _DT_I16
    if dt == torch.int8:
        return _DT_I8
    if dt == torch.uint32:
        return _DT_U32
    if dt == torch.uint64:
        return _DT_U64
    if dt == torch.uint16:
        return _DT_U16
    if dt == torch.uint8:
        return _DT_U8
    raise NotImplementedError(f"kthvalue: unsupported dtype {dt}")


@triton.jit
def _ord_of(v, DT: tl.constexpr):
    """Native-dtype tensor -> uint64 order-preserving ordinal (IEEE-style).
    +0.0 and -0.0 map to the same ordinal (they are numerically equal)."""
    if DT == 0:  # f32
        u = v.to(tl.uint32, bitcast=True)
        o = tl.where(u >= 0x80000000, ~u, u | 0x80000000)
        return tl.where(u == 0x80000000, 0x80000000, o).to(tl.uint64)
    elif DT == 1:  # f16
        u = v.to(tl.float32).to(tl.uint32, bitcast=True)
        o = tl.where(u >= 0x80000000, ~u, u | 0x80000000)
        return tl.where(u == 0x80000000, 0x80000000, o).to(tl.uint64)
    elif DT == 2:  # bf16
        u = v.to(tl.float32).to(tl.uint32, bitcast=True)
        o = tl.where(u >= 0x80000000, ~u, u | 0x80000000)
        return tl.where(u == 0x80000000, 0x80000000, o).to(tl.uint64)
    elif DT == 3:  # i32
        return (v.to(tl.uint32, bitcast=True) ^ 0x80000000).to(tl.uint64)
    elif DT == 4:  # i64
        return v.to(tl.uint64, bitcast=True) ^ 0x8000000000000000
    elif DT == 5:  # f64
        u = v.to(tl.uint64, bitcast=True)
        o = tl.where(u >= 0x8000000000000000, ~u, u | 0x8000000000000000)
        return tl.where(u == 0x8000000000000000, 0x8000000000000000, o)
    elif DT == 6:  # i16
        return (v.to(tl.int32).to(tl.uint32, bitcast=True) ^ 0x80000000).to(tl.uint64)
    elif DT == 7:  # i8
        return (v.to(tl.int32).to(tl.uint32, bitcast=True) ^ 0x80000000).to(tl.uint64)
    elif DT == 8:  # u32
        return v.to(tl.uint32, bitcast=True).to(tl.uint64)
    elif DT == 9:  # u64
        return v.to(tl.uint64, bitcast=True)
    elif DT == 10:  # u16
        return v.to(tl.uint32).to(tl.uint64)
    else:  # u8
        return v.to(tl.uint32).to(tl.uint64)


@triton.jit
def _val_of(o, DT: tl.constexpr):
    """uint64 ordinal (scalar/tensor) -> native dtype value."""
    if DT == 0:  # f32
        b = tl.where(o < 0x80000000, ~o, o & 0x7FFFFFFF).to(tl.uint32)
        return b.to(tl.float32, bitcast=True)
    elif DT == 1:  # f16
        b = tl.where(o < 0x80000000, ~o, o & 0x7FFFFFFF).to(tl.uint32)
        return b.to(tl.float32, bitcast=True).to(tl.float16)
    elif DT == 2:  # bf16
        b = tl.where(o < 0x80000000, ~o, o & 0x7FFFFFFF).to(tl.uint32)
        return b.to(tl.float32, bitcast=True).to(tl.bfloat16)
    elif DT == 3:  # i32
        return (o.to(tl.uint32) ^ 0x80000000).to(tl.int32, bitcast=True)
    elif DT == 4:  # i64
        return (o ^ 0x8000000000000000).to(tl.int64, bitcast=True)
    elif DT == 5:  # f64
        b = tl.where(o < 0x8000000000000000, ~o, o & 0x7FFFFFFFFFFFFFFF)
        return b.to(tl.float64, bitcast=True)
    elif DT == 6:  # i16
        return (o.to(tl.uint32) ^ 0x80000000).to(tl.int32, bitcast=True).to(tl.int16)
    elif DT == 7:  # i8
        return (o.to(tl.uint32) ^ 0x80000000).to(tl.int32, bitcast=True).to(tl.int8)
    elif DT == 8:  # u32
        return o.to(tl.uint32, bitcast=True)
    elif DT == 9:  # u64
        return o.to(tl.uint64, bitcast=True)
    elif DT == 10:  # u16
        return o.to(tl.uint32).to(tl.uint16)
    else:  # u8
        return o.to(tl.uint32).to(tl.uint8)


@triton.jit
def _base_of_slice(s, SHAPES, STRIDES, DIM: tl.constexpr, NDIM: tl.constexpr):
    """Linear element offset of slice `s` (0-d or vector) in the flat input."""
    inner = tl.full((), 1, tl.int64)
    for i in tl.static_range(DIM + 1, NDIM):
        inner = inner * SHAPES[i]
    outer = tl.full((), 1, tl.int64)
    for i in tl.static_range(0, DIM):
        outer = outer * SHAPES[i]
    inner_idx = s % inner
    outer_idx = s // inner
    base = tl.zeros_like(inner_idx)
    # pre-dim axes: axis 0 is most significant -> iterate from DIM-1 down to 0
    cum = tl.full((), 1, tl.int64)
    for i in tl.static_range(DIM - 1, -1, -1):
        coord = (outer_idx // cum) % SHAPES[i]
        base += coord * STRIDES[i]
        cum = cum * SHAPES[i]
    # post-dim axes: axis DIM+1 is most significant -> iterate from NDIM-1 down to DIM+1
    cum = tl.full((), 1, tl.int64)
    for i in tl.static_range(NDIM - 1, DIM, -1):
        coord = (inner_idx // cum) % SHAPES[i]
        base += coord * STRIDES[i]
        cum = cum * SHAPES[i]
    return base


@triton.jit
def _kth_smallk_kernel(
    inp,
    out_val,
    out_idx,
    k,
    N,
    num_slices,
    SHAPES,
    STRIDES,
    DIM: tl.constexpr,
    NDIM: tl.constexpr,
    V: tl.constexpr,
    BLOCK: tl.constexpr,
    DT: tl.constexpr,
    NB64: tl.constexpr,
):
    """k <= 8: sequential masked-min selection of the k-th smallest (ordinal,
    index) pair, then last-occurrence index scan."""
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    offs64 = offs.to(tl.int64)
    N64 = tl.full((), 0, tl.int64) + N
    k64 = tl.full((), 0, tl.int64) + k
    NS = tl.full((), 0, tl.int64) + num_slices
    vrow = tl.arange(0, V)
    row = (pid * V + vrow).to(tl.int64)
    valid = row < NS
    base = _base_of_slice(row, SHAPES, STRIDES, DIM, NDIM)
    sl_stride = STRIDES[DIM]
    nchunks = (N64 + BLOCK - 1) // BLOCK

    cur_ord = tl.full((V,), 0, tl.uint64)
    cur_idx = tl.full((V,), -1, tl.int64)
    for j in range(0, k64):
        best_o = tl.full((V,), 0xFFFFFFFFFFFFFFFF, tl.uint64)
        best_i = tl.full((V,), 1 << 62, tl.int64)
        for c in range(0, nchunks):
            off = c * BLOCK + offs64
            m = off < N64
            val = tl.load(
                inp + base[:, None] + off[None, :] * sl_stride, mask=m, other=0
            )
            ord_ = _ord_of(val, DT)
            gt = (ord_ > cur_ord[:, None]) | (
                (ord_ == cur_ord[:, None]) & (off[None, :] > cur_idx[:, None])
            )
            elig = m & gt
            chunk_o = tl.min(tl.where(elig, ord_, 0xFFFFFFFFFFFFFFFF), axis=1)
            is_min = elig & (ord_ == chunk_o[:, None])
            chunk_i = tl.min(tl.where(is_min, off[None, :], 1 << 62), axis=1)
            take = (chunk_o < best_o) | ((chunk_o == best_o) & (chunk_i < best_i))
            best_o = tl.where(take, chunk_o, best_o)
            best_i = tl.where(take, chunk_i, best_i)
        cur_ord = best_o
        cur_idx = best_i
    val_k = _val_of(cur_ord, DT)
    idx_k = tl.full((V,), -1, tl.int64)
    for c in range(0, nchunks):
        off = c * BLOCK + offs64
        m = off < N64
        val = tl.load(inp + base[:, None] + off[None, :] * sl_stride, mask=m, other=0)
        ord_ = _ord_of(val, DT)
        eq = m & (ord_ == cur_ord[:, None])
        cand = tl.max(tl.where(eq, off[None, :], -1), axis=1)
        idx_k = tl.maximum(idx_k, cand)
    tl.store(out_val + row, val_k, mask=valid)
    tl.store(out_idx + row, idx_k, mask=valid)


@triton.jit
def _kth_smallk_reg_kernel(
    inp,
    out_val,
    out_idx,
    k,
    N,
    num_slices,
    SHAPES,
    STRIDES,
    DIM: tl.constexpr,
    NDIM: tl.constexpr,
    V: tl.constexpr,
    BLOCK: tl.constexpr,
    DT: tl.constexpr,
    NB64: tl.constexpr,
):
    """k <= 8, single-chunk (N <= BLOCK): load the slice once into registers and
    run the k masked-min selection steps on register-resident ordinals. For
    32-bit ordinals each step is one packed (ord, idx) min; the final step uses
    a (ord, ~idx) packed min so the last-occurrence index comes out of the same
    reduction, avoiding a separate index scan."""
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    offs64 = offs.to(tl.int64)
    k64 = tl.full((), 0, tl.int64) + k
    NS = tl.full((), 0, tl.int64) + num_slices
    vrow = tl.arange(0, V)
    row = (pid * V + vrow).to(tl.int64)
    valid = row < NS
    base = _base_of_slice(row, SHAPES, STRIDES, DIM, NDIM)
    sl_stride = STRIDES[DIM]
    m = (offs[None, :] < N) & valid[:, None]
    val = tl.load(inp + base[:, None] + offs64[None, :] * sl_stride, mask=m, other=0)
    ord_reg = _ord_of(val, DT)

    if NB64:
        cur_ord = tl.full((V,), 0, tl.uint64)
        cur_idx = tl.full((V,), -1, tl.int64)
        for j in range(0, k64):
            gt = (ord_reg > cur_ord[:, None]) | (
                (ord_reg == cur_ord[:, None]) & (offs64[None, :] > cur_idx[:, None])
            )
            elig = m & gt
            best_o = tl.min(tl.where(elig, ord_reg, 0xFFFFFFFFFFFFFFFF), axis=1)
            is_min = elig & (ord_reg == best_o[:, None])
            best_i = tl.min(tl.where(is_min, offs64[None, :], 1 << 62), axis=1)
            cur_ord = best_o
            cur_idx = best_i
        val_k = _val_of(cur_ord, DT)
        idx_k = tl.max(
            tl.where(m & (ord_reg == cur_ord[:, None]), offs64[None, :], -1), axis=1
        )
    else:
        ord32 = ord_reg.to(tl.uint32)
        cur32 = tl.full((V,), 0, tl.uint32)
        cur_idx = tl.full((V,), -1, tl.int64)
        offu = offs64.to(tl.uint64)
        for j in range(0, k64 - 1):
            gt = (ord32 > cur32[:, None]) | (
                (ord32 == cur32[:, None])
                & (offu[None, :] > cur_idx[:, None].to(tl.uint64))
            )
            elig = m & gt
            key = (ord32.to(tl.uint64) << 32) | (offu & 0xFFFFFFFF)
            best_k = tl.min(tl.where(elig, key, 0xFFFFFFFFFFFFFFFF), axis=1)
            cur32 = (best_k >> 32).to(tl.uint32)
            cur_idx = (best_k & 0xFFFFFFFF).to(tl.int64)
        gt = (ord32 > cur32[:, None]) | (
            (ord32 == cur32[:, None]) & (offu[None, :] > cur_idx[:, None].to(tl.uint64))
        )
        elig = m & gt
        fkey = (ord32.to(tl.uint64) << 32) | (0xFFFFFFFF - offu)
        kth_k = tl.min(tl.where(elig, fkey, 0xFFFFFFFFFFFFFFFF), axis=1)
        ord_k = kth_k >> 32
        idx_k = (0xFFFFFFFF - (kth_k & 0xFFFFFFFF)).to(tl.int64)
        val_k = _val_of(ord_k, DT)

    tl.store(out_val + row, val_k, mask=valid)
    tl.store(out_idx + row, idx_k, mask=valid)


@triton.jit
def _kth_sort_kernel(
    inp,
    out_val,
    out_idx,
    k,
    N,
    num_slices,
    SHAPES,
    STRIDES,
    DIM: tl.constexpr,
    NDIM: tl.constexpr,
    V: tl.constexpr,
    BLOCK: tl.constexpr,
    DT: tl.constexpr,
    NB64: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    offs64 = offs.to(tl.int64)
    k64 = tl.full((), 0, tl.int64) + k
    NS = tl.full((), 0, tl.int64) + num_slices
    vrow = tl.arange(0, V)
    row = (pid * V + vrow).to(tl.int64)
    valid = row < NS
    base = _base_of_slice(row, SHAPES, STRIDES, DIM, NDIM)
    sl_stride = STRIDES[DIM]
    m = (offs[None, :] < N) & valid[:, None]
    val = tl.load(inp + base[:, None] + offs64[None, :] * sl_stride, mask=m, other=0)
    ord_ = _ord_of(val, DT)
    if NB64:
        key = (ord_ ^ 0x8000000000000000).to(tl.int64)
    else:
        key = (ord_.to(tl.uint32) ^ 0x80000000).to(tl.int32)
    if NB64:
        key_s = tl.where(m, key, 0x7FFFFFFFFFFFFFFF)
    else:
        key_s = tl.where(m, key, 0x7FFFFFFF)
    s = tl.sort(key_s, dim=1)
    is_k = offs64[None, :] == (k64 - 1)
    kth = tl.sum(tl.where(is_k, s, 0), axis=1)
    if NB64:
        ord_k = kth.to(tl.uint64) ^ 0x8000000000000000
    else:
        ord_k = (kth.to(tl.uint32) ^ 0x80000000).to(tl.uint64)
    val_k = _val_of(ord_k, DT)
    idx_k = tl.max(tl.where(m & (ord_ == ord_k[:, None]), offs64[None, :], -1), axis=1)
    tl.store(out_val + row, val_k, mask=valid)
    tl.store(out_idx + row, idx_k, mask=valid)


@triton.jit
def _kth_hist_kernel(
    inp,
    out_val,
    out_idx,
    k,
    N,
    SHAPES,
    STRIDES,
    DIM: tl.constexpr,
    NDIM: tl.constexpr,
    BLOCK: tl.constexpr,
    DT: tl.constexpr,
    NBP: tl.constexpr,
):
    pid = tl.program_id(0)
    s = pid.to(tl.int64)
    offs = tl.arange(0, BLOCK)
    offs64 = offs.to(tl.int64)
    N64 = tl.full((), 0, tl.int64) + N
    k64 = tl.full((), 0, tl.int64) + k
    base = _base_of_slice(s, SHAPES, STRIDES, DIM, NDIM)
    sl_stride = STRIDES[DIM]
    nchunks = (N64 + BLOCK - 1) // BLOCK
    arange256 = tl.arange(0, 256)
    prefix = tl.full((), 0, tl.uint64)
    k_local = k64
    for pb in tl.static_range(NBP):
        shift = (NBP - 1 - pb) * 8
        hist_g = tl.zeros((256,), tl.int32)
        not_group = tl.full((), 0, tl.int64)
        for c in range(0, nchunks):
            off = c * BLOCK + offs64
            m = off < N64
            val = tl.load(inp + base + off * sl_stride, mask=m, other=0)
            ord_ = _ord_of(val, DT)
            if pb == 0:
                group = m
            else:
                group = m & ((ord_ >> (shift + 8)) == (prefix >> (shift + 8)))
            byte = ((ord_ >> shift) & 0xFF).to(tl.int32)
            hv = tl.where(group, byte, 0)
            hist_g += tl.histogram(hv, 256)
            not_group += tl.sum((~group).to(tl.int64))
        hist_g0 = tl.sum(tl.where(arange256 == 0, hist_g, 0))
        hincl = tl.where(arange256 == 0, hist_g0 - not_group, hist_g)
        csum = tl.cumsum(hincl.to(tl.int64))
        ge = csum >= k_local
        xb = tl.min(tl.where(ge, arange256, 256))
        count_before = tl.sum(tl.where(arange256 < xb, hincl, 0).to(tl.int64))
        k_local = k_local - count_before
        prefix = prefix | (xb.to(tl.uint64) << shift)
    best = tl.full((), -1, tl.int64)
    for c in range(0, nchunks):
        off = c * BLOCK + offs64
        m = off < N64
        val = tl.load(inp + base + off * sl_stride, mask=m, other=0)
        ord_ = _ord_of(val, DT)
        eq = m & (ord_ == prefix)
        cand = tl.max(tl.where(eq, off, -1))
        best = tl.maximum(best, cand)
    val_k = _val_of(prefix, DT)
    tl.store(out_val + pid, val_k)
    tl.store(out_idx + pid, best)


_kthvalue_rt = collections.namedtuple("kthvalue", ["values", "indices"])


def kthvalue(inp, k, dim=-1, keepdim=False):
    if not isinstance(inp, torch.Tensor):
        raise TypeError("kthvalue(): input must be a tensor")
    if inp.dim() == 0:
        raise RuntimeError("kthvalue(): input must be at least 1-dimensional")
    orig_dim = dim
    if dim < 0:
        dim = dim + inp.dim()
    if dim < 0 or dim >= inp.dim():
        raise IndexError(
            f"Dimension out of range (expected to be in range of "
            f"[{-inp.dim()}, {inp.dim() - 1}], but got {orig_dim})"
        )
    N = inp.shape[dim]

    if keepdim:
        out_shape = list(inp.shape)
        out_shape[dim] = 1
    else:
        out_shape = list(inp.shape[:dim]) + list(inp.shape[dim + 1 :])

    if N == 0:
        raise IndexError(
            f"kthvalue(): Expected reduction dim {dim} to have non-zero size."
        )
    if k < 1 or k > N:
        raise RuntimeError(
            f"kthvalue(): selected number k out of range for dimension {dim}"
        )

    values = torch.empty(out_shape, dtype=inp.dtype, device=inp.device)
    indices = torch.empty(out_shape, dtype=torch.int64, device=inp.device)

    outer = 1
    for i in range(dim):
        outer *= inp.shape[i]
    inner = 1
    for i in range(dim + 1, inp.dim()):
        inner *= inp.shape[i]
    num_slices = outer * inner
    if num_slices == 0:
        return _kthvalue_rt(values, indices)

    DT = _dtype_code(inp.dtype)
    NB64 = DT in (_DT_I64, _DT_F64, _DT_U64)
    shapes = tuple(int(s) for s in inp.shape)
    strides = tuple(int(s) for s in inp.stride())
    ndim = inp.dim()

    if k <= 8:
        if N > 2048:
            # chunked small-k for very long slices
            BLOCK = 4096
            V = 1
            grid = (triton.cdiv(num_slices, V),)
            _kth_smallk_kernel[grid](
                inp,
                values,
                indices,
                k,
                N,
                num_slices,
                shapes,
                strides,
                DIM=dim,
                NDIM=ndim,
                V=V,
                BLOCK=BLOCK,
                DT=DT,
                NB64=NB64,
                num_warps=4,
            )
        elif N <= 8:
            # tiny N: bitonic sort wins and avoids tiny-BLOCK register issues
            BLOCK = triton.next_power_of_2(N)
            V = min(1024 // BLOCK, triton.next_power_of_2(num_slices))
            grid = (triton.cdiv(num_slices, V),)
            _kth_sort_kernel[grid](
                inp,
                values,
                indices,
                k,
                N,
                num_slices,
                shapes,
                strides,
                DIM=dim,
                NDIM=ndim,
                V=V,
                BLOCK=BLOCK,
                DT=DT,
                NB64=NB64,
            )
        else:
            # fused register-resident single-load small-k, per-N tuned config
            BLOCK = triton.next_power_of_2(N)
            if N <= 64:
                V = 16
                nw = 4
            elif N <= 128:
                V = 32
                nw = 8
            elif N <= 256:
                V = 4
                nw = 8
            else:
                V = 8 if num_slices >= 1024 else 4
                nw = 8
            V = max(1, min(V, triton.next_power_of_2(num_slices)))
            grid = (triton.cdiv(num_slices, V),)
            _kth_smallk_reg_kernel[grid](
                inp,
                values,
                indices,
                k,
                N,
                num_slices,
                shapes,
                strides,
                DIM=dim,
                NDIM=ndim,
                V=V,
                BLOCK=BLOCK,
                DT=DT,
                NB64=NB64,
                num_warps=nw,
            )
    elif N <= _SORT_MAX_N:
        BLOCK = triton.next_power_of_2(N)
        V = min(1024 // BLOCK, triton.next_power_of_2(num_slices))
        grid = (triton.cdiv(num_slices, V),)
        _kth_sort_kernel[grid](
            inp,
            values,
            indices,
            k,
            N,
            num_slices,
            shapes,
            strides,
            DIM=dim,
            NDIM=ndim,
            V=V,
            BLOCK=BLOCK,
            DT=DT,
            NB64=NB64,
        )
    else:
        BLOCK = min(_HIST_BLOCK, triton.next_power_of_2(N))
        nw = 2 if N <= 4096 else 8
        grid = (num_slices,)
        _kth_hist_kernel[grid](
            inp,
            values,
            indices,
            k,
            N,
            shapes,
            strides,
            DIM=dim,
            NDIM=ndim,
            BLOCK=BLOCK,
            DT=DT,
            NBP=8 if NB64 else 4,
            num_warps=nw,
        )

    return _kthvalue_rt(values, indices)
