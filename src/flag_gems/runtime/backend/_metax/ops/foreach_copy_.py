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

"""Triton implementation of ``aten::_foreach_copy_`` (in-place multi-tensor copy).

``run(self, src, non_blocking=False)`` copies every ``src[i]`` into the existing
storage of ``self[i]`` (dtype conversion + torch broadcasting semantics) and
returns ``None``.

Architecture: pairs are classified on the host into

* contiguous same-shape pairs -- grouped by exact (dst.dtype, src.dtype) and
  copied by a single fused launch over up to ``_SLOTS`` tensors at once
  (2D grid: element blocks x pair slot, typed pointers, no device metadata).
  When all slots share one numel that is a multiple of the tile, the mask-free
  16-argument kernel is used and the element count is derived from the grid;
* general (broadcast / non-contiguous) pairs -- one collapsed-stride kernel
  launch per pair using an int64 descriptor table.
"""

import torch
import triton
import triton.language as tl

MAX_DIMS = 8
META = 32
_SLOTS = 8
_TILE = 1024
_TILE_LARGE = 2048
_LARGE_SPLIT = 1 << 20
_I32_LIMIT = 2**31 - 2**20
_F16 = torch.float16
_BF16 = torch.bfloat16
_F64 = torch.float64


# ---------------------------------------------------------------------------
# kernels
# ---------------------------------------------------------------------------
@triton.jit
def _copy_exact_body(dst_ptr, src_ptr, offs, TWO_STEP: tl.constexpr):
    v = tl.load(src_ptr + offs)
    if TWO_STEP:
        v = v.to(tl.float32)
    tl.store(dst_ptr + offs, v.to(dst_ptr.dtype.element_ty))


@triton.jit
def _copy_block(dst_ptr, src_ptr, n, offs, TWO_STEP: tl.constexpr):
    mask = offs < n
    v = tl.load(src_ptr + offs, mask=mask)
    if TWO_STEP:
        v = v.to(tl.float32)
    tl.store(dst_ptr + offs, v.to(dst_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _copy_fused_exact(
    d0,
    s0,
    d1,
    s1,
    d2,
    s2,
    d3,
    s3,
    d4,
    s4,
    d5,
    s5,
    d6,
    s6,
    d7,
    s7,
    TWO_STEP: tl.constexpr,
    TILE: tl.constexpr,
):
    pid = tl.program_id(0)
    pair = tl.program_id(1)
    offs = pid * TILE + tl.arange(0, TILE)
    if pair == 0:
        _copy_exact_body(d0, s0, offs, TWO_STEP)
    elif pair == 1:
        _copy_exact_body(d1, s1, offs, TWO_STEP)
    elif pair == 2:
        _copy_exact_body(d2, s2, offs, TWO_STEP)
    elif pair == 3:
        _copy_exact_body(d3, s3, offs, TWO_STEP)
    elif pair == 4:
        _copy_exact_body(d4, s4, offs, TWO_STEP)
    elif pair == 5:
        _copy_exact_body(d5, s5, offs, TWO_STEP)
    elif pair == 6:
        _copy_exact_body(d6, s6, offs, TWO_STEP)
    else:
        _copy_exact_body(d7, s7, offs, TWO_STEP)


@triton.jit
def _copy_fused_kernel(
    d0,
    s0,
    n0,
    d1,
    s1,
    n1,
    d2,
    s2,
    n2,
    d3,
    s3,
    n3,
    d4,
    s4,
    n4,
    d5,
    s5,
    n5,
    d6,
    s6,
    n6,
    d7,
    s7,
    n7,
    TWO_STEP: tl.constexpr,
    I64: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    pair = tl.program_id(1)
    if I64:
        offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    else:
        offs = pid * BLOCK + tl.arange(0, BLOCK)
    if pair == 0:
        _copy_block(d0, s0, n0, offs, TWO_STEP)
    elif pair == 1:
        _copy_block(d1, s1, n1, offs, TWO_STEP)
    elif pair == 2:
        _copy_block(d2, s2, n2, offs, TWO_STEP)
    elif pair == 3:
        _copy_block(d3, s3, n3, offs, TWO_STEP)
    elif pair == 4:
        _copy_block(d4, s4, n4, offs, TWO_STEP)
    elif pair == 5:
        _copy_block(d5, s5, n5, offs, TWO_STEP)
    elif pair == 6:
        _copy_block(d6, s6, n6, offs, TWO_STEP)
    else:
        _copy_block(d7, s7, n7, offs, TWO_STEP)


@triton.jit
def _copy_strided_kernel(
    dst_ptr,
    src_ptr,
    meta_ptr,
    row,
    numel,
    TWO_STEP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = offs < numel.to(tl.int64)
    base = meta_ptr + row * 32
    rem = offs
    d_off = tl.zeros([BLOCK], tl.int64)
    s_off = tl.zeros([BLOCK], tl.int64)
    for k in tl.static_range(8):
        d = 7 - k
        sz = tl.load(base + 3 + d)
        ds = tl.load(base + 11 + d)
        ss = tl.load(base + 19 + d)
        idx = rem % sz
        rem = rem // sz
        d_off += idx * ds
        s_off += idx * ss
    v = tl.load(src_ptr + s_off, mask=mask)
    if TWO_STEP:
        v = v.to(tl.float32)
    tl.store(dst_ptr + d_off, v.to(dst_ptr.dtype.element_ty), mask=mask)


# ---------------------------------------------------------------------------
# host helpers
# ---------------------------------------------------------------------------
def _launch_fused(pairs, two_step):
    """One launch for up to _SLOTS contiguous same-shape pairs."""
    k = len(pairs)
    if k == _SLOTS:
        (
            (d0, s0, n0),
            (d1, s1, n1),
            (d2, s2, n2),
            (d3, s3, n3),
            (d4, s4, n4),
            (d5, s5, n5),
            (d6, s6, n6),
            (d7, s7, n7),
        ) = pairs
        if n0 > 0 and n0 == n1 == n2 == n3 == n4 == n5 == n6 == n7 and n0 < _I32_LIMIT:
            tile = _TILE_LARGE if n0 > _LARGE_SPLIT else _TILE
            if (n0 & (tile - 1)) == 0:
                _copy_fused_exact[(n0 // tile, _SLOTS)](
                    d0,
                    s0,
                    d1,
                    s1,
                    d2,
                    s2,
                    d3,
                    s3,
                    d4,
                    s4,
                    d5,
                    s5,
                    d6,
                    s6,
                    d7,
                    s7,
                    two_step,
                    tile,
                )
                return
        maxn = max(n0, n1, n2, n3, n4, n5, n6, n7)
        _copy_fused_kernel[((maxn + _TILE - 1) // _TILE, _SLOTS)](
            d0,
            s0,
            n0,
            d1,
            s1,
            n1,
            d2,
            s2,
            n2,
            d3,
            s3,
            n3,
            d4,
            s4,
            n4,
            d5,
            s5,
            n5,
            d6,
            s6,
            n6,
            d7,
            s7,
            n7,
            two_step,
            maxn > _I32_LIMIT,
            _TILE,
        )
        return
    d0, s0, n0 = pairs[0]
    args = []
    maxn = 0
    for i in range(_SLOTS):
        if i < k:
            d, s, nn = pairs[i]
            if nn > maxn:
                maxn = nn
        else:
            d = d0
            s = s0
            nn = 0
        args.append(d)
        args.append(s)
        args.append(nn)
    _copy_fused_kernel[((maxn + _TILE - 1) // _TILE, k)](
        args[0],
        args[1],
        args[2],
        args[3],
        args[4],
        args[5],
        args[6],
        args[7],
        args[8],
        args[9],
        args[10],
        args[11],
        args[12],
        args[13],
        args[14],
        args[15],
        args[16],
        args[17],
        args[18],
        args[19],
        args[20],
        args[21],
        args[22],
        args[23],
        two_step,
        maxn > _I32_LIMIT,
        _TILE,
    )


def _collapse(dst, src):
    """Collapse (dst shape, dst strides, src strides) to <= MAX_DIMS dims.

    Src broadcasting follows torch's right-aligned semantics (stride 0 on
    broadcast dims).  Raises RuntimeError on non-broadcastable shapes.
    """
    nd = dst.dim()
    sd = src.dim()
    if sd > nd:
        raise RuntimeError(
            "The size of tensor a (%d) must not be smaller than the size of "
            "tensor b (%d)" % (nd, sd)
        )
    off = nd - sd
    dshape = dst.shape
    dstride = dst.stride()
    sshape = src.shape
    sstride = src.stride()

    shape = []
    dsd = []
    sss = []
    for i in range(nd - 1, -1, -1):
        sz = dshape[i]
        if sz == 0:
            return None
        if sz == 1:
            continue
        j = i - off
        ssz = sshape[j] if j >= 0 else 1
        if ssz == 1:
            sst = 0
        elif ssz == sz:
            sst = sstride[j]
        else:
            raise RuntimeError(
                "The expanded size of the tensor (%d) must match the existing "
                "size (%d) at non-singleton dimension %d" % (ssz, sz, i)
            )
        dst_st = dstride[i]
        if shape:
            psz = shape[-1]
            if dst_st == dsd[-1] * psz and sst == sss[-1] * psz:
                shape[-1] = psz * sz
                continue
        shape.append(sz)
        dsd.append(dst_st)
        sss.append(sst)

    shape.reverse()
    dsd.reverse()
    sss.reverse()
    return shape, dsd, sss


def _split_pair(dst, src):
    """Return a list of (dst_view, src_view, dims) collapsing within MAX_DIMS."""
    dims = _collapse(dst, src)
    if dims is None or len(dims[0]) <= MAX_DIMS:
        return [(dst, src, dims)]
    # Rare: too many non-collapsible dims -> split the leading non-trivial dim.
    for i in range(dst.dim()):
        if dst.shape[i] > 1:
            break
    else:  # pragma: no cover - cannot happen for rank > MAX_DIMS
        raise RuntimeError("foreach_copy_: unsupported layout")
    out = []
    for j in range(dst.shape[i]):
        dj = dst.select(i, j)
        sj = src.select(i, j) if src.dim() == dst.dim() else src
        out.extend(_split_pair(dj, sj))
    return out


def _row(dims, numel):
    shape, dsd, sss = dims
    k = len(shape)
    pad = MAX_DIMS - k
    row = [0, 0, numel]
    row.extend(shape)
    row.extend([1] * pad)
    row.extend(dsd)
    row.extend([0] * pad)
    row.extend(sss)
    row.extend([0] * pad)
    row.append(k)
    row.extend([0] * (META - len(row)))
    return row


def _is_two_step(dd, sd):
    """torch's Half/BFloat16 constructors round double -> float -> half."""
    return sd is _F64 and (dd is _F16 or dd is _BF16)


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------
def foreach_copy_(self, src, non_blocking=False):
    if type(self) is list:
        dsts = self
    elif isinstance(self, torch.Tensor):
        dsts = [self]
    else:
        dsts = list(self)
    if type(src) is list:
        srcs = src
    elif isinstance(src, torch.Tensor):
        srcs = [src]
    else:
        srcs = list(src)

    n = len(dsts)
    if len(srcs) != n:
        raise RuntimeError("Tensor list must have same number of elements as src list")
    if n == 0:
        return None

    dev = dsts[0].device
    if dev.type != "cuda":
        raise RuntimeError("foreach_copy_: only CUDA tensors are supported")
    if dev.index is not None and dev.index != torch.cuda.current_device():
        torch.cuda.set_device(dev.index)

    contig = []
    key0 = None
    mixed = False
    gener = None

    for d, s in zip(dsts, srcs):
        if d.is_contiguous() and d.shape == s.shape and s.is_contiguous():
            numel = d.numel()
            if numel == 0:
                continue
            contig.append((d, s, numel))
            if key0 is None:
                key0 = (d.dtype, s.dtype)
            elif not mixed and (d.dtype, s.dtype) != key0:
                mixed = True
        else:
            if gener is None:
                gener = []
            gener.append((d, s))

    if contig:
        if mixed:
            buckets = {}
            for item in contig:
                key = (item[0].dtype, item[1].dtype)
                bucket = buckets.get(key)
                if bucket is None:
                    buckets[key] = [item]
                else:
                    bucket.append(item)
            for key, pairs in buckets.items():
                _launch_fused(pairs, _is_two_step(key[0], key[1]))
        else:
            _launch_fused(contig, _is_two_step(key0[0], key0[1]))

    if gener:
        rows = []
        flat = []
        for d, s in gener:
            for dv, sv, dims in _split_pair(d, s):
                pnumel = dv.numel()
                if pnumel == 0:
                    continue
                rows.append(_row(dims, pnumel))
                flat.append((dv, sv, len(rows) - 1))
        if flat:
            meta = torch.tensor(rows, dtype=torch.int64, device=dev)
            for dv, sv, row_idx in flat:
                pnumel = dv.numel()
                _copy_strided_kernel[((pnumel + _TILE - 1) // _TILE,)](
                    dv,
                    sv,
                    meta,
                    row_idx,
                    pnumel,
                    _is_two_step(dv.dtype, sv.dtype),
                    _TILE,
                )

    return None
