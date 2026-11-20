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

"""In-place multi-tensor cosine (``aten::_foreach_cos_``) in Triton.

Design
------
* ``run`` validates the TensorList like ATen (list/tuple, non-empty, floating
  dtypes only), then picks one of three launch paths.
* Uniform contiguous lists (all entries same dtype, same numel, contiguous) are
  served by a *single* fused launch: a 2D grid ``(K, tiles)`` where each CTA
  selects its own entry pointer.  This removes ``K-1`` Python/Triton launches,
  which dominate the small- and medium-shape cost of the operator.
* A contiguous single entry uses a minimal one-pointer kernel.
* Non-contiguous or heterogeneous lists fall back to per-entry launches: a flat
  kernel for contiguous entries and a collapsed shape/stride kernel otherwise.
  In every case the tensor's own storage is written in place.
"""

import torch
import triton
import triton.language as tl

_TL = {
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
    torch.float32: tl.float32,
    torch.float64: tl.float64,
}

_MAX_FUSED = 16
_MAX_RANK = 8


@triton.jit(do_not_specialize=["n"])
def _cos1(x_ptr, n, DT: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    if DT == tl.float64:
        tl.store(x_ptr + offs, tl.cos(x), mask=mask)
    else:
        tl.store(x_ptr + offs, tl.cos(x.to(tl.float32)).to(DT), mask=mask)


@triton.jit(
    do_not_specialize=[
        "n",
        "p0",
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
def _cosN(
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
    n,
    DT: tl.constexpr,
    BLOCK: tl.constexpr,
    K: tl.constexpr,
):
    """One launch for ``K`` equal-sized contiguous entries of one dtype.

    Axis 0 selects the entry, axis 1 the tile inside it.  Pointers travel as
    ``int64`` so that a uniform scalar ``select`` picks the base address; every
    CTA in the launch therefore performs identical vectorized work.
    """
    t = tl.program_id(0)
    offs = tl.program_id(1).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n

    p = p0
    if K > 1:
        p = tl.where(t == 1, p1, p)
        p = tl.where(t == 2, p2, p)
        p = tl.where(t == 3, p3, p)
        p = tl.where(t == 4, p4, p)
        p = tl.where(t == 5, p5, p)
        p = tl.where(t == 6, p6, p)
        p = tl.where(t == 7, p7, p)
        p = tl.where(t == 8, p8, p)
        p = tl.where(t == 9, p9, p)
        p = tl.where(t == 10, p10, p)
        p = tl.where(t == 11, p11, p)
        p = tl.where(t == 12, p12, p)
        p = tl.where(t == 13, p13, p)
        p = tl.where(t == 14, p14, p)
        p = tl.where(t == 15, p15, p)
    x_ptr = p.to(tl.pointer_type(DT))

    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    if DT == tl.float64:
        y = tl.cos(x)
    else:
        y = tl.cos(x.to(tl.float32)).to(DT)
    tl.store(x_ptr + offs, y, mask=mask)


@triton.jit(
    do_not_specialize=[
        "n",
        "s0",
        "s1",
        "s2",
        "s3",
        "s4",
        "s5",
        "s6",
        "s7",
        "t0",
        "t1",
        "t2",
        "t3",
        "t4",
        "t5",
        "t6",
        "t7",
    ]
)
def _cos_strided(
    x_ptr,
    n,
    s0,
    s1,
    s2,
    s3,
    s4,
    s5,
    s6,
    s7,
    t0,
    t1,
    t2,
    t3,
    t4,
    t5,
    t6,
    t7,
    DT: tl.constexpr,
    RANK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    rem = offs
    off = tl.zeros([BLOCK], tl.int64)
    if RANK >= 1:
        off += (rem % s0) * t0
        rem = rem // s0
    if RANK >= 2:
        off += (rem % s1) * t1
        rem = rem // s1
    if RANK >= 3:
        off += (rem % s2) * t2
        rem = rem // s2
    if RANK >= 4:
        off += (rem % s3) * t3
        rem = rem // s3
    if RANK >= 5:
        off += (rem % s4) * t4
        rem = rem // s4
    if RANK >= 6:
        off += (rem % s5) * t5
        rem = rem // s5
    if RANK >= 7:
        off += (rem % s6) * t6
        rem = rem // s6
    if RANK >= 8:
        off += (rem % s7) * t7
    x = tl.load(x_ptr + off, mask=mask, other=0.0)
    if DT == tl.float64:
        y = tl.cos(x)
    else:
        y = tl.cos(x.to(tl.float32)).to(DT)
    tl.store(x_ptr + off, y, mask=mask)


def _collapse(shape, stride):
    """Merge neighbouring logical dims that are adjacent in memory.

    Iteration starts at the innermost (last) logical dim, so ``out[0]`` is the
    fastest varying logical dim and the kernel can decompose a flat index with
    one modulo/division per entry.  A merge keeps the *inner* stride as the base
    stride of the merged block -- using the outer stride there would silently
    write outside the tensor.
    """
    dims = [(sz, st) for sz, st in zip(shape, stride) if sz != 1]
    out = []
    cur = None
    for sz, st in reversed(dims):
        if cur is None:
            cur = (sz, st)
        elif cur[1] * cur[0] == st:
            cur = (cur[0] * sz, cur[1])
        else:
            out.append(cur)
            cur = (sz, st)
    if cur is not None:
        out.append(cur)
    return out


def _bad_dtype(dtype):
    raise RuntimeError(
        "result type Float can't be cast to the desired output type "
        f"{str(dtype).replace('torch.', '').capitalize()}"
    )


def _strided(t, n, dt):
    dims = _collapse(t.shape, t.stride())
    if len(dims) > _MAX_RANK:
        tmp = t.contiguous()
        block = 256 if n <= 8192 else 1024
        _cos1[(triton.cdiv(n, block),)](tmp, n, dt, block, num_warps=4)
        t.copy_(tmp)
        return
    if not dims:
        dims = [(1, 1)]
    sizes = [d[0] for d in dims]
    strides = [d[1] for d in dims]
    pad = _MAX_RANK - len(dims)
    sizes += [1] * pad
    strides += [0] * pad
    block = 256
    _cos_strided[(triton.cdiv(n, block),)](
        t,
        n,
        sizes[0],
        sizes[1],
        sizes[2],
        sizes[3],
        sizes[4],
        sizes[5],
        sizes[6],
        sizes[7],
        strides[0],
        strides[1],
        strides[2],
        strides[3],
        strides[4],
        strides[5],
        strides[6],
        strides[7],
        dt,
        len(dims),
        block,
        num_warps=4,
    )


def _general(tensors):
    """Per-entry path for heterogeneous or non-contiguous lists."""
    for t in tensors:
        if not isinstance(t, torch.Tensor):
            raise TypeError("argument 'self' must be tuple of Tensors")
        if t.dtype not in _TL:
            _bad_dtype(t.dtype)
    for t in tensors:
        n = t.numel()
        if n == 0:
            continue
        dt = _TL[t.dtype]
        if t.is_contiguous():
            block = 256 if n <= 8192 else 1024
            _cos1[(triton.cdiv(n, block),)](t, n, dt, block, num_warps=4)
        else:
            _strided(t, n, dt)


def foreach_cos_(self):
    if not isinstance(self, (list, tuple)):
        raise TypeError("argument 'self' must be tuple of Tensors")
    k = len(self)
    if k == 0:
        raise RuntimeError("Tensor list must have at least one tensor.")

    first = self[0]
    if not isinstance(first, torch.Tensor):
        raise TypeError("argument 'self' must be tuple of Tensors")
    dt = first.dtype
    if dt not in _TL:
        _bad_dtype(dt)
    if not first.is_contiguous():
        _general(self)
        return None

    n = first.numel()
    if k == 1:
        if n:
            block = 256 if n <= 8192 else 1024
            _cos1[(triton.cdiv(n, block),)](first, n, _TL[dt], block, num_warps=4)
        return None

    for i in range(1, k):
        t = self[i]
        if t.dtype is not dt or t.numel() != n or not t.is_contiguous():
            _general(self)
            return None

    if n == 0:
        return None

    # Fused path: one launch per group of at most _MAX_FUSED entries.
    block = 256 if n <= 8192 else 1024
    tiles = triton.cdiv(n, block)
    if tiles > 65535:
        _general(self)
        return None
    tdt = _TL[dt]
    i = 0
    while i < k:
        chunk = k - i
        if chunk > _MAX_FUSED:
            chunk = _MAX_FUSED
        ptrs = [0] * _MAX_FUSED
        for j in range(chunk):
            ptrs[j] = self[i + j].data_ptr()
        _cosN[(chunk, tiles)](
            ptrs[0],
            ptrs[1],
            ptrs[2],
            ptrs[3],
            ptrs[4],
            ptrs[5],
            ptrs[6],
            ptrs[7],
            ptrs[8],
            ptrs[9],
            ptrs[10],
            ptrs[11],
            ptrs[12],
            ptrs[13],
            ptrs[14],
            ptrs[15],
            n,
            tdt,
            block,
            chunk,
            num_warps=4,
        )
        i += chunk
    return None
