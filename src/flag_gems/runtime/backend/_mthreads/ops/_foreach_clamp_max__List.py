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

import logging

import torch
import triton
import triton.language as tl
from triton.runtime import driver as _driver

from flag_gems.ops._foreach_binary import (
    _foreach_clamp_max__List as default__foreach_clamp_max__List,
)
from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)

# Moore Threads hardware does not support fp64.
_SUPPORTED_DTYPES = {torch.float16, torch.bfloat16, torch.float32}

_SLOTS = 16
_BLOCK = 1024
_BLOCK_WIDE = 2048
_WIDE_NUMEL = 1 << 18
_NUM_WARPS = 4
_MAX_RANK = 8

# (dtype, block, grid, numels, alignment) -> (compiled kernel, launch stream)
_FAST = {}


@triton.jit
def _apply(x, y):
    """MUSA body: fp32 opmath with a propagating NaN."""
    return tl.minimum(
        x.to(tl.float32), y.to(tl.float32), propagate_nan=tl.PropagateNan.ALL
    ).to(x.dtype)


@triton.jit
def _fused(
    a0,
    b0,
    o0,
    n0,
    a1,
    b1,
    o1,
    n1,
    a2,
    b2,
    o2,
    n2,
    a3,
    b3,
    o3,
    n3,
    a4,
    b4,
    o4,
    n4,
    a5,
    b5,
    o5,
    n5,
    a6,
    b6,
    o6,
    n6,
    a7,
    b7,
    o7,
    n7,
    a8,
    b8,
    o8,
    n8,
    a9,
    b9,
    o9,
    n9,
    a10,
    b10,
    o10,
    n10,
    a11,
    b11,
    o11,
    n11,
    a12,
    b12,
    o12,
    n12,
    a13,
    b13,
    o13,
    n13,
    a14,
    b14,
    o14,
    n14,
    a15,
    b15,
    o15,
    n15,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    slot = tl.program_id(1)

    pa = a0
    pb = b0
    po = o0
    np = n0
    pa = tl.where(slot == 1, a1, pa)
    pb = tl.where(slot == 1, b1, pb)
    po = tl.where(slot == 1, o1, po)
    np = tl.where(slot == 1, n1, np)
    pa = tl.where(slot == 2, a2, pa)
    pb = tl.where(slot == 2, b2, pb)
    po = tl.where(slot == 2, o2, po)
    np = tl.where(slot == 2, n2, np)
    pa = tl.where(slot == 3, a3, pa)
    pb = tl.where(slot == 3, b3, pb)
    po = tl.where(slot == 3, o3, po)
    np = tl.where(slot == 3, n3, np)
    pa = tl.where(slot == 4, a4, pa)
    pb = tl.where(slot == 4, b4, pb)
    po = tl.where(slot == 4, o4, po)
    np = tl.where(slot == 4, n4, np)
    pa = tl.where(slot == 5, a5, pa)
    pb = tl.where(slot == 5, b5, pb)
    po = tl.where(slot == 5, o5, po)
    np = tl.where(slot == 5, n5, np)
    pa = tl.where(slot == 6, a6, pa)
    pb = tl.where(slot == 6, b6, pb)
    po = tl.where(slot == 6, o6, po)
    np = tl.where(slot == 6, n6, np)
    pa = tl.where(slot == 7, a7, pa)
    pb = tl.where(slot == 7, b7, pb)
    po = tl.where(slot == 7, o7, po)
    np = tl.where(slot == 7, n7, np)
    pa = tl.where(slot == 8, a8, pa)
    pb = tl.where(slot == 8, b8, pb)
    po = tl.where(slot == 8, o8, po)
    np = tl.where(slot == 8, n8, np)
    pa = tl.where(slot == 9, a9, pa)
    pb = tl.where(slot == 9, b9, pb)
    po = tl.where(slot == 9, o9, po)
    np = tl.where(slot == 9, n9, np)
    pa = tl.where(slot == 10, a10, pa)
    pb = tl.where(slot == 10, b10, pb)
    po = tl.where(slot == 10, o10, po)
    np = tl.where(slot == 10, n10, np)
    pa = tl.where(slot == 11, a11, pa)
    pb = tl.where(slot == 11, b11, pb)
    po = tl.where(slot == 11, o11, po)
    np = tl.where(slot == 11, n11, np)
    pa = tl.where(slot == 12, a12, pa)
    pb = tl.where(slot == 12, b12, pb)
    po = tl.where(slot == 12, o12, po)
    np = tl.where(slot == 12, n12, np)
    pa = tl.where(slot == 13, a13, pa)
    pb = tl.where(slot == 13, b13, pb)
    po = tl.where(slot == 13, o13, po)
    np = tl.where(slot == 13, n13, np)
    pa = tl.where(slot == 14, a14, pa)
    pb = tl.where(slot == 14, b14, pb)
    po = tl.where(slot == 14, o14, po)
    np = tl.where(slot == 14, n14, np)
    pa = tl.where(slot == 15, a15, pa)
    pb = tl.where(slot == 15, b15, pb)
    po = tl.where(slot == 15, o15, po)
    np = tl.where(slot == 15, n15, np)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < np
    a = tl.load(pa + offs, mask=mask, other=0.0)
    b = tl.load(pb + offs, mask=mask, other=0.0)
    tl.store(po + offs, _apply(a, b), mask=mask)


@triton.jit(
    do_not_specialize=[
        "n",
        "d0",
        "d1",
        "d2",
        "d3",
        "d4",
        "d5",
        "d6",
        "d7",
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
def _strided(
    a_ptr,
    b_ptr,
    o_ptr,
    n,
    d0,
    d1,
    d2,
    d3,
    d4,
    d5,
    d6,
    d7,
    t0,
    t1,
    t2,
    t3,
    t4,
    t5,
    t6,
    t7,
    RANK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    rem = offs
    src = offs * 0
    if RANK >= 8:
        src += (rem % d7) * t7
        rem = rem // d7
    if RANK >= 7:
        src += (rem % d6) * t6
        rem = rem // d6
    if RANK >= 6:
        src += (rem % d5) * t5
        rem = rem // d5
    if RANK >= 5:
        src += (rem % d4) * t4
        rem = rem // d4
    if RANK >= 4:
        src += (rem % d3) * t3
        rem = rem // d3
    if RANK >= 3:
        src += (rem % d2) * t2
        rem = rem // d2
    if RANK >= 2:
        src += (rem % d1) * t1
        rem = rem // d1
    src += rem * t0
    a = tl.load(a_ptr + src, mask=mask, other=0.0)
    b = tl.load(b_ptr + src, mask=mask, other=0.0)
    tl.store(o_ptr + offs, _apply(a, b), mask=mask)


def _config(numel):
    return _BLOCK_WIDE if numel >= _WIDE_NUMEL else _BLOCK


def _dispatch(args, k, n_blocks, block, key):
    """Launch the fused kernel, replaying the compiled variant once it exists."""
    entry = _FAST.get(key)
    if entry is not None:
        compiled, stream = entry
        try:
            compiled.run(
                n_blocks,
                k,
                1,
                stream,
                compiled.function,
                compiled.packed_metadata,
                None,
                None,
                None,
                *args,
                block,
            )
            return
        except Exception:
            # A stale handle (driver reload, changed hook set) must not break the
            # call: drop it and take the ordinary launch path instead.
            _FAST.pop(key, None)

    compiled = _fused[(n_blocks, k)](*args, BLOCK=block, num_warps=_NUM_WARPS)
    try:
        drv = _driver.active
        _FAST[key] = (compiled, drv.get_current_stream(drv.get_current_device()))
    except Exception:
        _FAST.pop(key, None)


def _launch_chunk(rows, dtype):
    """One launch for up to ``_SLOTS`` contiguous (a, b, out, numel) rows."""
    k = len(rows)
    padded = list(rows) + [rows[0]] * (_SLOTS - k)
    args = []
    for s in range(_SLOTS):
        a, b, out, numel = padded[s]
        args += [a, b, out, numel if s < k else 0]

    numel_max = max(numel for _, _, _, numel in rows)
    block = _config(numel_max)
    n_blocks = (numel_max + block - 1) // block
    key = (
        dtype,
        block,
        (n_blocks, k),
        tuple(numel for _, _, _, numel in rows),
        tuple(t.data_ptr() & 15 for row in rows for t in row[:2]),
    )
    _dispatch(args, k, n_blocks, block, key)


def _collapse(t):
    """Metadata-only (shape, stride) compaction: drop 1s, merge adjacent dims."""
    dims = []
    strides = []
    for size, stride in zip(t.shape, t.stride()):
        if size == 1:
            continue
        if dims and strides[-1] == stride * size:
            dims[-1] *= size
        else:
            dims.append(size)
            strides.append(stride)
    return dims[::-1], strides[::-1]


def _launch_strided(a, b, out, numel):
    dims, strides = _collapse(a)
    rank = len(dims)
    if rank > _MAX_RANK:
        raise RuntimeError("foreach _foreach_clamp_max__List: unsupported tensor rank")
    pd = dims + [1] * (_MAX_RANK - rank)
    pt = strides + [0] * (_MAX_RANK - rank)
    _strided[((numel + _BLOCK - 1) // _BLOCK,)](
        a,
        b,
        out,
        numel,
        pd[0],
        pd[1],
        pd[2],
        pd[3],
        pd[4],
        pd[5],
        pd[6],
        pd[7],
        pt[0],
        pt[1],
        pt[2],
        pt[3],
        pt[4],
        pt[5],
        pt[6],
        pt[7],
        RANK=rank,
        BLOCK=_BLOCK,
        num_warps=_NUM_WARPS,
    )


def _use_triton_kernel(t):
    return t.device.type == "musa" and t.dtype in _SUPPORTED_DTYPES


def _foreach_clamp_max__List(self, other):
    logger.debug("GEMS_MTHREADS _FOREACH_CLAMP_MAX__LIST")

    if not isinstance(other, (list, tuple)):
        return default__foreach_clamp_max__List(self, other)

    nt = len(self)
    if nt != len(other):
        raise RuntimeError(
            "Tensor list must have same number of elements as tensor list"
        )
    if nt == 0:
        raise RuntimeError("Tensor list must have at least one tensor")

    tensors = list(self)
    others = list(other)
    for t, o in zip(tensors, others):
        if not isinstance(t, torch.Tensor) or not _use_triton_kernel(t):
            return default__foreach_clamp_max__List(self, other)
        if not isinstance(o, torch.Tensor) or not _use_triton_kernel(o):
            return default__foreach_clamp_max__List(self, other)

    if True:
        outs = tensors
    else:
        outs = [torch.empty_like(t) for t in tensors]
    device = tensors[0].device

    rows = []
    dtype = None
    with torch_device_fn.device(device):
        for t, o, out in zip(tensors, others, outs):
            if t.shape != o.shape:
                # Broadcasting needs the generic pairing rules.
                return default__foreach_clamp_max__List(self, other)
            numel = t.numel()
            if numel == 0:
                continue
            if not (t.is_contiguous() and o.is_contiguous()):
                if rows:
                    _launch_chunk(rows, dtype)
                    rows = []
                _launch_strided(t, o, out, numel)
                continue
            if dtype is not None and t.dtype is not dtype:
                if rows:
                    _launch_chunk(rows, dtype)
                    rows = []
            dtype = t.dtype
            rows.append((t, o, out, numel))
            if len(rows) == _SLOTS:
                _launch_chunk(rows, dtype)
                rows = []
        if rows:
            _launch_chunk(rows, dtype)

    return None if True else outs
