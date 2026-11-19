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
    _foreach_clamp_min__Scalar as default__foreach_clamp_min__Scalar,
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

# (dtype, block, grid, numels, scalars, alignment) -> (compiled kernel, stream)
_FAST = {}


@triton.jit
def _apply(x, s):
    """MUSA body: fp32 opmath, propagating NaN, rounded back to storage."""
    return tl.maximum(
        x.to(tl.float32), s.to(tl.float32), propagate_nan=tl.PropagateNan.ALL
    ).to(x.dtype)


@triton.jit
def _fused(
    x0,
    o0,
    n0,
    s0,
    x1,
    o1,
    n1,
    s1,
    x2,
    o2,
    n2,
    s2,
    x3,
    o3,
    n3,
    s3,
    x4,
    o4,
    n4,
    s4,
    x5,
    o5,
    n5,
    s5,
    x6,
    o6,
    n6,
    s6,
    x7,
    o7,
    n7,
    s7,
    x8,
    o8,
    n8,
    s8,
    x9,
    o9,
    n9,
    s9,
    x10,
    o10,
    n10,
    s10,
    x11,
    o11,
    n11,
    s11,
    x12,
    o12,
    n12,
    s12,
    x13,
    o13,
    n13,
    s13,
    x14,
    o14,
    n14,
    s14,
    x15,
    o15,
    n15,
    s15,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    slot = tl.program_id(1)

    px = x0
    po = o0
    np = n0
    sp = s0
    px = tl.where(slot == 1, x1, px)
    po = tl.where(slot == 1, o1, po)
    np = tl.where(slot == 1, n1, np)
    sp = tl.where(slot == 1, s1, sp)
    px = tl.where(slot == 2, x2, px)
    po = tl.where(slot == 2, o2, po)
    np = tl.where(slot == 2, n2, np)
    sp = tl.where(slot == 2, s2, sp)
    px = tl.where(slot == 3, x3, px)
    po = tl.where(slot == 3, o3, po)
    np = tl.where(slot == 3, n3, np)
    sp = tl.where(slot == 3, s3, sp)
    px = tl.where(slot == 4, x4, px)
    po = tl.where(slot == 4, o4, po)
    np = tl.where(slot == 4, n4, np)
    sp = tl.where(slot == 4, s4, sp)
    px = tl.where(slot == 5, x5, px)
    po = tl.where(slot == 5, o5, po)
    np = tl.where(slot == 5, n5, np)
    sp = tl.where(slot == 5, s5, sp)
    px = tl.where(slot == 6, x6, px)
    po = tl.where(slot == 6, o6, po)
    np = tl.where(slot == 6, n6, np)
    sp = tl.where(slot == 6, s6, sp)
    px = tl.where(slot == 7, x7, px)
    po = tl.where(slot == 7, o7, po)
    np = tl.where(slot == 7, n7, np)
    sp = tl.where(slot == 7, s7, sp)
    px = tl.where(slot == 8, x8, px)
    po = tl.where(slot == 8, o8, po)
    np = tl.where(slot == 8, n8, np)
    sp = tl.where(slot == 8, s8, sp)
    px = tl.where(slot == 9, x9, px)
    po = tl.where(slot == 9, o9, po)
    np = tl.where(slot == 9, n9, np)
    sp = tl.where(slot == 9, s9, sp)
    px = tl.where(slot == 10, x10, px)
    po = tl.where(slot == 10, o10, po)
    np = tl.where(slot == 10, n10, np)
    sp = tl.where(slot == 10, s10, sp)
    px = tl.where(slot == 11, x11, px)
    po = tl.where(slot == 11, o11, po)
    np = tl.where(slot == 11, n11, np)
    sp = tl.where(slot == 11, s11, sp)
    px = tl.where(slot == 12, x12, px)
    po = tl.where(slot == 12, o12, po)
    np = tl.where(slot == 12, n12, np)
    sp = tl.where(slot == 12, s12, sp)
    px = tl.where(slot == 13, x13, px)
    po = tl.where(slot == 13, o13, po)
    np = tl.where(slot == 13, n13, np)
    sp = tl.where(slot == 13, s13, sp)
    px = tl.where(slot == 14, x14, px)
    po = tl.where(slot == 14, o14, po)
    np = tl.where(slot == 14, n14, np)
    sp = tl.where(slot == 14, s14, sp)
    px = tl.where(slot == 15, x15, px)
    po = tl.where(slot == 15, o15, po)
    np = tl.where(slot == 15, n15, np)
    sp = tl.where(slot == 15, s15, sp)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < np
    x = tl.load(px + offs, mask=mask, other=0.0)
    tl.store(po + offs, _apply(x, sp), mask=mask)


@triton.jit(
    do_not_specialize=[
        "n",
        "s",
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
    x_ptr,
    o_ptr,
    n,
    s,
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
    x = tl.load(x_ptr + src, mask=mask, other=0.0)
    tl.store(o_ptr + src, _apply(x, s), mask=mask)


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


def _launch_chunk(rows, scalars, dtype):
    """One launch for up to ``_SLOTS`` contiguous (tensor, numel, scalar) rows."""
    k = len(rows)
    padded = list(rows) + [rows[0]] * (_SLOTS - k)
    args = []
    for s in range(_SLOTS):
        tensor, out, numel, _ = padded[s]
        args += [tensor, out, numel if s < k else 0, scalars[s] if s < k else 0.0]

    numel_max = max(numel for _, _, numel, _ in rows)
    block = _config(numel_max)
    n_blocks = (numel_max + block - 1) // block
    key = (
        dtype,
        block,
        (n_blocks, k),
        tuple(numel for _, _, numel, _ in rows),
        tuple(scalars),
        tuple(t.data_ptr() & 15 for t, _, _, _ in rows),
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


def _launch_strided(t, out, numel, value):
    dims, strides = _collapse(t)
    rank = len(dims)
    if rank > _MAX_RANK:
        raise RuntimeError(
            "foreach _foreach_clamp_min__Scalar: unsupported tensor rank"
        )
    pd = dims + [1] * (_MAX_RANK - rank)
    pt = strides + [0] * (_MAX_RANK - rank)
    _strided[((numel + _BLOCK - 1) // _BLOCK,)](
        t,
        out,
        numel,
        value,
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


def _foreach_clamp_min__Scalar(self, scalars):
    logger.debug("GEMS_MTHREADS _FOREACH_CLAMP_MIN__SCALAR")

    nt = len(self)
    per_tensor = isinstance(scalars, (list, tuple))
    if per_tensor and len(scalars) != nt:
        raise RuntimeError(
            "Tensor list must have same number of elements as scalar list."
        )
    if nt == 0:
        raise RuntimeError("Tensor list must have at least one tensor.")

    tensors = list(self)
    for t in tensors:
        if not isinstance(t, torch.Tensor) or not _use_triton_kernel(t):
            return default__foreach_clamp_min__Scalar(self, scalars)
    values = list(scalars) if per_tensor else [scalars] * nt
    for v in values:
        if not isinstance(v, (int, float)):
            return default__foreach_clamp_min__Scalar(self, scalars)

    device = tensors[0].device
    dsts = tensors if True else [torch.empty_like(t) for t in tensors]
    rows = []
    scalars_ = []
    dtype = None
    with torch_device_fn.device(device):
        for t, dst, v in zip(tensors, dsts, values):
            numel = t.numel()
            if numel == 0:
                continue
            value = float(v)
            if not t.is_contiguous():
                if rows:
                    _launch_chunk(rows, scalars_, dtype)
                    rows, scalars_ = [], []
                _launch_strided(t, dst, numel, value)
                continue
            if dtype is not None and t.dtype is not dtype:
                if rows:
                    _launch_chunk(rows, scalars_, dtype)
                    rows, scalars_ = [], []
            dtype = t.dtype
            rows.append((t, dst, numel, value))
            scalars_.append(value)
            if len(rows) == _SLOTS:
                _launch_chunk(rows, scalars_, dtype)
                rows, scalars_ = [], []
        if rows:
            _launch_chunk(rows, scalars_, dtype)

    return None if True else dsts
