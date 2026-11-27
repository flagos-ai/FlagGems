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


"""Moore Threads specialization of ``aten::_foreach_div.Tensor``.

Strategy
--------
* One launch per call covers up to ``_SLOTS`` contiguous tensors.  The second
  grid axis selects the slot and a comparison chain resolves that slot's
  pointers, so every operand is still passed as a typed pointer argument -- a
  pointer loaded from a metadata table loses the vectorized access on this
  target.
* The first call compiling a given variant goes through the ordinary subscript
  launch; the resulting ``CompiledKernel`` is captured with the device
  and replayed on the current stream afterwards, which skips Triton's per-call argument binding,
  specialization refinement and cache lookup.  That host cost -- tens of
  microseconds -- dominates a 16-tensor list whose device work is microseconds.
* Lists longer than ``_SLOTS`` are chunked.  Non-contiguous tensors take a
  per-tensor shape/stride kernel in which each operand carries its own strides,
  and unsupported dtypes or devices fall back to the generic executor.
"""

import logging

import torch
import triton
import triton.language as tl
from triton.runtime import driver as _driver

from flag_gems.ops._foreach_binary import (
    _foreach_div_Tensor as default_foreach_div_Tensor,
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
# Above this many elements per tensor the launch is bandwidth bound and the
# wider tile streams better; below it the call is dispatch bound and the
# narrower tile keeps more programs in flight.
_WIDE_NUMEL = 1 << 18
_NUM_WARPS = 4
_MAX_RANK = 8

# (device, variant key) -> compiled kernel; streams are resolved per launch.
_FAST = {}


@triton.jit
def _apply(x, y):
    """MUSA body: fp32 opmath, rounded once back to the storage dtype."""
    return (x.to(tl.float32) / y.to(tl.float32)).to(x.dtype)


@triton.jit
def _fused(
    x0,
    o0,
    n0,
    x1,
    o1,
    n1,
    x2,
    o2,
    n2,
    x3,
    o3,
    n3,
    x4,
    o4,
    n4,
    x5,
    o5,
    n5,
    x6,
    o6,
    n6,
    x7,
    o7,
    n7,
    x8,
    o8,
    n8,
    x9,
    o9,
    n9,
    x10,
    o10,
    n10,
    x11,
    o11,
    n11,
    x12,
    o12,
    n12,
    x13,
    o13,
    n13,
    x14,
    o14,
    n14,
    x15,
    o15,
    n15,
    yp,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    slot = tl.program_id(1)

    px = x0
    po = o0
    np = n0
    px = tl.where(slot == 1, x1, px)
    po = tl.where(slot == 1, o1, po)
    np = tl.where(slot == 1, n1, np)
    px = tl.where(slot == 2, x2, px)
    po = tl.where(slot == 2, o2, po)
    np = tl.where(slot == 2, n2, np)
    px = tl.where(slot == 3, x3, px)
    po = tl.where(slot == 3, o3, po)
    np = tl.where(slot == 3, n3, np)
    px = tl.where(slot == 4, x4, px)
    po = tl.where(slot == 4, o4, po)
    np = tl.where(slot == 4, n4, np)
    px = tl.where(slot == 5, x5, px)
    po = tl.where(slot == 5, o5, po)
    np = tl.where(slot == 5, n5, np)
    px = tl.where(slot == 6, x6, px)
    po = tl.where(slot == 6, o6, po)
    np = tl.where(slot == 6, n6, np)
    px = tl.where(slot == 7, x7, px)
    po = tl.where(slot == 7, o7, po)
    np = tl.where(slot == 7, n7, np)
    px = tl.where(slot == 8, x8, px)
    po = tl.where(slot == 8, o8, po)
    np = tl.where(slot == 8, n8, np)
    px = tl.where(slot == 9, x9, px)
    po = tl.where(slot == 9, o9, po)
    np = tl.where(slot == 9, n9, np)
    px = tl.where(slot == 10, x10, px)
    po = tl.where(slot == 10, o10, po)
    np = tl.where(slot == 10, n10, np)
    px = tl.where(slot == 11, x11, px)
    po = tl.where(slot == 11, o11, po)
    np = tl.where(slot == 11, n11, np)
    px = tl.where(slot == 12, x12, px)
    po = tl.where(slot == 12, o12, po)
    np = tl.where(slot == 12, n12, np)
    px = tl.where(slot == 13, x13, px)
    po = tl.where(slot == 13, o13, po)
    np = tl.where(slot == 13, n13, np)
    px = tl.where(slot == 14, x14, px)
    po = tl.where(slot == 14, o14, po)
    np = tl.where(slot == 14, n14, np)
    px = tl.where(slot == 15, x15, px)
    po = tl.where(slot == 15, o15, po)
    np = tl.where(slot == 15, n15, np)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < np
    x = tl.load(px + offs, mask=mask, other=0.0)
    y = tl.load(yp)
    tl.store(po + offs, _apply(x, y), mask=mask)


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
        "a0",
        "a1",
        "a2",
        "a3",
        "a4",
        "a5",
        "a6",
        "a7",
        "c0",
        "c1",
        "c2",
        "c3",
        "c4",
        "c5",
        "c6",
        "c7",
    ]
)
def _strided(
    x_ptr,
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
    a0,
    a1,
    a2,
    a3,
    a4,
    a5,
    a6,
    a7,
    c0,
    c1,
    c2,
    c3,
    c4,
    c5,
    c6,
    c7,
    yp,
    RANK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    rem = offs
    a_off = offs * 0
    c_off = offs * 0
    if RANK >= 8:
        q = rem % d7
        a_off += q * a7
        c_off += q * c7
        rem = rem // d7
    if RANK >= 7:
        q = rem % d6
        a_off += q * a6
        c_off += q * c6
        rem = rem // d6
    if RANK >= 6:
        q = rem % d5
        a_off += q * a5
        c_off += q * c5
        rem = rem // d5
    if RANK >= 5:
        q = rem % d4
        a_off += q * a4
        c_off += q * c4
        rem = rem // d4
    if RANK >= 4:
        q = rem % d3
        a_off += q * a3
        c_off += q * c3
        rem = rem // d3
    if RANK >= 3:
        q = rem % d2
        a_off += q * a2
        c_off += q * c2
        rem = rem // d2
    if RANK >= 2:
        q = rem % d1
        a_off += q * a1
        c_off += q * c1
        rem = rem // d1
    a_off += rem * a0
    c_off += rem * c0
    x = tl.load(x_ptr + a_off, mask=mask, other=0.0)
    y = tl.load(yp)
    tl.store(o_ptr + c_off, _apply(x, y), mask=mask)


def _config(numel):
    return _BLOCK_WIDE if numel >= _WIDE_NUMEL else _BLOCK


def _dispatch(args, k, n_blocks, block, key, other):
    """Launch the fused kernel, replaying the compiled variant once it exists."""
    drv = _driver.active
    device = drv.get_current_device()
    key = (device, key)
    entry = _FAST.get(key)
    if entry is not None:
        compiled = entry
        stream = drv.get_current_stream(device)
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
                other,
                block,
            )
            return
        except Exception:
            # A stale handle (driver reload, changed hook set) must not break
            # the call: drop it and take the ordinary launch path instead.
            _FAST.pop(key, None)

    compiled = _fused[(n_blocks, k)](*args, other, BLOCK=block, num_warps=_NUM_WARPS)
    try:
        _FAST[key] = compiled
    except Exception:
        _FAST.pop(key, None)


def _launch_chunk(chunk, dtype, other):
    """One launch for up to ``_SLOTS`` contiguous rows."""
    k = len(chunk)
    rows = list(chunk) + [chunk[0]] * (_SLOTS - k)
    args = []
    for s in range(_SLOTS):
        x, o, numel = rows[s]
        args += [x, o, numel if s < k else 0]

    numel_max = max(r[2] for r in chunk)
    block = _config(numel_max)
    n_blocks = (numel_max + block - 1) // block
    key = (
        dtype,
        block,
        (n_blocks, k),
        tuple(r[2] for r in chunk),
        tuple(r[0].data_ptr() & 15 for r in chunk),
        other.dtype,
        other.data_ptr() & 15,
    )

    _dispatch(args, k, n_blocks, block, key, other)


def _squeeze_dims(shape):
    """Logical dims with the size-1 axes dropped, plus the kept axis indices.

    Size-1 axes contribute no elements and any stride, so dropping them is
    always safe; adjacent axes are *not* merged, because a merge is only valid
    when every operand of the call agrees on it and the operands here may carry
    different strides for the same shape.
    """
    dims = []
    keep = []
    for i, size in enumerate(shape):
        if size != 1:
            dims.append(size)
            keep.append(i)
    if not dims:
        dims = [1]
        keep = [len(shape) - 1 if shape else 0]
    return dims[::-1], keep[::-1]


def _pad_strides(t, dims, keep):
    """``t``'s strides for the kept axes, padded out to ``_MAX_RANK``."""
    st = t.stride()
    out = [st[i] if i < len(st) else 0 for i in keep]
    return out + [0] * (_MAX_RANK - len(out))


def _launch_strided(t, out, numel, other):
    dims, keep = _squeeze_dims(tuple(t.shape))
    rank = len(dims)
    if rank > _MAX_RANK:
        raise RuntimeError("foreach _foreach_div_Tensor: unsupported tensor rank")
    pd = dims + [1] * (_MAX_RANK - rank)
    st_a = _pad_strides(t, dims, keep)
    st_c = _pad_strides(out, dims, keep)
    _strided[((numel + _BLOCK - 1) // _BLOCK,)](
        t,
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
        st_a[0],
        st_a[1],
        st_a[2],
        st_a[3],
        st_a[4],
        st_a[5],
        st_a[6],
        st_a[7],
        st_c[0],
        st_c[1],
        st_c[2],
        st_c[3],
        st_c[4],
        st_c[5],
        st_c[6],
        st_c[7],
        other,
        RANK=rank,
        BLOCK=_BLOCK,
        num_warps=_NUM_WARPS,
    )


def _use_triton_kernel(t):
    return (
        t.device.type == "musa"
        and t.dtype in _SUPPORTED_DTYPES
        and 0 < t.numel() < 2**31
        and (
            t.is_contiguous()
            or (
                sum(size != 1 for size in t.shape) <= _MAX_RANK
                and all(abs(stride) < 2**31 for stride in t.stride())
            )
        )
    )


def _foreach_div_Tensor(self, other):
    logger.debug("GEMS_MTHREADS _FOREACH_DIV_TENSOR")

    if not isinstance(self, (list, tuple)):
        return default_foreach_div_Tensor(self, other)
    nt = len(self)
    if nt == 0:
        raise RuntimeError("Tensor list must have at least one tensor.")
    if not isinstance(other, torch.Tensor) or other.numel() != 1:
        return default_foreach_div_Tensor(self, other)

    tensors = list(self)
    for t in tensors:
        if not isinstance(t, torch.Tensor) or not _use_triton_kernel(t):
            return default_foreach_div_Tensor(self, other)
    if not _use_triton_kernel(other) or not other.is_contiguous():
        return default_foreach_div_Tensor(self, other)

    device = tensors[0].device
    operands = tensors + [other]
    if any(t.device != device for t in operands):
        return default_foreach_div_Tensor(self, other)

    outs = [torch.empty_like(t) for t in tensors]

    device = tensors[0].device
    chunk = []
    dtype = None
    with torch_device_fn.device(device):
        for t, out in zip(tensors, outs):
            numel = t.numel()
            if numel == 0:
                continue
            if not (t.is_contiguous() and out.is_contiguous()):
                if chunk:
                    _launch_chunk(chunk, dtype, other)
                    chunk = []
                _launch_strided(t, out, numel, other)
                continue
            if dtype is not None and t.dtype != dtype:
                if chunk:
                    _launch_chunk(chunk, dtype, other)
                    chunk = []
            dtype = t.dtype
            chunk.append((t, out, numel))
            if len(chunk) == _SLOTS:
                _launch_chunk(chunk, dtype, other)
                chunk = []
        if chunk:
            _launch_chunk(chunk, dtype, other)

    return tuple(outs)
