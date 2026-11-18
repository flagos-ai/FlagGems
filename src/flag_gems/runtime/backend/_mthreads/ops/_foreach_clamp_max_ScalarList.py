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

"""Out-of-place foreach clamp_max (``aten::_foreach_clamp_max`` Scalar/ScalarList).

Semantics: ``out_i = clamp_max(self_i, s_i)`` elementwise, i.e.
``min(self_i, s_i)`` with a NaN in either operand propagating.  Fresh storage
per output, ``self`` untouched, results returned as a tuple.

Strategy
--------
* One launch per call covers up to ``_SLOTS`` contiguous tensors.  The second
  grid axis selects the slot, and the selected band pointers are resolved with
  a scalar comparison chain, so every branch still passes a typed pointer
  argument (a pointer loaded from a metadata table would lose the vectorized
  access on this target).
* The first call compiling a given variant goes through the ordinary subscript
  launch; the resulting ``CompiledKernel`` is then captured together with the
  current stream and reused, which skips Triton's per-call argument binding,
  specialization refinement and cache lookup.  The variant key holds only
  input-independent properties plus pointer alignment.
* Lists longer than ``_SLOTS`` are chunked; non-contiguous tensors and
  unsupported dtypes fall back to the generic executor.
"""

import logging

import torch
import triton
import triton.language as tl
from triton.runtime import driver as _driver

from flag_gems.ops._foreach_binary import (
    _foreach_clamp_max_ScalarList as default_foreach_clamp_max_ScalarList,
)
from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)

# Moore Threads hardware does not support fp64.
_SUPPORTED_DTYPES = {torch.float16, torch.bfloat16, torch.float32}

_SLOTS = 16
_BLOCK = 1024
_NUM_WARPS = 4
# Above this many elements per tensor the launch is fully bandwidth bound and
# the wider tile streams better; below it the call is dispatch bound and the
# narrower tile keeps more programs in flight.
_WIDE_NUMEL = 1 << 18
_BLOCK_WIDE = 2048
_MAX_RANK = 8

# (dtype, block, grid, numels, scalars, alignment) -> (kernel, stream)
_FAST = {}


@triton.jit
def _clamp_one(x, sc):
    """``min(x, sc)`` in fp32 opmath with a propagating NaN."""
    return tl.minimum(
        x.to(tl.float32), sc.to(tl.float32), propagate_nan=tl.PropagateNan.ALL
    ).to(x.dtype)


@triton.jit
def _clamp_max_fused(
    i0,
    o0,
    n0,
    s0,
    i1,
    o1,
    n1,
    s1,
    i2,
    o2,
    n2,
    s2,
    i3,
    o3,
    n3,
    s3,
    i4,
    o4,
    n4,
    s4,
    i5,
    o5,
    n5,
    s5,
    i6,
    o6,
    n6,
    s6,
    i7,
    o7,
    n7,
    s7,
    i8,
    o8,
    n8,
    s8,
    i9,
    o9,
    n9,
    s9,
    i10,
    o10,
    n10,
    s10,
    i11,
    o11,
    n11,
    s11,
    i12,
    o12,
    n12,
    s12,
    i13,
    o13,
    n13,
    s13,
    i14,
    o14,
    n14,
    s14,
    i15,
    o15,
    n15,
    s15,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    slot = tl.program_id(1)

    ip = i0
    op = o0
    np = n0
    sp = s0
    ip = tl.where(slot == 1, i1, ip)
    op = tl.where(slot == 1, o1, op)
    np = tl.where(slot == 1, n1, np)
    sp = tl.where(slot == 1, s1, sp)
    ip = tl.where(slot == 2, i2, ip)
    op = tl.where(slot == 2, o2, op)
    np = tl.where(slot == 2, n2, np)
    sp = tl.where(slot == 2, s2, sp)
    ip = tl.where(slot == 3, i3, ip)
    op = tl.where(slot == 3, o3, op)
    np = tl.where(slot == 3, n3, np)
    sp = tl.where(slot == 3, s3, sp)
    ip = tl.where(slot == 4, i4, ip)
    op = tl.where(slot == 4, o4, op)
    np = tl.where(slot == 4, n4, np)
    sp = tl.where(slot == 4, s4, sp)
    ip = tl.where(slot == 5, i5, ip)
    op = tl.where(slot == 5, o5, op)
    np = tl.where(slot == 5, n5, np)
    sp = tl.where(slot == 5, s5, sp)
    ip = tl.where(slot == 6, i6, ip)
    op = tl.where(slot == 6, o6, op)
    np = tl.where(slot == 6, n6, np)
    sp = tl.where(slot == 6, s6, sp)
    ip = tl.where(slot == 7, i7, ip)
    op = tl.where(slot == 7, o7, op)
    np = tl.where(slot == 7, n7, np)
    sp = tl.where(slot == 7, s7, sp)
    ip = tl.where(slot == 8, i8, ip)
    op = tl.where(slot == 8, o8, op)
    np = tl.where(slot == 8, n8, np)
    sp = tl.where(slot == 8, s8, sp)
    ip = tl.where(slot == 9, i9, ip)
    op = tl.where(slot == 9, o9, op)
    np = tl.where(slot == 9, n9, np)
    sp = tl.where(slot == 9, s9, sp)
    ip = tl.where(slot == 10, i10, ip)
    op = tl.where(slot == 10, o10, op)
    np = tl.where(slot == 10, n10, np)
    sp = tl.where(slot == 10, s10, sp)
    ip = tl.where(slot == 11, i11, ip)
    op = tl.where(slot == 11, o11, op)
    np = tl.where(slot == 11, n11, np)
    sp = tl.where(slot == 11, s11, sp)
    ip = tl.where(slot == 12, i12, ip)
    op = tl.where(slot == 12, o12, op)
    np = tl.where(slot == 12, n12, np)
    sp = tl.where(slot == 12, s12, sp)
    ip = tl.where(slot == 13, i13, ip)
    op = tl.where(slot == 13, o13, op)
    np = tl.where(slot == 13, n13, np)
    sp = tl.where(slot == 13, s13, sp)
    ip = tl.where(slot == 14, i14, ip)
    op = tl.where(slot == 14, o14, op)
    np = tl.where(slot == 14, n14, np)
    sp = tl.where(slot == 14, s14, sp)
    ip = tl.where(slot == 15, i15, ip)
    op = tl.where(slot == 15, o15, op)
    np = tl.where(slot == 15, n15, np)
    sp = tl.where(slot == 15, s15, sp)

    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < np
    x = tl.load(ip + offs, mask=mask, other=0.0)
    tl.store(op + offs, _clamp_one(x, sp), mask=mask)


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
def _clamp_max_strided(
    i_ptr,
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
    x = tl.load(i_ptr + src, mask=mask, other=0.0)
    tl.store(o_ptr + offs, _clamp_one(x, s), mask=mask)


def _config(numel):
    return (_BLOCK_WIDE, _NUM_WARPS) if numel >= _WIDE_NUMEL else (_BLOCK, _NUM_WARPS)


def _dispatch(args, k, numels, n_blocks, block, dtype, key, grid):
    """Launch the fused kernel, reusing the compiled variant once it exists."""
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
            )
            return
        except Exception:
            # A stale handle (driver reload, changed hook set) must not break
            # the call: drop it and take the ordinary launch path instead.
            _FAST.pop(key, None)

    compiled = _clamp_max_fused[(n_blocks, k)](*args, BLOCK=block, num_warps=_NUM_WARPS)
    try:
        drv = _driver.active
        _FAST[key] = (compiled, drv.get_current_stream(drv.get_current_device()))
    except Exception:
        _FAST.pop(key, None)


def _launch_chunk(chunk, numels, scalars, dtype, device):
    """One launch for up to ``_SLOTS`` contiguous (in, out, numel, scalar) rows."""
    k = len(chunk)
    pad = chunk[0]
    rows = list(chunk) + [pad] * (_SLOTS - k)
    args = []
    for s in range(_SLOTS):
        in_t, out_t, numel, val = rows[s]
        args += [in_t, out_t, numel, val]

    numel_max = max(numels)
    block, _ = _config(numel_max)
    n_blocks = (numel_max + block - 1) // block
    grid = (n_blocks, k)
    key = (
        dtype,
        block,
        grid,
        tuple(numels),
        tuple(scalars),
        tuple(t.data_ptr() & 15 for t, _, _, _ in chunk),
    )
    with torch_device_fn.device(device):
        _dispatch(args, k, numels, n_blocks, block, dtype, key, grid)


def _launch_single(in_t, out_t, numel, val):
    block, warps = _config(numel)
    grid = ((numel + block - 1) // block, 1)
    _clamp_max_fused[grid](
        in_t,
        out_t,
        numel,
        val,
        in_t,
        out_t,
        0,
        0.0,
        in_t,
        out_t,
        0,
        0.0,
        in_t,
        out_t,
        0,
        0.0,
        in_t,
        out_t,
        0,
        0.0,
        in_t,
        out_t,
        0,
        0.0,
        in_t,
        out_t,
        0,
        0.0,
        in_t,
        out_t,
        0,
        0.0,
        in_t,
        out_t,
        0,
        0.0,
        in_t,
        out_t,
        0,
        0.0,
        in_t,
        out_t,
        0,
        0.0,
        in_t,
        out_t,
        0,
        0.0,
        in_t,
        out_t,
        0,
        0.0,
        in_t,
        out_t,
        0,
        0.0,
        in_t,
        out_t,
        0,
        0.0,
        in_t,
        out_t,
        0,
        0.0,
        in_t,
        out_t,
        0,
        0.0,
        BLOCK=block,
        num_warps=warps,
    )


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


def _launch_strided(in_t, out_t, numel, val):
    dims, strides = _collapse(in_t)
    rank = len(dims)
    if rank > _MAX_RANK:
        raise RuntimeError("foreach clamp_max: unsupported tensor rank")
    block = _BLOCK
    pd = dims + [1] * (_MAX_RANK - rank)
    pt = strides + [0] * (_MAX_RANK - rank)
    _clamp_max_strided[((numel + block - 1) // block,)](
        in_t,
        out_t,
        numel,
        val,
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
        BLOCK=block,
        num_warps=_NUM_WARPS,
    )


def _use_triton_kernel(t, scalars):
    if t.device.type != "musa" or t.dtype not in _SUPPORTED_DTYPES:
        return False
    if not isinstance(scalars, (int, float)):
        return False
    return True


def _foreach_clamp_max_ScalarList(self, scalars):
    logger.debug("GEMS_MTHREADS _FOREACH_CLAMP_MAX_SCALARLIST")

    if not isinstance(self, (list, tuple)):
        raise TypeError(
            "_foreach_clamp_max() received an invalid combination of arguments - "
            "expected a tuple of Tensors"
        )
    nt = len(self)
    if nt == 0:
        raise RuntimeError("Tensor list must have at least one tensor.")

    per_tensor = isinstance(scalars, (list, tuple))
    if per_tensor:
        if len(scalars) != nt:
            raise RuntimeError(
                "Tensor list must have same number of elements as scalar list."
            )
    elif isinstance(scalars, torch.Tensor):
        if scalars.dim() != 0:
            raise TypeError("_foreach_clamp_max() scalar must be a Number")
        scalars = scalars.item()

    values = list(scalars) if per_tensor else [scalars] * nt
    tensors = list(self)

    # The specialized path needs every operand on the MUSA device, in a dtype
    # the kernels handle, and with a host scalar; anything else keeps the
    # generic executor's semantics (including its dtype-error messages).
    for t, v in zip(tensors, values):
        if not _use_triton_kernel(t, v):
            return default_foreach_clamp_max_ScalarList(self, scalars)

    outs = [torch.empty_like(t) for t in tensors]
    device = tensors[0].device

    chunk = []
    numels = []
    scalars_ = []
    dtype = None
    with torch_device_fn.device(device):
        for t, o, v in zip(tensors, outs, values):
            numel = t.numel()
            if numel == 0:
                continue
            if not t.is_contiguous():
                if chunk:
                    _launch_chunk(chunk, numels, scalars_, dtype, device)
                    chunk, numels, scalars_ = [], [], []
                _launch_strided(t, o, numel, float(v))
                continue
            if dtype is not None and t.dtype is not dtype:
                if chunk:
                    _launch_chunk(chunk, numels, scalars_, dtype, device)
                    chunk, numels, scalars_ = [], [], []
            dtype = t.dtype
            chunk.append((t, o, numel, float(v)))
            numels.append(numel)
            scalars_.append(float(v))
            if len(chunk) == _SLOTS:
                _launch_chunk(chunk, numels, scalars_, dtype, device)
                chunk, numels, scalars_ = [], [], []
        if chunk:
            _launch_chunk(chunk, numels, scalars_, dtype, device)

    return tuple(outs)
