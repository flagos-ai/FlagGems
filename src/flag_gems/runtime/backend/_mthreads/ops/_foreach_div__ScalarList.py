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

"""In-place foreach division by a scalar list (``_foreach_div_.ScalarList``).

Semantics: ``self[i] /= scalars[i]`` for every list entry, written back into the
entry's own storage.  Integral operands are promoted by ATen and rejected in
place; non-contiguous views are addressed directly rather than staged.

Strategy
--------
* One launch covers up to ``_SLOTS`` contiguous entries.  The second grid axis
  selects the entry, and the per-entry pointer, element count and divisor are
  resolved with a scalar comparison chain, so each branch still receives a
  typed pointer argument (an address loaded out of a metadata table would be
  scalarized on this target).
* The first call compiling a variant takes the ordinary subscript launch; the
  resulting ``CompiledKernel`` is then captured together with the current
  stream and replayed, which skips argument binding, specialization refinement
  and the kernel-cache lookup that ``JITFunction.run`` performs per call.
* Non-contiguous entries use a stride-decomposed kernel; unsupported dtypes,
  devices or layouts fall back to the generic executor.
"""

import logging

import torch
import triton
import triton.language as tl
from triton.runtime import driver as _driver

from flag_gems.ops._foreach_binary import (
    _foreach_div__ScalarList as default__foreach_div__ScalarList,
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

# (dtype, block, grid, numels, divisors, alignment) -> (kernel, stream)
_FAST = {}


@triton.jit
def _apply(x, d):
    """``x / d`` in fp32 opmath, rounded once to the storage dtype."""
    return (x.to(tl.float32) / d.to(tl.float32)).to(x.dtype)


@triton.jit
def _fused(
    x0,
    n0,
    d0,
    x1,
    n1,
    d1,
    x2,
    n2,
    d2,
    x3,
    n3,
    d3,
    x4,
    n4,
    d4,
    x5,
    n5,
    d5,
    x6,
    n6,
    d6,
    x7,
    n7,
    d7,
    x8,
    n8,
    d8,
    x9,
    n9,
    d9,
    x10,
    n10,
    d10,
    x11,
    n11,
    d11,
    x12,
    n12,
    d12,
    x13,
    n13,
    d13,
    x14,
    n14,
    d14,
    x15,
    n15,
    d15,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    slot = tl.program_id(1)

    px = x0
    np = n0
    dp = d0
    px = tl.where(slot == 1, x1, px)
    np = tl.where(slot == 1, n1, np)
    dp = tl.where(slot == 1, d1, dp)
    px = tl.where(slot == 2, x2, px)
    np = tl.where(slot == 2, n2, np)
    dp = tl.where(slot == 2, d2, dp)
    px = tl.where(slot == 3, x3, px)
    np = tl.where(slot == 3, n3, np)
    dp = tl.where(slot == 3, d3, dp)
    px = tl.where(slot == 4, x4, px)
    np = tl.where(slot == 4, n4, np)
    dp = tl.where(slot == 4, d4, dp)
    px = tl.where(slot == 5, x5, px)
    np = tl.where(slot == 5, n5, np)
    dp = tl.where(slot == 5, d5, dp)
    px = tl.where(slot == 6, x6, px)
    np = tl.where(slot == 6, n6, np)
    dp = tl.where(slot == 6, d6, dp)
    px = tl.where(slot == 7, x7, px)
    np = tl.where(slot == 7, n7, np)
    dp = tl.where(slot == 7, d7, dp)
    px = tl.where(slot == 8, x8, px)
    np = tl.where(slot == 8, n8, np)
    dp = tl.where(slot == 8, d8, dp)
    px = tl.where(slot == 9, x9, px)
    np = tl.where(slot == 9, n9, np)
    dp = tl.where(slot == 9, d9, dp)
    px = tl.where(slot == 10, x10, px)
    np = tl.where(slot == 10, n10, np)
    dp = tl.where(slot == 10, d10, dp)
    px = tl.where(slot == 11, x11, px)
    np = tl.where(slot == 11, n11, np)
    dp = tl.where(slot == 11, d11, dp)
    px = tl.where(slot == 12, x12, px)
    np = tl.where(slot == 12, n12, np)
    dp = tl.where(slot == 12, d12, dp)
    px = tl.where(slot == 13, x13, px)
    np = tl.where(slot == 13, n13, np)
    dp = tl.where(slot == 13, d13, dp)
    px = tl.where(slot == 14, x14, px)
    np = tl.where(slot == 14, n14, np)
    dp = tl.where(slot == 14, d14, dp)
    px = tl.where(slot == 15, x15, px)
    np = tl.where(slot == 15, n15, np)
    dp = tl.where(slot == 15, d15, dp)

    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < np
    v = tl.load(px + offs, mask=mask, other=0.0)
    tl.store(px + offs, _apply(v, dp), mask=mask)


@triton.jit(
    do_not_specialize=[
        "n",
        "d",
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
    n,
    d,
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
    v = tl.load(x_ptr + src, mask=mask, other=0.0)
    tl.store(x_ptr + src, _apply(v, d), mask=mask)


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


def _launch_chunk(rows, divisors, dtype):
    """One launch for up to ``_SLOTS`` contiguous (tensor, numel, divisor) rows."""
    k = len(rows)
    padded = list(rows) + [rows[0]] * (_SLOTS - k)
    args = []
    for s in range(_SLOTS):
        tensor, numel, _ = padded[s]
        args += [tensor, numel if s < k else 0, divisors[s] if s < k else 1.0]

    numel_max = max(numel for _, numel, _ in rows)
    block = _config(numel_max)
    n_blocks = (numel_max + block - 1) // block
    key = (
        dtype,
        block,
        (n_blocks, k),
        tuple(numel for _, numel, _ in rows),
        tuple(divisors),
        tuple(t.data_ptr() & 15 for t, _, _ in rows),
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


def _launch_strided(t, numel, divisor):
    dims, strides = _collapse(t)
    rank = len(dims)
    if rank > _MAX_RANK:
        raise RuntimeError("foreach div: unsupported tensor rank")
    pd = dims + [1] * (_MAX_RANK - rank)
    pt = strides + [0] * (_MAX_RANK - rank)
    _strided[((numel + _BLOCK - 1) // _BLOCK,)](
        t,
        numel,
        divisor,
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


def _is_number(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _foreach_div__ScalarList(self, scalars):
    logger.debug("GEMS_MTHREADS _FOREACH_DIV__SCALARLIST")

    if not isinstance(self, (list, tuple)):
        return default__foreach_div__ScalarList(self, scalars)

    nt = len(self)
    per_tensor = isinstance(scalars, (list, tuple))
    if per_tensor and len(scalars) != nt:
        raise RuntimeError(
            "Tensor list length %d does not match scalar list length %d"
            % (nt, len(scalars))
        )

    tensors = list(self)
    for t in tensors:
        if not isinstance(t, torch.Tensor) or not _use_triton_kernel(t):
            return default__foreach_div__ScalarList(self, scalars)
    values = list(scalars) if per_tensor else [scalars] * nt
    for v in values:
        if not _is_number(v):
            return default__foreach_div__ScalarList(self, scalars)

    device = tensors[0].device
    rows = []
    divisors = []
    dtype = None
    with torch_device_fn.device(device):
        for t, v in zip(tensors, values):
            numel = t.numel()
            if numel == 0:
                continue
            divisor = float(v)
            if not t.is_contiguous():
                if rows:
                    _launch_chunk(rows, divisors, dtype)
                    rows, divisors = [], []
                _launch_strided(t, numel, divisor)
                continue
            if dtype is not None and t.dtype is not dtype:
                if rows:
                    _launch_chunk(rows, divisors, dtype)
                    rows, divisors = [], []
            dtype = t.dtype
            rows.append((t, numel, divisor))
            divisors.append(divisor)
            if len(rows) == _SLOTS:
                _launch_chunk(rows, divisors, dtype)
                rows, divisors = [], []
        if rows:
            _launch_chunk(rows, divisors, dtype)

    return None
