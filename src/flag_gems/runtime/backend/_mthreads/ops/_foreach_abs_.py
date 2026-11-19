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

"""In-place ``aten::_foreach_abs_`` on the target device using Triton kernels.

Semantics reproduced from the target's own ATen build (probed on MUSA, torch
2.9.1): every tensor in the list is mutated in place, keeping its dtype,
shape, stride and storage; bool tensors are an identity (``abs`` on bool is
defined and returns the input values); the call returns ``None`` (the in-place
schema returns ``()``); an empty list raises ``RuntimeError``.

Nothing is allocated and nothing is copied: the kernels read and write through
each tensor's own ``data_ptr``, so a non-contiguous view keeps its layout.

Timing evidence (round 3, MUSA): large tensors sit on a size-invariant
~1.2 TB/s bandwidth plateau, while a 16-tensor list of 4096-element tensors
cost 330us of which ~19us per extra launch was pure dispatch.  So contiguous
tensors that share ``(dtype, numel)`` are processed by a single batched launch
instead of one launch each; everything else keeps the per-tensor path.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops._foreach_unary import _foreach_abs_ as default_foreach_abs_
from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)

# Moore Threads hardware does not support fp64.
_SUPPORTED_DTYPES = {torch.float16, torch.bfloat16, torch.float32}

_BLOCK = 1024
_WARPS = 4
_MAX_RANK = 8
_MAX_BATCH = 16


# ---------------------------------------------------------------------------
# kernels
# ---------------------------------------------------------------------------


@triton.jit
def _abs_flat_kernel(x_ptr, numel, BLOCK: tl.constexpr, USE_I64: tl.constexpr):
    """Sign-clearing absolute value over ``[0, numel)`` of one base pointer."""
    pid = tl.program_id(0)
    if USE_I64:
        offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    else:
        offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(x_ptr + offs, tl.abs(x), mask=mask)


@triton.jit
def _abs_slot(p, offs, mask):
    x = tl.load(p + offs, mask=mask)
    tl.store(p + offs, tl.abs(x), mask=mask)


@triton.jit
def _abs_batch_kernel(
    x0,
    x1,
    x2,
    x3,
    x4,
    x5,
    x6,
    x7,
    x8,
    x9,
    x10,
    x11,
    x12,
    x13,
    x14,
    x15,
    numel,
    nt,
    BLOCK: tl.constexpr,
    USE_I64: tl.constexpr,
):
    """``nt`` same-dtype, same-numel contiguous tensors in one launch.

    Every program touches the same offset window in each tensor, so no pointer
    table, no host-to-device metadata and no per-slot pointer branch is needed.
    Slots at or past ``nt`` are skipped, which lets one compiled kernel serve
    any group size without recompiling per list length.
    """
    pid = tl.program_id(0)
    if USE_I64:
        offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    else:
        offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    if 1 <= nt:
        _abs_slot(x0, offs, mask)
    if 2 <= nt:
        _abs_slot(x1, offs, mask)
    if 3 <= nt:
        _abs_slot(x2, offs, mask)
    if 4 <= nt:
        _abs_slot(x3, offs, mask)
    if 5 <= nt:
        _abs_slot(x4, offs, mask)
    if 6 <= nt:
        _abs_slot(x5, offs, mask)
    if 7 <= nt:
        _abs_slot(x6, offs, mask)
    if 8 <= nt:
        _abs_slot(x7, offs, mask)
    if 9 <= nt:
        _abs_slot(x8, offs, mask)
    if 10 <= nt:
        _abs_slot(x9, offs, mask)
    if 11 <= nt:
        _abs_slot(x10, offs, mask)
    if 12 <= nt:
        _abs_slot(x11, offs, mask)
    if 13 <= nt:
        _abs_slot(x12, offs, mask)
    if 14 <= nt:
        _abs_slot(x13, offs, mask)
    if 15 <= nt:
        _abs_slot(x14, offs, mask)
    if 16 <= nt:
        _abs_slot(x15, offs, mask)


@triton.jit
def _abs_strided_kernel(
    x_ptr,
    numel,
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
    RANK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Absolute value over a strided view addressed by (size, stride) pairs.

    Dims are supplied innermost-first (smallest stride first), so consecutive
    linear indices walk consecutive addresses in the common transpose case.
    """
    pid = tl.program_id(0)
    lin = pid * BLOCK + tl.arange(0, BLOCK)
    mask = lin < numel
    rem = lin.to(tl.int64)
    off = tl.zeros((BLOCK,), dtype=tl.int64)
    # Statically folded unrolling.  ``for i in range(RANK)`` is lowered to a
    # runtime scf.for by this Triton build, which makes tuple subscripting
    # illegal, and ``.to()`` on a scalar arg is illegal too because scalar args
    # equal to 0 or 1 become ``tl.constexpr``.  Constexpr ``if`` blocks avoid
    # both: int64 promotion comes from ``rem``/``off``.
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
        rem = rem // s7
    x = tl.load(x_ptr + off, mask=mask)
    tl.store(x_ptr + off, tl.abs(x), mask=mask)


# ---------------------------------------------------------------------------
# host side
# ---------------------------------------------------------------------------

_BOOL = torch.bool
_I64_LIMIT = 0x7FFFFFFF


def _check_list(self):
    if not isinstance(self, (list, tuple)):
        raise TypeError("argument 'self' must be tuple of Tensors")
    n = len(self)
    if n == 0:
        raise RuntimeError("Tensor list must have at least one tensor.")
    for i in range(n):
        if not isinstance(self[i], torch.Tensor):
            raise TypeError(
                f"expected Tensor as element {i} in argument 0, but got "
                f"{type(self[i]).__name__}"
            )
    return self


def _collapse(shape, stride):
    """Drop size-1 dims, order by ascending stride, merge contiguous runs."""
    dims = [(sz, st) for sz, st in zip(shape, stride) if sz != 1]
    dims.sort(key=lambda d: d[1])
    merged = []
    for sz, st in dims:
        if merged and merged[-1][1] * merged[-1][0] == st:
            psz, pst = merged[-1]
            merged[-1] = (psz * sz, pst)
        else:
            merged.append((sz, st))
    return merged


def _launch_flat(t, n):
    grid = ((n + _BLOCK - 1) // _BLOCK,)
    _abs_flat_kernel[grid](t, n, BLOCK=_BLOCK, USE_I64=n > _I64_LIMIT, num_warps=_WARPS)


def _launch_batch(ts, n):
    k = len(ts)
    if k == 1:
        _launch_flat(ts[0], n)
        return
    # Padded slots are never touched (``nt`` guards them) but must carry the
    # group's own dtype so the 16 pointer arguments share one pointer type.
    args = list(ts)
    args.extend([ts[0]] * (_MAX_BATCH - k))
    grid = ((n + _BLOCK - 1) // _BLOCK,)
    _abs_batch_kernel[grid](
        *args, n, k, BLOCK=_BLOCK, USE_I64=n > _I64_LIMIT, num_warps=_WARPS
    )


def _launch_strided(t, dims):
    n = 1
    for sz, _ in dims:
        n *= sz
    args = []
    for i in range(_MAX_RANK):
        args.append(dims[i][0] if i < len(dims) else 1)
    for i in range(_MAX_RANK):
        args.append(dims[i][1] if i < len(dims) else 0)
    grid = ((n + _BLOCK - 1) // _BLOCK,)
    _abs_strided_kernel[grid](
        t, n, *args, RANK=len(dims), BLOCK=_BLOCK, num_warps=_WARPS
    )


def _strided_one(t):
    dims = _collapse(t.shape, t.stride())
    if len(dims) <= _MAX_RANK:
        _launch_strided(t, dims)
        return
    # Rare: a view whose collapsed rank exceeds the kernel arity.  Peel one
    # real dim at a time (each step is a zero-copy view) until it fits.
    for i, sz in enumerate(t.shape):
        if sz != 1:
            for j in range(sz):
                _strided_one(t.select(i, j))
            return


def _use_triton_kernel(x):
    """Whether ``x`` can be handled by the MUSA kernels."""
    if not isinstance(x, torch.Tensor):
        return False
    if x.device.type != "musa" or x.dtype not in _SUPPORTED_DTYPES:
        return False
    if x.is_complex():
        return False
    return True


def _list_supported(seq):
    """A foreach call is specialized only if every tensor in it qualifies."""
    if seq is None:
        return True
    if isinstance(seq, torch.Tensor):
        return _use_triton_kernel(seq)
    for t in seq:
        if isinstance(t, torch.Tensor) and not _use_triton_kernel(t):
            return False
    return True


def _foreach_abs_(self):
    logger.debug("GEMS_MTHREADS _FOREACH_ABS_")

    if not _list_supported(self):
        return default_foreach_abs_(self)

    with torch_device_fn.device(
        self[0].device
        if isinstance(self, (list, tuple)) and self
        else torch.device("musa")
    ):
        return _launch_foreach_abs_(self)


def _launch_foreach_abs_(self):
    tensors = _check_list(self)
    groups = {}
    for t in tensors:
        dtype = t.dtype
        if dtype == _BOOL:
            # abs on bool is the identity: every stored value is already 0 or 1.
            continue
        if dtype.is_complex:
            raise RuntimeError("In-place abs is not supported for complex tensors.")
        n = t.numel()
        if n == 0:
            continue
        if not t.is_contiguous():
            _strided_one(t)
            continue
        key = (dtype, n)
        bucket = groups.get(key)
        if bucket is None:
            groups[key] = [t]
        else:
            bucket.append(t)

    for (_, n), ts in groups.items():
        k = len(ts)
        if k == 1:
            _launch_flat(ts[0], n)
        else:
            for i in range(0, k, _MAX_BATCH):
                _launch_batch(ts[i : i + _MAX_BATCH], n)

    # The in-place schema returns ``()``: the target's ATen build returns None
    # and the reference test asserts that observable behaviour.
    return None
