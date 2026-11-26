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
"""Triton implementation of ``aten::_foreach_reciprocal_``.

In-place multi-tensor unary map: every element of every tensor in the list is
replaced by its reciprocal, evaluated in the tensor's own precision, with the
storage / strides / dtype of each tensor preserved.  The call returns ``None``.

Layout strategy
---------------
Reciprocal is a pure element-wise map with no reduction, so the traversal order
over a tensor's elements is irrelevant.  That means a tensor which is
"non-overlapping and dense" (contiguous, transposed, permuted, offset views, ...)
can be addressed with a plain flat index over its storage span, without any
stride arithmetic and without materialising a copy.  Only genuinely gappy views
(``x[:, ::2]``) need per-dimension stride addressing, and those are the slow
path.

Launch strategy
---------------
The whole tensor list is fused into *one* kernel launch per dtype: up to
``SLOTS`` flat tensors of a dtype ride in one launch as individual pointer
arguments (no host->device metadata copy) on a ``(tiles, slot)`` 2-D grid, so a
program picks its tensor with a bounded branch chain.  Latency on this target is
dominated by per-launch host cost, so the number of launches must not grow with
``len(self)``, and the hottest variant carries no scalar arguments at all.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops._foreach_unary import (
    _foreach_reciprocal_ as default__foreach_reciprocal_,
)

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)

SLOTS = 16

# ---------------------------------------------------------------------------
# device math
# ---------------------------------------------------------------------------


@triton.jit
def _recip_value(x):
    """1/x with ATen's rounding: fp32/fp16/bf16 divide in fp32, fp64 in fp64."""
    if x.dtype == tl.float64:
        r = 1.0 / x
    else:
        r = (1.0 / x.to(tl.float32)).to(x.dtype)
    return r


@triton.jit(do_not_specialize=["n"])
def _recip_flat_kernel(x_ptr, n, BLOCK: tl.constexpr, EVEN: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    if EVEN:
        x = tl.load(x_ptr + offs)
        tl.store(x_ptr + offs, _recip_value(x))
    else:
        mask = offs < n
        x = tl.load(x_ptr + offs, mask=mask, other=1.0)
        tl.store(x_ptr + offs, _recip_value(x), mask=mask)


# NOTE: these slot helpers deliberately take no ``tl.constexpr`` parameter.
# In this Triton build a nested @triton.jit call that carries a constexpr
# argument and sits inside a runtime ``if`` mis-binds its positional arguments.


@triton.jit
def _slot_even(p, offs):
    x = tl.load(p + offs)
    tl.store(p + offs, _recip_value(x))


@triton.jit
def _recip_multi_kernel(
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
    BLOCK: tl.constexpr,
):
    """One launch for up to ``SLOTS`` equal-sized, block-aligned tensors."""
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    slot = tl.program_id(1)
    if slot == 0:
        _slot_even(x0, offs)
    elif slot == 1:
        _slot_even(x1, offs)
    elif slot == 2:
        _slot_even(x2, offs)
    elif slot == 3:
        _slot_even(x3, offs)
    elif slot == 4:
        _slot_even(x4, offs)
    elif slot == 5:
        _slot_even(x5, offs)
    elif slot == 6:
        _slot_even(x6, offs)
    elif slot == 7:
        _slot_even(x7, offs)
    elif slot == 8:
        _slot_even(x8, offs)
    elif slot == 9:
        _slot_even(x9, offs)
    elif slot == 10:
        _slot_even(x10, offs)
    elif slot == 11:
        _slot_even(x11, offs)
    elif slot == 12:
        _slot_even(x12, offs)
    elif slot == 13:
        _slot_even(x13, offs)
    elif slot == 14:
        _slot_even(x14, offs)
    else:
        _slot_even(x15, offs)


@triton.jit
def _recip_2d_kernel(x_ptr, n1, s0, s1, BLOCK: tl.constexpr):
    """Gappy view: ``off = i0 * s0 + i1 * s1`` with axis 1 innermost."""
    i0 = tl.program_id(1)
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n1
    ptrs = x_ptr + i0 * s0 + offs * s1
    x = tl.load(ptrs, mask=mask, other=1.0)
    tl.store(ptrs, _recip_value(x), mask=mask)


@triton.jit
def _recip_nd_kernel(x_ptr, n, n1, n2, n3, s0, s1, s2, s3, BLOCK: tl.constexpr):
    """Gappy view, up to four collapsed dimensions (outermost first)."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    i3 = offs % n3
    t = offs // n3
    i2 = t % n2
    t = t // n2
    i1 = t % n1
    i0 = t // n1
    off = i0 * s0 + i1 * s1 + i2 * s2 + i3 * s3
    x = tl.load(x_ptr + off, mask=mask, other=1.0)
    tl.store(x_ptr + off, _recip_value(x), mask=mask)


@triton.jit
def _recip_complex_flat_kernel(x_ptr, n, BLOCK: tl.constexpr, EVEN: tl.constexpr):
    """Dense complex tensor through its interleaved real/imag view."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    if EVEN:
        re = tl.load(x_ptr + 2 * offs)
        im = tl.load(x_ptr + 2 * offs + 1)
    else:
        mask = offs < n
        re = tl.load(x_ptr + 2 * offs, mask=mask, other=1.0)
        im = tl.load(x_ptr + 2 * offs + 1, mask=mask, other=0.0)
    if re.dtype == tl.float64:
        denom = re * re + im * im
        ore = re / denom
        oim = -im / denom
    else:
        ref = re.to(tl.float32)
        imf = im.to(tl.float32)
        denom = ref * ref + imf * imf
        ore = (ref / denom).to(re.dtype)
        oim = (-imf / denom).to(re.dtype)
    if EVEN:
        tl.store(x_ptr + 2 * offs, ore)
        tl.store(x_ptr + 2 * offs + 1, oim)
    else:
        tl.store(x_ptr + 2 * offs, ore, mask=mask)
        tl.store(x_ptr + 2 * offs + 1, oim, mask=mask)


# ---------------------------------------------------------------------------
# host helpers
# ---------------------------------------------------------------------------

_ATEN_TYPE_NAMES = {
    torch.int64: "Long",
    torch.int32: "Int",
    torch.int16: "Short",
    torch.int8: "Char",
    torch.uint8: "Byte",
    torch.bool: "Bool",
}


def _pick_block(n):
    if n >= 4096:
        return 1024
    block = 128
    while block < n:
        block *= 2
    return block


# ---------------------------------------------------------------------------
# host launch path
# ---------------------------------------------------------------------------
#
# Small lists are host-launch bound on this target: a 4096-element tensor and a
# 16-tensor list of them cost tens of microseconds, far more than the device
# work, so the per-call Python dispatch is what dominates.  Resolve the compiled
# kernel once per (dtype, BLOCK, constexpr, argument-alignment) signature and
# then call its launcher directly, falling back to the normal ``kernel[grid]``
# path when resolution is unavailable.  Alignment is part of the key because
# Triton bakes pointer alignment into the compiled variant.
#
_LAUNCHERS = {}
_MISSING = object()


def _launch_cached(fn, full_args, rt_args, grid, key):
    runner = _LAUNCHERS.get(key, _MISSING)
    if runner is _MISSING:
        runner = None
        try:
            g3 = (grid[0], grid[1] if len(grid) > 1 else 1, 1)
            compiled = fn.warmup(*full_args, grid=g3)
            if compiled is not None and hasattr(compiled, "__getitem__"):
                runner = compiled[g3]
                runner(*rt_args)
        except Exception:
            runner = None
        _LAUNCHERS[key] = runner
        if runner is not None:
            return
    elif runner is not None:
        runner(*rt_args)
        return
    fn[grid](*full_args)


def _is_dense(t):
    """Dense (non-overlapping, no gaps) test for a non-contiguous tensor."""
    if t.numel() <= 1:
        return True
    expected = 1
    for stride, size in sorted(
        (s, sz) for s, sz in zip(t.stride(), t.shape) if sz != 1
    ):
        if stride != expected:
            return False
        expected *= size
    return True


def _collapse(shape, strides):
    """Drop size-1 dims and merge adjacent dims that are contiguous in memory."""
    dims = [(s, st) for s, st in zip(shape, strides) if s != 1]
    out = []
    for s, st in reversed(dims):
        if out and out[-1][1] == s * st:
            ps, pst = out[-1]
            out[-1] = (ps * s, st)
        else:
            out.append((s, st))
    out.reverse()
    return out


def _launch_flat(t, n):
    block = _pick_block(n)
    even = n % block == 0
    _launch_cached(
        _recip_flat_kernel,
        (t, n, block, even),
        (t, n),
        ((n + block - 1) // block,),
        ("flat", t.dtype, block, even, t.data_ptr() % 16 == 0),
    )


def _launch_flat_group(group):
    nt = len(group)
    if nt == 1:
        t = group[0]
        _launch_flat(t, t.numel())
        return
    ns = [t.numel() for t in group]
    maxn = max(ns)
    block = _pick_block(maxn)
    if min(ns) == maxn and maxn % block == 0:
        args = list(group)
        pad = SLOTS - nt
        if pad:
            args += [group[0]] * pad
        _launch_cached(
            _recip_multi_kernel,
            args + [block],
            args,
            (maxn // block, nt),
            ("multi", group[0].dtype, block, tuple(t.data_ptr() % 16 for t in args)),
        )
        return
    # Heterogeneous or block-misaligned lists are not on the hot path; the
    # per-tensor launch keeps them correct without an extra kernel variant.
    for t in group:
        _launch_flat(t, t.numel())


def _launch_complex(t):
    n = t.numel()
    xr = torch.view_as_real(t)
    if not xr.is_contiguous():
        raise NotImplementedError(
            "foreach_reciprocal_: non-dense complex tensors are not supported"
        )
    block = _pick_block(n)
    _recip_complex_flat_kernel[((n + block - 1) // block,)](
        xr, n, block, n % block == 0
    )


def _launch_gappy(t):
    n = t.numel()
    dims = _collapse(t.shape, t.stride())
    if len(dims) > 4:
        for k in range(t.shape[0]):
            _launch_tensor(t.select(0, k))
        return
    if len(dims) == 1:
        block = _pick_block(n)
        _recip_2d_kernel[((n + block - 1) // block, 1)](t, n, 0, dims[0][1], block)
    elif len(dims) == 2:
        rows, s0 = dims[0]
        cols, s1 = dims[1]
        block = _pick_block(cols)
        _recip_2d_kernel[((cols + block - 1) // block, rows)](t, cols, s0, s1, block)
    else:
        dims = [(1, 0)] * (4 - len(dims)) + dims
        n1, s1 = dims[1]
        n2, s2 = dims[2]
        n3, s3 = dims[3]
        block = _pick_block(n)
        _recip_nd_kernel[((n + block - 1) // block,)](
            t, n, n1, n2, n3, dims[0][1], s1, s2, s3, block
        )


def _launch_tensor(t):
    n = t.numel()
    if n == 0:
        return
    if t.dtype.is_complex:
        _launch_complex(t)
    elif t.is_contiguous() or _is_dense(t):
        _launch_flat(t, n)
    else:
        _launch_gappy(t)


_OVERLAP_MSG = (
    "unsupported operation: more than one element of the written-to tensor "
    "refers to a single memory location. Please clone() the tensor before "
    "performing the operation."
)


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def _foreach_reciprocal_(self):
    logger.debug("GEMS_MTHREADS _FOREACH_RECIPROCAL_")
    _seq = self if isinstance(self, (list, tuple)) else [self]
    if any(not isinstance(t, torch.Tensor) or t.device.type != "musa" for t in _seq):
        return default__foreach_reciprocal_(self)
    if not isinstance(self, (list, tuple)):
        raise TypeError("argument 'self' must be tuple of Tensors")
    if len(self) == 0:
        raise RuntimeError("Tensor list must have at least one tensor.")

    groups = {}
    special = None
    for t in self:
        if not isinstance(t, torch.Tensor):
            raise TypeError("argument 'self' must be tuple of Tensors")
        dtype = t.dtype
        if dtype.is_floating_point:
            if t.is_contiguous():
                if t.numel() == 0:
                    continue
            else:
                if torch._debug_has_internal_overlap(t) == 1:
                    raise RuntimeError(_OVERLAP_MSG)
                if t.numel() == 0:
                    continue
                if not _is_dense(t):
                    if special is None:
                        special = []
                    special.append(t)
                    continue
            g = groups.get(dtype)
            if g is None:
                groups[dtype] = g = []
            g.append(t)
        elif dtype.is_complex:
            if not t.is_contiguous() and torch._debug_has_internal_overlap(t) == 1:
                raise RuntimeError(_OVERLAP_MSG)
            if special is None:
                special = []
            special.append(t)
        else:
            raise RuntimeError(
                "result type Float can't be cast to the desired output type "
                + _ATEN_TYPE_NAMES.get(dtype, str(dtype).capitalize())
            )

    for g in groups.values():
        if len(g) <= SLOTS:
            _launch_flat_group(g)
        else:
            for i in range(0, len(g), SLOTS):
                _launch_flat_group(g[i : i + SLOTS])

    if special is not None:
        for t in special:
            _launch_tensor(t)
    return None
