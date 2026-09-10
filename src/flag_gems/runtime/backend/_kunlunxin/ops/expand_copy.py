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

from .copy import copy_

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))

# Packed (uint64-wide) flat-copy tile for the identity/contiguous expand path.
# Measured 2026-09-05 on P800/KL3 (dev6): per-element flat loads cap at
# ~505 GB/s (fp32 16M -> 0.133ms vs torch 0.080ms = 0.60x) because each lane
# carries a single element.  Viewing the buffer as uint64 makes one lane move
# ELEM consecutive elements (fp32->2, fp16/bf16->4), reaching the wide-vector
# block-DMA path at ~845 GB/s (~1.0x torch native).  Bit-exact by construction
# (pure byte copy, no value reinterpretation).
_PACK_BLOCK = 32768

# Tile for the broadcast (stride-0) gather/repeat/replicate kernels.
_BCAST_BLOCK = 4096


@triton.jit
def _expand_flat_pack_kernel(
    src_ptr,
    dst_ptr,
    n,
    ELEM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Packed flat block-DMA copy for the contiguous (identity/full) expand.

    ``src`` and ``dst`` are contiguous same-dtype buffers of ``n`` elements.
    Each lane copies 8//ELEM consecutive elements through a uint64 view so the
    load/store hit the wide-vector path that per-element lanes cannot reach.
    """
    pid = tl.program_id(0)
    epl = 8 // ELEM
    nwide = n // epl
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < nwide
    src64 = src_ptr.to(tl.pointer_type(tl.uint64))
    dst64 = dst_ptr.to(tl.pointer_type(tl.uint64))
    vals = tl.load(src64 + offs, mask=mask)
    tl.store(dst64 + offs, vals, mask=mask)


@triton.jit
def _expand_repeat_kernel(
    src_ptr,
    dst_ptr,
    src_numel,
    BLOCK: tl.constexpr,
):
    """Leading broadcast: output = [rep, src_numel], a contiguous ``src_numel``
    source block repeated ``rep`` times along a grid axis.  No per-element index
    arithmetic: program (c, r) copies source chunk c to output row r.  ``src``
    must be a contiguous block of ``src_numel`` elements.
    """
    pid_c = tl.program_id(0)
    pid_r = tl.program_id(1)
    offs = pid_c * BLOCK + tl.arange(0, BLOCK)
    mask = offs < src_numel
    vals = tl.load(src_ptr + offs, mask=mask)
    tl.store(dst_ptr + pid_r * src_numel + offs, vals, mask=mask)


@triton.jit
def _expand_replicate_kernel(
    src_ptr,
    dst_ptr,
    rep,
    BLOCK: tl.constexpr,
):
    """Trailing broadcast: output = [src_rows, rep], each output row replicates
    source row ``pid_r`` (a scalar) ``rep`` times.  No per-element index
    arithmetic: program (c, r) stores a BLOCK-wide run of the row's value.
    """
    pid_r = tl.program_id(0)
    pid_c = tl.program_id(1)
    offs = pid_c * BLOCK + tl.arange(0, BLOCK)
    mask = offs < rep
    val = tl.load(src_ptr + pid_r)
    val_vec = tl.broadcast_to(val, (BLOCK,))
    tl.store(dst_ptr + pid_r * rep + offs, val_vec, mask=mask)


@triton.jit
def _expand_gather_kernel(
    src_ptr,
    dst_ptr,
    n_elements,
    s0,
    s1,
    s2,
    s3,
    s4,
    s5,
    st0,
    st1,
    st2,
    st3,
    st4,
    st5,
    NDIM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """General broadcast gather+store.  ``dst`` is contiguous, ``src`` is the
    expand view with stride 0 on the size-1 dims.  The flat output index is
    decomposed into per-dim indices from the innermost dim outwards and
    multiplied by the (possibly 0) source strides.  This is the correctness
    fallback for mixed broadcast layouts (e.g. broadcast on a middle dim); the
    pure leading/trailing broadcast cases use the cheaper repeat/replicate
    kernels above.
    """
    pid = tl.program_id(0)
    out_offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = out_offs < n_elements
    offs = out_offs
    src_off = tl.zeros([BLOCK], dtype=tl.int32)
    if NDIM >= 6:
        c = offs % s5
        offs = offs // s5
        src_off += c * st5
    if NDIM >= 5:
        c = offs % s4
        offs = offs // s4
        src_off += c * st4
    if NDIM >= 4:
        c = offs % s3
        offs = offs // s3
        src_off += c * st3
    if NDIM >= 3:
        c = offs % s2
        offs = offs // s2
        src_off += c * st2
    if NDIM >= 2:
        c = offs % s1
        offs = offs // s1
        src_off += c * st1
    if NDIM >= 1:
        c = offs % s0
        src_off += c * st0
    vals = tl.load(src_ptr + src_off, mask=mask, other=0.0)
    tl.store(dst_ptr + out_offs, vals, mask=mask)


def _can_pack_flat(x: torch.Tensor, out: torch.Tensor) -> bool:
    """Eligible for the packed uint64 flat-copy fast path: contiguous same-shape
    copy with 2- or 4-byte elements, numel divisible by the packing factor and
    8-byte-aligned storage (uint64 view).  Anything else (e8m0, 1/8-byte
    elements, odd numel, misaligned base) falls back to copy_.
    """
    elem = x.element_size()
    epl = {2: 4, 4: 2}.get(elem, 1)
    if epl <= 1:
        return False
    n = out.numel()
    return (
        n % epl == 0
        and (x.data_ptr() % 8 == 0)
        and (out.data_ptr() % 8 == 0)
        and x.dtype == out.dtype
    )


def _broadcast_layout(vshape, vstride, x_contiguous):
    """Classify an expand view's broadcast layout.

    Returns (kind, a):
      ("repeat", src_numel)   leading broadcast: out = [rep, src_numel]
      ("replicate", src_rows) trailing broadcast: out = [src_rows, rep]
      ("gather", ndim)        mixed layout -> _expand_gather_kernel
    """
    ndim = len(vshape)
    nz = [d for d in range(ndim) if vstride[d] != 0]
    if not nz:
        # scalar source: everything broadcasts from one element
        return ("repeat", 1)
    if not x_contiguous:
        # non-contiguous source: non-broadcast dims are not a contiguous block,
        # repeat/replicate would misread storage -> general gather
        return ("gather", ndim)
    t = len(nz)
    if nz == list(range(t)):
        # non-broadcast dims form the leading block [0..t-1]
        src_rows = 1
        for d in range(t):
            src_rows *= vshape[d]
        return ("replicate", src_rows)
    if nz == list(range(ndim - t, ndim)):
        # non-broadcast dims form the trailing block [ndim-t..ndim-1]
        src_numel = 1
        for d in range(ndim - t, ndim):
            src_numel *= vshape[d]
        return ("repeat", src_numel)
    return ("gather", ndim)


def _launch_gather(vshape, vstride, src, dst, n):
    ndim = len(vshape)
    shapes = (tuple(vshape) + (1,) * (6 - ndim))
    strided = (tuple(vstride) + (0,) * (6 - ndim))
    grid = (triton.cdiv(n, _BCAST_BLOCK),)
    _expand_gather_kernel[grid](
        src,
        dst,
        n,
        *shapes,
        *strided,
        NDIM=ndim,
        BLOCK=_BCAST_BLOCK,
        num_warps=4,
    )


def _launch_broadcast(view, out, x_contiguous):
    """Dispatch a broadcast (stride-0) expand to the cheapest correct kernel."""
    vshape = tuple(view.shape)
    vstride = tuple(view.stride())
    n = out.numel()
    kind, a = _broadcast_layout(vshape, vstride, x_contiguous)
    if kind == "repeat":
        src_numel = a
        rep = n // src_numel if src_numel > 1 else n
        grid = (triton.cdiv(src_numel, _BCAST_BLOCK), rep)
        _expand_repeat_kernel[grid](
            view, out, src_numel, BLOCK=_BCAST_BLOCK, num_warps=4
        )
        return
    if kind == "replicate":
        src_rows = a
        rep = n // src_rows
        grid = (src_rows, triton.cdiv(rep, _BCAST_BLOCK))
        _expand_replicate_kernel[grid](
            view, out, rep, BLOCK=_BCAST_BLOCK, num_warps=4
        )
        return
    # general gather
    _launch_gather(vshape, vstride, view, out, n)


def expand_copy(x: torch.Tensor, size) -> torch.Tensor:
    """Kunlunxin override for aten::expand_copy.

    The generic ``flag_gems.ops.expand_copy`` falls back to the pointwise
    ``copy_``, whose flat path caps at ~505 GB/s (0.60x torch) and whose strided
    broadcast path collapses to ~40ms on large shapes.  This override routes:
      * contiguous (identity/full) sources through a packed uint64 flat block-DMA
        kernel (~845 GB/s, ~1.0x torch);
      * pure leading/trailing broadcast sources through no-index-arithmetic
        repeat/replicate kernels (bandwidth-bound, no div/mod);
      * mixed layouts through an int32 flat gather (correctness fallback).
    """
    logger.debug("GEMS_KUNLUNXIN EXPAND_COPY")

    # Convert size to tuple and handle -1 (meaning keep original size)
    size_tuple = tuple(-1 if s is None else s for s in size)

    # Ensure input is on the correct device
    device = x.device

    # Create output tensor with target shape on the same device
    out = torch.empty(size_tuple, dtype=x.dtype, device=device)

    # Handle empty tensors
    if out.numel() == 0:
        return out

    # Use torch.expand to get a broadcasted view with correct strides
    view = x.expand(size_tuple)

    # Ensure view is on the right device (expand preserves device)
    if view.device != device:
        view = view.to(device)

    if view.is_contiguous():
        # Same-shape (or full) copy: packed uint64 flat block-DMA when the
        # layout allows it, otherwise the proven bounded-tile copy_ flat path.
        if _can_pack_flat(view, out):
            elem = view.element_size()
            epl = 8 // elem
            nwide = out.numel() // epl
            _expand_flat_pack_kernel[(triton.cdiv(nwide, _PACK_BLOCK),)](
                view, out, out.numel(), ELEM=elem, BLOCK=_PACK_BLOCK, num_warps=32
            )
            return out
        return copy_(out, view)

    # Broadcast (stride-0 dims): dedicated repeat/replicate/gather kernels.
    _launch_broadcast(view, out, x.is_contiguous())
    return out
