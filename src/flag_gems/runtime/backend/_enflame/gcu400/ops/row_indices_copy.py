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

"""Triton implementation of ``aten::row_indices_copy`` for GCU400.

``row_indices_copy`` materializes a fresh contiguous copy of the stored
row-indices buffer of a column-compressed sparse tensor (sparse_csc /
sparse_bsc).  The buffer is 1-D for plain layouts and ``batch_dims + (nnz,)``
for batched ones, and the copy honours an ``out=`` destination of any stride.

GCU400 adaptation
-----------------
* torch_gcu ships no ``SparseCsrgcu`` accessor kernels, so the source buffer
  is resolved through the process-wide component cache that
  ``values_copy.py`` attaches to ``torch`` for tensors built by the wrapped
  ``torch.sparse_*`` factories (P21); anything outside that cache falls back
  to the dispatcher accessor, which the ``sparse_compressed_tensor.py``
  bridge serves for every GCU compressed sparse tensor.
* A contiguous destination (every allocation path and nearly every ``out=``)
  is served by a flat kernel: the row-indices buffer is always stored
  row-major contiguous, so the generic rank-loop div/mod decode degenerates
  to the identity and is dropped, together with the per-call
  ``torch.tensor(layout, int64, device=gcu)`` host-to-device materialization
  of the stride vector, which costs ~85us per call -- an order of magnitude
  more than the copy itself.
* A strided destination scatters through a kernel whose sizes and strides
  travel as scalar launch parameters (the ``values_copy.py``
  ``_copy_nd_kernel`` idiom), so this path is also free of any host-to-device
  tensor materialization.  The arithmetic stays in the int32 safe domain
  whenever a host-side bound check proves every offset fits; only genuinely
  huge destinations fall back to int64 device arithmetic.
* Destinations of rank > 6 (6+ sparse batch dimensions; unreachable in
  practice) keep the generic flat-stride decode loop over an int64 layout
  buffer, preserving the operator's full generality.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)

# Launch-bound copy: a 2048-lane block with 8 warps measured fastest across
# the buffer sizes typical of sparse index arrays (1k-64k elements) -- fatter
# blocks keep winning on GCU400 down to 32 CTAs at 64k lanes (4.1us vs 5.5us
# at BLOCK=1024), while smaller blocks lose steadily (BLOCK=256: 15us).
_FLAT_BLOCK = 2048
_FLAT_WARPS = 8
_SCATTER_BLOCK = 1024
_SCATTER_WARPS = 4
_INT32_LIMIT = 2**31

# The component cache that ``values_copy.py`` installs on ``torch`` for
# GCU-resident compressed sparse tensors built by the wrapped factories.
# Reading it directly skips the dispatcher roundtrip of the accessor; the
# accessor itself remains the fallback for tensors outside the cache.
_COMPONENT_STATE_ATTR = "_flag_gems_gcu400_sparse_component_state"
_COL_COMPRESSED_LAYOUTS = (torch.sparse_csc, torch.sparse_bsc)


@triton.jit
def _row_indices_copy_flat_kernel(src_ptr, dst_ptr, n_elements, BLOCK: tl.constexpr):
    # The source buffer is always stored row-major contiguous (plain and
    # batched CSC/BSC alike), so a flat unit-stride read is exact regardless
    # of rank, and a contiguous destination makes the address decode the
    # identity.  The source is read exactly once, so its load bypasses L1.
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    vals = tl.load(src_ptr + offs, mask=mask, cache_modifier=".cg")
    tl.store(dst_ptr + offs, vals, mask=mask)


@triton.jit
def _row_indices_copy_scatter_kernel(
    src_ptr,
    dst_ptr,
    n_elements,
    d0,
    d1,
    d2,
    d3,
    d4,
    d5,
    e0,
    e1,
    e2,
    e3,
    e4,
    BLOCK: tl.constexpr,
    RANK: tl.constexpr,
    WIDE: tl.constexpr,
):
    # Copy ``n_elements`` values read flat from the row-major contiguous
    # source into a destination whose address is decoded from the flat index
    # through the destination's own sizes (``e*``, innermost first) and
    # strides (``d*``, innermost first).  All of them arrive as scalar launch
    # parameters, so no stride vector ever crosses the host-device boundary.
    # ``WIDE`` selects int64 accumulators for destinations whose offsets
    # exceed the int32 safe domain; everything else computes in int32.
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    if WIDE:
        cur = tl.where(mask, offs, 0).to(tl.int64)
        dst_off = tl.zeros([BLOCK], dtype=tl.int64)
    else:
        cur = tl.where(mask, offs, 0).to(tl.int32)
        dst_off = tl.zeros([BLOCK], dtype=tl.int32)
    if RANK >= 2:
        i0 = cur % e0
        cur = cur // e0
        dst_off += i0 * d0
    if RANK >= 3:
        i1 = cur % e1
        cur = cur // e1
        dst_off += i1 * d1
    if RANK >= 4:
        i2 = cur % e2
        cur = cur // e2
        dst_off += i2 * d2
    if RANK >= 5:
        i3 = cur % e3
        cur = cur // e3
        dst_off += i3 * d3
    if RANK >= 6:
        i4 = cur % e4
        cur = cur // e4
        dst_off += i4 * d4
    if RANK == 1:
        dst_off += cur * d0
    elif RANK == 2:
        dst_off += cur * d1
    elif RANK == 3:
        dst_off += cur * d2
    elif RANK == 4:
        dst_off += cur * d3
    elif RANK == 5:
        dst_off += cur * d4
    else:
        dst_off += cur * d5
    tl.store(dst_ptr + dst_off, tl.load(src_ptr + offs, mask=mask), mask=mask)


@triton.jit
def _row_indices_copy_decode_kernel(
    src_ptr,
    dst_ptr,
    layout_ptr,  # flat sizes followed by flat strides of the logical shape
    n_elements,
    RANK_DYN: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Generic fallback for destinations of rank > 6 (more sparse batch
    # dimensions than the scalar-parameter kernel covers): the flat offset is
    # decoded against the full stride vector loaded from a device buffer.
    # Only reachable for pathologically batched sparse layouts.
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    flat_offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = flat_offsets < n_elements

    vals = tl.load(src_ptr + flat_offsets, mask=mask)

    cur = tl.where(mask, flat_offsets, 0).to(tl.int64)
    dst_offsets = tl.zeros((BLOCK_SIZE,), dtype=tl.int64)
    for i in tl.static_range(RANK_DYN - 1, -1, -1):
        dim_size = tl.load(layout_ptr + i).to(tl.int64)
        dim_stride = tl.load(layout_ptr + RANK_DYN + i).to(tl.int64)
        dst_offsets += (cur % dim_size) * dim_stride
        cur = cur // dim_size
    tl.store(dst_ptr + dst_offsets, vals, mask=mask)


def _row_indices_source(self: torch.Tensor) -> torch.Tensor:
    """The stored row-indices buffer of a GCU column-compressed tensor.

    Table first: tensors built by the wrapped ``torch.sparse_*`` factories
    resolve straight from the component cache that ``values_copy.py`` keeps
    on ``torch``, skipping the dispatcher roundtrip.  Everything else (other
    backends, CPU, or cache misses) takes the accessor unchanged, which the
    ``sparse_compressed_tensor.py`` bridge serves on the GCU sparse keys.
    The cache holds each tensor inside its own entry, so a live tensor's id
    can only ever hit the entry it was recorded under.
    """
    state = getattr(torch, _COMPONENT_STATE_ATTR, None)
    if state is not None:
        entry = state.components.get(id(self))
        if entry is not None:
            return entry[2]
    return self.row_indices()


def _launch_flat(src, dst, n_elements):
    grid = ((n_elements + _FLAT_BLOCK - 1) // _FLAT_BLOCK,)
    with torch_device_fn.device(src.device):
        _row_indices_copy_flat_kernel[grid](
            src, dst, n_elements, BLOCK=_FLAT_BLOCK, num_warps=_FLAT_WARPS
        )


def _launch_scatter(src, dst, n_elements):
    # Sizes and strides are handed to the kernel innermost-first; a rank-1
    # destination uses d0 only and leaves the extents unused.
    sizes = list(dst.shape)
    strides = list(dst.stride())
    rank = len(sizes)
    extents = list(reversed(sizes))
    dst_strides = list(reversed(strides))
    # int32 stays the arithmetic domain only when every decoded address, the
    # element count and every size/stride value itself fit a signed int32;
    # one element past that and the kernel switches to int64 accumulators.
    max_offset = sum(
        (extent - 1) * abs(stride) for extent, stride in zip(extents, dst_strides)
    )
    wide = (
        n_elements >= _INT32_LIMIT
        or max_offset >= _INT32_LIMIT
        or any(abs(v) >= _INT32_LIMIT for v in sizes + strides)
    )
    # The kernel mods the five innermost extents; the outermost index is the
    # remainder and needs no extent slot.
    padded_strides = dst_strides + [0] * (6 - rank)
    padded_extents = (extents[:5] + [1] * 5)[:5]
    grid = ((n_elements + _SCATTER_BLOCK - 1) // _SCATTER_BLOCK,)
    with torch_device_fn.device(src.device):
        _row_indices_copy_scatter_kernel[grid](
            src,
            dst,
            n_elements,
            *padded_strides,
            *padded_extents,
            BLOCK=_SCATTER_BLOCK,
            RANK=rank,
            WIDE=wide,
            num_warps=_SCATTER_WARPS,
        )


def _row_indices_copy_impl(self: torch.Tensor, *, out=None):
    # Extract the row_indices buffer of a column-compressed sparse tensor
    # (sparse_csc / sparse_bsc) and return a contiguous strided copy.
    if self.layout not in _COL_COMPRESSED_LAYOUTS:
        raise RuntimeError(
            "row_indices expected sparse column compressed tensor layout but got "
            f"{str(self.layout).replace('torch.', '')}"
        )

    src = _row_indices_source(self)
    # The sparse constructor keeps whatever view it is handed, so a
    # defensively materialized source covers the (unobserved) strided-buffer
    # construction; a no-op for the contiguous common case.
    if not src.is_contiguous():
        src = src.contiguous()
    if out is not None:
        if out.dtype != src.dtype:
            raise RuntimeError(
                f"Expected out tensor to have dtype {src.dtype}, but got {out.dtype} instead"
            )
        # PyTorch resizes mismatched-shape outs to the required shape.
        if out.numel() != src.numel() or out.shape != src.shape:
            out.resize_(src.shape)
        dst = out
    else:
        # Allocate via ``empty_strided`` with contiguous strides to bypass the
        # GEMS ``aten::empty.memory_format`` override, which launches a Triton
        # zero-init kernel. The copy kernel overwrites every element under a
        # bounds mask, so an uninitialized allocation is safe.
        out_shape = tuple(src.shape)
        contig_strides = []
        stride = 1
        for dim_sz in reversed(out_shape):
            contig_strides.append(stride)
            stride *= dim_sz
        contig_strides.reverse()
        dst = torch.empty_strided(
            out_shape, contig_strides, dtype=src.dtype, device=src.device
        )

    n_elements = src.numel()
    if n_elements == 0:
        return dst

    # Contiguous destinations (freshly allocated ones always are) take the
    # flat fast path: no address decode, no stride vector on the device.
    if dst.is_contiguous():
        _launch_flat(src, dst, n_elements)
        return dst
    if dst.dim() <= 6:
        _launch_scatter(src, dst, n_elements)
        return dst
    # rank > 6: generic decode over a device-side layout buffer.
    layout = [int(d) for d in dst.shape] + [int(s) for s in dst.stride()]
    if dst.dim() == 0:
        layout = [1, 1]
    rank = len(layout) // 2
    layout_buf = torch.tensor(layout, dtype=torch.int64, device=src.device)
    BLOCK_SIZE = 4096
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    with torch_device_fn.device(src.device):
        _row_indices_copy_decode_kernel[grid](
            src, dst, layout_buf, n_elements, RANK_DYN=rank, BLOCK_SIZE=BLOCK_SIZE
        )
    return dst


def row_indices_copy(self: torch.Tensor):
    logger.debug("GEMS ROW_INDICES_COPY")
    return _row_indices_copy_impl(self)


def row_indices_copy_out(self: torch.Tensor, *, out: torch.Tensor):
    logger.debug("GEMS ROW_INDICES_COPY_OUT")
    return _row_indices_copy_impl(self, out=out)
