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

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

# Contiguous fast-path constants.
# The flat clone kernel is memory-bound; a large block keeps the number of
# programs low and saturates bandwidth on XPU (1024 -> ~150 GB/s, 8192 ->
# ~1 TB/s). Capped so the materialized tile stays small.
_COPY_BLOCK = 8192
# Inner fill block for the "contiguous slice" path (indexed dim is not the
# innermost, so every selected position is a contiguous run of `inner_size`
# elements). Capped to keep the materialized tile small and avoid IR explosion.
_SLICE_BLOCK = 4096
# Scatter block for the "indexed dim is the innermost" path. 1-D (one position
# per lane): the 2-D tile is mis-lowered on XPU when outer < BLOCK_M (drops
# lanes) and 1-D is slightly faster (7-20%).
_SCATTER_BLOCK = 256
# Threshold below which the "indexed dim is not innermost" path switches from
# the per-position slice kernel to a position-blocked kernel. For short inner
# runs (e.g. shape [200, 40999, 3], dim=1 -> inner=3) the slice kernel launches
# outer * index_len programs and becomes launch-bound (8.2M programs, ~480 ms);
# blocking positions amortizes the launch cost (~22x). The static inner loop is
# unrolled, so the threshold caps the unroll to avoid IR explosion.
_SMALL_INNER_LIMIT = 32
# Position block for the small-inner kernel. 512 beats 128 under cold-cache
# do_bench (fewer programs: ~16k vs ~64k); the 128 advantage only shows in
# hot-cache micro-benchmarks, an artifact.
_SMALL_INNER_BLOCK = 512

_FALLBACK_KEYSET = torch._C.DispatchKeySet(
    torch._C.DispatchKey.CompositeExplicitAutograd
)


def _native_clone(inp):
    # Clone without re-dispatching into FlagGems-registered ops.
    return torch.ops.aten.clone.default.redispatch(_FALLBACK_KEYSET, inp)


@libentry()
@triton.jit
def index_fill_copy_kernel(out, inp, N, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    # i64 offsets: i32 `pid * BLOCK` wraps for numel > 2^31 (OOB store).
    offsets = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    tl.store(out + offsets, tl.load(inp + offsets, mask=mask), mask=mask)


@libentry()
@triton.jit
def index_fill_slice_kernel(
    out,
    index,
    value,
    index_len,
    dim_size,
    inner_size,
    VALUE_IS_TENSOR: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # One program per (outer, index) slice. Each slice writes a contiguous run
    # of `inner_size` elements: out[o, idx[j], :] = value. The store base is a
    # scalar per program, so the per-iteration address is just `out + cols`,
    # which OffsetAnalysis can prove contiguous -> block DMA.
    pid = tl.program_id(axis=0)
    o = pid // index_len
    j = pid % index_len
    idx = tl.load(index + j).to(tl.int64)
    valid = (idx >= -dim_size) & (idx < dim_size)
    idx = tl.where(idx < 0, idx + dim_size, idx)
    # Clamp out-of-range indices so the pointer arithmetic below never leaves
    # the buffer; the store mask still skips them (see `eff` below).
    idx = tl.maximum(idx, 0)
    idx = tl.minimum(idx, dim_size - 1)
    if VALUE_IS_TENSOR:
        fill = tl.load(value)
    else:
        fill = value
    out += o.to(tl.int64) * dim_size * inner_size + idx * inner_size
    for c in range(0, inner_size, BLOCK):
        cols = c + tl.arange(0, BLOCK)
        # Fold the per-slice validity into the column index instead of AND-ing
        # a scalar `valid` into the store mask: `mask & valid` (vector & scalar
        # i1) is mis-lowered on XPU and turns the whole remainder of the buffer
        # into a contiguous store. `tl.where` keeps the mask a pure vector.
        eff = tl.where(valid, cols, inner_size + 1)
        mask = eff < inner_size
        tl.store(out + cols, fill, mask=mask)


@libentry()
@triton.jit
def index_fill_small_inner_kernel(
    out,
    index,
    value,
    total_pos,
    index_len,
    dim_size,
    INNER: tl.constexpr,
    VALUE_IS_TENSOR: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    # Indexed dim is not innermost but the inner run is short (<=
    # _SMALL_INNER_LIMIT). A per-position program (slice kernel) becomes
    # launch-bound when outer * index_len is large, so block over positions:
    # each lane writes the short contiguous inner run via a static unrolled
    # loop of 1-D stores (no 2-D offset tile, which mis-lowers on XPU).
    pid = tl.program_id(axis=0)
    m_offsets = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    # Clamp the position read instead of using a masked load: a masked load with
    # other=0 is lowered unreliably on XPU when the load mask differs from the
    # store mask. (i32 positions; i64 is applied below at the `base` multiply.)
    m_clamped = tl.minimum(m_offsets, total_pos - 1)
    outer_coord = m_clamped // index_len
    index_coord = m_clamped % index_len
    raw = tl.load(index + index_coord).to(tl.int64)
    valid = (raw >= -dim_size) & (raw < dim_size)
    idx = tl.where(raw < 0, raw + dim_size, raw)
    # Clamp out-of-range indices so the pointer arithmetic below never leaves
    # the buffer; the store mask still skips them (via `valid`).
    idx = tl.maximum(idx, 0)
    idx = tl.minimum(idx, dim_size - 1)
    if VALUE_IS_TENSOR:
        fill = tl.load(value)
    else:
        fill = value
    base = outer_coord.to(tl.int64) * dim_size * INNER + idx * INNER
    # OOB entries are clamped in-bounds and skipped by `valid`. Non-uniform
    # masked stores mis-lower on XPU (the tl.where alternative drops valid lanes).
    store_mask = (m_offsets < total_pos) & valid
    for c in tl.static_range(INNER):
        tl.store(out + base + c, fill, mask=store_mask)


@libentry()
@triton.jit
def index_fill_scatter_kernel(
    out,
    index,
    value,
    total_pos,
    index_len,
    dim_size,
    VALUE_IS_TENSOR: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Indexed dim is the innermost: out[o, idx[j]] = value. One position per
    # lane, 1-D gather-store (a 2-D tile mis-lowers on XPU when outer < BLOCK_M).
    pid = tl.program_id(axis=0)
    pos = pid * BLOCK + tl.arange(0, BLOCK)
    # Clamp the position instead of a masked load (other=0 is unreliable on XPU
    # when load/store masks differ). i32 pos; i64 is applied at `base` below.
    pos_clamped = tl.minimum(pos, total_pos - 1)
    o = pos_clamped // index_len
    j = pos_clamped % index_len
    idx = tl.load(index + j).to(tl.int64)
    valid = (idx >= -dim_size) & (idx < dim_size)
    idx = tl.where(idx < 0, idx + dim_size, idx)
    # Clamp out-of-range indices so pointer arithmetic stays in-bounds; the
    # store mask skips them via `valid`.
    idx = tl.maximum(idx, 0)
    idx = tl.minimum(idx, dim_size - 1)
    if VALUE_IS_TENSOR:
        fill = tl.load(value)
    else:
        fill = value
    base = o.to(tl.int64) * dim_size + idx
    # OOB handling: out-of-range entries are clamped in-bounds and skipped by
    # `valid`. Non-uniform masked stores mis-lower on XPU, so an OOB entry may
    # leak `value` to its clamped position (still UB: CPU raises, xdnn garbage).
    tl.store(out + base, fill, mask=(pos < total_pos) & valid)


def _fill_contiguous(out, dim, index, value, value_is_tensor):
    dim_size = out.size(dim)
    inner_size = 1
    for i in range(dim + 1, out.ndim):
        inner_size *= out.shape[i]
    outer = out.numel() // (dim_size * inner_size)

    if inner_size == 1:
        total_pos = outer * index.numel()
        grid = (triton.cdiv(total_pos, _SCATTER_BLOCK),)
        index_fill_scatter_kernel[grid](
            out,
            index,
            value,
            total_pos,
            index.numel(),
            dim_size,
            VALUE_IS_TENSOR=value_is_tensor,
            BLOCK=_SCATTER_BLOCK,
            num_warps=4,
        )
    elif inner_size <= _SMALL_INNER_LIMIT:
        total_pos = outer * index.numel()
        grid = (triton.cdiv(total_pos, _SMALL_INNER_BLOCK),)
        index_fill_small_inner_kernel[grid](
            out,
            index,
            value,
            total_pos,
            index.numel(),
            dim_size,
            INNER=inner_size,
            VALUE_IS_TENSOR=value_is_tensor,
            BLOCK_M=_SMALL_INNER_BLOCK,
            num_warps=8,
        )
    else:
        n_slices = outer * index.numel()
        block = min(triton.next_power_of_2(inner_size), _SLICE_BLOCK)
        grid = (n_slices,)
        index_fill_slice_kernel[grid](
            out,
            index,
            value,
            index.numel(),
            dim_size,
            inner_size,
            VALUE_IS_TENSOR=value_is_tensor,
            BLOCK=block,
            num_warps=8,
        )


def _prepare_index(inp, dim, index):
    if inp.ndim == 0:
        raise IndexError("index_fill expects self to have at least one dimension")
    if dim < -inp.ndim or dim >= inp.ndim:
        raise IndexError(
            f"Dimension out of range (expected to be in range of "
            f"[{-inp.ndim}, {inp.ndim - 1}], but got {dim})"
        )
    dim = dim % inp.ndim

    if index.dtype != torch.long:
        raise IndexError("index_fill_(): Expected dtype int64 for index.")
    if index.device != inp.device:
        raise RuntimeError(
            "Expected all tensors to be on the same device, but found at least "
            f"two devices, {inp.device} and {index.device}!"
        )
    if index.ndim > 1:
        raise IndexError("index_fill_(): Index is supposed to be a vector")
    if index.ndim == 0:
        index = index.reshape(1)

    return dim, index


def _prepare_tensor_value(inp, value):
    if value.ndim != 0:
        raise RuntimeError(
            "index_fill_ only supports a 0-dimensional value tensor, "
            f"but got tensor with {value.ndim} dimension(s)."
        )
    if value.device.type == "cpu":
        return False, value.item()
    if value.device != inp.device:
        raise RuntimeError(
            "Expected all tensors to be on the same device, but found at least "
            f"two devices, {inp.device} and {value.device}!"
        )
    return True, value


def index_fill(inp, dim, index, value):
    logger.debug("GEMS_KUNLUNXIN INDEX_FILL")
    dim, index = _prepare_index(inp, dim, index)
    if isinstance(value, torch.Tensor):
        value_is_tensor, value = _prepare_tensor_value(inp, value)
    else:
        value_is_tensor = False

    if inp.numel() == 0 or index.numel() == 0:
        return _native_clone(inp)

    if inp.is_contiguous():
        out = torch.empty_like(inp)
        with torch_device_fn.device(inp.device):
            grid = (triton.cdiv(out.numel(), _COPY_BLOCK),)
            index_fill_copy_kernel[grid](out, inp, out.numel(), BLOCK=_COPY_BLOCK)
    else:
        out = inp.contiguous()

    with torch_device_fn.device(inp.device):
        _fill_contiguous(out, dim, index, value, value_is_tensor)
    return out


def index_fill_(inp, dim, index, value):
    logger.debug("GEMS_KUNLUNXIN INDEX_FILL_")
    dim, index = _prepare_index(inp, dim, index)
    if isinstance(value, torch.Tensor):
        value_is_tensor, value = _prepare_tensor_value(inp, value)
    else:
        value_is_tensor = False

    if inp.numel() == 0 or index.numel() == 0:
        return inp

    if inp.is_contiguous():
        with torch_device_fn.device(inp.device):
            _fill_contiguous(inp, dim, index, value, value_is_tensor)
        return inp

    # Strided in-place path: materialize a contiguous copy, fill it, write back.
    contig = inp.contiguous()
    with torch_device_fn.device(inp.device):
        _fill_contiguous(contig, dim, index, value, value_is_tensor)
    torch.ops.aten.copy_.default.redispatch(_FALLBACK_KEYSET, inp, contig, False)
    return inp
