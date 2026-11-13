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

"""cummaxmin_backward — scatter-add of grad_output into a zero tensor along `dim`.

out[..., indices[..., t, ...], ...] += grad_output[..., t, ...] along `dim`,
accumulated in float32 and returned in grad_output.dtype.

Fast path (inner_size == 1, i.e. `dim` is the last axis):
  Indices from cummax/cummin are non-decreasing along `dim` and satisfy idx[t] <= t,
  so every destination p owns one contiguous run of t values and the p are exactly
  the run ends' index values.

  One program owns a whole row and walks it in tiles.  A key-carrying monoid scan
  yields the run prefix sum (the accumulation restarts where the index key
  changes), and the running state is carried across tiles.  Each tile is then
  published with one unmasked, vectorizable zero store followed by a block barrier
  and a sparse scatter of the run totals to `idx[run_end]`; a destination is always
  at a position <= its run end, so it lies in this or an already zeroed earlier
  tile.  One 8-byte index load, one grad load, one vectorized store: no atomics,
  no zero-fill pass, no cast pass.

  Long rows autotune (CHUNK, num_warps); short rows use a fixed single-tile config
  so launch-bound shapes pay no tuner dispatch cost.

Generic path (inner_size > 1): flat coalesced pass with fp32 atomic accumulation.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops.cummax import cummaxmin_backward as default_cummaxmin_backward
from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)

# Moore Threads hardware does not support fp64, so only these dtypes are
# specialized; the rest fall back to the generic implementation.
_SUPPORTED_DTYPES = {torch.float16, torch.bfloat16, torch.float32}


@triton.jit
def _key_combine(v1, k1, v2, k2):
    # Run-wise prefix sum: the accumulator carries the run key, so the
    # accumulation restarts exactly where the (monotone) key changes.  Comparing
    # adjacent keys is sufficient, which removes the separate run-start flag and
    # the shifted idx[t-1] load entirely.
    v = tl.where(k1 == k2, v1 + v2, v2)
    return v, k2


@triton.jit
def _seg_chunk(g_ptr, idx_ptr, pos, m, D, carry_idx, carry_sum, CHUNK: tl.constexpr):
    idx = tl.load(idx_ptr + pos, mask=m, other=0).to(tl.int32)
    idx_n = tl.load(idx_ptr + pos + 1, mask=m & (pos + 1 < D), other=-1).to(tl.int32)
    g = tl.load(g_ptr + pos, mask=m, other=0.0).to(tl.float32)

    acc, _k = tl.associative_scan((g, idx), 0, _key_combine)
    acc = tl.where(idx == carry_idx, acc + carry_sum, acc)
    return acc, idx, idx != idx_n


@triton.jit
def _seg_store(o_ptr, pos, m, acc, idx, at_end):
    # Zero the whole tile with one unmasked (vectorizable) store, then publish the
    # run totals.  The block barrier orders the zero writes before the scatter for
    # every thread in the CTA, and a run total only ever targets a position
    # <= its run end (idx[t] <= t), so the destination is always in this or an
    # already zeroed earlier tile.
    tl.store(o_ptr + pos, tl.zeros_like(acc).to(o_ptr.dtype.element_ty), mask=m)
    tl.debug_barrier()
    tl.store(o_ptr + idx, acc.to(o_ptr.dtype.element_ty), mask=m & at_end)


@triton.jit
def _bwd_last_kernel(
    grad_ptr,
    idx_ptr,
    out_ptr,
    D,
    CHUNK: tl.constexpr,
):
    row = tl.program_id(0)
    base = row.to(tl.int64) * D
    g_ptr = grad_ptr + base
    i_ptr = idx_ptr + base
    o_ptr = out_ptr + base
    lane = tl.arange(0, CHUNK)

    pos = lane
    m = pos < D
    acc, idx, at_end = _seg_chunk(g_ptr, i_ptr, pos, m, D, -1, 0.0, CHUNK)
    _seg_store(o_ptr, pos, m, acc, idx, at_end)


_TUNE_CONFIGS = [
    triton.Config({"CHUNK": 512}, num_warps=2),
    triton.Config({"CHUNK": 512}, num_warps=4),
    triton.Config({"CHUNK": 1024}, num_warps=4),
    triton.Config({"CHUNK": 1024}, num_warps=8),
    triton.Config({"CHUNK": 2048}, num_warps=8),
    triton.Config({"CHUNK": 4096}, num_warps=8),
]

_ONE_TILE_CONFIGS = [
    triton.Config({"CHUNK": 1024}, num_warps=4),
    triton.Config({"CHUNK": 1024}, num_warps=8),
]


@triton.autotune(configs=_ONE_TILE_CONFIGS, key=["D", "elem_bytes"])
@triton.jit
def _bwd_one_tile_kernel(
    grad_ptr,
    idx_ptr,
    out_ptr,
    D,
    elem_bytes,
    CHUNK: tl.constexpr,
):
    # Whole row fits in one tile: no loop, no cross-tile carry, no reduction.
    row = tl.program_id(0)
    base = row.to(tl.int64) * D
    lane = tl.arange(0, CHUNK)
    o_ptr = out_ptr + base

    pos = lane
    m = pos < D
    acc, idx, at_end = _seg_chunk(
        grad_ptr + base, idx_ptr + base, pos, m, D, -1, 0.0, CHUNK
    )
    _seg_store(o_ptr, pos, m, acc, idx, at_end)


@triton.autotune(configs=_TUNE_CONFIGS, key=["D", "elem_bytes"])
@triton.jit
def _bwd_last_tuned_kernel(
    grad_ptr,
    idx_ptr,
    out_ptr,
    D,
    elem_bytes,
    CHUNK: tl.constexpr,
):
    row = tl.program_id(0)
    base = row.to(tl.int64) * D
    g_ptr = grad_ptr + base
    i_ptr = idx_ptr + base
    o_ptr = out_ptr + base
    lane = tl.arange(0, CHUNK)

    carry_idx = tl.full([], -1, tl.int32)
    carry_sum = tl.zeros([], tl.float32)
    for start in tl.range(0, D, CHUNK):
        pos = start + lane
        m = pos < D
        acc, idx, at_end = _seg_chunk(
            g_ptr, i_ptr, pos, m, D, carry_idx, carry_sum, CHUNK
        )
        _seg_store(o_ptr, pos, m, acc, idx, at_end)
        # Carry the running run state to the next tile.  The key is re-read with a
        # single scalar load instead of a block reduction.
        carry_idx = tl.load(
            i_ptr + start + CHUNK - 1, mask=start + CHUNK - 1 < D, other=-1
        ).to(tl.int32)
        carry_sum = tl.sum(tl.where(lane == (CHUNK - 1), acc, 0.0))


@triton.jit
def _scatter_add_kernel(
    grad_ptr,
    idx_ptr,
    out_ptr,
    n_elements,
    reduction_size,
    inner_size,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    g = tl.load(grad_ptr + offs, mask=mask, other=0.0)
    idx = tl.load(idx_ptr + offs, mask=mask, other=0)
    t = (offs // inner_size) % reduction_size
    dst = offs + (idx - t) * inner_size
    tl.atomic_add(out_ptr + dst, g.to(tl.float32), mask=mask)


@triton.jit
def _cast_kernel(src_ptr, dst_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    v = tl.load(src_ptr + offs, mask=mask, other=0.0)
    tl.store(dst_ptr + offs, v.to(dst_ptr.dtype.element_ty), mask=mask)


def _use_triton_kernel(grad_output, indices):
    """Whether the call can be handled by the MUSA kernels."""
    if not isinstance(grad_output, torch.Tensor):
        return False
    if not isinstance(indices, torch.Tensor):
        return False
    if grad_output.device.type != "musa" or indices.device.type != "musa":
        return False
    if grad_output.dtype not in _SUPPORTED_DTYPES:
        return False
    if grad_output.is_complex():
        return False
    return True


def cummaxmin_backward(grad_output, input, indices, dim):
    logger.debug("GEMS_MTHREADS CUMMAXMIN_BACKWARD")

    if not _use_triton_kernel(grad_output, indices):
        return default_cummaxmin_backward(grad_output, input, indices, dim)

    ndim = grad_output.ndim
    if dim < 0:
        dim += ndim
    shape = grad_output.shape
    reduction_size = shape[dim]
    inner_size = 1
    for i in range(dim + 1, ndim):
        inner_size *= shape[i]
    n_elements = grad_output.numel()
    dtype = grad_output.dtype
    device = grad_output.device

    if n_elements == 0 or reduction_size == 0:
        return torch.zeros(shape, dtype=dtype, device=device)

    with torch_device_fn.device(device):
        return _launch_cummaxmin_backward(
            grad_output,
            indices,
            shape,
            dtype,
            device,
            reduction_size,
            inner_size,
            n_elements,
        )


def _launch_cummaxmin_backward(
    grad_output,
    indices,
    shape,
    dtype,
    device,
    reduction_size,
    inner_size,
    n_elements,
):
    if inner_size == 1:
        rows = n_elements // reduction_size
        out = torch.empty(shape, dtype=dtype, device=device)
        if reduction_size <= 256:
            chunk, num_warps = (64, 1) if reduction_size <= 64 else (256, 2)
            _bwd_last_kernel[(rows,)](
                grad_output,
                indices,
                out,
                reduction_size,
                chunk,
                num_warps=num_warps,
            )
        elif reduction_size <= 1024:
            _bwd_one_tile_kernel[(rows,)](
                grad_output,
                indices,
                out,
                reduction_size,
                dtype.itemsize,
            )
        else:
            _bwd_last_tuned_kernel[(rows,)](
                grad_output,
                indices,
                out,
                reduction_size,
                dtype.itemsize,
            )
        return out

    acc = torch.zeros(shape, dtype=torch.float32, device=device)
    BLOCK = 1024
    _scatter_add_kernel[(triton.cdiv(n_elements, BLOCK),)](
        grad_output,
        indices,
        acc,
        n_elements,
        reduction_size,
        inner_size,
        BLOCK,
    )
    if dtype == torch.float32:
        return acc
    out = torch.empty(shape, dtype=dtype, device=device)
    _cast_kernel[(triton.cdiv(n_elements, 4096),)](acc, out, n_elements, 4096)
    return out
