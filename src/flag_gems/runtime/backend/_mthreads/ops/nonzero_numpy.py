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

from flag_gems.ops.nonzero_numpy import nonzero_numpy as default_nonzero_numpy
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)

# Threshold for single-block vs two-pass strategy. A block-local cumsum is cheap
# up to a few thousand lanes; past that the two-pass count + fill avoids the
# full-tensor scan bottleneck of the generic nonzero() path.
SINGLE_BLOCK_THRESHOLD = 8192

# Block size for the two-pass count + fill kernels.
TWO_PASS_BLOCK = 2048

# Per-axis sizes are passed as scalar kernel arguments (up to MAX_NDIM). Building
# a device tensor of the shape would cost host + H2D overhead per call that would
# dominate on the small shapes.
MAX_NDIM = 4


@libentry()
@triton.jit
def nonzero_single_kernel(
    inp,
    out,
    count_ptr,
    n_elements,
    s0,
    s1,
    s2,
    s3,
    ndim: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Single-block nonzero for small tensors (grid == (1,)).

    Fuses the (!=0) test, block-local prefix-sum, per-axis index decomposition
    and structure-of-arrays scatter into one launch, so the local cumsum is
    already the global prefix sum -- no cross-block scan needed.
    """
    pid = tl.program_id(0)
    off = pid * BLOCK + tl.arange(0, BLOCK)
    mask = off < n_elements

    vals = tl.load(inp + off, mask=mask, other=0)
    nz = (vals != 0) & mask

    local_pos = tl.cumsum(nz.to(tl.int32))
    local_pos = tl.where(nz, local_pos - 1, 0)

    cnt = tl.sum(nz.to(tl.int32))
    if pid == 0:
        tl.store(count_ptr, cnt)

    idx_flat = off
    if ndim >= 4:
        d = idx_flat % s3
        idx_flat = idx_flat // s3
        tl.store(out + 3 * n_elements + local_pos, d.to(tl.int64), mask=nz)
    if ndim >= 3:
        d = idx_flat % s2
        idx_flat = idx_flat // s2
        tl.store(out + 2 * n_elements + local_pos, d.to(tl.int64), mask=nz)
    if ndim >= 2:
        d = idx_flat % s1
        idx_flat = idx_flat // s1
        tl.store(out + 1 * n_elements + local_pos, d.to(tl.int64), mask=nz)
    if ndim >= 1:
        d = idx_flat % s0
        tl.store(out + 0 * n_elements + local_pos, d.to(tl.int64), mask=nz)


@libentry()
@triton.jit
def nonzero_count_kernel(
    inp,
    block_counts,
    n_elements,
    BLOCK: tl.constexpr,
):
    """First pass for large tensors: count nonzeros per block."""
    pid = tl.program_id(0)
    off = pid * BLOCK + tl.arange(0, BLOCK)
    mask = off < n_elements
    vals = tl.load(inp + off, mask=mask, other=0)
    nz = (vals != 0) & mask
    tl.store(block_counts + pid, tl.sum(nz.to(tl.int32)))


@libentry()
@triton.jit
def nonzero_fill_kernel(
    inp,
    block_counts,
    offsets,
    out,
    n_elements,
    s0,
    s1,
    s2,
    s3,
    n_total,
    ndim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Second pass: scatter indices with a per-block exclusive-prefix offset.

    ``offsets`` holds the inclusive prefix sum of per-block counts; a block's
    exclusive base is ``offsets[pid] - block_counts[pid]``. This preserves stable
    input order without atomics.
    """
    pid = tl.program_id(0)
    off = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = off < n_elements
    vals = tl.load(inp + off, mask=mask, other=0)
    nz = (vals != 0) & mask

    cnt = tl.load(block_counts + pid)
    incl = tl.load(offsets + pid)
    base = incl - cnt

    local_pos = tl.cumsum(nz.to(tl.int32))
    local_pos = tl.where(nz, local_pos - 1, 0)
    pos = base + local_pos

    idx_flat = off
    if ndim >= 4:
        d = idx_flat % s3
        idx_flat = idx_flat // s3
        tl.store(out + 3 * n_total + pos, d.to(tl.int64), mask=nz)
    if ndim >= 3:
        d = idx_flat % s2
        idx_flat = idx_flat // s2
        tl.store(out + 2 * n_total + pos, d.to(tl.int64), mask=nz)
    if ndim >= 2:
        d = idx_flat % s1
        idx_flat = idx_flat // s1
        tl.store(out + 1 * n_total + pos, d.to(tl.int64), mask=nz)
    if ndim >= 1:
        d = idx_flat % s0
        tl.store(out + 0 * n_total + pos, d.to(tl.int64), mask=nz)


def _shape_scalars(inp):
    """Per-axis sizes as scalar ints, outermost axis first, padded to MAX_NDIM."""
    shape = list(inp.shape)
    return shape + [1] * (MAX_NDIM - len(shape))


def _nonzero_single(inp, n_elements, ndim, shape):
    flat = inp.contiguous().view(n_elements)
    block = triton.next_power_of_2(max(n_elements, 4))
    num_warps = 4 if block <= 4096 else 8
    out = torch.empty(ndim, n_elements, dtype=torch.int64, device=inp.device)
    count = torch.empty(1, dtype=torch.int32, device=inp.device)
    with torch_device_fn.device(inp.device):
        nonzero_single_kernel[(1,)](
            flat,
            out,
            count,
            n_elements,
            shape[0],
            shape[1],
            shape[2],
            shape[3],
            ndim,
            BLOCK=block,
            num_warps=num_warps,
        )
    nnz = int(count.item())
    return [out[d, :nnz] for d in range(ndim)]


def _nonzero_two_pass(inp, n_elements, ndim, shape):
    flat = inp.contiguous().view(n_elements)
    block = TWO_PASS_BLOCK
    grid = triton.cdiv(n_elements, block)
    block_counts = torch.empty(grid, dtype=torch.int32, device=inp.device)
    with torch_device_fn.device(inp.device):
        nonzero_count_kernel[(grid,)](
            flat,
            block_counts,
            n_elements,
            BLOCK=block,
        )
    offsets = block_counts.cumsum(axis=0)
    nnz = int(offsets[grid - 1].item())
    if nnz == 0:
        return [inp.new_empty(0, dtype=torch.int64) for _ in range(ndim)]
    out = torch.empty(ndim, nnz, dtype=torch.int64, device=inp.device)
    with torch_device_fn.device(inp.device):
        nonzero_fill_kernel[(grid,)](
            flat,
            block_counts,
            offsets,
            out,
            n_elements,
            shape[0],
            shape[1],
            shape[2],
            shape[3],
            nnz,
            ndim,
            BLOCK_SIZE=block,
            num_warps=8,
        )
    return [out[d] for d in range(ndim)]


def _use_triton_kernel(inp) -> bool:
    if not isinstance(inp, torch.Tensor):
        return False
    if inp.device.type != "musa":
        return False
    # The optimized kernels decompose the flat index using per-axis scalar
    # arguments up to MAX_NDIM; higher-rank inputs defer to the generic path.
    if inp.ndim == 0 or inp.ndim > MAX_NDIM:
        return False
    return True


def nonzero_numpy(inp):
    """Moore Threads (MUSA) specialized ``nonzero_numpy``.

    Two-tier strategy (matching the thead/iluvatar backends):

    * small tensor (<= SINGLE_BLOCK_THRESHOLD, <= 4-D): single-block fused
      kernel (count + index decompose + scatter, one launch)
    * large tensor (<= 4-D): two-pass count + cumsum + fill, avoiding the
      full-tensor scan of the generic path
    * otherwise: fall back to the generic ``nonzero()`` + ``unbind()``.

    The generic implementation delegates to ``nonzero()`` which issues three
    separate kernels per call (elementwise !=0, full-tensor cumsum, scatter);
    that launch overhead dominates on the benchmark shapes.
    """
    logger.debug("GEMS_MTHREADS NONZERO_NUMPY")
    if not _use_triton_kernel(inp):
        return default_nonzero_numpy(inp)

    inp = inp.contiguous()
    n_elements = inp.numel()
    ndim = inp.ndim

    if n_elements == 0:
        return [inp.new_empty(0, dtype=torch.int64) for _ in range(ndim)]

    shape = _shape_scalars(inp)
    if n_elements <= SINGLE_BLOCK_THRESHOLD:
        return _nonzero_single(inp, n_elements, ndim, shape)
    return _nonzero_two_pass(inp, n_elements, ndim, shape)
