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
from flag_gems.utils import dim_compress, libentry
from flag_gems.utils import triton_lang_extension as ext

from ..utils.block_size_utils import get_block_size_1d

logger = logging.getLogger(__name__)


# torch.all: Tests if all elements in input evaluate to True. If the dtype of input
#            is not BOOL, then test if all elements in input evaluate to non-zero value
# In triton function, test if all elements in input evaluate to non-zero value is ok.

cluster_num = 12
core_num = 64
buf_len_per_core = 2048
vector_size = 16


# Tile budget = the current max tile (BLOCK_M=64 * BLOCK_N=512). We keep this
# constant so the [BLOCK_M, BLOCK_N] tile never grows past the size that already
# compiles cleanly (no XPU struct explosion), we only RESHAPE it.
TILE_BUDGET = 64 * 512


# Large flat torch.all(inp) reductions reshape to a [M, K] grid and reduce via the
# wider all_kernel_dim (axis=1) path, which is ~1.75x the flat all_kernel_1 at 1G.
# The reshape wins only above ~8M elements (below that the extra launch dominates).
_GLOBAL_2D_MIN = 1 << 23


def _pick_2d_cols(n):
    # Largest power-of-2 column width in [8192, 65536] that divides n, so the flat
    # buffer can be view()'d as a dense [n // K, K] grid with no copy. 0 = no clean fit.
    for K in (65536, 32768, 16384, 8192):
        if n % K == 0:
            return K
    return 0


def _heur_n_raw(N):
    # For N <= 8192 keep the historical cap of 512 (square / small-N shapes are
    # already near the reduce-bandwidth ceiling with BLOCK_M=64, BLOCK_N=512).
    # For very wide N, a 512-wide tile forces N/512 serial chunks (e.g. 128 for
    # N=65536); widening BLOCK_N to 4096 cuts the loop count ~8x. Measured on XPU
    # (proto): [1024,65536] 113 -> 165 GB/s (+46%) at the SAME tile budget.
    if N <= 8192:
        block_n = min(N, 512)
    else:
        block_n = min(triton.next_power_of_2(N), 4096)
    return triton.next_power_of_2(max(block_n, 1))


def heur_m_block_size(args):
    M = args["M"]
    block_n = _heur_n_raw(args["N"])
    # For very small M, use minimum BLOCK_M of 1
    block_m = min(triton.cdiv(M, cluster_num), core_num)
    # Keep BLOCK_M * BLOCK_N <= TILE_BUDGET: if BLOCK_N was widened for large N,
    # shrink BLOCK_M so the tile stays the same size (constant compile footprint).
    block_m = min(block_m, max(TILE_BUDGET // block_n, 1))
    return triton.next_power_of_2(max(block_m, 1))


def heur_n_block_size(args):
    return _heur_n_raw(args["N"])


@triton.jit
def reduce_all(a, b):
    return a and b


@libentry()
@triton.jit
def all_kernel_1(
    inp,
    mid,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Stage 1 of the global-all reduction: each program reduces one
    BLOCK_SIZE-sized chunk of the flattened input into a single bool in `mid`.
    Splitting the work across `cdiv(n_elements, BLOCK_SIZE)` programs restores
    parallelism (the old single-program loop ran at ~7 GB/s on one core)."""
    pid = ext.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements
    val = tl.load(inp + offset, mask=mask, other=1)
    # masked-out lanes must be True (identity for AND); do not rely on `other`.
    nz = tl.where(mask, val != 0, True)
    result = tl.reduce(nz, axis=0, combine_fn=reduce_all)
    tl.store(mid + pid, result)


@libentry()
@triton.jit
def all_kernel_2(
    mid,
    out,
    mid_size,
    BLOCK_MID: tl.constexpr,
):
    """Stage 2: a single program reduces the per-chunk bools from stage 1."""
    offset = tl.arange(0, BLOCK_MID)
    mask = offset < mid_size
    val = tl.load(mid + offset, mask=mask, other=1)
    nz = tl.where(mask, val != 0, True)
    result = tl.reduce(nz, axis=0, combine_fn=reduce_all)
    tl.store(out, result)


@libentry()
@triton.heuristics(
    values={
        "BLOCK_M": heur_m_block_size,
        "BLOCK_N": heur_n_block_size,
    },
)
@triton.jit
def all_kernel_dim(
    inp,
    out,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Map the program id to the row of inp it should compute.
    pid = ext.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    inp = inp + rows * N
    out = out + rows
    row_mask = rows < M

    _all = tl.full([BLOCK_M, BLOCK_N], value=1, dtype=tl.int1)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        col_mask = cols < N
        mask = row_mask and col_mask

        a = tl.load(inp + cols, mask, other=1.0)
        _all = _all and (a != 0)
    all = tl.reduce(_all, axis=1, combine_fn=reduce_all)
    tl.store(out, all[:, None], row_mask)


def all(inp):
    logger.debug("GEMS_KUNLUNXIN ALL")
    n_elements = inp.numel()

    # Fast path for large flat reductions. The 1D all_kernel_1 (flat BLOCK_SIZE tile +
    # axis=0 reduce) tops out ~115-230 GB/s on XPU. Viewing the contiguous buffer as a
    # [M, K] grid and reducing along axis=1 via all_kernel_dim coalesces far better
    # (measured ~1.75x at 1G, crossover ~8M elements). Falls back to the flat path when
    # the buffer is small, non-contiguous, or has no clean power-of-2 column width.
    if n_elements >= _GLOBAL_2D_MIN and inp.is_contiguous():
        K = _pick_2d_cols(n_elements)
        if K:
            M = n_elements // K
            inp2d = inp.view(M, K)
            mid = torch.empty((M,), dtype=torch.bool, device=inp.device)
            out = torch.empty([], dtype=torch.bool, device=inp.device)
            block_mid = triton.next_power_of_2(M)
            grid = lambda meta: (max(triton.cdiv(M, meta["BLOCK_M"]), 1),)
            with torch_device_fn.device(inp.device):
                all_kernel_dim[grid](inp2d, mid, M, K, buffer_size_limit=2048)
                all_kernel_2[(1, 1, 1)](mid, out, M, block_mid, buffer_size_limit=2048)
            return out

    block_size = get_block_size_1d(n_elements, inp.element_size())
    mid_size = triton.cdiv(n_elements, block_size)
    block_mid = triton.next_power_of_2(mid_size)

    mid = torch.empty((mid_size,), dtype=torch.bool, device=inp.device)
    out = torch.empty([], dtype=torch.bool, device=inp.device)
    with torch_device_fn.device(inp.device):
        all_kernel_1[(mid_size, 1, 1)](
            inp, mid, n_elements, block_size, buffer_size_limit=2048
        )
        if mid_size == 1:
            return mid.reshape([])
        all_kernel_2[(1, 1, 1)](mid, out, mid_size, block_mid, buffer_size_limit=2048)
    return out


def all_dim(inp, dim=None, keepdim=False):
    logger.debug("GEMS_KUNLUNXIN ALL_DIM")
    shape = list(inp.shape)
    orig_ndim = inp.ndim

    if dim is None:
        out = all(inp)
        if keepdim:
            out = torch.reshape(out, [1] * orig_ndim)
        return out

    assert dim >= -orig_ndim and dim < orig_ndim, "Invalid dim"
    dim = dim % orig_ndim
    N = shape[dim]
    inp = dim_compress(inp, dim)
    shape[dim] = 1
    M = inp.numel() // N

    if inp.dtype != torch.bool and M * N <= 64:
        inp = inp != 0

    out = torch.empty(shape, dtype=torch.bool, device=inp.device)
    grid = lambda meta: (max(triton.cdiv(M, meta["BLOCK_M"]), 1),)
    with torch_device_fn.device(inp.device):
        all_kernel_dim[grid](inp, out, M, N, buffer_size_limit=2048)

    if not keepdim and out.ndim > 0:
        out = out.squeeze(dim) if dim < out.ndim else out
    return out


def all_dims(inp, dim=None, keepdim=False):
    logger.debug("GEMS_KUNLUNXIN ALL_DIMS")

    if dim is None or isinstance(dim, int):
        return all_dim(inp, dim=dim, keepdim=keepdim)
    orig_ndim = inp.ndim
    assert ((i >= -orig_ndim and i < orig_ndim) for i in dim), "Invalid dim"

    shape = list(inp.shape)
    dim = [d % orig_ndim for d in dim]
    inp = dim_compress(inp, dim)
    N = 1
    for i in dim:
        N *= shape[i]
        shape[i] = 1
    M = inp.numel() // N

    if inp.dtype != torch.bool and M * N <= 64:
        inp = inp != 0

    out = torch.empty(shape, dtype=torch.bool, device=inp.device)
    grid = lambda meta: (max(triton.cdiv(M, meta["BLOCK_M"]), 1),)
    with torch_device_fn.device(inp.device):
        all_kernel_dim[grid](inp, out, M, N, buffer_size_limit=2048)

    if not keepdim:
        # Squeeze reduced axes from highest to lowest. Removing a low axis first
        # shifts the positions of the remaining (still size-1) reduced axes, so a
        # later `squeeze(dim=d)` would target the wrong axis and silently leave a
        # leading size-1 dim (e.g. dim=[1,0] on (7,4,11,1) gave [1,11,1] vs [11,1]).
        for d in sorted(dim, reverse=True):
            if out.ndim > 0:
                out = out.squeeze(dim=d)
    return out
