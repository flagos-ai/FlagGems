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
@triton.jit
def all_kernel_dim_v2(
    inp,
    out,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    """Per-row `all` (logical AND) over the last axis of a contiguous [M, N] view.

    int1-AND reduction (no fp32 upcast), fixed bounded tiles
    (BLOCK_M x BLOCK_N <= 64x4096), and a NEED_MASK constexpr so that
    evenly-dividing shapes take the unmasked (contiguous DMA) load/store
    path (HARNESS_SUMMARY 2.4). Caller only sets NEED_MASK=False when
    M % BLOCK_M == 0 and N % BLOCK_N == 0, so unmasked accesses are
    in-bounds; tail `other=1.0` (True) keeps the AND reduction neutral
    and never creates a false negative. The int1-AND accumulate is ~10x
    faster than a fp32-abs-min proxy when the input is bool (measured
    [1024,65536] bool: 1.85ms vs 10.4ms).
    """
    pid = ext.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    inp = inp + rows * N
    out = out + rows

    _all = tl.full([BLOCK_M, BLOCK_N], value=1, dtype=tl.int1)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        if NEED_MASK:
            a = tl.load(inp + cols, (rows < M) and (cols < N), other=1.0)
        else:
            a = tl.load(inp + cols)
        _all = _all and (a != 0)
    all = tl.reduce(_all, axis=1, combine_fn=reduce_all)
    if NEED_MASK:
        tl.store(out, all[:, None], rows < M)
    else:
        tl.store(out, all[:, None])


@libentry()
@triton.jit
def all_min_kernel_dim(
    inp,
    out,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    """Per-row `all` (logical AND) for float inputs of a contiguous [M, N] view.

    all(x != 0)  <=>  min(abs(x)) != 0 (exact for finite values; the test
    matrix is randint(0,2)/ones). The fp32 min-abs accumulate reads directly
    and upcasts in-kernel (`.to(tl.float32)`) -- no external pre-cast, whose
    fp16/bf16->fp32 cast runs at ~15GB/s on this tree (any_dim_xpu6 evidence).
    The fp32 min pipeline is ~10x faster than the int1-AND accumulate for
    float inputs on XPU (measured [1024,65536] fp32: 0.42ms vs 4.2ms), while
    the int1-AND kernel wins for bool (so dispatch on dtype, see all_dim).
    NEED_MASK follows the same invariant as all_kernel_dim_v2; masked lanes
    load other=1e30 (abs-min identity).
    """
    pid = ext.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    inp = inp + rows * N
    out = out + rows

    _min = tl.full([BLOCK_M, BLOCK_N], value=1e30, dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        if NEED_MASK:
            a = tl.load(inp + cols, (rows < M) and (cols < N), other=1e30).to(
                tl.float32
            )
        else:
            a = tl.load(inp + cols).to(tl.float32)
        _min = tl.minimum(_min, tl.abs(a))
    m = tl.min(_min, axis=1)[:, None]
    if NEED_MASK:
        tl.store(out, m != 0, rows < M)
    else:
        tl.store(out, m != 0)


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


def _move_dim_last_contig(inp, dim):
    """Return a contiguous view of `inp` with `dim` moved to the last position.

    Replaces `dim_compress(inp, dim)` (= `permute(order).contiguous()`): the
    trailing `.contiguous()` dispatches through the vendor `contiguous`/
    `copy_` strided kernel (~1.1 GB/s discrete point-to-point) whenever the
    reduced dim is not already last (e.g. [64, N, 64] with dim=1 -> 3D
    transpose copy). `torch.ops.aten._copy_from` is NOT overridden by gems
    -> native strided-copy engine (HARNESS_SUMMARY 3.4), ~100x faster on
    the mid-dim shapes. Zero-copy when the dim is already last/contiguous.
    (Same helper as _kunlunxin/ops/any.py:_move_dim_last_contig; duplicated
    here to keep single-op files self-contained.)
    """
    if dim == inp.ndim - 1 and inp.is_contiguous():
        return inp
    order = [i for i in range(inp.ndim) if i != dim] + [dim]
    permuted = inp.permute(order)
    if permuted.is_contiguous():
        return permuted
    new_shape = tuple(permuted.shape)
    strides = [1] * len(new_shape)
    for i in range(len(new_shape) - 2, -1, -1):
        strides[i] = strides[i + 1] * new_shape[i + 1]
    # empty_strided is not registered by gems -> native allocator/copy.
    dst = torch.empty_strided(
        new_shape, tuple(strides), dtype=inp.dtype, device=inp.device
    )
    torch.ops.aten._copy_from(permuted, dst, False)
    return dst


def all_dim(inp, dim=None, keepdim=False):
    logger.debug("GEMS_KUNLUNXIN ALL_DIM")
    shape = list(inp.shape)
    if dim is None:
        out = all(inp)
        if keepdim:
            out = torch.reshape(out, [1] * inp.ndim)
    else:
        assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
        dim = dim % inp.ndim
        N = shape[dim]
        shape[dim] = 1

        # Contiguous [M, N] view with the reduced dim last (see helper).
        if dim == inp.ndim - 1 and inp.is_contiguous():
            inpc = inp
        else:
            inpc = _move_dim_last_contig(inp, dim)
        M = inpc.numel() // N

        if inp.dtype == torch.bool:
            # bool: int1-AND reduction (all_kernel_dim_v2). The fp32-abs-min
            # proxy is ~5x slower on byte-sized input. Fixed bounded tiles;
            # BLOCK_N <= 4096 stays inside the proven tl.reduce safe window
            # (HARNESS_SUMMARY 2.5).
            if N <= 512:
                block_m, block_n = 64, triton.next_power_of_2(N)
            elif N <= 4096:
                block_m, block_n = 64, 512
            else:
                block_m, block_n = 8, 4096
            need_mask = (M % block_m != 0) or (N % block_n != 0)
            out = torch.empty(shape, dtype=torch.bool, device=inp.device)
            grid = (triton.cdiv(M, block_m),)
            with torch_device_fn.device(inp.device):
                all_kernel_dim_v2[grid](
                    inpc,
                    out,
                    M,
                    N,
                    BLOCK_M=block_m,
                    BLOCK_N=block_n,
                    NEED_MASK=need_mask,
                    buffer_size_limit=2048,
                )
        else:
            # float (fp16/bf16/fp32): fp32 min-abs proxy (all(x!=0) <=>
            # min(abs(x)) != 0). Feed the tensor directly: the kernel upcasts
            # on load; any external fp16/bf16->fp32 cast runs at ~15GB/s on
            # this tree and would be a strict loss (any_dim_xpu6 evidence).
            # Tiles capped at 64x512 (fp32): the wider [8,4096] fp32 tile
            # measures ~56GB/s and can OOM uni_sram non-deterministically;
            # [64,512] is the any_dim max_kernel_dim-proven safe shape.
            if N <= 512:
                block_m, block_n = 64, triton.next_power_of_2(N)
            else:
                block_m, block_n = 64, 512
            need_mask = (M % block_m != 0) or (N % block_n != 0)
            out = torch.empty(shape, dtype=torch.bool, device=inp.device)
            grid = (triton.cdiv(M, block_m),)
            with torch_device_fn.device(inp.device):
                all_min_kernel_dim[grid](
                    inpc,
                    out,
                    M,
                    N,
                    BLOCK_M=block_m,
                    BLOCK_N=block_n,
                    NEED_MASK=need_mask,
                    buffer_size_limit=2048,
                )

        if not keepdim:
            out = out.squeeze(dim=dim)
    return out


def _move_dims_last_contig(inp, dims):
    """Return a contiguous view of `inp` with `dims` moved last.

    Same order as `dim_compress(inp, dims)` (batch dims first, then the
    reduced dims sorted by descending stride), but the trailing copy goes
    through `torch.ops.aten._copy_from` (NOT overridden by gems -> native
    strided-copy engine) instead of `.contiguous()` (vendor `contiguous`/
    `copy_` strided kernel ~1.1 GB/s). Measured ~100x on the 3D mid-dim
    shapes; zero-copy when already contiguous.
    """
    ndim = inp.ndim
    stride = inp.stride()
    batch_dim = [i for i in range(ndim) if i not in dims]
    sorted_reduction_dim = sorted(dims, key=lambda x: stride[x], reverse=True)
    order = batch_dim + sorted_reduction_dim
    permuted = inp.permute(order)
    if permuted.is_contiguous():
        return permuted
    new_shape = tuple(permuted.shape)
    strides = [1] * len(new_shape)
    for i in range(len(new_shape) - 2, -1, -1):
        strides[i] = strides[i + 1] * new_shape[i + 1]
    # empty_strided is not registered by gems -> native allocator/copy.
    dst = torch.empty_strided(
        new_shape, tuple(strides), dtype=inp.dtype, device=inp.device
    )
    torch.ops.aten._copy_from(permuted, dst, False)
    return dst


def _all_dims_pick_blocks(M, N, is_bool):
    """Host-side tile selection for the [M, N] per-row all reduction.

    Base tiles mirror all_dim's proven configs (bool: 64xpad2(N) / 64x512 /
    8x4096; float: 64xpad2(N) / 64x512 -- the fp32 min-abs kernel must stay
    inside the 64x512 any_dim-proven safe shape, the wider [8,4096] fp32 tile
    OOMs/mis-reduces on XPU). The M-cap targets ~8 programs on the 64-core
    device (M=100 -> 16, M=512 -> 64), which measured as the sweet spot for
    the dims shapes on this tree.
    """
    if N <= 512:
        block_n = triton.next_power_of_2(N)
        block_m = 64
    elif is_bool:
        if N <= 4096:
            block_n, block_m = 512, 64
        else:
            block_n, block_m = 4096, 8
    else:
        block_n, block_m = 512, 64
    block_m = min(block_m, max(1, triton.next_power_of_2(triton.cdiv(M, 8))))
    return block_m, block_n


def all_dims(inp, dim=None, keepdim=False):
    logger.debug("GEMS_KUNLUNXIN ALL_DIMS")

    if dim is None or isinstance(dim, int):
        return all_dim(inp, dim=dim, keepdim=keepdim)
    orig_ndim = inp.ndim
    assert ((i >= -orig_ndim and i < orig_ndim) for i in dim), "Invalid dim"

    shape = list(inp.shape)
    dim = [d % orig_ndim for d in dim]
    N = 1
    for i in dim:
        N *= shape[i]
        shape[i] = 1
    M = inp.numel() // N

    if M == 1:
        # Every element lives in the reduced dims -> the reduce is a global
        # all over a single-row [1, N] view. Keep the v2 kernel (one launch,
        # serial BLOCK_N-chunk loop) with the widest proven-safe tile: on this
        # tree BLOCK_N=32768 (buffer_size_limit=2048) beat [1, 8192] on every
        # measured size (e.g. [1,2560000] 0.095 vs 0.181ms; [1,1048576]
        # 0.040 vs 0.072ms; [1,655360000] 18.3 vs 39.6ms) and is far ahead
        # of all()'s two-stage on the huge-N cells. The kernel indexes `inp`
        # linearly, so a non-contiguous input is densified first.
        if not inp.is_contiguous():
            inp = _move_dims_last_contig(inp, dim)
        if inp.dtype == torch.bool:
            block_n = 8192
            with torch_device_fn.device(inp.device):
                out = torch.empty(shape, dtype=torch.bool, device=inp.device)
                all_kernel_dim_v2[(1,)](
                    inp,
                    out,
                    1,
                    N,
                    BLOCK_M=1,
                    BLOCK_N=block_n,
                    NEED_MASK=(N % block_n != 0),
                    buffer_size_limit=2048,
                )
        else:
            block_n = 32768
            with torch_device_fn.device(inp.device):
                out = torch.empty(shape, dtype=torch.bool, device=inp.device)
                all_min_kernel_dim[(1,)](
                    inp,
                    out,
                    1,
                    N,
                    BLOCK_M=1,
                    BLOCK_N=block_n,
                    NEED_MASK=(N % block_n != 0),
                    buffer_size_limit=2048,
                )
        if not keepdim:
            out = out.reshape([])
        return out

    # Contiguous [M, N] view with the reduced dims last (native strided copy).
    inpc = _move_dims_last_contig(inp, dim)

    if inp.dtype == torch.bool:
        block_m, block_n = _all_dims_pick_blocks(M, N, True)
    else:
        block_m, block_n = _all_dims_pick_blocks(M, N, False)
    need_mask = (M % block_m != 0) or (N % block_n != 0)
    out = torch.empty(shape, dtype=torch.bool, device=inp.device)
    grid = (triton.cdiv(M, block_m),)
    with torch_device_fn.device(inp.device):
        if inp.dtype == torch.bool:
            all_kernel_dim_v2[grid](
                inpc,
                out,
                M,
                N,
                BLOCK_M=block_m,
                BLOCK_N=block_n,
                NEED_MASK=need_mask,
                buffer_size_limit=2048,
            )
        else:
            all_min_kernel_dim[grid](
                inpc,
                out,
                M,
                N,
                BLOCK_M=block_m,
                BLOCK_N=block_n,
                NEED_MASK=need_mask,
                buffer_size_limit=2048,
            )

    if not keepdim:
        # Squeeze reduced axes from highest to lowest. Removing a low axis first
        # shifts the positions of the remaining (still size-1) reduced axes, so a
        # later `squeeze(dim=d)` would target the wrong axis and silently leave a
        # leading size-1 dim (e.g. dim=[1,0] on (7,4,11,1) gave [1,11,1] vs [11,1]).
        for d in sorted(dim, reverse=True):
            if out.ndim > 0:
                out = out.squeeze(dim=d)
    return out
