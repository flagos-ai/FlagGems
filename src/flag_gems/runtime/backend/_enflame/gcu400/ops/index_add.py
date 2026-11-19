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
#
# GCU400-specific implementation of index_add / index_add_ (zero-atomic version).
#
# The original uses scalar atomic accumulation to add each src element into out[index[j]], but GCU400 scalar
# atomic throughput is only ~0.39 G elem/s (random plain writes 21.5 G elem/s, 55x difference), measured
# core benchmark at only 0.03~0.07x. This implementation completely removes atomic writes from the data path:
#
# 1. No-duplicate cases (all benchmark randperm hits) use 1D/2D scatter: duplicate detection
#    happens on device (inverse map inv[d]=j + comparing j with inv[d]); once
# duplicates are ruled out each output address has a single writer, so load + add + store replaces atomic accumulation
#    (item 30 (1) exclusive address writes, zero extra traffic, zero atomics).
# 2. Duplicates with small scale use index_add_reduce_small_kernel: output-driven one-hot
# weighted reduce_sum, with each output address written by a unique program, naturally supporting any duplicate pattern.
# 3. General duplicate cases use index_add_reduce_kernel: the owner program designated by inv[d]
#    reduces all contributions for that target with tl.sum over j and writes once, zero atomics throughout.
# 4. Detection never uses torch.sort (measured device-side sort + sync drags the 2048-element case to
#    2.6 seconds); only two int32 scalars are read for one host sync, which also completes bounds validation;
#    with J <= 4096 a single program detects and clears flags itself, saving a zeros launch.
# 5. Static 1D / 2D kernels (no runtime codegen), (outer, j, i) flattened into a 1D grid,
#    with an in-kernel grid-stride loop as fallback when grid.x exceeds 65535.
import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

_INT32_OFFSET_LIMIT = 1 << 31

_BLOCK_J = 2048  # index elements handled per program by the 1D kernel
_BLOCK_J_2D = 4  # tile along the index dimension for the 2D kernel
_BLOCK_I_2D = 256  # tile along the inner dimension for the 2D kernel

# GCU400 hardware limit: grid.x <= 65535
_MAX_GRID_X = 65535
_MAX_BLOCK_J = 16384
_MAX_2D_TILE_ELEMENTS = 4096  # DSM budget cap for a 2D tile

_BLOCK_PROBE = 2048  # j tile of the large-J probe kernel
_SINGLE_CHECK_MAX_J = 4096  # maximum index length a single program can cover
_MAX_REDUCE_TILE = 32768  # (BLOCK_D * BLOCK_J) element cap of reduce_small
# With duplicate indices and modest scale, output-driven reduce_sum is faster than owner scanning
_REDUCE_BUDGET = 1 << 25
_BLOCK_P_REDUCE = 256  # j tile for owner reduce_sum scanning
_BLOCK_I_REDUCE = 32  # inner-dimension tile for owner reduce_sum
_NUM_WARPS = 4  # uniform launch parameter, convenient for sweeping the optimal warp count by shape


def _next_pow2(value):
    power = 1
    while power < value:
        power <<= 1
    return power


@libentry()
@triton.jit
def index_add_check_kernel(
    index_ptr,
    inv_ptr,
    flags_ptr,
    J,
    D,
    BLOCK_J: tl.constexpr,
):
    """A single program covers all indices: build the inverse map, check bounds, check duplicates.

    debug_barrier guarantees that inv stores are visible to later loads within the same program (on GCU400
    this semantics was verified in the sort operator). flags is zero-initialized on the host; only 1 is written here.
    """
    rj = tl.arange(0, BLOCK_J)
    mask = rj < J
    # A single program owns flags exclusively, so it can clear them directly, saving a torch.zeros launch
    tl.store(flags_ptr, 0)
    tl.store(flags_ptr + 1, 0)
    tl.debug_barrier()
    d = tl.load(index_ptr + rj, mask=mask, other=-1)
    valid = mask & (d >= 0) & (d < D)
    tl.store(inv_ptr + d, rj.to(tl.int32), mask=valid)
    bad = tl.sum((mask & ~valid).to(tl.int32), axis=0) > 0
    tl.store(flags_ptr, 1, mask=bad)
    tl.debug_barrier()
    owner = tl.load(inv_ptr + d, mask=valid, other=rj.to(tl.int32))
    dup = tl.sum((valid & (owner != rj.to(tl.int32))).to(tl.int32), axis=0) > 0
    tl.store(flags_ptr + 1, 1, mask=dup)


@libentry()
@triton.jit
def index_add_probe_kernel(
    index_ptr,
    inv_ptr,
    flags_ptr,
    J,
    D,
    total_tiles,
    BLOCK_J: tl.constexpr,
    GRID_SIZE: tl.constexpr,
):
    """Large-J probe: inv[index[j]] = j, with bounds checking."""
    pid = tl.program_id(0)
    for tile_id in tl.range(pid, total_tiles, GRID_SIZE):
        rj = tile_id * BLOCK_J + tl.arange(0, BLOCK_J)
        mask = rj < J
        d = tl.load(index_ptr + rj, mask=mask, other=0)
        valid = mask & (d >= 0) & (d < D)
        tl.store(inv_ptr + d, rj.to(tl.int32), mask=valid)
        bad = tl.sum((mask & ~valid).to(tl.int32), axis=0) > 0
        tl.store(flags_ptr, 1, mask=bad)


@libentry()
@triton.jit
def index_add_dup_kernel(
    index_ptr,
    inv_ptr,
    flags_ptr,
    J,
    D,
    total_tiles,
    BLOCK_J: tl.constexpr,
    GRID_SIZE: tl.constexpr,
):
    """Large-J duplicate detection: when several j point to the same d, inv[d] keeps only one of them."""
    pid = tl.program_id(0)
    for tile_id in tl.range(pid, total_tiles, GRID_SIZE):
        rj = tile_id * BLOCK_J + tl.arange(0, BLOCK_J)
        mask = rj < J
        d = tl.load(index_ptr + rj, mask=mask, other=0)
        safe = mask & (d >= 0) & (d < D)
        owner = tl.load(inv_ptr + d, mask=safe, other=rj.to(tl.int32))
        mismatch = safe & (owner != rj.to(tl.int32))
        dup = tl.sum(mismatch.to(tl.int32), axis=0) > 0
        tl.store(flags_ptr + 1, 1, mask=dup)


def _inspect_index(index, D, flags, device):
    """Determine the duplicate pattern and validate bounds, returning (need_reduce, inv).

    Small J is covered by one program over all indices (1 launch); large J uses probe + dup
    two kernels. flags[0]=out-of-bounds, flags[1]=duplicate; one tolist() reads both scalars,
    the only host sync in the whole path.
    """
    J = index.numel()
    inv = torch.empty((D,), dtype=torch.int32, device=device)
    if J <= _SINGLE_CHECK_MAX_J:
        index_add_check_kernel[(1,)](
            index,
            inv,
            flags,
            J,
            D,
            BLOCK_J=_next_pow2(J),
            num_warps=_NUM_WARPS,
        )
    else:
        block_j = min(_BLOCK_PROBE, _next_pow2(J))
        total = triton.cdiv(J, block_j)
        grid = min(total, _MAX_GRID_X)
        index_add_probe_kernel[(grid,)](
            index,
            inv,
            flags,
            J,
            D,
            total,
            BLOCK_J=block_j,
            GRID_SIZE=grid,
            num_warps=_NUM_WARPS,
        )
        index_add_dup_kernel[(grid,)](
            index,
            inv,
            flags,
            J,
            D,
            total,
            BLOCK_J=block_j,
            GRID_SIZE=grid,
            num_warps=_NUM_WARPS,
        )
    err, dup = flags.tolist()
    assert err == 0, "0 <= index < self.size(dim)"
    return dup > 0, inv


def _tile_layout(J, inner, outer):
    """Choose tiles by the grid.x limit, returning (block_j, block_i, total_tiles, grid_size)."""
    if inner == 1:
        block_i = 1
        block_j = min(_BLOCK_J, _next_pow2(J))
        cap_j = _MAX_BLOCK_J
    else:
        block_i = min(_BLOCK_I_2D, _next_pow2(inner))
        block_j = _BLOCK_J_2D
        cap_j = max(1, _MAX_2D_TILE_ELEMENTS // block_i)

    blocks_i = triton.cdiv(inner, block_i)
    max_blocks_j = max(1, _MAX_GRID_X // (outer * blocks_i))
    needed_j = _next_pow2(triton.cdiv(J, max_blocks_j))
    block_j = min(cap_j, max(block_j, needed_j))

    total_tiles = outer * triton.cdiv(J, block_j) * blocks_i
    return block_j, block_i, total_tiles, min(total_tiles, _MAX_GRID_X)


@libentry()
@triton.jit
def index_add_reduce_small_kernel(
    index_ptr,
    src_ptr,
    out_ptr,
    J,
    D,
    alpha,
    src_so,
    out_oo,
    total_tiles,
    BLOCK_D: tl.constexpr,
    BLOCK_J: tl.constexpr,
    GRID_SIZE: tl.constexpr,
):
    """Small-shape output-driven reduce_sum: each output address is written by a unique program.

    match[d, j] = (index[j] == d), reduced with weighted tl.sum over j, naturally supporting
    any duplicate pattern; bounds validity is already checked by _inspect_index.
    """
    pid = tl.program_id(0)
    blocks_d = tl.cdiv(D, BLOCK_D)
    for tile_id in tl.range(pid, total_tiles, GRID_SIZE):
        o = tile_id // blocks_d
        db = tile_id % blocks_d

        rd = db * BLOCK_D + tl.arange(0, BLOCK_D)
        rj = tl.arange(0, BLOCK_J)
        jmask = rj < J
        dj = tl.load(index_ptr + rj, mask=jmask, other=-1)

        s = tl.load(src_ptr + o * src_so + rj, mask=jmask, other=0).to(tl.float32)
        match = (dj[None, :] == rd[:, None]) & jmask[None, :]
        acc = tl.sum(tl.where(match, s[None, :], 0.0), axis=1) * alpha

        out_off = o * out_oo + rd
        dmask = rd < D
        cur = tl.load(out_ptr + out_off, mask=dmask, other=0).to(tl.float32)
        tl.store(
            out_ptr + out_off,
            (cur + acc).to(out_ptr.dtype.element_ty),
            mask=dmask,
        )


@libentry()
@triton.jit
def index_add_1d_kernel(
    index_ptr,
    src_ptr,
    out_ptr,
    J,
    alpha,
    src_so,
    out_oo,
    total_tiles,
    USE_INT64: tl.constexpr,
    BLOCK_J: tl.constexpr,
    GRID_SIZE: tl.constexpr,
):
    """Fast path when dim is the last dimension (inner == 1).

    With no duplicate indices each out[o, index[j]] has a single writer, so load-add-store
    fully replaces scalar atomic accumulation; accumulate in fp32 then cast to avoid a second fp16 rounding.
    """
    pid = tl.program_id(0)
    blocks_j = tl.cdiv(J, BLOCK_J)

    for tile_id in tl.range(pid, total_tiles, GRID_SIZE):
        o = tile_id // blocks_j
        jb = tile_id % blocks_j

        rj = jb * BLOCK_J + tl.arange(0, BLOCK_J)
        mask = rj < J

        src_off = o * src_so + rj
        idx = tl.load(index_ptr + rj, mask=mask, other=0)
        val = tl.load(src_ptr + src_off, mask=mask, other=0).to(tl.float32)

        if USE_INT64:
            out_off = o.to(tl.int64) * out_oo + idx.to(tl.int64)
        else:
            out_off = o * out_oo + idx

        cur = tl.load(out_ptr + out_off, mask=mask, other=0).to(tl.float32)
        tl.store(
            out_ptr + out_off,
            (cur + val * alpha).to(out_ptr.dtype.element_ty),
            mask=mask,
        )


@libentry()
@triton.jit
def index_add_kernel(
    index_ptr,
    src_ptr,
    out_ptr,
    J,
    inner,
    alpha,
    src_so,
    src_sj,
    out_oo,
    out_od,
    total_tiles,
    USE_INT64: tl.constexpr,
    BLOCK_J: tl.constexpr,
    BLOCK_I: tl.constexpr,
    GRID_SIZE: tl.constexpr,
):
    """Generic path: (outer, j, i) flattened into a 1D grid with exclusive address writes."""
    pid = tl.program_id(0)
    blocks_i = tl.cdiv(inner, BLOCK_I)
    blocks_j = tl.cdiv(J, BLOCK_J)
    per_outer = blocks_j * blocks_i

    for tile_id in tl.range(pid, total_tiles, GRID_SIZE):
        o = tile_id // per_outer
        rem = tile_id % per_outer
        jb = rem // blocks_i
        ib = rem % blocks_i

        rj = jb * BLOCK_J + tl.arange(0, BLOCK_J)
        ri = ib * BLOCK_I + tl.arange(0, BLOCK_I)
        mask = (rj < J)[:, None] & (ri < inner)[None, :]

        src_off = o * src_so + rj[:, None] * src_sj + ri[None, :]
        idx = tl.load(index_ptr + rj, mask=rj < J, other=0)

        val = tl.load(src_ptr + src_off, mask=mask, other=0).to(tl.float32)

        if USE_INT64:
            base = o.to(tl.int64) * out_oo + idx.to(tl.int64) * out_od
        else:
            base = o * out_oo + idx * out_od
        out_off = base[:, None] + ri[None, :]

        cur = tl.load(out_ptr + out_off, mask=mask, other=0).to(tl.float32)
        tl.store(
            out_ptr + out_off,
            (cur + val * alpha).to(out_ptr.dtype.element_ty),
            mask=mask,
        )


@libentry()
@triton.jit
def index_add_reduce_kernel(
    index_ptr,
    inv_ptr,
    src_ptr,
    out_ptr,
    J,
    inner,
    alpha,
    src_so,
    src_sj,
    out_oo,
    out_od,
    total_tiles,
    USE_INT64: tl.constexpr,
    BLOCK_P: tl.constexpr,
    BLOCK_I: tl.constexpr,
    GRID_SIZE: tl.constexpr,
):
    """reduce_sum path for duplicate indices (zero atomics throughout).

    Each program corresponds to one (o, j, i_tile); only the owner with j == inv[index[j]]
    reduces all contributions for that target with tl.sum over j and writes once, guaranteeing a unique writer.
    """
    pid = tl.program_id(0)
    blocks_i = tl.cdiv(inner, BLOCK_I)
    per_outer = J * blocks_i

    for tile_id in tl.range(pid, total_tiles, GRID_SIZE):
        o = tile_id // per_outer
        rem = tile_id % per_outer
        j = rem // blocks_i
        ib = rem % blocks_i

        d = tl.load(index_ptr + j)
        owner = tl.load(inv_ptr + d)
        if j == owner:
            ri = ib * BLOCK_I + tl.arange(0, BLOCK_I)
            acc = tl.zeros((BLOCK_I,), dtype=tl.float32)
            for q0 in tl.range(0, J, BLOCK_P):
                rq = q0 + tl.arange(0, BLOCK_P)
                rd = tl.load(index_ptr + rq, mask=rq < J, other=-1)
                hit = (rd == d) & (rq < J)
                src_off = o * src_so + rq[:, None] * src_sj + ri[None, :]
                vals = tl.load(
                    src_ptr + src_off,
                    mask=hit[:, None] & (ri[None, :] < inner),
                    other=0,
                ).to(tl.float32)
                acc += tl.sum(vals, axis=0)

            if USE_INT64:
                base = o.to(tl.int64) * out_oo + d.to(tl.int64) * out_od
            else:
                base = o * out_oo + d * out_od
            out_off = base + ri
            mask_i = ri < inner
            cur = tl.load(out_ptr + out_off, mask=mask_i, other=0).to(tl.float32)
            tl.store(
                out_ptr + out_off,
                (cur + acc * alpha).to(out_ptr.dtype.element_ty),
                mask=mask_i,
            )


def _check_args(inp, dim, index, src):
    assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
    assert (
        inp.ndim == src.ndim
    ), "Self and source should have the same number of dimensions"
    assert index.numel() == src.size(dim), (
        "The dimth dimension of source must have the same size as "
        "the length of index"
    )
    for d in range(inp.ndim):
        if d != dim % inp.ndim:
            assert inp.size(d) == src.size(
                d
            ), "src.size(d) == self.size(d) for all dimensions d != dim"


def _index_add_impl(out, index, src, dim, alpha):
    """Write index_add's sum into out (out is already a copy of inp or the in-place target)."""
    dim = dim % out.ndim
    shape = out.shape
    D = shape[dim]
    outer = 1
    for size in shape[:dim]:
        outer *= size
    inner = 1
    for size in shape[dim + 1 :]:
        inner *= size
    J = index.numel()
    if J == 0 or inner == 0 or outer == 0:
        return

    # Flattening (outer, D, inner) requires row-major contiguity; non-contiguous inputs get one copy,
    # and in-place cases copy back at the end (so the caller still sees the same tensor)
    out_work = out if out.is_contiguous() else out.contiguous()
    src_work = src if src.is_contiguous() else src.contiguous()

    idx = index if index.dtype == torch.int32 else index.to(torch.int32)

    use_int64 = out_work.numel() >= _INT32_OFFSET_LIMIT
    src_so = J * inner
    src_sj = inner
    out_oo = D * inner
    out_od = inner

    # Single-program detection can clear flags itself (saving a zeros launch); large-J multi-
    # program detection must initialize flags to 0 first
    if J <= _SINGLE_CHECK_MAX_J:
        flags = torch.empty((2,), dtype=torch.int32, device=out.device)
    else:
        flags = torch.zeros((2,), dtype=torch.int32, device=out.device)

    with torch_device_fn.device(out.device):
        need_reduce, inv = _inspect_index(idx, D, flags, out.device)
        if need_reduce and inner == 1 and outer * D * J <= _REDUCE_BUDGET:
            # Duplicates with small scale: output-driven reduce_sum, each d written by a unique program
            block_j = _next_pow2(J)
            block_d = min(_next_pow2(D), max(1, _MAX_REDUCE_TILE // block_j))
            total = outer * triton.cdiv(D, block_d)
            grid = min(total, _MAX_GRID_X)
            index_add_reduce_small_kernel[(grid,)](
                idx,
                src_work,
                out_work,
                J,
                D,
                alpha,
                src_so,
                out_oo,
                total,
                BLOCK_D=block_d,
                BLOCK_J=block_j,
                GRID_SIZE=grid,
                num_warps=_NUM_WARPS,
            )
        elif need_reduce:
            # General duplicate path: the owner program reduces with tl.sum then writes exclusively
            block_i = min(_BLOCK_I_REDUCE, _next_pow2(inner))
            total = outer * J * triton.cdiv(inner, block_i)
            grid = min(total, _MAX_GRID_X)
            index_add_reduce_kernel[(grid,)](
                idx,
                inv,
                src_work,
                out_work,
                J,
                inner,
                alpha,
                src_so,
                src_sj,
                out_oo,
                out_od,
                total,
                USE_INT64=use_int64,
                BLOCK_P=_BLOCK_P_REDUCE,
                BLOCK_I=block_i,
                GRID_SIZE=grid,
                num_warps=_NUM_WARPS,
            )
        else:
            block_j, block_i, total, grid = _tile_layout(J, inner, outer)
            if inner == 1:
                index_add_1d_kernel[(grid,)](
                    idx,
                    src_work,
                    out_work,
                    J,
                    alpha,
                    src_so,
                    out_oo,
                    total,
                    USE_INT64=use_int64,
                    BLOCK_J=block_j,
                    GRID_SIZE=grid,
                    num_warps=_NUM_WARPS,
                )
            else:
                index_add_kernel[(grid,)](
                    idx,
                    src_work,
                    out_work,
                    J,
                    inner,
                    alpha,
                    src_so,
                    src_sj,
                    out_oo,
                    out_od,
                    total,
                    USE_INT64=use_int64,
                    BLOCK_J=block_j,
                    BLOCK_I=block_i,
                    GRID_SIZE=grid,
                    num_warps=_NUM_WARPS,
                )

    if out_work is not out:
        out.copy_(out_work)


def index_add(inp, dim, index, src, alpha=1):
    logger.debug("GEMS_ENFLAME INDEX_ADD")
    _check_args(inp, dim, index, src)
    out = inp.clone()
    _index_add_impl(out, index, src, dim, alpha)
    return out


def index_add_(inp, dim, index, src, alpha=1):
    logger.debug("GEMS_ENFLAME INDEX_ADD_")
    _check_args(inp, dim, index, src)
    _index_add_impl(inp, index, src, dim, alpha)
    return inp
