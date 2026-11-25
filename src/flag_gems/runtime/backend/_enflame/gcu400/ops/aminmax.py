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
from flag_gems.utils.limits import get_dtype_max, get_dtype_min

logger = logging.getLogger(__name__)

# GCU400 reduction-friendly parameters (values from on-board A/B sweeps on 2026-11-13, see task
# j20261113032234_423a8985): flat reduction uses striped persistent CTAs + single load +
# nomask hot loop; row reduction removes the per-element where predicate tax; non-last-dim reduction
# uses a materialization-free striding kernel, avoiding the read amplification of dim_compress transpose copies.
_FLAT_BLOCK = 16384
_FLAT_GRID_SMALL = 48
_FLAT_GRID_LARGE = 512
_FLAT_TILES_CROSSOVER = 2048
_ROW_BLOCK_M = 64
_ROW_GRID_LIMIT = 65535
_STRIDED_BLOCK_R = 32


@libentry()
@triton.jit
def aminmax_flat_partial_kernel(
    inp,
    min_partial,
    max_partial,
    M,
    BLOCK: tl.constexpr,
    GRID: tl.constexpr,
    HAS_TAIL: tl.constexpr,
    IS_FLOAT: tl.constexpr,
):
    pid = ext.program_id(0)
    acc_dtype = tl.float32 if IS_FLOAT else inp.type.element_ty
    dtype_max = get_dtype_max(acc_dtype)
    dtype_min = get_dtype_min(acc_dtype)
    acc_min = tl.full((BLOCK,), dtype_max, acc_dtype)
    acc_max = tl.full((BLOCK,), dtype_min, acc_dtype)
    offsets = tl.arange(0, BLOCK)
    num_full = M // BLOCK
    for tile_index in range(pid, num_full, GRID):
        values = tl.load(inp + tile_index * BLOCK + offsets).to(acc_dtype)
        acc_min = tl.minimum(acc_min, values)
        acc_max = tl.maximum(acc_max, values)
    if HAS_TAIL:
        if pid == num_full % GRID:
            tail_offsets = num_full * BLOCK + offsets
            tail_mask = tail_offsets < M
            values = tl.load(inp + tail_offsets, mask=tail_mask, other=0).to(acc_dtype)
            acc_min = tl.minimum(acc_min, tl.where(tail_mask, values, dtype_max))
            acc_max = tl.maximum(acc_max, tl.where(tail_mask, values, dtype_min))
    tl.store(min_partial + pid, tl.min(acc_min).to(min_partial.type.element_ty))
    tl.store(max_partial + pid, tl.max(acc_max).to(max_partial.type.element_ty))


@libentry()
@triton.jit
def aminmax_flat_final_kernel(
    min_partial,
    max_partial,
    min_out,
    max_out,
    P: tl.constexpr,
    BLOCK_F: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_F)
    valid = offsets < P
    dtype = min_partial.type.element_ty
    partial_min = tl.load(min_partial + offsets, mask=valid, other=get_dtype_max(dtype))
    partial_max = tl.load(max_partial + offsets, mask=valid, other=get_dtype_min(dtype))
    tl.store(min_out, tl.min(partial_min))
    tl.store(max_out, tl.max(partial_max))


@libentry()
@triton.jit
def aminmax_row_kernel(
    inp,
    min_out,
    max_out,
    M,
    N,
    NUM_GROUPS,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    GRID: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
):
    pid = ext.program_id(0)
    dtype = inp.type.element_ty
    acc_dtype = tl.float32 if dtype is tl.bfloat16 else dtype
    dtype_max = get_dtype_max(acc_dtype)
    dtype_min = get_dtype_min(acc_dtype)
    fill_min = tl.full((1, 1), dtype_min, acc_dtype)
    num_full = N // BLOCK_N
    for group in range(pid, NUM_GROUPS, GRID):
        rows = group * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
        base = inp + rows * N
        acc_min = tl.full((BLOCK_M, BLOCK_N), dtype_max, acc_dtype)
        acc_max = tl.full((BLOCK_M, BLOCK_N), dtype_min, acc_dtype)
        if EVEN_M:
            for tile_index in range(0, num_full):
                cols = tile_index * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
                values = tl.load(base + cols).to(acc_dtype)
                acc_min = tl.minimum(acc_min, values)
                acc_max = tl.maximum(acc_max, values)
            if not EVEN_N:
                cols = num_full * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
                col_mask = cols < N
                values = tl.load(base + cols, mask=col_mask, other=dtype_max).to(
                    acc_dtype
                )
                acc_min = tl.minimum(acc_min, values)
                acc_max = tl.maximum(acc_max, tl.where(col_mask, values, fill_min))
            row_min = tl.min(acc_min, axis=1)[:, None].to(dtype)
            row_max = tl.max(acc_max, axis=1)[:, None].to(dtype)
            tl.store(min_out + rows, row_min)
            tl.store(max_out + rows, row_max)
        else:
            row_mask = rows < M
            for tile_index in range(0, num_full):
                cols = tile_index * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
                values = tl.load(base + cols, mask=row_mask, other=dtype_max).to(
                    acc_dtype
                )
                acc_min = tl.minimum(acc_min, values)
                acc_max = tl.maximum(acc_max, tl.where(row_mask, values, fill_min))
            if not EVEN_N:
                cols = num_full * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
                tile_mask = row_mask & (cols < N)
                values = tl.load(base + cols, mask=tile_mask, other=dtype_max).to(
                    acc_dtype
                )
                acc_min = tl.minimum(acc_min, values)
                acc_max = tl.maximum(acc_max, tl.where(tile_mask, values, fill_min))
            row_min = tl.min(acc_min, axis=1)[:, None].to(dtype)
            row_max = tl.max(acc_max, axis=1)[:, None].to(dtype)
            tl.store(min_out + rows, row_min, mask=row_mask)
            tl.store(max_out + rows, row_max, mask=row_mask)


@libentry()
@triton.jit
def aminmax_strided_kernel(
    inp,
    min_out,
    max_out,
    R,
    INNER,
    N_ICHUNKS,
    BR: tl.constexpr,
    BI: tl.constexpr,
    EVEN_I: tl.constexpr,
):
    pid = ext.program_id(0)
    outer_index = pid // N_ICHUNKS
    inner_chunk = pid % N_ICHUNKS
    inner_cols = inner_chunk * BI + tl.arange(0, BI)
    base = inp + outer_index * R * INNER
    dtype = inp.type.element_ty
    acc_dtype = tl.float32 if dtype is tl.bfloat16 else dtype
    dtype_max = get_dtype_max(acc_dtype)
    dtype_min = get_dtype_min(acc_dtype)
    fill_min = tl.full((1,), dtype_min, acc_dtype)
    acc_min = tl.full((BI,), dtype_max, acc_dtype)
    acc_max = tl.full((BI,), dtype_min, acc_dtype)
    num_full = R // BR
    if EVEN_I:
        for tile_index in range(0, num_full):
            rows = tile_index * BR + tl.arange(0, BR)[:, None]
            values = tl.load(base + rows * INNER + inner_cols[None, :]).to(acc_dtype)
            acc_min = tl.minimum(acc_min, tl.min(values, axis=0).to(acc_dtype))
            acc_max = tl.maximum(acc_max, tl.max(values, axis=0).to(acc_dtype))
        start = num_full * BR
        for row in range(start, R):
            values = tl.load(base + row * INNER + inner_cols).to(acc_dtype)
            acc_min = tl.minimum(acc_min, values)
            acc_max = tl.maximum(acc_max, values)
        out_base = min_out + outer_index * INNER + inner_cols
        tl.store(out_base, acc_min.to(dtype))
        tl.store(max_out + outer_index * INNER + inner_cols, acc_max.to(dtype))
    else:
        inner_mask = inner_cols < INNER
        for tile_index in range(0, num_full):
            rows = tile_index * BR + tl.arange(0, BR)[:, None]
            values = tl.load(
                base + rows * INNER + inner_cols[None, :],
                mask=inner_mask[None, :],
                other=dtype_max,
            ).to(acc_dtype)
            acc_min = tl.minimum(acc_min, tl.min(values, axis=0).to(acc_dtype))
            acc_max = tl.maximum(
                acc_max,
                tl.where(inner_mask, tl.max(values, axis=0).to(acc_dtype), fill_min),
            )
        start = num_full * BR
        for row in range(start, R):
            values = tl.load(
                base + row * INNER + inner_cols, mask=inner_mask, other=dtype_max
            ).to(acc_dtype)
            acc_min = tl.minimum(acc_min, values)
            acc_max = tl.maximum(acc_max, tl.where(inner_mask, values, fill_min))
        tl.store(
            min_out + outer_index * INNER + inner_cols,
            acc_min.to(dtype),
            mask=inner_mask,
        )
        tl.store(
            max_out + outer_index * INNER + inner_cols,
            acc_max.to(dtype),
            mask=inner_mask,
        )


def _aminmax_flat_kernels(inp, min_out, max_out):
    """Launch the striped-persistent partial + single-CTA final pair."""
    numel = inp.numel()
    dtype = inp.dtype
    num_tiles = triton.cdiv(numel, _FLAT_BLOCK)
    if num_tiles == 1:
        block = triton.next_power_of_2(numel)
    else:
        block = _FLAT_BLOCK
    if num_tiles <= _FLAT_TILES_CROSSOVER:
        grid = min(num_tiles, _FLAT_GRID_SMALL)
        num_warps = 4
    else:
        grid = _FLAT_GRID_LARGE
        num_warps = 2
    min_partial = torch.empty((grid,), dtype=dtype, device=inp.device)
    max_partial = torch.empty((grid,), dtype=dtype, device=inp.device)
    with torch_device_fn.device(inp.device):
        aminmax_flat_partial_kernel[(grid,)](
            inp,
            min_partial,
            max_partial,
            numel,
            block,
            grid,
            (numel % block) != 0,
            inp.is_floating_point(),
            num_warps=num_warps,
        )
        aminmax_flat_final_kernel[(1,)](
            min_partial,
            max_partial,
            min_out,
            max_out,
            grid,
            triton.next_power_of_2(grid),
            num_warps=4,
        )


def _aminmax_flat(inp, keepdim, out):
    """Whole-tensor reduction: striped persistent CTAs + one final CTA."""
    numel = inp.numel()
    if numel == 0:
        raise RuntimeError(
            "aminmax(): cannot compute aminmax over an empty dimension as the "
            "operation has no identity."
        )
    if not inp.is_contiguous():
        inp = inp.contiguous()
    dtype = inp.dtype
    if out is not None:
        min_out = out[0] if isinstance(out, tuple) else out
        max_out = out[1] if isinstance(out, tuple) else out
        if not keepdim:
            min_out = min_out.squeeze()
            max_out = max_out.squeeze()
    else:
        if not keepdim:
            min_out = torch.empty([], dtype=dtype, device=inp.device)
            max_out = torch.empty([], dtype=dtype, device=inp.device)
        else:
            shape = [1] * inp.dim()
            min_out = torch.empty(shape, dtype=dtype, device=inp.device)
            max_out = torch.empty(shape, dtype=dtype, device=inp.device)
    _aminmax_flat_kernels(inp, min_out, max_out)
    return min_out, max_out


def _aminmax_row(view, min_out, max_out, M, N):
    block_m = min(_ROW_BLOCK_M, triton.next_power_of_2(M))
    if N >= 2048:
        block_n = 512
    else:
        wide_acc = view.element_size() >= 4 or view.dtype is torch.bfloat16
        cap = 256 if wide_acc else 512
        block_n = min(triton.next_power_of_2(N), cap)
    num_groups = triton.cdiv(M, block_m)
    grid = min(num_groups, _ROW_GRID_LIMIT)
    with torch_device_fn.device(view.device):
        aminmax_row_kernel[(grid,)](
            view,
            min_out,
            max_out,
            M,
            N,
            num_groups,
            block_m,
            block_n,
            grid,
            (M % block_m) == 0,
            (N % block_n) == 0,
            num_warps=4,
        )


def _aminmax_strided(inp, min_out, max_out, R, inner, outer):
    dtype = inp.dtype
    cap = 2048 if dtype is torch.bfloat16 else 1024
    block_inner = min(triton.next_power_of_2(inner), cap)
    n_ichunks = triton.cdiv(inner, block_inner)
    if outer * n_ichunks > _ROW_GRID_LIMIT:
        return False
    grid = (outer * n_ichunks,)
    with torch_device_fn.device(inp.device):
        aminmax_strided_kernel[grid](
            inp,
            min_out,
            max_out,
            R,
            inner,
            n_ichunks,
            _STRIDED_BLOCK_R,
            block_inner,
            (inner % block_inner) == 0,
            num_warps=4,
        )
    return True


def aminmax(inp, dim=None, keepdim=False, *, out=None):
    logger.debug("GEMS_ENFLAME_GCU400 AMINMAX")

    if dim is None:
        return _aminmax_flat(inp, keepdim, out)

    if isinstance(dim, int):
        dim = [dim]
    assert ((i >= -inp.ndim and i < inp.ndim) for i in dim), "Invalid dim"
    dtype = inp.dtype

    shape = list(inp.shape)
    dim = [d % inp.ndim for d in dim]
    N = 1
    for i in dim:
        N *= shape[i]
        shape[i] = 1
    M = inp.numel() // N

    if M == 1:
        # All reduced elements form the whole tensor, but non-reduced dims may
        # still exist (they all have size 1). Keep the dim-path output shape.
        if out is not None:
            min_out = out[0] if isinstance(out, tuple) else out
            max_out = out[1] if isinstance(out, tuple) else out
        else:
            min_out = torch.empty(shape, dtype=dtype, device=inp.device)
            max_out = torch.empty(shape, dtype=dtype, device=inp.device)
        if not inp.is_contiguous():
            inp = inp.contiguous()
        _aminmax_flat_kernels(inp, min_out, max_out)
        if not keepdim:
            min_out = min_out.squeeze(dim=dim)
            max_out = max_out.squeeze(dim=dim)
        return min_out, max_out

    if out is not None:
        min_out = out[0] if isinstance(out, tuple) else out
        max_out = out[1] if isinstance(out, tuple) else out
    else:
        min_out = torch.empty(shape, dtype=dtype, device=inp.device)
        max_out = torch.empty(shape, dtype=dtype, device=inp.device)

    ndim = inp.ndim
    single_dim = dim[0] if len(dim) == 1 else None
    if single_dim is not None and single_dim != ndim - 1 and inp.is_contiguous():
        inner = 1
        for i in range(single_dim + 1, ndim):
            inner *= shape[i]
        if inner >= 16384 or inp.numel() <= 65536:
            applied = _aminmax_strided(
                inp, min_out, max_out, inp.shape[single_dim], inner, M // inner
            )
            if applied:
                if not keepdim:
                    min_out = min_out.squeeze(dim=dim)
                    max_out = max_out.squeeze(dim=dim)
                return min_out, max_out

    view = dim_compress(inp, dim)
    _aminmax_row(view, min_out, max_out, M, N)
    if not keepdim:
        min_out = min_out.squeeze(dim=dim)
        max_out = max_out.squeeze(dim=dim)
    return min_out, max_out
