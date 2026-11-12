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

logger = logging.getLogger(__name__)

_FLAT_BLOCK = 8192
_MAX_TILE_LANES = 2048


@libentry()
@triton.jit
def _weight_norm_dot_rows_kernel(
    saved_v,
    grad_w,
    dot,
    N: tl.constexpr,
    ROWS: tl.constexpr,
):
    block = ext.program_id(0) * (ROWS * N)
    lane = tl.arange(0, ROWS * N)
    offs = block + lane
    products = tl.load(saved_v + offs).to(tl.float32) * tl.load(grad_w + offs).to(
        tl.float32
    )
    for r in tl.static_range(ROWS):
        sel = (lane >= r * N) & (lane < (r + 1) * N)
        tl.store(
            dot + ext.program_id(0) * ROWS + r, tl.sum(tl.where(sel, products, 0.0))
        )


@libentry()
@triton.jit
def _weight_norm_apply_rows_kernel(
    grad_v,
    grad_g,
    grad_w,
    saved_v,
    saved_g,
    saved_norms,
    dot,
    GROUPS: tl.constexpr,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offs = ext.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row = offs // N
    v = tl.load(saved_v + offs).to(tl.float32)
    w = tl.load(grad_w + offs).to(tl.float32)
    g = tl.load(saved_g + row).to(tl.float32)
    norm = tl.load(saved_norms + row).to(tl.float32)
    d = tl.load(dot + row)
    tl.store(grad_v + offs, (g / norm) * (w - v * (d / (norm * norm))))
    if ext.program_id(0) == 0:
        idx = tl.arange(0, GROUPS)
        mask = idx < GROUPS
        tl.store(
            grad_g + idx,
            tl.load(dot + idx, mask=mask) / tl.load(saved_norms + idx, mask=mask),
            mask=mask,
        )


@libentry()
@triton.jit
def _weight_norm_backward_rows_kernel(
    grad_v,
    grad_g,
    grad_w,
    saved_v,
    saved_g,
    saved_norms,
    N,
    BLOCK: tl.constexpr,
    ROWS: tl.constexpr,
):
    """General first-dimension path for reduction extents that are not powers of two."""
    base = ext.program_id(0) * ROWS
    cols = tl.arange(0, BLOCK)
    for r in tl.static_range(ROWS):
        row = base + r
        acc = tl.zeros((BLOCK,), dtype=tl.float32)
        for start in range(0, N, BLOCK):
            offs = row * N + start + cols
            in_col = start + cols < N
            acc += tl.where(in_col, tl.load(saved_v + offs, mask=in_col), 0.0).to(
                tl.float32
            ) * tl.where(in_col, tl.load(grad_w + offs, mask=in_col), 0.0).to(
                tl.float32
            )
        dot = tl.sum(acc, 0)
        g = tl.load(saved_g + row).to(tl.float32)
        norm = tl.load(saved_norms + row).to(tl.float32)
        scale = g / norm
        projection = dot / (norm * norm)
        for start in range(0, N, BLOCK):
            offs = row * N + start + cols
            in_col = start + cols < N
            v = tl.where(in_col, tl.load(saved_v + offs, mask=in_col), 0.0).to(
                tl.float32
            )
            w = tl.where(in_col, tl.load(grad_w + offs, mask=in_col), 0.0).to(
                tl.float32
            )
            tl.store(grad_v + offs, scale * (w - v * projection), mask=in_col)
        tl.store(grad_g + row, dot / norm)


@libentry()
@triton.jit
def _weight_norm_backward_cols_partial_kernel(
    grad_w,
    saved_v,
    partials,
    N,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
):
    block = ext.program_id(1)
    col = ext.program_id(0) * COLS + tl.arange(0, COLS)
    acc = tl.zeros((COLS,), dtype=tl.float32)
    for i in tl.static_range(ROWS):
        offs = (block * ROWS + i) * N + col
        acc += tl.load(saved_v + offs).to(tl.float32) * tl.load(grad_w + offs).to(
            tl.float32
        )
    tl.store(partials + block * N + col, acc)


@libentry()
@triton.jit
def _weight_norm_backward_cols_partial_masked_kernel(
    grad_w,
    saved_v,
    partials,
    M,
    N,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
):
    block = ext.program_id(1)
    col = ext.program_id(0) * COLS + tl.arange(0, COLS)
    safe_col = tl.where(col < N, col, 0)
    in_col = col < N
    acc = tl.zeros((COLS,), dtype=tl.float32)
    for i in tl.static_range(ROWS):
        row = tl.minimum(block * ROWS + i, M - 1)
        offs = row * N + safe_col
        v = tl.where(in_col, tl.load(saved_v + offs), 0.0).to(tl.float32)
        w = tl.where(in_col, tl.load(grad_w + offs), 0.0).to(tl.float32)
        acc += v * w
    tl.store(partials + block * N + safe_col, acc, mask=in_col)


@libentry()
@triton.jit
def _weight_norm_backward_cols_scale_kernel(
    saved_g,
    saved_norms,
    partials,
    grad_g,
    scale,
    proj,
    N,
    EXACT: tl.constexpr,
    BLOCKS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    col = ext.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    safe_col = tl.where(col < N, col, 0)
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for b in tl.static_range(BLOCKS):
        acc += tl.load(partials + b * N + safe_col)
    g = tl.load(saved_g + safe_col).to(tl.float32)
    norm = tl.load(saved_norms + safe_col).to(tl.float32)
    in_col = col < N
    if EXACT:
        tl.store(grad_g + col, acc / norm)
        tl.store(scale + col, g / norm)
        tl.store(proj + col, acc / (norm * norm))
    else:
        tl.store(grad_g + safe_col, acc / norm, mask=in_col)
        tl.store(scale + safe_col, g / norm, mask=in_col)
        tl.store(proj + safe_col, acc / (norm * norm), mask=in_col)


@libentry()
@triton.jit
def _weight_norm_backward_cols_apply_kernel(
    grad_v,
    grad_w,
    saved_v,
    scale,
    proj,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offs = ext.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    col = offs % N
    v = tl.load(saved_v + offs).to(tl.float32)
    w = tl.load(grad_w + offs).to(tl.float32)
    s = tl.load(scale + col)
    p = tl.load(proj + col)
    tl.store(grad_v + offs, s * (w - v * p))


@libentry()
@triton.jit
def _weight_norm_backward_cols_apply_fused_kernel(
    grad_v,
    grad_g,
    grad_w,
    saved_v,
    saved_g,
    saved_norms,
    partials,
    GROUPS: tl.constexpr,
    BLOCKS: tl.constexpr,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offs = ext.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    col = offs % N
    d = tl.zeros((BLOCK,), dtype=tl.float32)
    for b in tl.static_range(BLOCKS):
        d += tl.load(partials + b * N + col)
    v = tl.load(saved_v + offs).to(tl.float32)
    w = tl.load(grad_w + offs).to(tl.float32)
    g = tl.load(saved_g + col).to(tl.float32)
    norm = tl.load(saved_norms + col).to(tl.float32)
    tl.store(grad_v + offs, (g / norm) * (w - v * (d / (norm * norm))))
    if ext.program_id(0) == 0:
        idx = tl.arange(0, GROUPS)
        mask = idx < GROUPS
        total = tl.zeros((GROUPS,), dtype=tl.float32)
        for b in tl.static_range(BLOCKS):
            total += tl.load(partials + b * N + idx, mask=mask)
        tl.store(
            grad_g + idx,
            total / tl.load(saved_norms + idx, mask=mask),
            mask=mask,
        )


@libentry()
@triton.jit
def _weight_norm_backward_cols_apply_masked_kernel(
    grad_v,
    grad_w,
    saved_v,
    scale,
    proj,
    TOTAL,
    N,
    BLOCK: tl.constexpr,
):
    offs = ext.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    in_range = offs < TOTAL
    safe = tl.where(in_range, offs, 0)
    col = safe % N
    v = tl.where(in_range, tl.load(saved_v + safe), 0.0).to(tl.float32)
    w = tl.where(in_range, tl.load(grad_w + safe), 0.0).to(tl.float32)
    s = tl.load(scale + col)
    p = tl.load(proj + col)
    tl.store(grad_v + safe, s * (w - v * p), mask=in_range)


def _composite_backward(grad_w, saved_v, saved_g, saved_norms, dim):
    """PyTorch's composite formula, retained for double backward and rare dtypes."""
    broadcast_shape = [1] * saved_v.ndim
    broadcast_shape[dim] = saved_v.shape[dim]
    norms = saved_norms.to(saved_g.dtype)
    products = grad_w * saved_v
    if dim == 0:
        per_dim_sums = products.reshape(saved_v.shape[0], -1).sum(1)
    else:
        per_dim_sums = products.reshape(-1, saved_v.shape[-1]).sum(0)
    per_dim_sums = per_dim_sums.reshape(broadcast_shape)
    grad_v = (saved_g / norms) * (grad_w - saved_v * (per_dim_sums / (norms * norms)))
    grad_g = per_dim_sums / norms
    return grad_v, grad_g


def _is_pow2(value):
    return value > 0 and value & (value - 1) == 0


def _first_dim_launch(grad_v, grad_g, grad_w, saved_v, saved_g, saved_norms):
    groups = saved_v.shape[0]
    total = saved_v.numel()
    reduce_size = total // groups
    if total <= 1024:
        block = triton.next_power_of_2(max(reduce_size, 1))
        _weight_norm_backward_rows_kernel[(groups,)](
            grad_v,
            grad_g,
            grad_w,
            saved_v,
            saved_g,
            saved_norms,
            reduce_size,
            BLOCK=block,
            ROWS=1,
        )
        return

    if (
        _is_pow2(reduce_size)
        and _is_pow2(total)
        and reduce_size <= _MAX_TILE_LANES
        and total <= _FLAT_BLOCK * _FLAT_BLOCK
    ):
        rows = 1
        while (
            rows < groups
            and groups % (rows * 2) == 0
            and (rows * 2) * reduce_size <= _MAX_TILE_LANES
        ):
            rows *= 2
        block = min(_FLAT_BLOCK, total)
        if block < reduce_size:
            block = reduce_size
        dot = torch.empty(groups, dtype=torch.float32, device=saved_v.device)
        _weight_norm_dot_rows_kernel[(groups // rows,)](
            saved_v,
            grad_w,
            dot,
            N=reduce_size,
            ROWS=rows,
        )
        _weight_norm_apply_rows_kernel[(total // block,)](
            grad_v,
            grad_g,
            grad_w,
            saved_v,
            saved_g,
            saved_norms,
            dot,
            GROUPS=triton.next_power_of_2(groups),
            N=reduce_size,
            BLOCK=block,
        )
        return

    block = min(_FLAT_BLOCK, triton.next_power_of_2(max(reduce_size, 1)))
    rows = 1
    while rows < 8 and groups % (rows * 2) == 0:
        rows *= 2
    _weight_norm_backward_rows_kernel[(triton.cdiv(groups, rows),)](
        grad_v,
        grad_g,
        grad_w,
        saved_v,
        saved_g,
        saved_norms,
        reduce_size,
        BLOCK=block,
        ROWS=rows,
    )


def _last_dim_launch(grad_v, grad_g, grad_w, saved_v, saved_g, saved_norms):
    groups = saved_v.shape[-1]
    total = saved_v.numel()
    reduce_size = total // groups
    rows = 32 if reduce_size % 32 == 0 else (8 if reduce_size % 8 == 0 else 1)
    blocks = triton.cdiv(reduce_size, rows)
    partials = torch.empty((blocks, groups), dtype=torch.float32, device=saved_v.device)
    scale = torch.empty(groups, dtype=torch.float32, device=saved_v.device)
    proj = torch.empty(groups, dtype=torch.float32, device=saved_v.device)
    if _is_pow2(groups):
        _weight_norm_backward_cols_partial_kernel[(1, blocks)](
            grad_w,
            saved_v,
            partials,
            groups,
            ROWS=rows,
            COLS=groups,
        )
    else:
        cols = min(1024, triton.next_power_of_2(max(groups, 1)))
        _weight_norm_backward_cols_partial_masked_kernel[
            (triton.cdiv(groups, cols), blocks)
        ](
            grad_w,
            saved_v,
            partials,
            reduce_size,
            groups,
            ROWS=rows,
            COLS=cols,
        )
    if _is_pow2(groups) and total % _FLAT_BLOCK == 0 and blocks <= 4:
        _weight_norm_backward_cols_apply_fused_kernel[(total // _FLAT_BLOCK,)](
            grad_v,
            grad_g,
            grad_w,
            saved_v,
            saved_g,
            saved_norms,
            partials,
            GROUPS=triton.next_power_of_2(groups),
            BLOCKS=blocks,
            N=groups,
            BLOCK=_FLAT_BLOCK,
        )
        return

    scale_block = min(_MAX_TILE_LANES, triton.next_power_of_2(max(groups, 1)))
    _weight_norm_backward_cols_scale_kernel[(triton.cdiv(groups, scale_block),)](
        saved_g,
        saved_norms,
        partials,
        grad_g,
        scale,
        proj,
        groups,
        EXACT=(groups % scale_block == 0),
        BLOCKS=blocks,
        BLOCK=scale_block,
    )
    if _is_pow2(groups) and total % _FLAT_BLOCK == 0:
        _weight_norm_backward_cols_apply_kernel[(total // _FLAT_BLOCK,)](
            grad_v,
            grad_w,
            saved_v,
            scale,
            proj,
            N=groups,
            BLOCK=_FLAT_BLOCK,
        )
    else:
        _weight_norm_backward_cols_apply_masked_kernel[
            (triton.cdiv(total, _FLAT_BLOCK),)
        ](
            grad_v,
            grad_w,
            saved_v,
            scale,
            proj,
            total,
            groups,
            BLOCK=_FLAT_BLOCK,
        )


def weight_norm_differentiable_backward(grad_w, saved_v, saved_g, saved_norms, dim):
    logger.debug("GEMS_KUNLUNXIN _WEIGHT_NORM_DIFFERENTIABLE_BACKWARD")

    for name, tensor in (
        ("grad_w", grad_w),
        ("saved_v", saved_v),
        ("saved_g", saved_g),
        ("saved_norms", saved_norms),
    ):
        if not tensor.is_contiguous():
            raise RuntimeError(f"{name} must be contiguous")
    if dim != 0 and dim != saved_v.ndim - 1:
        raise RuntimeError("Expected dim to be the first or last dimension")
    if saved_v.ndim == 0:
        raise IndexError("Dimension specified as -1 but tensor has no dimensions")

    broadcast_shape = [1] * saved_v.ndim
    broadcast_shape[dim] = saved_v.shape[dim]
    expected_norm_dtype = (
        torch.float32
        if saved_v.dtype in (torch.float16, torch.bfloat16)
        else saved_v.dtype
    )
    can_fuse = (
        not (
            torch.is_grad_enabled()
            and any(x.requires_grad for x in (grad_w, saved_v, saved_g, saved_norms))
        )
        and saved_v.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and grad_w.dtype == saved_v.dtype
        and saved_g.dtype == saved_v.dtype
        and saved_norms.dtype == expected_norm_dtype
        and grad_w.shape == saved_v.shape
        and list(saved_g.shape) == broadcast_shape
        and list(saved_norms.shape) == broadcast_shape
        and grad_w.device == saved_v.device == saved_g.device == saved_norms.device
        and saved_v.numel() > 0
    )
    if not can_fuse:
        return _composite_backward(grad_w, saved_v, saved_g, saved_norms, dim)

    grad_v = torch.empty_like(saved_v)
    grad_g = torch.empty_like(saved_g)
    with torch_device_fn.device(saved_v.device):
        if dim == 0:
            _first_dim_launch(grad_v, grad_g, grad_w, saved_v, saved_g, saved_norms)
        else:
            _last_dim_launch(grad_v, grad_g, grad_w, saved_v, saved_g, saved_norms)
    return grad_v, grad_g
