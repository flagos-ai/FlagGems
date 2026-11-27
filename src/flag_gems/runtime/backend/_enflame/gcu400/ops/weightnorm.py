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
import math

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, tl_extra_shim
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)


@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("weight_norm_kernel_last"), key=["M", "N"]
)
@triton.jit(do_not_specialize=["eps"])
def weight_norm_kernel_last(
    output,
    norm,
    v,
    g,
    M,
    N,
    eps,
    BLOCK_ROW_SIZE: tl.constexpr,
    BLOCK_COL_SIZE: tl.constexpr,
):
    tx = tl.arange(0, BLOCK_COL_SIZE)[:, None]
    bx = ext.program_id(axis=0) * BLOCK_COL_SIZE
    col_offset = bx + tx
    col_mask = col_offset < N

    ty = tl.arange(0, BLOCK_ROW_SIZE)[None, :]
    v_block = tl.zeros([BLOCK_COL_SIZE, BLOCK_ROW_SIZE], dtype=tl.float32)
    for base in range(0, M, BLOCK_ROW_SIZE):
        row_offset = base + ty
        mask = row_offset < M and col_mask
        v_value = tl.load(v + row_offset * N + col_offset, mask=mask).to(tl.float32)
        v_block += v_value * v_value

    normalized = tl.sqrt(tl.sum(v_block, axis=1) + eps)
    tl.store(norm + col_offset, normalized[:, None], mask=col_mask)
    g_value = tl.load(g + col_offset, mask=col_mask).to(tl.float32)

    for base in range(0, M, BLOCK_ROW_SIZE):
        row_offset = base + ty
        mask = row_offset < M and col_mask
        v_value = tl.load(v + row_offset * N + col_offset, mask=mask).to(tl.float32)
        v_vec = v_value / normalized[:, None]
        out = v_vec * g_value
        tl.store(output + row_offset * N + col_offset, out, mask=mask)


@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("weight_norm_kernel_first"), key=["M", "N"]
)
@triton.jit(do_not_specialize=["eps"])
def weight_norm_kernel_first(
    output,
    norm,
    v,
    g,
    M,
    N,
    eps,
    BLOCK_ROW_SIZE: tl.constexpr,
    BLOCK_COL_SIZE: tl.constexpr,
):
    ty = tl.arange(0, BLOCK_ROW_SIZE)[:, None]
    by = ext.program_id(axis=0) * BLOCK_ROW_SIZE
    row_offset = by + ty
    row_mask = row_offset < M

    tx = tl.arange(0, BLOCK_COL_SIZE)[None, :]
    v_block = tl.zeros([BLOCK_ROW_SIZE, BLOCK_COL_SIZE], dtype=tl.float32)
    for base in range(0, N, BLOCK_COL_SIZE):
        col_offset = base + tx
        mask = col_offset < N and row_mask
        v_value = tl.load(v + row_offset * N + col_offset, mask=mask).to(tl.float32)
        v_block += v_value * v_value

    normalized = tl.sqrt(tl.sum(v_block, axis=1) + eps)
    tl.store(norm + row_offset, normalized[:, None], mask=row_mask)
    g_value = tl.load(g + row_offset, mask=row_mask).to(tl.float32)

    for base in range(0, N, BLOCK_COL_SIZE):
        col_offset = base + tx
        mask = col_offset < N and row_mask
        v_value = tl.load(v + row_offset * N + col_offset, mask=mask).to(tl.float32)
        v_vec = v_value / normalized[:, None]
        out = v_vec * g_value
        tl.store(output + row_offset * N + col_offset, out, mask=mask)


@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("weight_norm_kernel_last"), key=["M", "N"]
)
@triton.jit(do_not_specialize=["eps"])
def weight_norm_bwd_kernel_last(
    v_grad,
    g_grad,
    w,
    v,
    g,
    norm,
    M,
    N,
    eps,
    BLOCK_ROW_SIZE: tl.constexpr,
    BLOCK_COL_SIZE: tl.constexpr,
):
    tx = tl.arange(0, BLOCK_COL_SIZE)[:, None]
    bx = ext.program_id(axis=0) * BLOCK_COL_SIZE
    col_offset = tx + bx
    col_mask = col_offset < N

    g_value = tl.load(g + col_offset, mask=col_mask).to(tl.float32)
    norm_value = tl.load(norm + col_offset, mask=col_mask).to(tl.float32)
    # norm_value is already sqrt(sum(v^2) + eps) from forward, guaranteed > 0
    norm_1 = 1 / norm_value
    norm_3 = tl_extra_shim.pow(norm_1, 3)

    ty = tl.arange(0, BLOCK_ROW_SIZE)[None, :]

    vw_block = tl.zeros([BLOCK_COL_SIZE, BLOCK_ROW_SIZE], dtype=tl.float32)
    for base in range(0, M, BLOCK_ROW_SIZE):
        row_offset = base + ty
        mask = row_offset < M and col_mask
        v_value = tl.load(v + row_offset * N + col_offset, mask=mask).to(tl.float32)
        w_value = tl.load(w + row_offset * N + col_offset, mask=mask).to(tl.float32)
        vw_block += v_value * w_value
    vw_sum = tl.sum(vw_block, 1)[:, None]

    for base in range(0, M, BLOCK_ROW_SIZE):
        row_offset = base + ty
        mask = row_offset < M and col_mask
        v_value = tl.load(v + row_offset * N + col_offset, mask=mask).to(tl.float32)
        w_value = tl.load(w + row_offset * N + col_offset, mask=mask).to(tl.float32)
        v_grad_value = g_value * (w_value * norm_1 - v_value * norm_3 * vw_sum)
        tl.store(v_grad + row_offset * N + col_offset, v_grad_value, mask=mask)

    g_grad_value = vw_sum * norm_1
    tl.store(g_grad + col_offset, g_grad_value, mask=col_mask)


@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("weight_norm_kernel_first"), key=["M", "N"]
)
@triton.jit(do_not_specialize=["eps"])
def weight_norm_bwd_kernel_first(
    v_grad,
    g_grad,
    w,
    v,
    g,
    norm,
    M,
    N,
    eps,
    BLOCK_ROW_SIZE: tl.constexpr,
    BLOCK_COL_SIZE: tl.constexpr,
):
    ty = tl.arange(0, BLOCK_ROW_SIZE)[:, None]
    by = ext.program_id(axis=0) * BLOCK_ROW_SIZE
    row_offset = by + ty
    row_mask = row_offset < M

    g_value = tl.load(g + row_offset, mask=row_mask).to(tl.float32)
    norm_value = tl.load(norm + row_offset, mask=row_mask).to(tl.float32)
    # norm_value is already sqrt(sum(v^2) + eps) from forward, guaranteed > 0
    norm_1 = 1 / norm_value
    norm_3 = tl_extra_shim.pow(norm_1, 3)

    tx = tl.arange(0, BLOCK_COL_SIZE)[None, :]

    v_block = tl.zeros([BLOCK_ROW_SIZE, BLOCK_COL_SIZE], dtype=tl.float32)
    for base in range(0, N, BLOCK_COL_SIZE):
        col_offset = base + tx
        mask = col_offset < N and row_mask
        v_value = tl.load(v + row_offset * N + col_offset, mask=mask).to(tl.float32)
        w_value = tl.load(w + row_offset * N + col_offset, mask=mask).to(tl.float32)
        v_block += v_value * w_value
    vw_sum = tl.sum(v_block, 1)[:, None]

    for base in range(0, N, BLOCK_COL_SIZE):
        col_offset = base + tx
        mask = col_offset < N and row_mask
        v_value = tl.load(v + row_offset * N + col_offset, mask=mask).to(tl.float32)
        w_value = tl.load(w + row_offset * N + col_offset, mask=mask).to(tl.float32)
        v_grad_value = g_value * (w_value * norm_1 - v_value * norm_3 * vw_sum)
        tl.store(v_grad + row_offset * N + col_offset, v_grad_value, mask=mask)

    g_grad_value = vw_sum * norm_1
    tl.store(g_grad + row_offset, g_grad_value, mask=row_mask)


# Per-program DSM budget of GCU400 in bytes.  The single-pass kernel holds the
# fp32 input tile and the fp32 output tile at the same time, hence the factor 2.
_DSM_BYTES = 917504
# Widest row the single-pass kernel can keep resident: 2 * 4 bytes * BLOCK_ROW *
# N <= _DSM_BYTES with BLOCK_ROW >= 8 still allowed.
_ONEPASS_MAX_N = 4096
# Measured crossover: at N=64 the autotuned two-pass kernel wins (8.2us against
# 15.4us for one pass), at N=4096 one pass wins (0.086ms against 0.153ms), so
# narrow rows keep the tuned two-pass form.
_ONEPASS_MIN_N = 2048
# num_warps of the single-pass kernel, matching the warps the tuned two-pass
# configs use so the per-program vector width stays unchanged.
_FIRST_NUM_WARPS = 4
# Minimal program count of the single-pass path: one wave of the 24 GCU400 SIPs.
_MIN_PROGRAMS = 24
# Row count from which the split last-dim path pays off; below it the extra
# kernel launch and partial buffer cost more than the idle SIPs.
_SPLIT_MIN_ROWS = 256
# Column tile of the split last-dim path.
_LAST_BLOCK_COL = 512


def _onepass_block_row(M, N):
    """Rows per program so the fp32 in/out tiles stay inside the DSM."""
    budget = _DSM_BYTES // (2 * 4 * max(1, N))
    # BLOCK_ROW is a tile axis and must stay a power of two.
    budget = 1 << max(0, budget.bit_length() - 1)
    block_row = max(1, min(triton.next_power_of_2(M), budget))
    while block_row > 1 and triton.cdiv(M, block_row) < _MIN_PROGRAMS:
        block_row //= 2
    return block_row


@triton.jit(do_not_specialize=["eps"])
def _wn_first_onepass_kernel(
    output,
    norm,
    v,
    g,
    M,
    N,
    eps,
    BLOCK_ROW: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_ROW + tl.arange(0, BLOCK_ROW)
    cols = tl.arange(0, BLOCK_N)
    row_mask = rows < M
    mask = row_mask[:, None] & (cols < N)[None, :]
    offs = rows[:, None] * N + cols[None, :]

    v_value = tl.load(v + offs, mask=mask, other=0.0).to(tl.float32)
    sumsq = tl.sum(v_value * v_value, axis=1)
    norm_value = tl.sqrt(sumsq + eps)
    g_value = tl.load(g + rows, mask=row_mask, other=0.0).to(tl.float32)
    tl.store(output + offs, v_value * (g_value / norm_value)[:, None], mask=mask)
    tl.store(norm + rows, norm_value, mask=row_mask)


def _last_plan(M, N):
    """Split the last-dim reduction so the grid can fill the SIPs.

    Returns (rows_per_block, BLOCK_ROW, BLOCK_COL).  The row blocks give the
    parallelism the plain kernel lacks, BLOCK_ROW is the height of one streamed
    tile and stays inside the DSM budget.
    """
    block_col = min(triton.next_power_of_2(N), _LAST_BLOCK_COL)
    col_blocks = triton.cdiv(N, block_col)
    row_blocks = max(1, -(-_MIN_PROGRAMS // col_blocks))
    rows_per_block = max(1, triton.cdiv(M, row_blocks))
    dsm_cap = max(1, _DSM_BYTES // (3 * 4 * block_col))
    # The tile height is an arange bound, so it must be a power of two.
    dsm_cap = 1 << max(0, dsm_cap.bit_length() - 1)
    block_row = min(triton.next_power_of_2(rows_per_block), dsm_cap)
    return rows_per_block, block_row, block_col


@triton.jit
def _wn_last_partial_kernel(
    partial,
    v,
    M,
    N,
    rows_per_block,
    BLOCK_ROW: tl.constexpr,
    BLOCK_COL: tl.constexpr,
):
    row_block = tl.program_id(0)
    col_block = tl.program_id(1)
    cols = col_block * BLOCK_COL + tl.arange(0, BLOCK_COL)
    col_mask = cols < N
    lane_rows = tl.arange(0, BLOCK_ROW)
    row_start = row_block * rows_per_block

    acc = tl.zeros([BLOCK_COL], dtype=tl.float32)
    for offset in range(0, rows_per_block, BLOCK_ROW):
        row_index = row_start + offset + lane_rows
        mask = (row_index < M)[:, None] & (offset + lane_rows < rows_per_block)[:, None]
        mask &= col_mask[None, :]
        x = tl.load(
            v + row_index[:, None] * N + cols[None, :], mask=mask, other=0.0
        ).to(tl.float32)
        acc += tl.sum(x * x, axis=0)
    tl.store(partial + row_block * N + cols, acc, mask=col_mask)


@triton.jit(do_not_specialize=["eps"])
def _wn_last_apply_kernel(
    output,
    norm,
    v,
    g,
    partial,
    M,
    N,
    eps,
    n_row_blocks,
    rows_per_block,
    ROW_BLOCKS: tl.constexpr,
    BLOCK_ROW: tl.constexpr,
    BLOCK_COL: tl.constexpr,
):
    row_block = tl.program_id(0)
    col_block = tl.program_id(1)
    cols = col_block * BLOCK_COL + tl.arange(0, BLOCK_COL)
    col_mask = cols < N

    # Every program needs the whole partial column, so the buffer stays tiny
    # (row blocks are capped at one SIP wave's worth of programs).
    part_rows = tl.arange(0, ROW_BLOCKS)
    part_mask = (part_rows < n_row_blocks)[:, None] & col_mask[None, :]
    total = tl.sum(
        tl.load(
            partial + part_rows[:, None] * N + cols[None, :], mask=part_mask, other=0.0
        ),
        axis=0,
    )
    norm_value = tl.sqrt(total + eps)
    if row_block == 0:
        # One writer per column publishes the norm consumed by the backward pass.
        tl.store(norm + cols, norm_value, mask=col_mask)
    scale = tl.load(g + cols, mask=col_mask, other=0.0).to(tl.float32) / norm_value

    lane_rows = tl.arange(0, BLOCK_ROW)
    row_start = row_block * rows_per_block
    for offset in range(0, rows_per_block, BLOCK_ROW):
        row_index = row_start + offset + lane_rows
        mask = (row_index < M)[:, None] & (offset + lane_rows < rows_per_block)[:, None]
        mask &= col_mask[None, :]
        offs = row_index[:, None] * N + cols[None, :]
        x = tl.load(v + offs, mask=mask, other=0.0).to(tl.float32)
        tl.store(output + offs, x * scale[None, :], mask=mask)


def weight_norm_interface(v, g, dim=0):
    logger.debug("GEMS_ENFLAME WEIGHT_NORM_INTERFACE_FORWARD")
    v = v.contiguous()
    g = g.contiguous()
    output = torch.empty_like(v)
    norm = torch.empty_like(g, dtype=torch.float32)
    if dim == 0:
        M = v.shape[0]
        N = math.prod(v.shape[1:])
        eps = torch.finfo(torch.float32).tiny
        if _ONEPASS_MIN_N <= N <= _ONEPASS_MAX_N:
            # One pass: the whole row block stays in the DSM, so `v` is read
            # once instead of twice.
            block_row = _onepass_block_row(M, N)
            block_n = triton.next_power_of_2(N)
            with torch_device_fn.device(v.device):
                _wn_first_onepass_kernel[(triton.cdiv(M, block_row),)](
                    output,
                    norm,
                    v,
                    g,
                    M,
                    N,
                    eps,
                    BLOCK_ROW=block_row,
                    BLOCK_N=block_n,
                    num_warps=_FIRST_NUM_WARPS,
                )
        else:
            grid = lambda META: (triton.cdiv(M, META["BLOCK_ROW_SIZE"]),)
            with torch_device_fn.device(v.device):
                weight_norm_kernel_first[grid](output, norm, v, g, M, N, eps=eps)
    elif dim == v.ndim - 1:
        M = math.prod(v.shape[:-1])
        N = v.shape[dim]
        eps = torch.finfo(torch.float32).tiny
        if M >= _SPLIT_MIN_ROWS and triton.cdiv(N, _LAST_BLOCK_COL) < _MIN_PROGRAMS:
            # The plain last-dim kernel is gridded over columns only, so wide
            # rows leave the SIPs idle: 8 programs at (4096, 4096) and a single
            # program at (64, 512, 512).  Split the row reduction across
            # programs through a per-row-block partial buffer instead.
            rows_per_block, block_row, block_col = _last_plan(M, N)
            row_blocks = max(1, triton.cdiv(M, rows_per_block))
            partial = torch.empty((row_blocks, N), dtype=torch.float32, device=v.device)
            grid = (row_blocks, triton.cdiv(N, block_col))
            with torch_device_fn.device(v.device):
                _wn_last_partial_kernel[grid](
                    partial,
                    v,
                    M,
                    N,
                    rows_per_block,
                    BLOCK_ROW=block_row,
                    BLOCK_COL=block_col,
                    num_warps=_FIRST_NUM_WARPS,
                )
                _wn_last_apply_kernel[grid](
                    output,
                    norm,
                    v,
                    g,
                    partial,
                    M,
                    N,
                    eps,
                    row_blocks,
                    rows_per_block,
                    ROW_BLOCKS=triton.next_power_of_2(row_blocks),
                    BLOCK_ROW=block_row,
                    BLOCK_COL=block_col,
                    num_warps=_FIRST_NUM_WARPS,
                )
        else:
            grid = lambda META: (triton.cdiv(N, META["BLOCK_COL_SIZE"]),)
            with torch_device_fn.device(v.device):
                weight_norm_kernel_last[grid](output, norm, v, g, M, N, eps=eps)
    return output, norm


def weight_norm_interface_backward(w_grad, saved_v, saved_g, saved_norms, dim):
    logger.debug("GEMS_ENFLAME WEIGHT_NORM_INTERFACE_BACKWARD")
    w_grad = w_grad.contiguous()
    saved_v = saved_v.contiguous()
    saved_g = saved_g.contiguous()
    saved_norms = saved_norms.contiguous()
    v_grad = torch.empty_like(saved_v)
    g_grad = torch.empty_like(saved_g)

    if dim == 0:
        M = saved_v.shape[0]
        N = math.prod(saved_v.shape[1:])
        grid = lambda META: (triton.cdiv(M, META["BLOCK_ROW_SIZE"]),)
        with torch_device_fn.device(saved_v.device):
            weight_norm_bwd_kernel_first[grid](
                v_grad,
                g_grad,
                w_grad,
                saved_v,
                saved_g,
                saved_norms,
                M,
                N,
                eps=torch.finfo(torch.float32).tiny,
            )
    elif dim == saved_v.ndim - 1:
        M = math.prod(saved_v.shape[:dim])
        N = saved_v.shape[dim]
        grid = lambda META: (triton.cdiv(N, META["BLOCK_COL_SIZE"]),)
        with torch_device_fn.device(saved_v.device):
            weight_norm_bwd_kernel_last[grid](
                v_grad,
                g_grad,
                w_grad,
                saved_v,
                saved_g,
                saved_norms,
                M,
                N,
                eps=torch.finfo(torch.float32).tiny,
            )
    return v_grad, g_grad
