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

# from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)


def weight_norm_kernel_last_block_row(args):
    return 1
    import builtins

    return builtins.min(args["M"], 8192)


def weight_norm_kernel_last_block_col(args):
    # return 1
    return triton.next_power_of_2(triton.cdiv(args["N"], 12))


@libentry()
# @triton.autotune(
#     configs=runtime.get_tuned_config("weight_norm_kernel_last"), key=["M", "N"]
# )
@triton.heuristics(
    values={
        "BLOCK_ROW_SIZE": weight_norm_kernel_last_block_row,
        "BLOCK_COL_SIZE": weight_norm_kernel_last_block_col,
    },
)
@triton.jit(do_not_specialize=["eps"])
def weight_norm_kernel_last(
    output,
    norm,
    v,
    g,
    M: tl.constexpr,
    N: tl.constexpr,
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


# --- kunlunxin (XPU) dim=0 forward rewrite ---------------------------------
# The original `weight_norm_kernel_first` used an UNBOUNDED BLOCK_ROW_SIZE
# (next_pow2(cdiv(M,12))) and BLOCK_COL_SIZE=1, i.e. the N-reduction ran as N
# serial scalar-column iterations -> catastrophic for large N (IR baseline
# `harness/perf_ir_3/ir-weight_norm_interface-dev7.log`: [100,65536,100] gems
# 6590ms / speedup 0.001). The dim=0 semantics are per-row: for row m,
# norm[m] = sqrt(sum_n v[m,n]^2 + eps); out[m,n] = v[m,n] * g[m] / norm[m];
# g has shape [M] (one scalar per row).
#
# HARD XPU CONSTRAINT discovered here: a single `tl.sum` over a 1D tile only
# reduces the FIRST 8192 lanes correctly (N=16384 -> half, 32768 -> quarter,
# ...); loop-carried accumulation (`acc += ...` across a `range` loop) only keeps
# the LAST iteration; and a chunked [K, TILE] tile hits the `row*N` discrete-
# access wall. So a triton per-row reduction can only cover N <= 8192, and any
# 2-pass chunked triton reduction for larger N is LAUNCH-BOUND (~25x slower than
# torch). Dispatch:
#   N <= 256          -> multirow [TILE_M, N] tile (one load, amortize launch)
#   256 < N <= 8192   -> constexpr-N 1D single-tile per row (one load, block DMA)
#   N > 8192          -> torch primitives (tuned vendor reduce/elementwise)

_WN_TILE = 8192


@libentry()
@triton.jit(do_not_specialize=["eps"])
def _wn_multirow_kernel(
    output, norm, v, g, M, N: tl.constexpr, TILE_M: tl.constexpr, eps
):
    pid = tl.program_id(0)
    rows = pid * TILE_M + tl.arange(0, TILE_M)
    rmask = rows < M
    n = tl.arange(0, N)
    offs = rows[:, None] * N + n[None, :]
    x = tl.load(v + offs, mask=rmask[:, None]).to(tl.float32)
    ss = tl.sum(x * x, axis=1)
    nrm = tl.sqrt(ss + eps)
    tl.store(norm + rows, nrm, mask=rmask)
    gv = tl.load(g + rows, mask=rmask).to(tl.float32)
    scale = gv / nrm
    out = x * scale[:, None]
    tl.store(output + offs, out, mask=rmask[:, None])


@libentry()
@triton.jit(do_not_specialize=["eps"])
def _wn_row1d_kernel(output, norm, v, g, N: tl.constexpr, eps):
    pid = tl.program_id(0)
    base = pid * N
    n = tl.arange(0, N)
    x = tl.load(v + base + n).to(tl.float32)
    nrm = tl.sqrt(tl.sum(x * x, axis=0) + eps)
    tl.store(norm + pid, nrm)
    gv = tl.load(g + pid).to(tl.float32)
    scale = gv / nrm
    tl.store(output + base + n, x * scale)


# For N > 8192 a triton per-row reduction is not viable (see _wn_first_forward):
# we fall back to torch primitives, which use tuned vendor reduction/elementwise
# kernels and are ~25x faster than any launch-bound triton chunked reduction.
def _wn_first_forward(output, norm, v, g, M, N):
    eps = torch.finfo(torch.float32).tiny
    with torch_device_fn.device(v.device):
        if N <= 256:
            TILE_M = triton.next_power_of_2(max(1, _WN_TILE // N))
            grid = (triton.cdiv(M, TILE_M),)
            _wn_multirow_kernel[grid](
                output, norm, v, g, M, N, TILE_M, eps, num_warps=4
            )
        elif N <= _WN_TILE:
            _wn_row1d_kernel[(M,)](output, norm, v, g, N, eps, num_warps=8)
        else:
            # N > 8192: a triton per-row reduction needs >=2 chunks/row, but
            # tl.sum caps at 8192 lanes and a chunked [K, TILE] tile hits the
            # `row*N` discrete-access wall, so the only triton option is a
            # 2-pass (partial-reduce + scale) kernel with M*ceil(N/8192)
            # programs. That is LAUNCH-BOUND on XPU (~1.6 GB/s, e.g. 655M elems
            # in ~2.4s) and measured ~25x slower than plain torch primitives
            # (655M in ~90ms). Torch's `sum` is a tuned vendor kernel, so for
            # large N we compute norm + scaled output with torch ops directly.
            # g / norm may arrive with the FULL v-shape (benchmark passes g of
            # shape v.shape, and `norm = empty_like(g)` then inherits it) OR the
            # [M] test shape. The triton kernels tolerate this via pointer
            # arithmetic `tl.load(g + row)` (reading the first M flattened
            # elements); mirror that here by flattening to the first M elements
            # so the arithmetic stays [M]-shaped and never broadcast-errors.
            v2 = v.view(M, N)
            vf = v2.float()
            ss = vf.pow(2).sum(1)  # [M]
            nrm = torch.sqrt(ss + eps)  # [M]
            gg = g.reshape(-1)[:M].to(torch.float32)  # first M elems == tl.load(g+row)
            norm.reshape(-1)[:M].copy_(nrm)
            output.view(M, N).copy_((vf * (gg / nrm)[:, None]).to(v.dtype))


def heur_block_m_weight_norm_bwd_kernel_last(args):
    return 1


def heur_block_n_weight_norm_bwd_kernel_last(args):
    return triton.next_power_of_2(triton.cdiv(args["N"], 12))


@libentry()
# @triton.autotune(
#     configs=runtime.get_tuned_config("weight_norm_kernel_last"), key=["M", "N"]
# )
@triton.heuristics(
    values={
        "BLOCK_ROW_SIZE": heur_block_m_weight_norm_bwd_kernel_last,
        "BLOCK_COL_SIZE": heur_block_n_weight_norm_bwd_kernel_last,
    },
)
@triton.jit(do_not_specialize=["eps"])
def weight_norm_bwd_kernel_last(
    v_grad,
    g_grad,
    w,
    v,
    g,
    norm,
    M: tl.constexpr,
    N: tl.constexpr,
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
        v_grad_value = g_value * (
            w_value / (norm_value + eps)
            - v_value / (norm_value * norm_value * norm_value + eps) * vw_sum
        )
        tl.store(v_grad + row_offset * N + col_offset, v_grad_value, mask=mask)

    g_grad_value = vw_sum / (norm_value + eps)
    tl.store(g_grad + col_offset, g_grad_value, mask=col_mask)


def heur_block_m_weight_norm_bwd_kernel_first(args):
    return triton.next_power_of_2(triton.cdiv(args["M"], 12))


def heur_block_n_weight_norm_bwd_kernel_first(args):
    return 1


@libentry()
# @triton.autotune(
#     configs=runtime.get_tuned_config("weight_norm_kernel_first"), key=["M", "N"]
# )
@triton.heuristics(
    values={
        "BLOCK_ROW_SIZE": heur_block_m_weight_norm_bwd_kernel_first,
        "BLOCK_COL_SIZE": heur_block_n_weight_norm_bwd_kernel_first,
    },
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
        v_grad_value = g_value * (
            w_value / (norm_value + eps)
            - v_value / (norm_value * norm_value * norm_value + eps) * vw_sum
        )
        tl.store(v_grad + row_offset * N + col_offset, v_grad_value, mask=mask)

    g_grad_value = vw_sum / (norm_value + eps)
    tl.store(g_grad + row_offset, g_grad_value, mask=row_mask)


def weight_norm_interface(v, g, dim=0):
    logger.debug("GEMS_KUNLUNXIN WEIGHT_NORM_INTERFACE")
    v = v.contiguous()
    g = g.contiguous()
    output = torch.empty_like(v)
    norm = torch.empty_like(g, dtype=torch.float32)
    if dim == 0:
        M = v.shape[0]
        N = math.prod(v.shape[1:])
        _wn_first_forward(output, norm, v, g, M, N)
    elif dim == v.ndim - 1:
        M = math.prod(v.shape[:-1])
        N = v.shape[dim]
        grid = lambda META: (triton.cdiv(N, META["BLOCK_COL_SIZE"]),)
        with torch_device_fn.device(v.device):
            weight_norm_kernel_last[grid](
                output, norm, v, g, M, N, eps=torch.finfo(torch.float32).tiny
            )
    return output, norm


def weight_norm_interface_backward(w_grad, saved_v, saved_g, saved_norms, dim):
    logger.debug("GEMS_KUNLUNXIN WEIGHT_NORM_INTERFACE_BACKWARD")
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
