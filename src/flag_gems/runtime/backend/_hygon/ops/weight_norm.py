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

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, tl_extra_shim
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)


# The generic weight_norm autotune space (see conf tune_configs for
# weight_norm_kernel_first / _last / _kernel) enumerates block combinations up
# to BLOCK_ROW_SIZE=32 x BLOCK_COL_SIZE=2048 with num_warps up to 16. On Hygon
# those largest tiles run fine for a single forward pass, but they trigger a
# KERNEL VMFault when Triton's autotuner benchmarks them in a tight timing loop
# (do_bench replays each candidate hundreds of times), aborting the process.
# We provide a curated, conservative config space that keeps the reduction tile
# small (<= 8 x 1024 fp32) while still covering the shapes exercised by the
# op. This matches the generic numerics exactly, only the launch parameters
# differ.
def _configs_first():
    # dim == 0: reduce over columns (N). Grid over rows (M).
    return [
        triton.Config({"BLOCK_ROW_SIZE": br, "BLOCK_COL_SIZE": bc}, num_warps=w)
        for br in (1, 4, 8)
        for bc in (512, 1024)
        for w in (4, 8)
    ]


def _configs_last():
    # dim == ndim - 1: reduce over rows (M). Grid over columns (N).
    # BLOCK_ROW_SIZE is the reduction chunk; large tensors are bandwidth bound
    # and benefit from a bigger chunk / more warps. This path is *not* the
    # kernel_first VMFault site, so a wider space is safe here.
    return [
        triton.Config({"BLOCK_ROW_SIZE": br, "BLOCK_COL_SIZE": bc}, num_warps=w)
        for bc in (1, 4, 8)
        for br in (512, 1024, 2048)
        for w in (4, 8, 16)
    ]


def _configs_except_dim():
    # arbitrary middle dim: rows over v_shape1, cols over v_shape0 * v_shape2.
    return [
        triton.Config({"BLOCK_ROW_SIZE": br, "BLOCK_COL_SIZE": bc}, num_warps=w)
        for br in (1, 4, 8)
        for bc in (256, 512, 1024)
        for w in (4, 8)
    ]


@libentry()
@triton.autotune(configs=_configs_first(), key=["M", "N"])
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
@triton.autotune(configs=_configs_last(), key=["M", "N"])
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
@triton.autotune(configs=_configs_last(), key=["M", "N"])
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
@triton.autotune(configs=_configs_first(), key=["M", "N"])
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


@libentry()
@triton.autotune(
    configs=_configs_except_dim(), key=["v_shape0", "v_shape1", "v_shape2"]
)
@triton.jit(do_not_specialize=["eps"])
def weight_norm_except_dim_kernel(
    output,
    norm,
    v,
    g,
    v_shape0,
    v_shape1,
    v_shape2,
    eps,
    BLOCK_ROW_SIZE: tl.constexpr,
    BLOCK_COL_SIZE: tl.constexpr,
):
    tid_m = tl.arange(0, BLOCK_ROW_SIZE)[:, None]
    pid = ext.program_id(axis=0) * BLOCK_ROW_SIZE
    row_offset = pid + tid_m
    row_mask = row_offset < v_shape1

    tid_n = tl.arange(0, BLOCK_COL_SIZE)[None, :]
    v_block = tl.zeros([BLOCK_ROW_SIZE, BLOCK_COL_SIZE], dtype=tl.float32)
    for base in range(0, v_shape0 * v_shape2, BLOCK_COL_SIZE):
        col_offset = base + tid_n
        m_idx = col_offset // v_shape2
        n_idx = row_offset
        k_idx = col_offset % v_shape2

        mask = m_idx < v_shape0 and row_mask

        v_offsets = m_idx * v_shape1 * v_shape2 + n_idx * v_shape2 + k_idx
        v_value = tl.load(v + v_offsets, mask=mask)
        v_block += v_value * v_value
    v_sum = tl.sum(v_block, axis=1) + eps
    v_norm = tl.sqrt(v_sum[:, None])
    tl.store(norm + row_offset, v_norm, mask=row_mask)

    g_value = tl.load(g + row_offset, mask=row_mask)
    for base in range(0, v_shape0 * v_shape2, BLOCK_COL_SIZE):
        col_offset = base + tid_n
        m_idx = col_offset // v_shape2
        n_idx = row_offset
        k_idx = col_offset % v_shape2

        mask = m_idx < v_shape0 and row_mask

        v_offsets = m_idx * v_shape1 * v_shape2 + n_idx * v_shape2 + k_idx
        v_value = tl.load(v + v_offsets, mask=mask)
        out = v_value * g_value / v_norm
        tl.store(output + v_offsets, out, mask=mask)


@libentry()
@triton.autotune(
    configs=_configs_except_dim(), key=["v_shape0", "v_shape1", "v_shape2"]
)
@triton.jit(do_not_specialize=["eps"])
def weight_norm_except_dim_bwd_kernel(
    v_grad,
    g_grad,
    grad,
    v,
    g,
    norm,
    v_shape0,
    v_shape1,
    v_shape2,
    eps,
    BLOCK_ROW_SIZE: tl.constexpr,
    BLOCK_COL_SIZE: tl.constexpr,
):
    tid_m = tl.arange(0, BLOCK_ROW_SIZE)[:, None]
    pid = ext.program_id(axis=0) * BLOCK_ROW_SIZE
    row_offset = pid + tid_m
    row_mask = row_offset < v_shape1

    g_value = tl.load(g + row_offset, mask=row_mask).to(tl.float32)
    norm_value = tl.load(norm + row_offset, mask=row_mask).to(tl.float32)

    tid_n = tl.arange(0, BLOCK_COL_SIZE)[None, :]

    # norm_value is already sqrt(sum(v^2) + eps) from forward, safe to divide directly
    norm_1 = 1 / norm_value
    norm_cu = tl_extra_shim.pow(norm_value, 3)

    v_block = tl.zeros([BLOCK_ROW_SIZE, BLOCK_COL_SIZE], dtype=tl.float32)
    for base in range(0, v_shape0 * v_shape2, BLOCK_COL_SIZE):
        col_offset = base + tid_n
        m_idx = col_offset // v_shape2
        n_idx = row_offset
        k_idx = col_offset % v_shape2

        mask = m_idx < v_shape0 and row_mask

        v_offsets = m_idx * v_shape1 * v_shape2 + n_idx * v_shape2 + k_idx
        v_value = tl.load(v + v_offsets, mask=mask).to(tl.float32)
        grad_value = tl.load(grad + v_offsets, mask=mask).to(tl.float32)
        v_block += v_value * grad_value
    vw_sum = tl.sum(v_block, axis=1)[:, None]

    for base in range(0, v_shape0 * v_shape2, BLOCK_COL_SIZE):
        col_offset = base + tid_n
        m_idx = col_offset // v_shape2
        n_idx = row_offset
        k_idx = col_offset % v_shape2

        mask = m_idx < v_shape0 and row_mask

        v_offsets = m_idx * v_shape1 * v_shape2 + n_idx * v_shape2 + k_idx
        v_value = tl.load(v + v_offsets, mask=mask).to(tl.float32)
        grad_value = tl.load(grad + v_offsets, mask=mask).to(tl.float32)
        # v_grad = g * (grad / norm - v * <v, grad> / norm^3)
        v_grad_value = g_value * (grad_value * norm_1 - v_value * vw_sum / norm_cu)
        tl.store(v_grad + v_offsets, v_grad_value, mask=mask)

    # g_grad = <v, grad> / norm
    g_grad_value = vw_sum * norm_1
    tl.store(g_grad + row_offset, g_grad_value, mask=row_mask)


def weight_norm_interface(v, g, dim=0):
    logger.debug("GEMS_HYGON WEIGHT NORM INTERFACE FORWARD")
    v = v.contiguous()
    g = g.contiguous()
    output = torch.empty_like(v)
    norm = torch.empty_like(g, dtype=torch.float32)
    if dim == 0:
        M = v.shape[0]
        N = math.prod(v.shape[1:])
        grid = lambda META: (triton.cdiv(M, META["BLOCK_ROW_SIZE"]),)
        with torch_device_fn.device(v.device):
            weight_norm_kernel_first[grid](
                output, norm, v, g, M, N, eps=torch.finfo(torch.float32).tiny
            )
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
    logger.debug("GEMS_HYGON WEIGHT NORM INTERFACE BACKWARD")
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


def weight_norm_except_dim(v, g, dim):
    logger.debug("GEMS_HYGON WEIGHT NORM EXCEPT DIM FORWARD")
    v = v.contiguous()
    output = torch.empty_like(v)
    norm = torch.empty_like(g, dtype=torch.float32)
    v_shape = [
        math.prod(v.shape[:dim]),
        v.shape[dim],
        math.prod(v.shape[dim + 1 :]),
    ]

    grid = lambda META: (triton.cdiv(v_shape[1], META["BLOCK_ROW_SIZE"]),)

    with torch_device_fn.device(v.device):
        weight_norm_except_dim_kernel[grid](
            output,
            norm,
            v,
            g,
            v_shape[0],
            v_shape[1],
            v_shape[2],
            eps=torch.finfo(torch.float32).tiny,
        )
    return output, norm


def weight_norm_except_dim_backward(grad, v, g, norm, dim):
    logger.debug("GEMS_HYGON WEIGHT NORM EXCEPT DIM BACKWARD")
    grad = grad.contiguous()
    v_grad = torch.empty_like(v)
    g_grad = torch.empty_like(g)
    v_shape = [
        math.prod(v.shape[:dim]),
        v.shape[dim],
        math.prod(v.shape[dim + 1 :]),
    ]

    grid = lambda META: (triton.cdiv(v_shape[1], META["BLOCK_ROW_SIZE"]),)
    with torch_device_fn.device(v.device):
        weight_norm_except_dim_bwd_kernel[grid](
            v_grad,
            g_grad,
            grad,
            v,
            g,
            norm,
            *v_shape,
            eps=torch.finfo(torch.float32).tiny,
        )
    return v_grad, g_grad


class WeightNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v, g, dim=0):
        logger.debug("GEMS_HYGON WEIGHT NORM")
        dim = dim % v.ndim
        can_use_fused = dim == 0 or dim == v.ndim - 1
        if can_use_fused:
            output, norm = weight_norm_interface(v, g, dim)
        else:
            output, norm = weight_norm_except_dim(v, g, dim)
        ctx.save_for_backward(v, g, norm)
        ctx.dim = dim
        ctx.can_use_fused = can_use_fused
        return output

    @staticmethod
    def backward(ctx, grad):
        logger.debug("GEMS_HYGON WEIGHT NORM BACKWARD")
        v, g, norm = ctx.saved_tensors
        dim = ctx.dim
        if ctx.can_use_fused:
            v_grad, g_grad = weight_norm_interface_backward(grad, v, g, norm, dim)
        else:
            v_grad, g_grad = weight_norm_except_dim_backward(grad, v, g, norm, dim)
        return v_grad, g_grad, None


def weight_norm(v, g, dim=0):
    return WeightNorm.apply(v, g, dim)
