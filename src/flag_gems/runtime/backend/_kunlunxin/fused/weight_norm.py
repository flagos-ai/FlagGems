import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

from ..ops import weight_norm_interface, weight_norm_interface_backward

logger = logging.getLogger(__name__)


def heur_row_weight_norm_except_dim_kernel(args):
    return triton.next_power_of_2(triton.cdiv(args["v_shape1"], 12))


def heur_col_weight_norm_except_dim_kernel(args):
    return 1


@libentry()
# @triton.autotune(
#     configs=runtime.get_tuned_config("weight_norm_kernel"),
#     key=["v_shape0", "v_shape1", "v_shape2"],
# )
@triton.heuristics(
    values={
        "BLOCK_ROW_SIZE": heur_row_weight_norm_except_dim_kernel,
        "BLOCK_COL_SIZE": heur_col_weight_norm_except_dim_kernel,
    },
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
    pid = tle.program_id(axis=0) * BLOCK_ROW_SIZE
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
# @triton.autotune(
#     configs=runtime.get_tuned_config("weight_norm_kernel"),
#     key=["v_shape0", "v_shape1", "v_shape2"],
# )
@triton.heuristics(
    values={
        "BLOCK_ROW_SIZE": heur_row_weight_norm_except_dim_kernel,
        "BLOCK_COL_SIZE": heur_col_weight_norm_except_dim_kernel,
    },
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
    pid = tle.program_id(axis=0) * BLOCK_ROW_SIZE
    row_offset = pid + tid_m
    row_mask = row_offset < v_shape1

    g_value = tl.load(g + row_offset, mask=row_mask).to(tl.float32)
    norm_value = tl.load(norm + row_offset, mask=row_mask).to(tl.float32)

    tid_n = tl.arange(0, BLOCK_COL_SIZE)[None, :]

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
        v_grad_value = g_value * (
            grad_value / (norm_value + eps)
            - v_value / (norm_value * norm_value * norm_value + eps) * vw_sum
        )
        tl.store(v_grad + v_offsets, v_grad_value, mask=mask)

    g_grad_value = vw_sum / (norm_value + eps)
    tl.store(g_grad + row_offset, g_grad_value, mask=row_mask)


def weight_norm_except_dim(v, g, dim):
    logger.debug("GEMS WEIGHT NORM EXCEPT DIM FORWARD")
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
    logger.debug("GEMS WEIGHT NORM EXCEPT DIM BACKWARD")
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
        logger.debug("GEMS WEIGHT NORM")
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
        logger.debug("GEMS WEIGHT NORM BACKWARD")
        v, g, norm = ctx.saved_tensors
        dim = ctx.dim
        if ctx.can_use_fused:
            v_grad, g_grad = weight_norm_interface_backward(grad, v, g, norm, dim)
        else:
            v_grad, g_grad = weight_norm_except_dim_backward(grad, v, g, norm, dim)
        return v_grad, g_grad, None


def weight_norm(v, g, dim=0):
    return WeightNorm.apply(v, g, dim)
