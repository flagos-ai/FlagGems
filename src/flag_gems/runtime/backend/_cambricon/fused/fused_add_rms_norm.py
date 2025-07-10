import logging
import math

import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

from ..utils import TOTAL_CORE_NUM

logger = logging.getLogger(__name__)


def get_configs():
    configs = []
    for BLOCK_SIZE in [2048, 1024, 512]:
        for M_BLOCK in range(1, 10, 2):
            for num_stages in [1, 5]:
                configs.append(
                    triton.Config(
                        {"M_BLOCK": M_BLOCK, "BLOCK_SIZE": BLOCK_SIZE},
                        num_stages=num_stages,
                        num_warps=1,
                    )
                )
    return configs


@triton.autotune(
    configs=get_configs(),
    key=["M", "N_COLS"],
    restore_value=["x_ptr", "r_ptr"],
)
@libentry()
@triton.jit(do_not_specialize=["eps"])
def fused_add_rms_norm_kernel(
    x_ptr,
    r_ptr,
    w_ptr,
    eps,
    stride,
    M,
    N_COLS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    M_BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    pnum = tl.num_programs(axis=0)
    M_OUT_BLOCK = tl.cdiv(M, pnum)

    lb = pid * M_OUT_BLOCK
    ub = tl.minimum((pid + 1) * M_OUT_BLOCK, M)
    for m_start in range(lb, ub, M_BLOCK):
        m_offset = m_start + tl.arange(0, M_BLOCK)
        mx_ptr = x_ptr + stride * m_offset
        mr_ptr = r_ptr + stride * m_offset
        _mean = tl.zeros([M_BLOCK, BLOCK_SIZE], dtype=tl.float32)
        for offset in range(0, N_COLS, BLOCK_SIZE):
            cols = offset + tl.arange(0, BLOCK_SIZE)
            row_mask = m_offset < ub
            col_mask = cols < N_COLS
            mask = row_mask[:, None] & col_mask[None, :]
            x = tl.load(mx_ptr[:, None] + cols[None, :], mask=mask, other=0.0).to(
                tl.float32
            )
            r = tl.load(mr_ptr[:, None] + cols[None, :], mask=mask, other=0.0).to(
                tl.float32
            )
            xpr = x + r
            tl.store(mr_ptr[:, None] + cols[None, :], xpr, mask=mask)
            _mean += xpr * xpr

        # Since `_mean * (1 / N_COLS)` performs better, make this change.
        # var = tl.sum(_mean / N_COLS, axis=1)
        var = tl.sum(_mean * (1.0 / N_COLS), axis=1)
        rrms = 1.0 / tl.sqrt(var + eps)

        for offset in range(0, N_COLS, BLOCK_SIZE):
            cols = offset + tl.arange(0, BLOCK_SIZE)
            row_mask = m_offset < ub
            col_mask = cols < N_COLS
            mask = row_mask[:, None] & col_mask[None, :]

            xpr = tl.load(mr_ptr[:, None] + cols[None, :], mask=mask, other=0.0).to(
                tl.float32
            )
            w = tl.load(w_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
            y = xpr * rrms[:, None]
            y = y * w
            y = y.to(x_ptr.dtype.element_ty)
            tl.store(mx_ptr[:, None] + cols[None, :], y, mask=mask)


def fused_add_rms_norm(x, residual, normalized_shape, weight, eps=1e-5):
    """
    This function performs fused residual addition and RMS normalization **in-place**.
    Both `x` and `residual` tensors will be modified. Use with caution if these tensors
    are reused elsewhere or require gradients.
    """
    logger.debug("GEMS_CAMBRICON FUSED_ADD_RMSNORM FORWARD")
    dim = x.ndim - len(normalized_shape)
    M = math.prod(x.shape[:dim])
    N = math.prod(normalized_shape)

    x = x.contiguous()
    residual = residual.contiguous()
    weight = weight.contiguous()

    with torch_device_fn.device(x.device):
        fused_add_rms_norm_kernel[TOTAL_CORE_NUM,](
            x, residual, weight, eps, x.stride(dim - 1), M, N
        )
    return x, residual
