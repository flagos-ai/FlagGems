import logging
import math

import torch
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
)
@libentry()
@triton.jit(do_not_specialize=["eps"])
def skip_rms_norm_kernel(
    x_ptr,
    y_ptr,
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
        my_ptr = y_ptr + stride * m_offset
        _mean = tl.zeros([M_BLOCK, BLOCK_SIZE], dtype=tl.float32)
        for offset in range(0, N_COLS, BLOCK_SIZE):
            cols = offset + tl.arange(0, BLOCK_SIZE)
            mask = (m_offset < M)[:, None] & (cols < N_COLS)[None, :]
            x = tl.load(
                mx_ptr[:, None] + cols[None, :],
                mask=mask,
                other=0.0,
                eviction_policy="evict_last",
            ).to(tl.float32)
            r = tl.load(
                mr_ptr[:, None] + cols[None, :],
                mask=mask,
                other=0.0,
                eviction_policy="evict_last",
            ).to(tl.float32)
            x += r
            _mean += x * x

        # Since `_mean * (1 / N_COLS)` performs better, make this change.
        # var = tl.sum(_mean / N_COLS, axis=1)
        var = tl.sum(_mean * (1 / N_COLS), axis=1)
        rrms = 1 / tl.sqrt(var + eps)

        for offset in range(0, N_COLS, BLOCK_SIZE):
            cols = offset + tl.arange(0, BLOCK_SIZE)
            mask = (m_offset < M)[:, None] & (cols < N_COLS)[None, :]
            x = tl.load(
                mx_ptr[:, None] + cols[None, :],
                mask=mask,
                other=0.0,
                eviction_policy="evict_first",
            ).to(tl.float32)
            r = tl.load(
                mr_ptr[:, None] + cols[None, :],
                mask=mask,
                other=0.0,
                eviction_policy="evict_last",
            ).to(tl.float32)
            x += r
            w = tl.load(
                w_ptr + cols,
                mask=cols < N_COLS,
                other=0.0,
                eviction_policy="evict_first",
            )
            y = (x * rrms[:, None]).to(y_ptr.dtype.element_ty) * w
            tl.store(my_ptr[:, None] + cols[None, :], y, mask=mask)


class SkipRmsNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, residual, normalized_shape, weight, eps=1e-5):
        logger.debug("GEMS_CAMBRICON SKIP RMSNORM FORWARD")
        dim = x.ndim - len(normalized_shape)
        M = math.prod(x.shape[:dim])
        N = math.prod(normalized_shape)

        x = x.contiguous()
        residual = residual.contiguous()
        weight = weight.contiguous()
        y = torch.empty_like(x)

        with torch_device_fn.device(x.device):
            skip_rms_norm_kernel[TOTAL_CORE_NUM,](
                x, y, residual, weight, eps, x.stride(dim - 1), M, N
            )
        return y


def skip_rms_norm(x, residual, normalized_shape, weight, eps=1e-5):
    return SkipRmsNorm.apply(x, residual, normalized_shape, weight, eps)
