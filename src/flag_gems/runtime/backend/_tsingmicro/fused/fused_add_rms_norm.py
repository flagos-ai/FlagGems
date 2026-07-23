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

import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

TOTAL_CORE_NUM = torch_device_fn.get_device_properties().multi_processor_count

logger = logging.getLogger(__name__)


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


@libentry()
@triton.jit(do_not_specialize=["eps"])
def fused_add_rms_norm_fast_kernel(
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
    cols = tl.arange(0, BLOCK_SIZE)
    col_mask = cols < N_COLS
    w = tl.load(w_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
    for m_start in range(lb, ub, M_BLOCK):
        m_offset = m_start + tl.arange(0, M_BLOCK)
        mx_ptr = x_ptr + stride * m_offset
        mr_ptr = r_ptr + stride * m_offset
        _mean = tl.zeros([M_BLOCK, BLOCK_SIZE], dtype=tl.float32)
        row_mask = m_offset < ub
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
        var = tl.sum(_mean * (1.0 / N_COLS), axis=1)
        rrms = 1.0 / tl.sqrt(var + eps)
        y = xpr * rrms[:, None]
        y = y * w
        y = y.to(x_ptr.dtype.element_ty)
        tl.store(mx_ptr[:, None] + cols[None, :], y, mask=mask)


def _next_pow2(n):
    """Smallest power of 2 >= n (n >= 1)."""
    return 1 << (n - 1).bit_length()


def _m_block(M, max_m_block=64):
    """Power-of-2 M_BLOCK for tl.arange(0, M_BLOCK).

    Rounding up (next_pow2) gives larger tiles and fewer loop iterations;
    any rows past the actual per-program count are masked by row_mask.
    """
    return max(1, _next_pow2(min(triton.cdiv(M, TOTAL_CORE_NUM), max_m_block)))


def fused_add_rms_norm(x, residual, normalized_shape, weight, eps=1e-5):
    """
    This function performs fused residual addition and RMS normalization **in-place**.
    Both `x` and `residual` tensors will be modified. Use with caution if these tensors
    are reused elsewhere or require gradients.
    """
    logger.debug(
        "GEMS_TSINGMICRO FUSED_ADD_RMS_NORM, [input shape]: %s, [residual shape]: %s, [weight shape]: %s",
        x.size(),
        residual.size(),
        weight.size(),
    )
    dim = x.ndim - len(normalized_shape)
    M = math.prod(x.shape[:dim])
    N = math.prod(normalized_shape)

    x = x.contiguous()
    residual = residual.contiguous()
    weight = weight.contiguous()

    with torch_device_fn.device(x.device):
        if N <= 4096:
            BLOCK_SIZE = triton.next_power_of_2(N)
            M_BLOCK = _m_block(M)
            fused_add_rms_norm_fast_kernel[TOTAL_CORE_NUM,](
                x, residual, weight, eps, x.stride(dim - 1), M, N, BLOCK_SIZE, M_BLOCK
            )
        else:
            BLOCK_SIZE = 4096
            M_BLOCK = _m_block(M)
            fused_add_rms_norm_kernel[TOTAL_CORE_NUM,](
                x, residual, weight, eps, x.stride(dim - 1), M, N, BLOCK_SIZE, M_BLOCK
            )
    return x, residual
