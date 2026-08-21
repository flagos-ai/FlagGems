import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def _pdist_forward_kernel(
    input,
    output,
    N: tl.constexpr,
    M: tl.constexpr,
    P: tl.constexpr,
    P_IS_INF: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    i = ext.program_id(0)
    j = ext.program_id(1)
    if j <= i:
        return

    columns = tl.arange(0, BLOCK_M)
    mask = columns < M
    lhs = tl.load(input + i * M + columns, mask=mask, other=0.0).to(tl.float32)
    rhs = tl.load(input + j * M + columns, mask=mask, other=0.0).to(tl.float32)
    difference = tl.abs(lhs - rhs)

    if P_IS_INF:
        distance = tl.max(difference, axis=0)
    elif P == 0.0:
        distance = 0.0
        for column in tl.static_range(M):
            lhs_value = tl.load(input + i * M + column).to(tl.float32)
            rhs_value = tl.load(input + j * M + column).to(tl.float32)
            distance += tl.where(lhs_value != rhs_value, 1.0, 0.0)
    elif P == 1.0:
        distance = tl.sum(difference, axis=0)
    elif P == 2.0:
        distance = tl.sqrt(tl.sum(difference * difference, axis=0))
    else:
        distance = 0.0

    output_offset = i * N - i * (i + 1) // 2 + j - i - 1
    tl.store(output + output_offset, distance)


@libentry()
@triton.jit
def _pdist_general_partial_kernel(
    input,
    partials,
    N: tl.constexpr,
    M: tl.constexpr,
    P: tl.constexpr,
    N_CHUNKS: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    i = ext.program_id(0)
    j = ext.program_id(1)
    chunk = ext.program_id(2)
    if j <= i:
        return

    output_offset = i * N - i * (i + 1) // 2 + j - i - 1
    columns = chunk * BLOCK_M + tl.arange(0, BLOCK_M)
    mask = columns < M
    lhs = tl.load(input + i * M + columns, mask=mask, other=0.0).to(tl.float32)
    rhs = tl.load(input + j * M + columns, mask=mask, other=0.0).to(tl.float32)
    difference = tl.abs(lhs - rhs)
    active = mask & (difference > 0.0)
    safe_difference = tl.where(active, difference, 1.0)
    powered = tl.where(active, tl.exp(P * tl.log(safe_difference)), 0.0)
    partial = tl.sum(powered, axis=0)
    tl.store(partials + output_offset * N_CHUNKS + chunk, partial)


@libentry()
@triton.jit
def _pdist_general_finalize_kernel(
    partials,
    output,
    N: tl.constexpr,
    P: tl.constexpr,
    N_CHUNKS: tl.constexpr,
):
    i = ext.program_id(0)
    j = ext.program_id(1)
    if j <= i:
        return

    output_offset = i * N - i * (i + 1) // 2 + j - i - 1
    power_sum = 0.0
    for chunk in tl.static_range(N_CHUNKS):
        power_sum += tl.load(partials + output_offset * N_CHUNKS + chunk)
    safe_sum = tl.where(power_sum > 0.0, power_sum, 1.0)
    distance = tl.where(power_sum > 0.0, tl.exp(tl.log(safe_sum) / P), 0.0)
    tl.store(output + output_offset, distance)


def _pdist_forward(input, p=2.0):
    logger.debug("GEMS_KUNLUNXIN _PDIST_FORWARD")
    if input.ndim != 2:
        raise RuntimeError("pdist only supports 2D input")
    if input.dtype not in (torch.float32, torch.float64):
        raise RuntimeError(
            f"pdist only supports float32 and float64, got {input.dtype}"
        )
    if p < 0:
        raise RuntimeError("pdist only supports non-negative p values")

    row_count, feature_count = input.shape
    pair_count = row_count * (row_count - 1) // 2
    output = torch.empty((pair_count,), dtype=input.dtype, device=input.device)
    if pair_count == 0:
        return output

    input_contiguous = input.contiguous()
    block_m = triton.next_power_of_2(feature_count)
    grid = (row_count, row_count)
    is_general = not math.isinf(p) and p not in (0.0, 1.0, 2.0)
    with torch_device_fn.device(input.device):
        if is_general:
            general_block_m = min(128, block_m)
            chunk_count = triton.cdiv(feature_count, general_block_m)
            partials = torch.empty(
                (pair_count, chunk_count), dtype=torch.float32, device=input.device
            )
            _pdist_general_partial_kernel[(row_count, row_count, chunk_count)](
                input_contiguous,
                partials,
                N=row_count,
                M=feature_count,
                P=float(p),
                N_CHUNKS=chunk_count,
                BLOCK_M=general_block_m,
                isCloseVectorization=True,
                buffer_size_limit=2048,
            )
            _pdist_general_finalize_kernel[grid](
                partials,
                output,
                N=row_count,
                P=float(p),
                N_CHUNKS=chunk_count,
                isCloseVectorization=True,
                buffer_size_limit=2048,
            )
        else:
            _pdist_forward_kernel[grid](
                input_contiguous,
                output,
                N=row_count,
                M=feature_count,
                P=1.0 if math.isinf(p) else float(p),
                P_IS_INF=math.isinf(p),
                BLOCK_M=block_m,
                isCloseVectorization=True,
                buffer_size_limit=2048,
            )
    return output


def pdist(input, p=2.0):
    logger.debug("GEMS_KUNLUNXIN PDIST")
    return _pdist_forward(input, p)
