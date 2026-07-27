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
rsqrt = tl_extra_shim.rsqrt


@libentry()
@triton.jit
def _add_rms_norm_partial_kernel(
    x1,
    x2,
    partials,
    N: tl.constexpr,
    N_CHUNKS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = ext.program_id(0)
    row = pid // N_CHUNKS
    chunk = pid % N_CHUNKS
    columns = chunk * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = columns < N
    safe_columns = tl.minimum(columns, N - 1)
    offsets = row * N + safe_columns
    lhs = tl.load(x1 + offsets).to(tl.float32)
    rhs = tl.load(x2 + offsets).to(tl.float32)
    value = tl.where(mask, lhs + rhs, 0.0)
    partial = tl.sum(value * value, axis=0)
    tl.store(partials + row * N_CHUNKS + chunk, partial)


@libentry()
@triton.jit(do_not_specialize=["eps"])
def _add_rms_norm_rstd_kernel(
    partials,
    rstd,
    eps,
    N: tl.constexpr,
    N_CHUNKS: tl.constexpr,
    BLOCK_CHUNKS: tl.constexpr,
):
    row = ext.program_id(0)
    chunks = tl.arange(0, BLOCK_CHUNKS)
    mask = chunks < N_CHUNKS
    safe_chunks = tl.minimum(chunks, N_CHUNKS - 1)
    partial = tl.load(partials + row * N_CHUNKS + safe_chunks).to(tl.float32)
    partial = tl.where(mask, partial, 0.0)
    mean_square = tl.sum(partial, axis=0) / N
    tl.store(rstd + row, rsqrt(mean_square + eps))


@libentry()
@triton.jit
def _add_rms_norm_output_kernel(
    x1,
    x2,
    weight,
    rstd,
    output,
    N: tl.constexpr,
    N_CHUNKS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = ext.program_id(0)
    row = pid // N_CHUNKS
    chunk = pid % N_CHUNKS
    columns = chunk * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = columns < N
    safe_columns = tl.minimum(columns, N - 1)
    offsets = row * N + safe_columns
    lhs = tl.load(x1 + offsets).to(tl.float32)
    rhs = tl.load(x2 + offsets).to(tl.float32)
    scale = tl.load(weight + safe_columns).to(tl.float32)
    inv_rms = tl.load(rstd + row)
    result = tl.where(mask, (lhs + rhs) * inv_rms * scale, 0.0)
    tl.store(output + offsets, result, mask=mask)


@libentry()
@triton.jit(do_not_specialize=["eps"])
def _add_rms_norm_kernel(
    x1,
    x2,
    weight,
    output,
    eps,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    row = ext.program_id(0)
    row_offset = row * N
    total = tl.zeros((), dtype=tl.float32)

    for offset in range(0, N, BLOCK_SIZE):
        columns = offset + tl.arange(0, BLOCK_SIZE)
        if NEED_MASK:
            mask = columns < N
            safe_columns = tl.minimum(columns, N - 1)
            lhs = tl.load(x1 + row_offset + safe_columns).to(tl.float32)
            rhs = tl.load(x2 + row_offset + safe_columns).to(tl.float32)
        else:
            lhs = tl.load(x1 + row_offset + columns).to(tl.float32)
            rhs = tl.load(x2 + row_offset + columns).to(tl.float32)
        value = lhs + rhs
        if NEED_MASK:
            value = tl.where(mask, value, 0.0)
        total += tl.sum(value * value, axis=0)

    inv_rms = rsqrt(total / N + eps)
    for offset in range(0, N, BLOCK_SIZE):
        columns = offset + tl.arange(0, BLOCK_SIZE)
        if NEED_MASK:
            mask = columns < N
            safe_columns = tl.minimum(columns, N - 1)
            lhs = tl.load(x1 + row_offset + safe_columns).to(tl.float32)
            rhs = tl.load(x2 + row_offset + safe_columns).to(tl.float32)
            scale = tl.load(weight + safe_columns).to(tl.float32)
            result = (lhs + rhs) * inv_rms * scale
            tl.store(output + row_offset + columns, result, mask=mask)
        else:
            lhs = tl.load(x1 + row_offset + columns).to(tl.float32)
            rhs = tl.load(x2 + row_offset + columns).to(tl.float32)
            scale = tl.load(weight + columns).to(tl.float32)
            result = (lhs + rhs) * inv_rms * scale
            tl.store(output + row_offset + columns, result)


@libentry()
@triton.jit(do_not_specialize=["eps"])
def _add_rms_norm_multirow_kernel(
    x1,
    x2,
    weight,
    output,
    rows,
    eps,
    TILE_M: tl.constexpr,
    N: tl.constexpr,
):
    pid = ext.program_id(0)
    columns = tl.arange(0, N)
    row_ids = pid * TILE_M + tl.arange(0, TILE_M)
    row_mask = row_ids < rows
    offsets = row_ids[:, None] * N + columns[None, :]
    lhs = tl.load(x1 + offsets, mask=row_mask[:, None], other=0.0).to(tl.float32)
    rhs = tl.load(x2 + offsets, mask=row_mask[:, None], other=0.0).to(tl.float32)
    value = lhs + rhs
    inv_rms = rsqrt(tl.sum(value * value, axis=1) / N + eps)
    scale = tl.load(weight + columns).to(tl.float32)
    result = value * inv_rms[:, None] * scale[None, :]
    tl.store(output + offsets, result, mask=row_mask[:, None])


def add_rms_norm(x1, x2, normalized_shape, weight, eps=1e-5):
    logger.debug("GEMS_KUNLUNXIN ADD_RMS_NORM")
    if x1.shape != x2.shape:
        raise RuntimeError(f"input shapes must match: {x1.shape} vs {x2.shape}")

    normalized_shape = tuple(normalized_shape)
    normalized_size = math.prod(normalized_shape)
    if tuple(x1.shape[-len(normalized_shape) :]) != normalized_shape:
        raise RuntimeError("normalized_shape must match the trailing input dimensions")
    if weight is None or weight.numel() != normalized_size:
        raise RuntimeError("weight must contain one value per normalized element")

    x1_contiguous = x1.contiguous()
    x2_contiguous = x2.contiguous()
    weight_contiguous = weight.contiguous()
    output = torch.empty_like(x1_contiguous)
    n_elements = x1_contiguous.numel()
    if n_elements == 0:
        return output

    rows = n_elements // normalized_size
    if normalized_size <= 256 and rows >= 4096:
        tile_m = max(1, min(128, 8192 // normalized_size))
        with torch_device_fn.device(x1.device):
            _add_rms_norm_multirow_kernel[(triton.cdiv(rows, tile_m),)](
                x1_contiguous,
                x2_contiguous,
                weight_contiguous,
                output,
                rows,
                eps,
                TILE_M=tile_m,
                N=normalized_size,
                isCloseVectorization=True,
            )
    else:
        block_size = min(64 * 128, triton.next_power_of_2(normalized_size))
        need_mask = normalized_size % block_size != 0
        with torch_device_fn.device(x1.device):
            _add_rms_norm_kernel[(rows,)](
                x1_contiguous,
                x2_contiguous,
                weight_contiguous,
                output,
                eps,
                N=normalized_size,
                BLOCK_SIZE=block_size,
                NEED_MASK=need_mask,
                isCloseVectorization=True,
                buffer_size_limit=2048,
            )
    return output
