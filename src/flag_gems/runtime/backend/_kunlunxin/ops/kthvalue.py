# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def _kthvalue_stage_kernel(
    input_ptr,
    selected_ptr,
    partial_value_ptr,
    partial_index_ptr,
    M,
    N,
    CHUNKS,
    CHUNK_OFFSET,
    CHUNK_ID,
    ROW_OFFSET,
    BLOCK_N: tl.constexpr,
):
    pid_m = ROW_OFFSET + ext.program_id(0)
    offsets = tl.arange(0, BLOCK_N)
    indices = CHUNK_OFFSET + offsets
    valid = indices < N
    input_base = input_ptr + pid_m * N

    previous0 = tl.load(selected_ptr + pid_m)
    previous1 = tl.load(selected_ptr + M + pid_m)
    previous2 = tl.load(selected_ptr + 2 * M + pid_m)
    values = tl.load(
        input_base + indices,
        mask=valid,
        other=float("inf"),
    ).to(tl.float32)
    values = tl.where(indices == previous0, float("inf"), values)
    values = tl.where(indices == previous1, float("inf"), values)
    values = tl.where(indices == previous2, float("inf"), values)

    chunk_value = tl.min(values, axis=0)
    chunk_index = tl.min(
        tl.where(values == chunk_value, indices, N), axis=0
    )
    partial_offset = pid_m * CHUNKS + CHUNK_ID
    tl.store(partial_value_ptr + partial_offset, chunk_value)
    tl.store(partial_index_ptr + partial_offset, chunk_index)


@libentry()
@triton.jit
def _kthvalue_finalize_kernel(
    partial_value_ptr,
    partial_index_ptr,
    selected_output_ptr,
    value_ptr,
    index_ptr,
    M,
    CHUNKS,
    ROW_OFFSET,
    BLOCK_C: tl.constexpr,
):
    pid = ROW_OFFSET + ext.program_id(0)
    offsets = tl.arange(0, BLOCK_C)
    valid = offsets < CHUNKS
    base = pid * CHUNKS
    values = tl.load(
        partial_value_ptr + base + offsets,
        mask=valid,
        other=float("inf"),
    )
    indices = tl.load(
        partial_index_ptr + base + offsets,
        mask=valid,
        other=-1,
    )
    best_value = tl.min(values, axis=0)
    best_index = tl.min(
        tl.where(values == best_value, indices, 2147483647), axis=0
    )
    tl.store(selected_output_ptr + pid, best_index)
    tl.store(value_ptr + pid, best_value)
    tl.store(index_ptr + pid, best_index)


def kthvalue(inp, k, dim=-1, keepdim=False):
    logger.debug("GEMS_KUNLUNXIN KTHVALUE")

    ndim = inp.ndim
    if dim < -ndim or dim >= ndim:
        raise IndexError(
            f"Dimension out of range (expected to be in range of [{-ndim}, {ndim - 1}], but got {dim})"
        )
    dim %= ndim
    dim_size = inp.shape[dim]
    if dim_size == 0:
        raise IndexError(
            f"kthvalue(): Expected reduction dim {dim} to have non-zero size."
        )
    if k < 1 or k > dim_size:
        raise RuntimeError(
            f"kthvalue(): selected number k out of range for dimension {dim}"
        )

    if inp.numel() == 0:
        out_shape = list(inp.shape)
        if keepdim:
            out_shape[dim] = 1
        else:
            del out_shape[dim]
        return (
            torch.empty(out_shape, dtype=inp.dtype, device=inp.device),
            torch.empty(out_shape, dtype=torch.int64, device=inp.device),
        )

    perm = [axis for axis in range(ndim) if axis != dim] + [dim]
    transposed = inp.permute(perm).contiguous()
    M = transposed.numel() // dim_size
    values = torch.empty((M,), dtype=inp.dtype, device=inp.device)
    indices = torch.empty((M,), dtype=torch.int64, device=inp.device)
    selected = torch.full((4, M), -1, dtype=torch.int32, device=inp.device)
    block_n = 512
    chunks = triton.cdiv(dim_size, block_n)
    block_c = triton.next_power_of_2(chunks)
    partial_values = torch.empty(
        (M, chunks), dtype=torch.float32, device=inp.device
    )
    partial_indices = torch.empty(
        (M, chunks), dtype=torch.int32, device=inp.device
    )

    max_programs = 16384
    with torch_device_fn.device(inp.device):
        torch_device_fn.synchronize()
        for rank in range(k):
            for chunk_id in range(chunks):
                for row_offset in range(0, M, max_programs):
                    row_count = min(max_programs, M - row_offset)
                    _kthvalue_stage_kernel[(row_count,)](
                        transposed,
                        selected,
                        partial_values,
                        partial_indices,
                        M,
                        dim_size,
                        chunks,
                        chunk_id * block_n,
                        chunk_id,
                        row_offset,
                        BLOCK_N=block_n,
                        num_warps=4,
                        buffer_size_limit=2048,
                        isCloseVectorization=True,
                    )
                    torch_device_fn.synchronize()
            for row_offset in range(0, M, max_programs):
                row_count = min(max_programs, M - row_offset)
                _kthvalue_finalize_kernel[(row_count,)](
                    partial_values,
                    partial_indices,
                    selected[rank],
                    values,
                    indices,
                    M,
                    chunks,
                    row_offset,
                    BLOCK_C=block_c,
                    num_warps=4,
                    buffer_size_limit=2048,
                    isCloseVectorization=True,
                )
                torch_device_fn.synchronize()

    out_shape = list(inp.shape)
    if keepdim:
        out_shape[dim] = 1
    else:
        del out_shape[dim]
    return values.reshape(out_shape), indices.reshape(out_shape)
