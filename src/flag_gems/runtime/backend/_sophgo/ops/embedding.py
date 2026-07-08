import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops.embedding import embedding as _fallback_embedding
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle


_SOPHGO_GRID_CAP = 64


@libentry()
@triton.jit
def _embedding_contig_masked_kernel(
    out_ptr,
    indices_ptr,
    weight_ptr,
    M,
    N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_COL_TILES: tl.constexpr,
    TOTAL_TILES,
    GRID_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)

    # Keep the physical launch to at most sg2260's 64 programs.  A program
    # visits multiple row/column tiles when M is large, avoiding additional
    # scheduler waves and loading each index once per row tile.
    for tile_id in range(pid, TOTAL_TILES, GRID_SIZE):
        tile_m = tile_id // NUM_COL_TILES
        tile_n = tile_id - tile_m * NUM_COL_TILES
        rows = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
        cols = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)

        row_mask = rows < M
        col_mask = cols < N
        mask = row_mask[:, None] & col_mask[None, :]

        row_idx = tl.load(indices_ptr + rows, mask=row_mask, other=0).to(tl.int32)
        values = tl.load(
            weight_ptr + row_idx[:, None] * N + cols[None, :],
            mask=mask,
            other=0.0,
        )
        tl.store(out_ptr + rows[:, None] * N + cols[None, :], values, mask=mask)


@libentry()
@triton.jit
def _embedding_contig_nomask_kernel(
    out_ptr,
    indices_ptr,
    weight_ptr,
    N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_COL_TILES: tl.constexpr,
    TOTAL_TILES,
    GRID_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)

    for tile_id in range(pid, TOTAL_TILES, GRID_SIZE):
        tile_m = tile_id // NUM_COL_TILES
        tile_n = tile_id - tile_m * NUM_COL_TILES
        rows = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
        cols = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
        row_idx = tl.load(indices_ptr + rows).to(tl.int32)
        values = tl.load(weight_ptr + row_idx[:, None] * N + cols[None, :])
        tl.store(out_ptr + rows[:, None] * N + cols[None, :], values)


@libentry()
@triton.jit
def indice_freq_kernel(
    indices_freq,
    indices,
    elem_cnt: tl.constexpr,
    INDICE_BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    block_start = pid * INDICE_BLOCK_SIZE

    offsets = block_start + tl.arange(0, INDICE_BLOCK_SIZE)
    mask = offsets < elem_cnt

    index_element = tl.load(indices + offsets, mask=mask)
    tl.atomic_add(indices_freq + index_element, 1, mask=mask)


@libentry()
@triton.jit(do_not_specialize=["padding_idx"])
def embedding_backward_kernel(
    grad_in,
    grad_out,
    indices,
    padding_idx,
    HAS_PADDING_IDX: tl.constexpr,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tle.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    row_idx = tl.load(indices + pid).to(tl.int32)
    if not HAS_PADDING_IDX:
        embedding_grad = tl.load(grad_out + pid * N + cols, mask=mask, other=0.0)
        if tl.constexpr(embedding_grad.dtype.is_bf16()):
            embedding_grad = embedding_grad.to(tl.float32)
        tl.atomic_add(grad_in + row_idx * N + cols, embedding_grad, mask=mask)
    else:
        if row_idx != padding_idx:
            embedding_grad = tl.load(grad_out + pid * N + cols, mask=mask, other=0.0)
            if tl.constexpr(embedding_grad.dtype.is_bf16()):
                embedding_grad = embedding_grad.to(tl.float32)
            tl.atomic_add(grad_in + row_idx * N + cols, embedding_grad, mask=mask)


@libentry()
@triton.jit(do_not_specialize=["n_rows"])
def embedding_grad_scale_kernel(
    grad_out,
    indice_freq,
    n_rows,
    N,
    BLOCK_N: tl.constexpr,
):
    row_start = tle.program_id(0)
    row_step = tle.num_programs(0)

    for row_idx in range(row_start, n_rows, row_step):
        embedding_scale = 1.0
        indice_freq_val = tl.load(indice_freq + row_idx)
        if indice_freq_val > 1:
            embedding_scale = 1.0 / indice_freq_val

        cols = tl.arange(0, BLOCK_N)
        mask = cols < N
        embedding_grad = tl.load(grad_out + row_idx * N + cols, mask=mask)
        scaled_embedding_grad = embedding_grad * embedding_scale
        tl.store(grad_out + row_idx * N + cols, scaled_embedding_grad, mask=mask)


def _select_embedding_tile_shape(M, N):
    # An embedding is memory-bound.  Keep a program's two-dimensional tile at
    # 8192 values, which lets D=128/D=256 consume 64/32 rows per tile.  This
    # bounds the live vector size for the Sophgo lowering.
    block_n = min(triton.next_power_of_2(N), 4096)
    block_m = min(64, 8192 // block_n)
    while block_m > M and block_m > 1:
        block_m //= 2
    # Predicate generation and masked gather need additional local buffers on
    # sg2260.  Keep tail tiles at 2048 elements so non-aligned dimensions can
    # compile while the common divisible path retains its 8192-value tile.
    if M % block_m != 0 or N % block_n != 0:
        block_m = min(block_m, max(1, 2048 // block_n))
    return block_m, block_n


def embedding(weight, indices, padding_idx=-1, scale_grad_by_freq=False, sparse=False):
    logging.debug("GEMS_SOPHGO_TPU EMBEDDING FORWARD")
    assert not sparse, "Currently do not support sparse format"

    M = indices.numel()
    N = weight.shape[-1]

    if M == 0:
        return torch.empty((*indices.shape, N), device=indices.device, dtype=weight.dtype)

    block_m, block_n = _select_embedding_tile_shape(M, N)
    # Wide-vector and tail paths do not outperform the generic one-row Triton
    # implementation in CModel.  Keep that implementation as a safe fallback
    # and reserve the specialized gather for the measured D=128/256 fast path.
    if N not in (128, 256) or M % block_m != 0 or N % block_n != 0:
        return _fallback_embedding(weight, indices, padding_idx, scale_grad_by_freq, sparse)

    indices = indices.contiguous()
    weight = weight.contiguous()
    output = torch.empty((*indices.shape, N), device=indices.device, dtype=weight.dtype)

    num_col_tiles = triton.cdiv(N, block_n)
    total_tiles = triton.cdiv(M, block_m) * num_col_tiles
    grid_size = min(total_tiles, _SOPHGO_GRID_CAP)
    kernel = _embedding_contig_nomask_kernel

    with torch_device_fn.device(weight.device):
        kernel[(grid_size,)](
            output,
            indices,
            weight,
            N=N,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            NUM_COL_TILES=num_col_tiles,
            TOTAL_TILES=total_tiles,
            GRID_SIZE=grid_size,
            num_warps=4,
        )

    return output


def embedding_backward(
    grad_outputs,
    indices,
    num_weights,
    padding_idx=-1,
    scale_grad_by_freq=False,
    sparse=False,
):
    logging.debug("GEMS_SOPHGO_TPU EMBEDDING BACKWARD")
    assert not sparse, "Currently do not support sparse format"

    M = indices.numel()
    N = grad_outputs.shape[-1]

    grad_inputs = torch.zeros(
        (num_weights, grad_outputs.shape[-1]),
        device=grad_outputs.device,
        dtype=(
            torch.float32
            if grad_outputs.dtype is torch.bfloat16
            else grad_outputs.dtype
        ),
    )

    if M == 0:
        return (
            grad_inputs.to(torch.bfloat16)
            if grad_outputs.dtype is torch.bfloat16
            else grad_inputs
        )

    indices = indices.contiguous()
    grad_outputs = grad_outputs.contiguous()

    if scale_grad_by_freq:
        indice_freq = torch.zeros(
            (num_weights,),
            requires_grad=False,
            device=grad_outputs.device,
            dtype=torch.int32,
        )
        indice_block_size = 256
        indice_grid = (triton.cdiv(M, indice_block_size),)

        with torch_device_fn.device(grad_outputs.device):
            indice_freq_kernel[indice_grid](
                indice_freq,
                indices,
                M,
                indice_block_size,
                isCLOSE_TTXPU_O_ATOMIC_SIM=True,
            )
    else:
        indice_freq = None

    block_n = triton.next_power_of_2(N)
    has_padding_idx = padding_idx is not None

    with torch_device_fn.device(grad_outputs.device):
        embedding_backward_kernel[(M,)](
            grad_inputs,
            grad_outputs,
            indices,
            padding_idx,
            has_padding_idx,
            N,
            BLOCK_N=block_n,
        )

    if scale_grad_by_freq:
        with torch_device_fn.device(grad_outputs.device):
            embedding_grad_scale_kernel[(M,)](
                grad_inputs,
                indice_freq,
                num_weights,
                N,
                BLOCK_N=block_n,
            )

    return (
        grad_inputs.to(torch.bfloat16)
        if grad_outputs.dtype is torch.bfloat16
        else grad_inputs
    )
