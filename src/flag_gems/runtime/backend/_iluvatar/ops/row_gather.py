# SPDX-License-Identifier: Apache-2.0
"""Single-axis gathers with runtime row counts for the BI-V150."""

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["rows", "count"])
def _gather_vector(X, IDX, OUT, rows, count, TILE: tl.constexpr):
    positions = tl.program_id(0) * TILE + tl.arange(0, TILE)
    present = positions < count
    index = tl.load(IDX + positions, present, other=0)
    index = tl.where(index < 0, index + rows, index)
    valid = present & (index >= 0) & (index < rows)
    value = tl.load(X + index, valid, other=0)
    tl.store(OUT + positions, value, valid)


@triton.jit(do_not_specialize=["rows", "count"])
def _gather_rows(
    X,
    IDX,
    OUT,
    rows,
    count,
    WIDTH: tl.constexpr,
    ROW_TILE: tl.constexpr,
    COL_TILE: tl.constexpr,
):
    positions = tl.program_id(0) * ROW_TILE + tl.arange(0, ROW_TILE)
    columns = tl.program_id(1) * COL_TILE + tl.arange(0, COL_TILE)
    index = tl.load(IDX + positions, positions < count, other=0)
    index = tl.where(index < 0, index + rows, index)
    valid = ((positions < count) & (index >= 0) & (index < rows))[:, None] & (
        columns < WIDTH
    )[None, :]
    value = tl.load(X + index[:, None] * WIDTH + columns[None, :], valid, other=0)
    tl.store(OUT + positions[:, None] * WIDTH + columns[None, :], value, valid)


_prepared = set()


def prepare_row_gathers(device, hidden_size):
    key = (device.index, hidden_size)
    if key in _prepared:
        return
    with torch.cuda.device(device):
        for index_type in (torch.int32, torch.int64):
            indices = torch.empty(64, device=device, dtype=index_type)
            for dtype in (torch.int32, torch.int64, torch.bfloat16):
                sample = torch.empty((2, 32), device=device, dtype=dtype)
                vector = sample.view(-1)
                binary = _gather_vector.warmup(
                    vector, indices, vector, 64, 64, 128, grid=(1,), num_warps=2
                )
                _ = binary.run
                columns = min(256, triton.next_power_of_2(hidden_size))
                binary = _gather_rows.warmup(
                    sample,
                    indices,
                    sample,
                    64,
                    64,
                    hidden_size,
                    4,
                    columns,
                    grid=(16, triton.cdiv(hidden_size, columns)),
                    num_warps=4,
                )
                _ = binary.run
    _prepared.add(key)


def _supported(input, indices):
    if (
        not input.is_cuda
        or input.ndim not in (1, 2)
        or not input.is_contiguous()
        or input.device != indices.device
        or input.dtype not in (torch.int32, torch.int64, torch.bfloat16)
        or indices.dtype not in (torch.int32, torch.int64)
    ):
        return False
    width = input.shape[1] if input.ndim == 2 else 1
    return width <= 4096 and max(input.numel(), indices.numel() * width) < 2**31


def _launch(input, indices, output):
    rows = input.shape[0]
    count = indices.numel()
    width = input.shape[1] if input.ndim == 2 else 1
    if count and width and rows:
        if width == 1:
            _gather_vector[(triton.cdiv(count, 128),)](
                input, indices, output, rows, count, 128, num_warps=2
            )
        else:
            columns = min(256, triton.next_power_of_2(width))
            _gather_rows[(triton.cdiv(count, 4), triton.cdiv(width, columns))](
                input, indices, output, rows, count, width, 4, columns, num_warps=4
            )


def gather(input, indices, output):
    if (
        not _supported(input, indices)
        or not indices.is_contiguous()
        or not output.is_contiguous()
        or output.dtype != input.dtype
        or input.device != output.device
        or output.shape != indices.shape + input.shape[1:]
    ):
        return False
    _launch(input, indices, output)
    return True


def index_first_dim(input, indices):
    if not _supported(input, indices) or (input.shape[0] == 0 and indices.numel() > 0):
        return None
    out = input.new_empty(indices.shape + input.shape[1:])
    _launch(input, indices.contiguous(), out)
    return out
