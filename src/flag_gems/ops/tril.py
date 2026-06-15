import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)


@triton.jit
def _tril_tile_kernel(
    in_ptr,
    out_ptr,
    diag: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_b = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
    mask = (offs_m < M) & (offs_n < N)

    base = pid_b * (M * N)
    idxs = base + offs_m * N + offs_n

    row_start = pid_m * BLOCK_M
    row_end = row_start + BLOCK_M - 1
    col_start = pid_n * BLOCK_N
    col_end = col_start + BLOCK_N - 1

    if col_start > row_end + diag:
        tl.store(out_ptr + idxs, 0.0, mask=mask)
        return

    if col_end <= row_start + diag:
        x = tl.load(in_ptr + idxs, mask=mask, other=0.0)
        tl.store(out_ptr + idxs, x, mask=mask)
        return

    keep = offs_n <= (offs_m + diag)
    x = tl.load(in_ptr + idxs, mask=mask & keep, other=0.0)
    tl.store(out_ptr + idxs, x, mask=mask)


@triton.jit
def _tril_rows_kernel(
    in_ptr,
    out_ptr,
    diag,
    M: tl.constexpr,
    N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    row_mask = offs_m < M
    base = pid_b * (M * N)
    row_base = base + offs_m * N
    row_start = pid_m * BLOCK_M
    row_end = row_start + BLOCK_M - 1

    for col_start in range(0, N, BLOCK_N):
        offs_n = col_start + tl.arange(0, BLOCK_N)[None, :]
        mask = row_mask & (offs_n < N)
        idxs = row_base + offs_n

        col_end = col_start + BLOCK_N - 1
        if col_start > row_end + diag:
            tl.store(out_ptr + idxs, 0.0, mask=mask)
        elif col_end <= row_start + diag:
            x = tl.load(in_ptr + idxs, mask=mask, other=0.0)
            tl.store(out_ptr + idxs, x, mask=mask)
        else:
            keep = offs_n <= (offs_m + diag)
            x = tl.load(in_ptr + idxs, mask=mask & keep, other=0.0)
            tl.store(out_ptr + idxs, x, mask=mask)


@triton.jit
def _tril_flat_kernel(
    in_ptr,
    out_ptr,
    total,
    diag,
    M: tl.constexpr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total

    matrix_offsets = offsets % (M * N)
    rows = matrix_offsets // N
    cols = matrix_offsets - rows * N
    keep = cols <= rows + diag

    x = tl.load(in_ptr + offsets, mask=mask & keep, other=0.0)
    tl.store(out_ptr + offsets, x, mask=mask)


@triton.jit
def _tril_exact_row_kernel(
    in_ptr,
    out_ptr,
    diag,
    M: tl.constexpr,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)

    offs_n = tl.arange(0, BLOCK_N)
    idxs = pid_b * (M * N) + pid_m * N + offs_n
    keep = offs_n <= pid_m + diag
    x = tl.load(in_ptr + idxs, mask=keep, other=0.0)
    tl.store(out_ptr + idxs, x)


@triton.jit
def _tril_exact_diag0_tile_kernel(
    in_ptr,
    out_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_b = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
    idxs = pid_b * (M * N) + offs_m * N + offs_n

    row_start = pid_m * BLOCK_M
    row_end = row_start + BLOCK_M - 1
    col_start = pid_n * BLOCK_N
    col_end = col_start + BLOCK_N - 1

    if col_start > row_end:
        tl.store(out_ptr + idxs, 0.0)
        return

    if col_end <= row_start:
        x = tl.load(in_ptr + idxs)
        tl.store(out_ptr + idxs, x)
        return

    keep = offs_n <= offs_m
    x = tl.load(in_ptr + idxs, mask=keep, other=0.0)
    tl.store(out_ptr + idxs, x)


@triton.jit
def _tril_inplace_zero_tile_kernel(
    ptr,
    diag: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_b = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
    mask = (offs_m < M) & (offs_n < N)
    idxs = pid_b * (M * N) + offs_m * N + offs_n

    row_start = pid_m * BLOCK_M
    col_end = pid_n * BLOCK_N + BLOCK_N - 1
    if col_end <= row_start + diag:
        return

    row_end = row_start + BLOCK_M - 1
    col_start = pid_n * BLOCK_N
    if col_start > row_end + diag:
        tl.store(ptr + idxs, 0.0, mask=mask)
        return

    zero = offs_n > offs_m + diag
    tl.store(ptr + idxs, 0.0, mask=mask & zero)


@triton.jit
def _tril_inplace_zero_strided_tile_kernel(
    ptr,
    diag: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    B0: tl.constexpr,
    B1: tl.constexpr,
    B2: tl.constexpr,
    B3: tl.constexpr,
    B4: tl.constexpr,
    B5: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    S2: tl.constexpr,
    S3: tl.constexpr,
    S4: tl.constexpr,
    S5: tl.constexpr,
    STRIDE_M: tl.constexpr,
    STRIDE_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_b = tl.program_id(2)

    b = pid_b
    i5 = b % B5
    b = b // B5
    i4 = b % B4
    b = b // B4
    i3 = b % B3
    b = b // B3
    i2 = b % B2
    b = b // B2
    i1 = b % B1
    i0 = b // B1
    batch_offset = i0 * S0 + i1 * S1 + i2 * S2 + i3 * S3 + i4 * S4 + i5 * S5

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
    mask = (offs_m < M) & (offs_n < N)
    idxs = batch_offset + offs_m * STRIDE_M + offs_n * STRIDE_N

    row_start = pid_m * BLOCK_M
    col_end = pid_n * BLOCK_N + BLOCK_N - 1
    if col_end <= row_start + diag:
        return

    row_end = row_start + BLOCK_M - 1
    col_start = pid_n * BLOCK_N
    if col_start > row_end + diag:
        tl.store(ptr + idxs, 0.0, mask=mask)
        return

    zero = offs_n > offs_m + diag
    tl.store(ptr + idxs, 0.0, mask=mask & zero)


@triton.jit
def _tril_strided_out_tile_kernel(
    in_ptr,
    out_ptr,
    diag,
    M: tl.constexpr,
    N: tl.constexpr,
    B0: tl.constexpr,
    B1: tl.constexpr,
    B2: tl.constexpr,
    B3: tl.constexpr,
    B4: tl.constexpr,
    B5: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    S2: tl.constexpr,
    S3: tl.constexpr,
    S4: tl.constexpr,
    S5: tl.constexpr,
    STRIDE_M: tl.constexpr,
    STRIDE_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_b = tl.program_id(2)

    b = pid_b
    i5 = b % B5
    b = b // B5
    i4 = b % B4
    b = b // B4
    i3 = b % B3
    b = b // B3
    i2 = b % B2
    b = b // B2
    i1 = b % B1
    i0 = b // B1
    out_batch_offset = i0 * S0 + i1 * S1 + i2 * S2 + i3 * S3 + i4 * S4 + i5 * S5

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
    mask = (offs_m < M) & (offs_n < N)
    in_idxs = pid_b * (M * N) + offs_m * N + offs_n
    out_idxs = out_batch_offset + offs_m * STRIDE_M + offs_n * STRIDE_N

    row_start = pid_m * BLOCK_M
    row_end = row_start + BLOCK_M - 1
    col_start = pid_n * BLOCK_N
    col_end = col_start + BLOCK_N - 1

    if col_start > row_end + diag:
        tl.store(out_ptr + out_idxs, 0.0, mask=mask)
        return

    if col_end <= row_start + diag:
        x = tl.load(in_ptr + in_idxs, mask=mask, other=0.0)
        tl.store(out_ptr + out_idxs, x, mask=mask)
        return

    keep = offs_n <= (offs_m + diag)
    x = tl.load(in_ptr + in_idxs, mask=mask & keep, other=0.0)
    tl.store(out_ptr + out_idxs, x, mask=mask)


def _check_input(input: torch.Tensor):
    if input.dim() < 2:
        raise RuntimeError("tril: input tensor must have at least 2 dimensions")


def _empty_contiguous_like(input: torch.Tensor):
    if input.is_contiguous():
        return torch.empty_like(input)
    return torch.empty_like(input, memory_format=torch.contiguous_format)


def _zero_out(out: torch.Tensor):
    if out.numel() == 0:
        return out
    if out.is_contiguous():
        return out.zero_()
    return out.fill_(0)


def _is_power_of_2(value: int):
    return value > 0 and (value & (value - 1)) == 0


def _has_internal_overlap_from_strides(tensor: torch.Tensor):
    span = 1
    strides_and_sizes = sorted(
        (stride, size)
        for size, stride in zip(tensor.shape, tensor.stride())
        if size > 1
    )
    for stride, size in strides_and_sizes:
        if stride < span:
            return True
        span += stride * (size - 1)
    return False


def _tensors_overlap(left: torch.Tensor, right: torch.Tensor):
    try:
        return torch._C._overlaps(left, right)
    except AttributeError:
        return True


def _can_use_strided_out_kernel(input: torch.Tensor, out: torch.Tensor):
    if out.is_contiguous() or out.numel() == 0:
        return False
    if out.dim() - 2 > 6:
        return False
    if _has_internal_overlap_from_strides(out):
        return False
    if input.is_contiguous() and _tensors_overlap(input, out):
        return False
    return True


_WIDE_EXACT_ROW_MIN_N = 2048
_WIDE_EXACT_ROW_MAX_N = 8192
_WIDE_EXACT_ROW_MIN_ROWS = 256
_WIDE_EXACT_ROW_ALWAYS_ROW_M = 512
_TINY_BATCHED_TILE_MIN_BATCH = 128


def _use_wide_exact_row(M: int, N: int, batch: int):
    # One exact-row program covers one matrix row with BLOCK_N == N.  Use it for
    # wide power-of-two rows where it avoids the flat kernel's div/mod indexing,
    # but require enough row programs to keep occupancy reasonable.
    if N < _WIDE_EXACT_ROW_MIN_N or N > _WIDE_EXACT_ROW_MAX_N or not _is_power_of_2(N):
        return False

    rows = M * batch
    if M >= _WIDE_EXACT_ROW_ALWAYS_ROW_M:
        return True
    return N <= 4096 and rows >= _WIDE_EXACT_ROW_MIN_ROWS


def _use_tiny_batched_tile(M: int, N: int, batch: int):
    return batch >= _TINY_BATCHED_TILE_MIN_BATCH and M <= 32 and N <= 32


def _wide_exact_row_warps(N: int):
    if N <= 4096:
        return 2
    return 4


def _launch_tile(
    input: torch.Tensor,
    out: torch.Tensor,
    diagonal: int,
    block_m: int = 32,
    block_n: int = 32,
    num_warps: int = 4,
    num_stages: int = 2,
):
    M, N = input.shape[-2:]
    total = input.numel()
    if total == 0:
        return out

    batch = total // (M * N)
    grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n), batch)
    with torch_device_fn.device(input.device):
        _tril_tile_kernel[grid](
            input,
            out,
            int(diagonal),
            M,
            N,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    return out


def _launch_flat(
    input: torch.Tensor,
    out: torch.Tensor,
    diagonal: int,
    block_size: int = 1024,
    num_warps: int = 4,
    num_stages: int = 2,
):
    M, N = input.shape[-2:]
    total = input.numel()
    if total == 0:
        return out

    grid = (triton.cdiv(total, block_size),)
    with torch_device_fn.device(input.device):
        _tril_flat_kernel[grid](
            input,
            out,
            total,
            int(diagonal),
            M,
            N,
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    return out


def _launch_rows(
    input: torch.Tensor,
    out: torch.Tensor,
    diagonal: int,
    block_m: int = 32,
    block_n: int = 64,
    num_warps: int = 4,
    num_stages: int = 2,
):
    M, N = input.shape[-2:]
    total = input.numel()
    if total == 0:
        return out

    batch = total // (M * N)
    grid = (triton.cdiv(M, block_m), batch)
    with torch_device_fn.device(input.device):
        _tril_rows_kernel[grid](
            input,
            out,
            int(diagonal),
            M,
            N,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    return out


def _launch_exact_row(
    input: torch.Tensor,
    out: torch.Tensor,
    diagonal: int,
    num_warps: int = 4,
    num_stages: int = 2,
):
    M, N = input.shape[-2:]
    total = input.numel()
    if total == 0:
        return out

    batch = total // (M * N)
    grid = (M, batch)
    with torch_device_fn.device(input.device):
        _tril_exact_row_kernel[grid](
            input,
            out,
            int(diagonal),
            M,
            N,
            BLOCK_N=N,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    return out


def _launch_exact_diag0_tile(
    input: torch.Tensor,
    out: torch.Tensor,
    block_m: int,
    block_n: int,
    num_warps: int = 4,
    num_stages: int = 2,
):
    M, N = input.shape[-2:]
    total = input.numel()
    if total == 0:
        return out

    batch = total // (M * N)
    grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n), batch)
    with torch_device_fn.device(input.device):
        _tril_exact_diag0_tile_kernel[grid](
            input,
            out,
            M,
            N,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    return out


def _launch_tril_inplace_contiguous(
    input: torch.Tensor,
    diagonal: int,
    block_m: int = 16,
    block_n: int = 64,
    num_warps: int = 4,
    num_stages: int = 2,
):
    M, N = input.shape[-2:]
    if input.numel() == 0:
        return input

    active_rows = min(M, max(0, N - 1 - diagonal))
    if active_rows == 0:
        return input

    batch = input.numel() // (M * N)
    grid = (triton.cdiv(active_rows, block_m), triton.cdiv(N, block_n), batch)
    with torch_device_fn.device(input.device):
        _tril_inplace_zero_tile_kernel[grid](
            input,
            int(diagonal),
            M,
            N,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    return input


def _launch_tril_inplace_strided(
    input: torch.Tensor,
    diagonal: int,
    block_m: int = 16,
    block_n: int = 64,
    num_warps: int = 4,
    num_stages: int = 2,
):
    M, N = input.shape[-2:]
    if input.numel() == 0:
        return input

    active_rows = min(M, max(0, N - 1 - diagonal))
    if active_rows == 0:
        return input

    batch_shape = list(input.shape[:-2])
    batch_strides = list(input.stride()[:-2])
    batch = 1
    for size in batch_shape:
        batch *= size

    if len(batch_shape) > 6:
        tmp = _empty_contiguous_like(input)
        _launch_tril(input, tmp, diagonal)
        input.copy_(tmp)
        return input

    batch_shape.extend([1] * (6 - len(batch_shape)))
    batch_strides.extend([0] * (6 - len(batch_strides)))
    stride_m, stride_n = input.stride()[-2:]

    grid = (triton.cdiv(active_rows, block_m), triton.cdiv(N, block_n), batch)
    with torch_device_fn.device(input.device):
        _tril_inplace_zero_strided_tile_kernel[grid](
            input,
            int(diagonal),
            M,
            N,
            B0=batch_shape[0],
            B1=batch_shape[1],
            B2=batch_shape[2],
            B3=batch_shape[3],
            B4=batch_shape[4],
            B5=batch_shape[5],
            S0=batch_strides[0],
            S1=batch_strides[1],
            S2=batch_strides[2],
            S3=batch_strides[3],
            S4=batch_strides[4],
            S5=batch_strides[5],
            STRIDE_M=stride_m,
            STRIDE_N=stride_n,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    return input


def _launch_tril_strided_out(
    input: torch.Tensor,
    out: torch.Tensor,
    diagonal: int,
    block_m: int = 32,
    block_n: int = 64,
    num_warps: int = 4,
    num_stages: int = 2,
):
    M, N = input.shape[-2:]
    if input.numel() == 0:
        return out

    input_to_use = input if input.is_contiguous() else input.contiguous()
    batch_shape = list(out.shape[:-2])
    batch_strides = list(out.stride()[:-2])
    batch = 1
    for size in batch_shape:
        batch *= size

    batch_shape.extend([1] * (6 - len(batch_shape)))
    batch_strides.extend([0] * (6 - len(batch_strides)))
    stride_m, stride_n = out.stride()[-2:]

    grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n), batch)
    with torch_device_fn.device(input.device):
        _tril_strided_out_tile_kernel[grid](
            input_to_use,
            out,
            int(diagonal),
            M,
            N,
            B0=batch_shape[0],
            B1=batch_shape[1],
            B2=batch_shape[2],
            B3=batch_shape[3],
            B4=batch_shape[4],
            B5=batch_shape[5],
            S0=batch_strides[0],
            S1=batch_strides[1],
            S2=batch_strides[2],
            S3=batch_strides[3],
            S4=batch_strides[4],
            S5=batch_strides[5],
            STRIDE_M=stride_m,
            STRIDE_N=stride_n,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    return out


def _launch_tril(input: torch.Tensor, out: torch.Tensor, diagonal: int):
    M, N = input.shape[-2:]
    if input.numel() == 0:
        return out

    if diagonal <= -M:
        return _zero_out(out)
    if diagonal >= N - 1:
        out.copy_(input)
        return out

    input_to_use = input if input.is_contiguous() else input.contiguous()
    batch = input_to_use.numel() // (M * N)
    if _use_wide_exact_row(M, N, batch):
        return _launch_exact_row(
            input_to_use,
            out,
            diagonal,
            num_warps=_wide_exact_row_warps(N),
        )
    if batch == 1 and M == 1024 and N == 1024 and diagonal == 0:
        return _launch_exact_diag0_tile(
            input_to_use,
            out,
            block_m=32,
            block_n=64,
            num_warps=4,
        )
    if batch >= 1 and M == 512 and N == 512 and diagonal == 0:
        return _launch_exact_diag0_tile(
            input_to_use,
            out,
            block_m=16,
            block_n=128,
            num_warps=4,
        )
    if _use_tiny_batched_tile(M, N, batch):
        return _launch_tile(
            input_to_use,
            out,
            diagonal,
            block_m=16,
            block_n=64,
            num_warps=2,
        )
    if M <= 64 and N <= 64:
        return _launch_rows(
            input_to_use,
            out,
            diagonal,
            block_m=2,
            block_n=64,
            num_warps=1,
        )
    if N >= 2048:
        return _launch_flat(
            input_to_use,
            out,
            diagonal,
            block_size=4096,
            num_warps=4,
        )
    if batch > 1:
        if M >= 256 and N >= 256:
            return _launch_tile(
                input_to_use,
                out,
                diagonal,
                block_m=16,
                block_n=64,
                num_warps=4,
            )
        return _launch_rows(
            input_to_use,
            out,
            diagonal,
            block_m=8,
            block_n=512,
            num_warps=1,
        )
    if N >= 512:
        return _launch_tile(
            input_to_use,
            out,
            diagonal,
            block_m=64,
            block_n=64,
            num_warps=4,
        )
    if M == 256 and N == 256:
        return _launch_rows(
            input_to_use,
            out,
            diagonal,
            block_m=8,
            block_n=256,
            num_warps=2,
        )
    return _launch_rows(
        input_to_use,
        out,
        diagonal,
        block_m=8,
        block_n=512,
        num_warps=1,
    )


def tril(input: torch.Tensor, diagonal: int = 0):
    logger.debug("GEMS TRIL")
    _check_input(input)

    out = _empty_contiguous_like(input)
    return _launch_tril(input, out, int(diagonal))


def tril_(input: torch.Tensor, diagonal: int = 0):
    logger.debug("GEMS TRIL_")
    _check_input(input)

    diagonal = int(diagonal)
    if input.numel() == 0:
        return input

    M, N = input.shape[-2:]
    if diagonal >= N - 1:
        return input
    if diagonal <= -M:
        return _zero_out(input)

    if input.is_contiguous():
        return _launch_tril_inplace_contiguous(input, diagonal)

    return _launch_tril_inplace_strided(input, diagonal)


def tril_out(input: torch.Tensor, diagonal: int = 0, *, out: torch.Tensor = None):
    logger.debug("GEMS TRIL.OUT")

    if out is None:
        return tril(input, diagonal)

    _check_input(input)
    if out.dtype != input.dtype:
        raise RuntimeError(
            f"Expected out tensor to have dtype {input.dtype}, but got {out.dtype} instead"
        )
    if out.device != input.device:
        raise RuntimeError(
            f"Expected out tensor to be on device {input.device}, but got {out.device} instead"
        )
    if out.shape != input.shape:
        out.resize_(input.shape)

    if out.is_contiguous():
        return _launch_tril(input, out, int(diagonal))

    if input.numel() == 0:
        return out
    M, N = input.shape[-2:]
    if diagonal <= -M:
        return _zero_out(out)
    if diagonal >= N - 1:
        out.copy_(input)
        return out

    if _can_use_strided_out_kernel(input, out):
        batch = input.numel() // (M * N)
        if M <= 64 and N <= 64:
            return _launch_tril_strided_out(
                input,
                out,
                int(diagonal),
                block_m=16,
                block_n=64,
                num_warps=2,
            )
        if batch > 1 and M >= 256 and N >= 256:
            return _launch_tril_strided_out(
                input,
                out,
                int(diagonal),
                block_m=16,
                block_n=64,
                num_warps=4,
            )
        return _launch_tril_strided_out(
            input,
            out,
            int(diagonal),
            block_m=32,
            block_n=64,
            num_warps=4,
        )

    tmp = _empty_contiguous_like(input)
    _launch_tril(input, tmp, int(diagonal))
    out.copy_(tmp)
    return out
