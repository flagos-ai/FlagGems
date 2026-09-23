import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

from ..utils.tle_copy import tle_copy

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
    keep = offs_n <= (offs_m + diag)

    in_ptr += pid_b * (M * N) + offs_m * N
    out_ptr += pid_b * (M * N) + offs_m * N

    x = tl.load(in_ptr + offs_n, mask=mask, other=0.0)
    result = tl.where(keep, x, 0.0)
    tl.store(out_ptr + offs_n, result, mask=mask)


@triton.jit
def _tril_rows_kernel(
    in_ptr,
    out_ptr,
    diag: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    row_mask = offs_m < M
    in_ptr += pid_b * (M * N) + offs_m * N
    out_ptr += pid_b * (M * N) + offs_m * N

    for col_start in range(0, N, BLOCK_N):
        offs_n = col_start + tl.arange(0, BLOCK_N)[None, :]
        mask = row_mask & (offs_n < N)
        keep = offs_n <= (offs_m + diag)
        x = tl.load(in_ptr + offs_n, mask=mask, other=0.0)
        result = tl.where(keep, x, 0.0)
        tl.store(out_ptr + offs_n, result, mask=mask)


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
    x = tl.load(in_ptr + idxs)
    result = tl.where(keep, x, 0.0)
    tl.store(out_ptr + idxs, result)


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
    mask = (offs_m < M) & (offs_n < N)
    keep = offs_n <= offs_m
    offsets = pid_b * (M * N) + offs_m * N + offs_n
    x = tl.load(in_ptr + offsets, mask=mask & keep, other=0.0)
    tl.store(out_ptr + offsets, x, mask=mask)


@libentry()
@triton.jit
def _tril_flat_inplace_kernel(
    ptr,
    active_total,
    MN,
    diag,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_b = tl.program_id(1)
    base = pid_b * MN

    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < active_total
    rows = offsets // N
    cols = offsets - rows * N
    keep = cols <= rows + diag

    x = tl.load(ptr + base + offsets, mask=mask, other=0.0)
    y = tl.where(keep, x, 0.0)
    tl.store(ptr + base + offsets, y, mask=mask)


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

    row = pid_m
    first_zero_col = tl.maximum(row + diag + 1, 0)
    offs_n = first_zero_col + pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs_n < N
    ptr += batch_offset + row * STRIDE_M
    tl.store(ptr + offs_n * STRIDE_N, 0.0, mask=mask)


@libentry()
@triton.jit
def _tril_strided_out_tile_kernel(
    in_ptr,
    out_ptr,
    diag,
    M,
    N,
    B0,
    B1,
    B2,
    B3,
    B4,
    B5,
    S0,
    S1,
    S2,
    S3,
    S4,
    S5,
    STRIDE_M,
    STRIDE_N,
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
    keep = offs_n <= (offs_m + diag)

    in_ptr += pid_b * (M * N) + offs_m * N
    out_ptr += out_batch_offset + offs_m * STRIDE_M

    x = tl.load(in_ptr + offs_n, mask=mask, other=0.0)
    result = tl.where(keep, x, 0.0)
    tl.store(out_ptr + offs_n * STRIDE_N, result, mask=mask)


@triton.jit
def _tril_flat2d_kernel(
    in_ptr,
    out_ptr,
    total,
    diag,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    rows = offsets // N
    cols = offsets - rows * N
    keep = cols <= rows + diag
    if NEED_MASK:
        mask = offsets < total
        x = tl.load(in_ptr + offsets, mask=mask, other=0.0)
        y = tl.where(keep, x, 0.0)
        tl.store(out_ptr + offsets, y, mask=mask)
    else:
        x = tl.load(in_ptr + offsets)
        y = tl.where(keep, x, 0.0)
        tl.store(out_ptr + offsets, y)


@triton.jit
def _tril_flat_batched_kernel(
    in_ptr,
    out_ptr,
    total,
    diag,
    MN,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    matrix_offsets = offsets % MN
    rows = matrix_offsets // N
    cols = matrix_offsets - rows * N
    keep = cols <= rows + diag
    if NEED_MASK:
        mask = offsets < total
        x = tl.load(in_ptr + offsets, mask=mask, other=0.0)
        y = tl.where(keep, x, 0.0)
        tl.store(out_ptr + offsets, y, mask=mask)
    else:
        x = tl.load(in_ptr + offsets)
        y = tl.where(keep, x, 0.0)
        tl.store(out_ptr + offsets, y)


@triton.jit
def _tril_flat_batchgrid_kernel(
    in_ptr,
    out_ptr,
    diag,
    N: tl.constexpr,
    MN: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_b = tl.program_id(1)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    rows = offsets // N
    cols = offsets - rows * N
    keep = cols <= rows + diag
    base = pid_b * MN
    if NEED_MASK:
        mask = offsets < MN
        x = tl.load(in_ptr + base + offsets, mask=mask, other=0.0)
        y = tl.where(keep, x, 0.0)
        tl.store(out_ptr + base + offsets, y, mask=mask)
    else:
        x = tl.load(in_ptr + base + offsets)
        y = tl.where(keep, x, 0.0)
        tl.store(out_ptr + base + offsets, y)


@triton.jit
def _tril_wide_scalar_kernel(
    in_ptr,
    out_ptr,
    diag,
    MN: tl.constexpr,
    LOG2_BPR: tl.constexpr,
    BPR_MASK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_b = tl.program_id(1)
    lane = tl.arange(0, BLOCK_SIZE)
    row = pid >> LOG2_BPR
    blk = pid & BPR_MASK
    s = row + diag - blk * BLOCK_SIZE
    base = pid_b * MN
    offsets = pid * BLOCK_SIZE + lane
    x = tl.load(in_ptr + base + offsets)
    tl.store(out_ptr + base + offsets, tl.where(lane <= s, x, 0.0))


@triton.jit
def _tril_flat_pow2_kernel(
    in_ptr,
    out_ptr,
    active_total,
    diag,
    MN,
    LOG2N: tl.constexpr,
    NMASK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_b = tl.program_id(1)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    keep = (offsets & NMASK) <= (offsets >> LOG2N) + diag
    base = pid_b * MN
    if NEED_MASK:
        mask = offsets < active_total
        x = tl.load(in_ptr + base + offsets, mask=mask, other=0.0)
        y = tl.where(keep, x, 0.0)
        tl.store(out_ptr + base + offsets, y, mask=mask)
    else:
        x = tl.load(in_ptr + base + offsets)
        y = tl.where(keep, x, 0.0)
        tl.store(out_ptr + base + offsets, y)


@triton.jit
def _tril_band_batchgrid_kernel(
    in_ptr,
    out_ptr,
    active_total,
    diag,
    MN,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_b = tl.program_id(1)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    rows = offsets // N
    cols = offsets - rows * N
    keep = cols <= rows + diag
    base = pid_b * MN
    if NEED_MASK:
        mask = offsets < active_total
        x = tl.load(in_ptr + base + offsets, mask=mask, other=0.0)
        y = tl.where(keep, x, 0.0)
        tl.store(out_ptr + base + offsets, y, mask=mask)
    else:
        x = tl.load(in_ptr + base + offsets)
        y = tl.where(keep, x, 0.0)
        tl.store(out_ptr + base + offsets, y)


@triton.jit
def _tril_row2d_kernel(
    in_ptr,
    out_ptr,
    M,
    N,
    diag,
    BLOCK_N: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    pid = tl.program_id(0)
    row = pid % M
    base = pid * N
    for c0 in range(0, N, BLOCK_N):
        cols = c0 + tl.arange(0, BLOCK_N)
        keep = cols <= row + diag
        if NEED_MASK:
            m = cols < N
            x = tl.load(in_ptr + base + cols, mask=m, other=0.0)
            tl.store(out_ptr + base + cols, tl.where(keep, x, 0.0), mask=m)
        else:
            x = tl.load(in_ptr + base + cols)
            tl.store(out_ptr + base + cols, tl.where(keep, x, 0.0))


@triton.jit
def _tril_zero_flat_kernel(
    ptr,
    total,
    BLOCK_SIZE: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    if NEED_MASK:
        mask = offsets < total
        tl.store(ptr + offsets, 0.0, mask=mask)
    else:
        tl.store(ptr + offsets, 0.0)


_BLOCK_SIZE = 16384
_ROW_N_THRESHOLD = 2048
_FLAT_WIDE_N = 8192
_SMALL_TOTAL_ZERO = 1 << 20
_BAND_MIN_TOTAL = 1 << 20


def _vendor_copy_from(src: torch.Tensor, dst: torch.Tensor):
    if not tle_copy(src, dst):
        torch.ops.aten._copy_from(src, dst)
    return dst


def _launch_v2_flat(
    input: torch.Tensor,
    out: torch.Tensor,
    diagonal: int,
    total: int = None,
    block_size: int = _BLOCK_SIZE,
    num_warps: int = 8,
):
    if total is None:
        total = input.numel()
    grid = (triton.cdiv(total, block_size),)
    need_mask = total % block_size != 0
    with torch_device_fn.device(input.device):
        _tril_flat2d_kernel[grid](
            input,
            out,
            total,
            int(diagonal),
            input.shape[-1],
            block_size,
            need_mask,
            num_warps=num_warps,
        )
    return out


def _launch_v2_flat_batched(
    input: torch.Tensor,
    out: torch.Tensor,
    diagonal: int,
    block_size: int = _BLOCK_SIZE,
    num_warps: int = 8,
):
    total = input.numel()
    M, N = input.shape[-2:]
    MN = M * N
    grid = (triton.cdiv(total, block_size),)
    need_mask = total % block_size != 0
    with torch_device_fn.device(input.device):
        _tril_flat_batched_kernel[grid](
            input,
            out,
            total,
            int(diagonal),
            MN,
            N,
            block_size,
            need_mask,
            num_warps=num_warps,
        )
    return out


def _launch_v2_flat_batchgrid(
    input: torch.Tensor,
    out: torch.Tensor,
    diagonal: int,
    block_size: int = _BLOCK_SIZE,
    num_warps: int = 8,
):
    M, N = input.shape[-2:]
    MN = M * N
    batch = input.numel() // MN
    tiles = triton.cdiv(MN, block_size)
    need_mask = MN % block_size != 0
    with torch_device_fn.device(input.device):
        _tril_flat_batchgrid_kernel[(tiles, batch)](
            input,
            out,
            int(diagonal),
            N,
            MN,
            block_size,
            need_mask,
            num_warps=num_warps,
        )
    return out


def _launch_v2_rows(
    input: torch.Tensor,
    out: torch.Tensor,
    diagonal: int,
    num_rows: int,
    num_warps: int = 4,
):
    M, N = input.shape[-2:]
    block_n = min(triton.next_power_of_2(N), _BLOCK_SIZE)
    need_mask = N % block_n != 0
    with torch_device_fn.device(input.device):
        _tril_row2d_kernel[(num_rows,)](
            input,
            out,
            M,
            N,
            int(diagonal),
            block_n,
            need_mask,
            num_warps=num_warps,
        )


def _launch_v2_zero(
    out: torch.Tensor,
    block_size: int = _BLOCK_SIZE,
    num_warps: int = 8,
):
    total = out.numel()
    grid = (triton.cdiv(total, block_size),)
    need_mask = total % block_size != 0
    with torch_device_fn.device(out.device):
        _tril_zero_flat_kernel[grid](
            out,
            total,
            block_size,
            need_mask,
            num_warps=num_warps,
        )


_WIDE_SCALAR_BLOCK = _BLOCK_SIZE


def _use_wide_scalar(N: int):
    return _is_power_of_2(N) and N > _FLAT_WIDE_N and N >= _WIDE_SCALAR_BLOCK


def _launch_v2_wide_scalar(
    input: torch.Tensor,
    out: torch.Tensor,
    diagonal: int,
    block_size: int = _WIDE_SCALAR_BLOCK,
    num_warps: int = 4,
):
    M, N = input.shape[-2:]
    MN = M * N
    batch = input.numel() // MN
    bpr = N // block_size
    grid = (M * bpr, batch)
    with torch_device_fn.device(input.device):
        _tril_wide_scalar_kernel[grid](
            input,
            out,
            int(diagonal),
            MN,
            bpr.bit_length() - 1,
            bpr - 1,
            block_size,
            num_warps=num_warps,
        )
    return out


def _launch_v2_pow2(
    input: torch.Tensor,
    out: torch.Tensor,
    diagonal: int,
    active_rows: int = None,
    block_size: int = _BLOCK_SIZE,
    num_warps: int = 4,
):
    M, N = input.shape[-2:]
    MN = M * N
    batch = input.numel() // MN
    rows = M if active_rows is None else active_rows
    active_total = rows * N
    grid = (triton.cdiv(active_total, block_size), batch)
    need_mask = active_total % block_size != 0
    with torch_device_fn.device(input.device):
        _tril_flat_pow2_kernel[grid](
            input,
            out,
            active_total,
            int(diagonal),
            MN,
            N.bit_length() - 1,
            N - 1,
            block_size,
            need_mask,
            num_warps=num_warps,
        )
    return out


def _launch_v2_band_batchgrid(
    input: torch.Tensor,
    out: torch.Tensor,
    diagonal: int,
    band_lo: int,
    block_size: int = _BLOCK_SIZE,
    num_warps: int = 8,
):
    M, N = input.shape[-2:]
    MN = M * N
    batch = input.numel() // MN
    active_total = band_lo * N
    grid = (triton.cdiv(active_total, block_size), batch)
    need_mask = active_total % block_size != 0
    with torch_device_fn.device(input.device):
        _tril_band_batchgrid_kernel[grid](
            input,
            out,
            active_total,
            int(diagonal),
            MN,
            N,
            block_size,
            need_mask,
            num_warps=num_warps,
        )
    return out


def _launch_v2_band(
    input: torch.Tensor,
    out: torch.Tensor,
    diagonal: int,
    band_lo: int,
):
    M, N = input.shape[-2:]
    batch = input.numel() // (M * N)
    total = band_lo * N
    if _is_power_of_2(N):
        if batch == 1:
            if total > 0:
                _launch_v2_pow2(input, out, diagonal, active_rows=band_lo)
            if band_lo < M and input.data_ptr() != out.data_ptr():
                _vendor_copy_from(input[band_lo:], out[band_lo:])
            return out
        if band_lo < M and input.data_ptr() != out.data_ptr():
            _vendor_copy_from(input, out)
        if total > 0:
            _launch_v2_pow2(input, out, diagonal, active_rows=band_lo)
        return out
    if batch == 1:
        if total > 0:
            if N >= _ROW_N_THRESHOLD:
                _launch_v2_rows(input, out, diagonal, num_rows=band_lo)
            else:
                _launch_v2_flat(input, out, diagonal, total)
        if band_lo < M and input.data_ptr() != out.data_ptr():
            _vendor_copy_from(input[band_lo:], out[band_lo:])
        return out
    if band_lo < M and input.data_ptr() != out.data_ptr():
        _vendor_copy_from(input, out)
    if total > 0:
        _launch_v2_band_batchgrid(input, out, diagonal, band_lo)
    return out


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


_INPLACE_FLAT_BLOCK = 8192
_INPLACE_POW2_MIN_TOTAL = 1 << 17


def _launch_tril_inplace_contiguous(
    input: torch.Tensor,
    diagonal: int,
    block_size: int = _INPLACE_FLAT_BLOCK,
    num_warps: int = 8,
    num_stages: int = 2,
):
    M, N = input.shape[-2:]
    if input.numel() == 0:
        return input

    active_rows = min(M, max(0, N - 1 - diagonal))
    if active_rows == 0:
        return input

    if _is_power_of_2(N) and active_rows * N >= _INPLACE_POW2_MIN_TOTAL:
        return _launch_v2_pow2(input, input, int(diagonal), active_rows=active_rows)

    MN = M * N
    active_total = active_rows * N
    batch = input.numel() // MN

    grid = (triton.cdiv(active_total, block_size), batch)
    with torch_device_fn.device(input.device):
        _tril_flat_inplace_kernel[grid](
            input,
            active_total,
            MN,
            int(diagonal),
            N,
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    return input


def _launch_tril_inplace_strided(
    input: torch.Tensor,
    diagonal: int,
    block_m: int = 1,
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
    total = input.numel()
    if total == 0:
        return out

    if diagonal <= -M:
        if total <= _SMALL_TOTAL_ZERO:
            _launch_v2_zero(out)
        else:
            out.zero_()
        return out
    if diagonal >= N - 1:
        if input.data_ptr() != out.data_ptr():
            _vendor_copy_from(input, out)
        return out

    input_to_use = input if input.is_contiguous() else input.contiguous()
    batch = input_to_use.numel() // (M * N)

    band_lo = min(M, max(0, N - 1 - diagonal))
    if band_lo < M and band_lo * N <= (M * N) // 4 and total >= _BAND_MIN_TOTAL:
        _launch_v2_band(input_to_use, out, diagonal, band_lo)
        return out

    if _use_wide_scalar(N):
        return _launch_v2_wide_scalar(input_to_use, out, diagonal)

    if _is_power_of_2(N) and not (batch > 1 and M * N <= 4096):
        return _launch_v2_pow2(input_to_use, out, diagonal)

    if batch == 1:
        if _use_wide_exact_row(M, N, batch):
            return _launch_exact_row(
                input_to_use,
                out,
                diagonal,
                num_warps=_wide_exact_row_warps(N),
            )
        if N >= _ROW_N_THRESHOLD:
            _launch_v2_rows(input_to_use, out, diagonal, num_rows=M)
            return out
        return _launch_v2_flat(input_to_use, out, diagonal)
    if M * N <= 4096:
        return _launch_v2_flat_batched(input_to_use, out, diagonal)
    _launch_v2_flat_batchgrid(input_to_use, out, diagonal)
    return out


def tril(input: torch.Tensor, diagonal: int = 0):
    logger.debug("GEMS_KUNLUNXIN TRIL")
    _check_input(input)

    out = _empty_contiguous_like(input)
    return _launch_tril(input, out, int(diagonal))


def tril_(input: torch.Tensor, diagonal: int = 0):
    logger.debug("GEMS_KUNLUNXIN TRIL_")
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


_ZC_BLOCK = 8192
_ZC_RPC = 8
_ZC_MIN_TOTAL = 1 << 21
_ZC_SLICE_M = 8


def _tensors_may_overlap(a: torch.Tensor, b: torch.Tensor) -> bool:
    if a.data_ptr() == b.data_ptr():
        return True
    try:
        return bool(torch._C._overlaps(a, b))
    except Exception:
        a0, b0 = a.data_ptr(), b.data_ptr()
        asz = a.numel() * a.element_size()
        bsz = b.numel() * b.element_size()
        return a0 < b0 + bsz and b0 < a0 + asz


@triton.jit
def _zc_zero_upper_storage_kernel(
    ptr, M, N, diag, JOFF, B, NJ, BLOCK: tl.constexpr, RPC: tl.constexpr
):
    """Dense storage [B, N, M] (out.transpose(-2,-1) contiguous view): zero
    [0, min(j - diag, M)) of storage row (b, j). Only rows j with j - diag > 0
    are launched: j = JOFF + (row % NJ) with JOFF = diag + 1 (diag >= 0) or 0.
    Single-compare lane mask (2-compare masks are ~900x slower on this XPU).
    grid = (cdiv(M, BLOCK), cdiv(B * NJ, RPC))."""
    i0 = tl.program_id(0) * BLOCK
    g = tl.program_id(1)
    for r in tl.static_range(RPC):
        row = g * RPC + r
        if row < B * NJ:
            j = JOFF + (row % NJ)
            end = tl.minimum(j - diag, M)
            if end > i0:
                m = tl.arange(0, BLOCK) < (end - i0)
                tl.store(
                    ptr + (row // NJ) * (N * M) + j * M + i0 + tl.arange(0, BLOCK),
                    0.0,
                    mask=m,
                )


@triton.jit
def _zc_zero_upper_row_kernel(
    ptr,
    M,
    N,
    diag,
    B,
    BATCH_STRIDE,
    ROW_STRIDE,
    NI,
    BLOCK: tl.constexpr,
    RPC: tl.constexpr,
):
    """out [B, M, N] with unit col stride: zero [i + diag + 1, N) of logical row
    (b, i). NI = min(M, N - diag - 1) = rows per batch with nonzero work.
    Single-compare lane mask. grid = (cdiv(N, BLOCK), cdiv(B * NI, RPC))."""
    j0 = tl.program_id(0) * BLOCK
    g = tl.program_id(1)
    for r in tl.static_range(RPC):
        row = g * RPC + r
        if row < B * NI:
            i = row % NI
            start = i + diag + 1
            lo = tl.maximum(start, j0)
            hi = tl.minimum(N, j0 + BLOCK)
            if lo < hi:
                m = tl.arange(0, BLOCK) < (hi - lo)
                tl.store(
                    ptr
                    + (row // NI) * BATCH_STRIDE
                    + i * ROW_STRIDE
                    + lo
                    + tl.arange(0, BLOCK),
                    0.0,
                    mask=m,
                )


def _launch_tril_out_copied_zero(
    input: torch.Tensor, out: torch.Tensor, diagonal: int
) -> bool:
    """Copy-first + store-only-zero-upper for non-contiguous tril_out.

    Returns True if the fast path was taken. The vendor native strided copy
    (`_vendor_copy_from`) fills the whole output, then a store-only kernel
    (no load, no vselect) zeroes the strict-upper region; measured ~2-5x
    faster than the legacy tmp+launch_tril+copy for large matrices
    ([4096,4096] fp16: 172us vs 500us, [10000,65536] 5.4ms vs 14.4ms).
    Keeps the legacy path for batched-small shapes (zero pass is launch-bound
    there) and for any aliasing/irregular layout (safety first)."""
    M, N = input.shape[-2:]
    transposed = out.transpose(-2, -1).is_contiguous()
    if transposed:
        dense = True
    elif out.stride(-1) == 1:
        dense = False
    else:
        return False

    if M * N < _ZC_MIN_TOTAL and not (dense is False and M < _ZC_SLICE_M):
        return False
    if _tensors_may_overlap(input, out):
        return False
    total = input.numel()
    if total >= (1 << 31):
        return False

    batch = total // (M * N)
    if transposed:
        if diagonal >= 0:
            nj = N - 1 - diagonal
            joff = diagonal + 1
        else:
            nj = N
            joff = 0
        grid = (triton.cdiv(M, _ZC_BLOCK), triton.cdiv(batch * nj, _ZC_RPC))
        with torch_device_fn.device(input.device):
            _vendor_copy_from(input, out)
            _zc_zero_upper_storage_kernel[grid](
                out, M, N, diagonal, joff, batch, nj, _ZC_BLOCK, _ZC_RPC, num_warps=4
            )
    else:
        ni = min(M, N - 1 - diagonal)
        if ni <= 0:
            return False
        grid = (triton.cdiv(N, _ZC_BLOCK), triton.cdiv(batch * ni, _ZC_RPC))
        with torch_device_fn.device(input.device):
            _vendor_copy_from(input, out)
            _zc_zero_upper_row_kernel[grid](
                out,
                M,
                N,
                diagonal,
                batch,
                out.stride(0) if out.dim() > 2 else 0,
                out.stride(-2),
                ni,
                _ZC_BLOCK,
                _ZC_RPC,
                num_warps=4,
            )
    return True


def tril_out(input: torch.Tensor, diagonal: int = 0, *, out: torch.Tensor = None):
    logger.debug("GEMS_KUNLUNXIN TRIL_OUT")

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
        if input.data_ptr() != out.data_ptr():
            _vendor_copy_from(input, out)
        return out

    if _launch_tril_out_copied_zero(input, out, int(diagonal)):
        return out

    tmp = _empty_contiguous_like(input)
    _launch_tril(input, tmp, int(diagonal))
    _vendor_copy_from(tmp, out)
    return out
