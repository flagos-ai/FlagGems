import logging
import operator

import torch
import triton
import triton.language as tl

import flag_gems
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

DEFAULT_BLOCK_SIZE = 1024
SINGLE_BLOCK_MAX_NUMEL = 16384
SMALL_COUNTS_MAX_BLOCKS = 1024


def _check_int_arg(value, name):
    if isinstance(value, bool):
        raise TypeError(f"nonzero_static(): argument '{name}' must be int, not bool")

    try:
        return operator.index(value)
    except TypeError as exc:
        raise TypeError(
            f"nonzero_static(): argument '{name}' must be int, "
            f"not {type(value).__name__}"
        ) from exc


@triton.jit
def _load_nonzero_flags(x_ptr, offsets, mask, IS_COMPLEX: tl.constexpr):
    if IS_COMPLEX:
        base_offsets = offsets * 2
        real = tl.load(x_ptr + base_offsets, mask=mask, other=0)
        imag = tl.load(x_ptr + base_offsets + 1, mask=mask, other=0)
        return (real != 0) | (imag != 0)

    vals = tl.load(x_ptr + offsets, mask=mask, other=0)
    return vals != 0


@libentry()
@triton.jit
def _nonzero_static_count_kernel(
    x_ptr,
    counts_ptr,
    numel: tl.constexpr,
    IS_COMPLEX: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel

    flags = _load_nonzero_flags(x_ptr, offsets, mask, IS_COMPLEX)

    cnt = tl.sum(flags.to(tl.int32), axis=0)
    tl.store(counts_ptr + pid, cnt.to(tl.int64))


@libentry()
@triton.jit
def _nonzero_static_fill_kernel(
    out_ptr,
    total_out: tl.constexpr,
    fill_value: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_out

    vals = tl.full((BLOCK_SIZE,), fill_value, tl.int64)
    tl.store(out_ptr + offsets, vals, mask=mask)


@libentry()
@triton.jit
def _nonzero_static_fill_tail_kernel(
    out_ptr,
    prefix_ptr,
    num_blocks: tl.constexpr,
    size: tl.constexpr,
    ndim: tl.constexpr,
    fill_value: tl.constexpr,
    fill_offset: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    total_nnz = tl.load(prefix_ptr + num_blocks - 1)
    valid_rows = tl.minimum(total_nnz, size)
    tail_start = valid_rows * ndim + fill_offset
    total_out = size * ndim

    pid = tl.program_id(0)
    offsets = tail_start + pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_out

    vals = tl.full((BLOCK_SIZE,), fill_value, tl.int64)
    tl.store(out_ptr + offsets, vals, mask=mask)


@libentry()
@triton.jit
def _nonzero_static_single_block_kernel(
    x_ptr,
    out_ptr,
    size: tl.constexpr,
    numel: tl.constexpr,
    ndim: tl.constexpr,
    D0: tl.constexpr,
    D1: tl.constexpr,
    D2: tl.constexpr,
    D3: tl.constexpr,
    fill_value: tl.constexpr,
    total_out: tl.constexpr,
    IS_COMPLEX: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel

    flags = _load_nonzero_flags(x_ptr, offsets, mask, IS_COMPLEX)

    local_rank = tl.cumsum(flags.to(tl.int32), 0) - 1
    write_mask = mask & flags & (local_rank < size)
    linear = offsets.to(tl.int64)
    global_rank = local_rank.to(tl.int64)

    if ndim == 1:
        c0 = linear
        tl.store(out_ptr + global_rank * ndim, c0, mask=write_mask)

    if ndim == 2:
        c0 = linear // D1
        c1 = linear % D1
        tl.store(out_ptr + global_rank * ndim, c0, mask=write_mask)
        tl.store(out_ptr + global_rank * ndim + 1, c1, mask=write_mask)

    if ndim == 3:
        s0 = D1 * D2
        c0 = linear // s0
        r0 = linear % s0
        c1 = r0 // D2
        c2 = r0 % D2
        tl.store(out_ptr + global_rank * ndim, c0, mask=write_mask)
        tl.store(out_ptr + global_rank * ndim + 1, c1, mask=write_mask)
        tl.store(out_ptr + global_rank * ndim + 2, c2, mask=write_mask)

    if ndim == 4:
        s0 = D1 * D2 * D3
        s1 = D2 * D3
        c0 = linear // s0
        r0 = linear % s0
        c1 = r0 // s1
        r1 = r0 % s1
        c2 = r1 // D3
        c3 = r1 % D3
        tl.store(out_ptr + global_rank * ndim, c0, mask=write_mask)
        tl.store(out_ptr + global_rank * ndim + 1, c1, mask=write_mask)
        tl.store(out_ptr + global_rank * ndim + 2, c2, mask=write_mask)
        tl.store(out_ptr + global_rank * ndim + 3, c3, mask=write_mask)

    total_nnz = tl.sum(flags.to(tl.int32), axis=0)
    valid_rows = tl.minimum(total_nnz, size)
    tail_offsets = valid_rows * ndim + offsets
    tail_mask = tail_offsets < total_out
    tail_vals = tl.full((BLOCK_SIZE,), fill_value, tl.int64)
    tl.store(out_ptr + tail_offsets, tail_vals, mask=tail_mask)


@libentry()
@triton.jit
def _nonzero_static_single_block_generic_kernel(
    x_ptr,
    shape_ptr,
    out_ptr,
    size: tl.constexpr,
    numel: tl.constexpr,
    ndim: tl.constexpr,
    fill_value: tl.constexpr,
    total_out: tl.constexpr,
    IS_COMPLEX: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel

    flags = _load_nonzero_flags(x_ptr, offsets, mask, IS_COMPLEX)

    local_rank = tl.cumsum(flags.to(tl.int32), 0) - 1
    global_rank = local_rank.to(tl.int64)
    write_mask = mask & flags & (local_rank < size)

    idx_flat = offsets.to(tl.int64)
    for dim in range(ndim - 1, -1, -1):
        dim_size = tl.load(shape_ptr + dim)
        coord = idx_flat % dim_size
        idx_flat //= dim_size
        tl.store(out_ptr + global_rank * ndim + dim, coord, mask=write_mask)

    total_nnz = tl.sum(flags.to(tl.int32), axis=0)
    valid_rows = tl.minimum(total_nnz, size)
    tail_offsets = valid_rows * ndim + offsets
    tail_mask = tail_offsets < total_out
    tail_vals = tl.full((BLOCK_SIZE,), fill_value, tl.int64)
    tl.store(out_ptr + tail_offsets, tail_vals, mask=tail_mask)


@libentry()
@triton.jit
def _nonzero_static_write_kernel(
    x_ptr,
    prefix_ptr,
    counts_ptr,
    out_ptr,
    size: tl.constexpr,
    numel: tl.constexpr,
    num_blocks: tl.constexpr,
    ndim: tl.constexpr,
    D0: tl.constexpr,
    D1: tl.constexpr,
    D2: tl.constexpr,
    D3: tl.constexpr,
    fill_value: tl.constexpr,
    total_out: tl.constexpr,
    IS_COMPLEX: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel

    flags = _load_nonzero_flags(x_ptr, offsets, mask, IS_COMPLEX)

    block_nnz = tl.load(counts_ptr + pid)
    prefix = tl.load(prefix_ptr + pid) - block_nnz

    local_rank = tl.cumsum(flags.to(tl.int32), 0) - 1
    global_rank = prefix + local_rank.to(tl.int64)

    write_mask = mask & flags & (global_rank < size)
    linear = offsets.to(tl.int64)

    if ndim == 1:
        c0 = linear
        tl.store(out_ptr + global_rank * ndim, c0, mask=write_mask)

    if ndim == 2:
        c0 = linear // D1
        c1 = linear % D1
        tl.store(out_ptr + global_rank * ndim, c0, mask=write_mask)
        tl.store(out_ptr + global_rank * ndim + 1, c1, mask=write_mask)

    if ndim == 3:
        s0 = D1 * D2
        c0 = linear // s0
        r0 = linear % s0
        c1 = r0 // D2
        c2 = r0 % D2
        tl.store(out_ptr + global_rank * ndim, c0, mask=write_mask)
        tl.store(out_ptr + global_rank * ndim + 1, c1, mask=write_mask)
        tl.store(out_ptr + global_rank * ndim + 2, c2, mask=write_mask)

    if ndim == 4:
        s0 = D1 * D2 * D3
        s1 = D2 * D3
        c0 = linear // s0
        r0 = linear % s0
        c1 = r0 // s1
        r1 = r0 % s1
        c2 = r1 // D3
        c3 = r1 % D3
        tl.store(out_ptr + global_rank * ndim, c0, mask=write_mask)
        tl.store(out_ptr + global_rank * ndim + 1, c1, mask=write_mask)
        tl.store(out_ptr + global_rank * ndim + 2, c2, mask=write_mask)
        tl.store(out_ptr + global_rank * ndim + 3, c3, mask=write_mask)

    total_nnz = tl.load(prefix_ptr + num_blocks - 1)
    valid_rows = tl.minimum(total_nnz, size)
    tail_offsets = valid_rows * ndim + pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    tail_mask = tail_offsets < total_out
    tail_vals = tl.full((BLOCK_SIZE,), fill_value, tl.int64)
    tl.store(out_ptr + tail_offsets, tail_vals, mask=tail_mask)


@libentry()
@triton.jit
def _nonzero_static_write_small_counts_kernel(
    x_ptr,
    counts_ptr,
    out_ptr,
    size: tl.constexpr,
    numel: tl.constexpr,
    num_blocks: tl.constexpr,
    ndim: tl.constexpr,
    D0: tl.constexpr,
    D1: tl.constexpr,
    D2: tl.constexpr,
    D3: tl.constexpr,
    fill_value: tl.constexpr,
    total_out: tl.constexpr,
    IS_COMPLEX: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    PREFIX_BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel

    count_offsets = tl.arange(0, PREFIX_BLOCK_SIZE)
    count_mask = count_offsets < num_blocks
    count_vals = tl.load(counts_ptr + count_offsets, mask=count_mask, other=0)
    prefix = tl.sum(tl.where(count_offsets < pid, count_vals, 0), axis=0)
    total_nnz = tl.sum(count_vals, axis=0)

    flags = _load_nonzero_flags(x_ptr, offsets, mask, IS_COMPLEX)

    local_rank = tl.cumsum(flags.to(tl.int32), 0) - 1
    global_rank = prefix + local_rank.to(tl.int64)

    write_mask = mask & flags & (global_rank < size)
    linear = offsets.to(tl.int64)

    if ndim == 1:
        c0 = linear
        tl.store(out_ptr + global_rank * ndim, c0, mask=write_mask)

    if ndim == 2:
        c0 = linear // D1
        c1 = linear % D1
        tl.store(out_ptr + global_rank * ndim, c0, mask=write_mask)
        tl.store(out_ptr + global_rank * ndim + 1, c1, mask=write_mask)

    if ndim == 3:
        s0 = D1 * D2
        c0 = linear // s0
        r0 = linear % s0
        c1 = r0 // D2
        c2 = r0 % D2
        tl.store(out_ptr + global_rank * ndim, c0, mask=write_mask)
        tl.store(out_ptr + global_rank * ndim + 1, c1, mask=write_mask)
        tl.store(out_ptr + global_rank * ndim + 2, c2, mask=write_mask)

    if ndim == 4:
        s0 = D1 * D2 * D3
        s1 = D2 * D3
        c0 = linear // s0
        r0 = linear % s0
        c1 = r0 // s1
        r1 = r0 % s1
        c2 = r1 // D3
        c3 = r1 % D3
        tl.store(out_ptr + global_rank * ndim, c0, mask=write_mask)
        tl.store(out_ptr + global_rank * ndim + 1, c1, mask=write_mask)
        tl.store(out_ptr + global_rank * ndim + 2, c2, mask=write_mask)
        tl.store(out_ptr + global_rank * ndim + 3, c3, mask=write_mask)

    valid_rows = tl.minimum(total_nnz, size)
    tail_offsets = valid_rows * ndim + pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    tail_mask = tail_offsets < total_out
    tail_vals = tl.full((BLOCK_SIZE,), fill_value, tl.int64)
    tl.store(out_ptr + tail_offsets, tail_vals, mask=tail_mask)


@libentry()
@triton.jit
def _nonzero_static_write_generic_kernel(
    x_ptr,
    shape_ptr,
    prefix_ptr,
    counts_ptr,
    out_ptr,
    size: tl.constexpr,
    numel: tl.constexpr,
    num_blocks: tl.constexpr,
    ndim: tl.constexpr,
    fill_value: tl.constexpr,
    total_out: tl.constexpr,
    IS_COMPLEX: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel

    flags = _load_nonzero_flags(x_ptr, offsets, mask, IS_COMPLEX)

    block_nnz = tl.load(counts_ptr + pid)
    prefix = tl.load(prefix_ptr + pid) - block_nnz

    local_rank = tl.cumsum(flags.to(tl.int32), 0) - 1
    global_rank = prefix + local_rank.to(tl.int64)
    write_mask = mask & flags & (global_rank < size)

    idx_flat = offsets.to(tl.int64)
    for dim in range(ndim - 1, -1, -1):
        dim_size = tl.load(shape_ptr + dim)
        coord = idx_flat % dim_size
        idx_flat //= dim_size
        tl.store(out_ptr + global_rank * ndim + dim, coord, mask=write_mask)

    total_nnz = tl.load(prefix_ptr + num_blocks - 1)
    valid_rows = tl.minimum(total_nnz, size)
    tail_offsets = valid_rows * ndim + pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    tail_mask = tail_offsets < total_out
    tail_vals = tl.full((BLOCK_SIZE,), fill_value, tl.int64)
    tl.store(out_ptr + tail_offsets, tail_vals, mask=tail_mask)


def _prepare_nonzero_static_out(input, size, out):
    expected_shape = (size, input.dim())
    if out.dtype != torch.int64:
        raise RuntimeError(
            f"Expected out tensor to have dtype torch.int64, but got {out.dtype} instead"
        )
    if out.device != input.device:
        raise RuntimeError(
            f"Expected out tensor to be on {input.device}, but got {out.device} instead"
        )
    if tuple(out.shape) != expected_shape:
        out.resize_(expected_shape)
    return out


def nonzero_static_ref(
    x: torch.Tensor, *, size: int, fill_value: int = -1, out: torch.Tensor = None
):
    size = _check_int_arg(size, "size")
    fill_value = _check_int_arg(fill_value, "fill_value")

    if size < 0:
        raise RuntimeError("nonzero_static: size must be non-negative")

    ndim = x.dim()
    if out is None:
        out = torch.empty((size, ndim), device=x.device, dtype=torch.long)
    else:
        out = _prepare_nonzero_static_out(x, size, out)

    if size == 0:
        return out

    if ndim == 0:
        return out

    nz = torch.nonzero(x, as_tuple=False)
    copy_len = min(size, nz.shape[0])

    if copy_len > 0:
        out[:copy_len].copy_(nz[:copy_len])

    if copy_len < size:
        out[copy_len:].fill_(fill_value)

    return out


def _finish_nonzero_static_out(out, work_out):
    if out is not work_out:
        out.copy_(work_out)
    return out


def _nonzero_static_impl(
    input: torch.Tensor, *, size: int, fill_value: int = -1, out: torch.Tensor = None
):
    size = _check_int_arg(size, "size")
    fill_value = _check_int_arg(fill_value, "fill_value")

    if size < 0:
        raise RuntimeError("nonzero_static: size must be non-negative")

    ndim = input.dim()

    if input.device.type != flag_gems.device:
        return nonzero_static_ref(input, size=size, fill_value=fill_value, out=out)

    if out is None:
        out = torch.empty((size, ndim), device=input.device, dtype=torch.int64)
    else:
        out = _prepare_nonzero_static_out(input, size, out)

    if size == 0:
        return out

    if ndim == 0:
        return out

    work_out = out
    if not work_out.is_contiguous():
        work_out = torch.empty((size, ndim), device=input.device, dtype=torch.int64)

    is_complex = input.is_complex()
    source = input.contiguous()
    if is_complex:
        x = torch.view_as_real(source).reshape(-1)
        numel = source.numel()
    else:
        x = source
        numel = x.numel()

    block_size = DEFAULT_BLOCK_SIZE
    total_out = size * ndim

    if numel == 0:
        fill_grid = (triton.cdiv(total_out, block_size),)
        with torch_device_fn.device(input.device):
            _nonzero_static_fill_kernel[fill_grid](
                work_out,
                total_out,
                fill_value,
                BLOCK_SIZE=block_size,
            )
        return _finish_nonzero_static_out(out, work_out)

    num_blocks = triton.cdiv(numel, block_size)
    use_generic_ndim = ndim > 4

    if use_generic_ndim:
        shape = torch.tensor(input.shape, dtype=torch.int64, device=input.device)
    else:
        shape = list(input.shape) + [1] * (4 - ndim)

    single_block_elems = max(numel, total_out)
    if single_block_elems <= SINGLE_BLOCK_MAX_NUMEL:
        single_block_size = 1 << (single_block_elems - 1).bit_length()
        with torch_device_fn.device(input.device):
            if use_generic_ndim:
                _nonzero_static_single_block_generic_kernel[(1,)](
                    x,
                    shape,
                    work_out,
                    size,
                    numel,
                    ndim,
                    fill_value,
                    total_out,
                    IS_COMPLEX=is_complex,
                    BLOCK_SIZE=single_block_size,
                )
            else:
                _nonzero_static_single_block_kernel[(1,)](
                    x,
                    work_out,
                    size,
                    numel,
                    ndim,
                    shape[0],
                    shape[1],
                    shape[2],
                    shape[3],
                    fill_value,
                    total_out,
                    IS_COMPLEX=is_complex,
                    BLOCK_SIZE=single_block_size,
                )
        return _finish_nonzero_static_out(out, work_out)

    counts = torch.empty((num_blocks,), device=input.device, dtype=torch.int64)

    with torch_device_fn.device(input.device):
        _nonzero_static_count_kernel[(num_blocks,)](
            x,
            counts,
            numel,
            IS_COMPLEX=is_complex,
            BLOCK_SIZE=block_size,
        )

    if (
        not use_generic_ndim
        and num_blocks <= SMALL_COUNTS_MAX_BLOCKS
        and total_out <= num_blocks * block_size
    ):
        prefix_block_size = 1 << (num_blocks - 1).bit_length()
        with torch_device_fn.device(input.device):
            _nonzero_static_write_small_counts_kernel[(num_blocks,)](
                x,
                counts,
                work_out,
                size,
                numel,
                num_blocks,
                ndim,
                shape[0],
                shape[1],
                shape[2],
                shape[3],
                fill_value,
                total_out,
                IS_COMPLEX=is_complex,
                BLOCK_SIZE=block_size,
                PREFIX_BLOCK_SIZE=prefix_block_size,
            )
        return _finish_nonzero_static_out(out, work_out)

    prefix = torch.cumsum(counts, dim=0)

    with torch_device_fn.device(input.device):
        if use_generic_ndim:
            _nonzero_static_write_generic_kernel[(num_blocks,)](
                x,
                shape,
                prefix,
                counts,
                work_out,
                size,
                numel,
                num_blocks,
                ndim,
                fill_value,
                total_out,
                IS_COMPLEX=is_complex,
                BLOCK_SIZE=block_size,
            )
        else:
            _nonzero_static_write_kernel[(num_blocks,)](
                x,
                prefix,
                counts,
                work_out,
                size,
                numel,
                num_blocks,
                ndim,
                shape[0],
                shape[1],
                shape[2],
                shape[3],
                fill_value,
                total_out,
                IS_COMPLEX=is_complex,
                BLOCK_SIZE=block_size,
            )

    if total_out > num_blocks * block_size:
        filled_tail_elems = num_blocks * block_size
        fill_grid = (triton.cdiv(total_out - filled_tail_elems, block_size),)
        with torch_device_fn.device(input.device):
            _nonzero_static_fill_tail_kernel[fill_grid](
                work_out,
                prefix,
                num_blocks,
                size,
                ndim,
                fill_value,
                filled_tail_elems,
                BLOCK_SIZE=block_size,
            )

    return _finish_nonzero_static_out(out, work_out)


def nonzero_static(input: torch.Tensor, *, size: int, fill_value: int = -1):
    logger.debug("GEMS NONZERO_STATIC")
    return _nonzero_static_impl(input, size=size, fill_value=fill_value)


def nonzero_static_out(
    input: torch.Tensor, *, size: int, fill_value: int = -1, out: torch.Tensor
):
    logger.debug("GEMS NONZERO_STATIC_OUT")
    return _nonzero_static_impl(input, size=size, fill_value=fill_value, out=out)
