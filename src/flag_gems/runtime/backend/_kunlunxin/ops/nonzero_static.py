import operator

import torch
import triton
import triton.language as tl

import flag_gems
from flag_gems.ops.nonzero_static import nonzero_static as _nonzero_static
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

_SMALL_INPUT_MAX_NUMEL = 8192
_MULTI_BLOCK_TILE_SIZE = 8192
_FILL_BLOCK_SIZE = 256


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


@libentry()
@triton.jit
def _nonzero_static_small_kernel(
    x_ptr,
    workspace_ptr,
    count_ptr,
    size: tl.constexpr,
    numel: tl.constexpr,
    ndim: tl.constexpr,
    D0: tl.constexpr,
    D1: tl.constexpr,
    D2: tl.constexpr,
    D3: tl.constexpr,
    D4: tl.constexpr,
    D5: tl.constexpr,
    IS_COMPLEX: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    if IS_COMPLEX:
        real = tl.load(x_ptr + offsets * 2, mask=mask, other=0)
        imag = tl.load(x_ptr + offsets * 2 + 1, mask=mask, other=0)
        flags = (real != 0) | (imag != 0)
    else:
        flags = tl.load(x_ptr + offsets, mask=mask, other=0) != 0

    valid = flags & (offsets < numel)
    rank = tl.cumsum(valid.to(tl.int32), axis=0) - 1
    destination = tl.where(
        valid & (rank < size), rank.to(tl.int64), (size + offsets).to(tl.int64)
    )
    linear = offsets.to(tl.int64)

    if ndim == 1:
        tl.store(workspace_ptr + destination, linear)
    if ndim == 2:
        tl.store(workspace_ptr + destination * 2, linear // D1)
        tl.store(workspace_ptr + destination * 2 + 1, linear % D1)
    if ndim == 3:
        d12 = D1 * D2
        rem = linear % d12
        tl.store(workspace_ptr + destination * 3, linear // d12)
        tl.store(workspace_ptr + destination * 3 + 1, rem // D2)
        tl.store(workspace_ptr + destination * 3 + 2, rem % D2)
    if ndim == 4:
        d123 = D1 * D2 * D3
        d23 = D2 * D3
        rem = linear % d123
        tl.store(workspace_ptr + destination * 4, linear // d123)
        tl.store(workspace_ptr + destination * 4 + 1, rem // d23)
        tl.store(workspace_ptr + destination * 4 + 2, (rem % d23) // D3)
        tl.store(workspace_ptr + destination * 4 + 3, rem % D3)
    if ndim == 5:
        d1234 = D1 * D2 * D3 * D4
        d234 = D2 * D3 * D4
        d34 = D3 * D4
        rem = linear % d1234
        tl.store(workspace_ptr + destination * 5, linear // d1234)
        tl.store(workspace_ptr + destination * 5 + 1, rem // d234)
        tl.store(workspace_ptr + destination * 5 + 2, (rem % d234) // d34)
        tl.store(workspace_ptr + destination * 5 + 3, (rem % d34) // D4)
        tl.store(workspace_ptr + destination * 5 + 4, rem % D4)
    if ndim == 6:
        d12345 = D1 * D2 * D3 * D4 * D5
        d2345 = D2 * D3 * D4 * D5
        d345 = D3 * D4 * D5
        d45 = D4 * D5
        rem = linear % d12345
        tl.store(workspace_ptr + destination * 6, linear // d12345)
        tl.store(workspace_ptr + destination * 6 + 1, rem // d2345)
        tl.store(workspace_ptr + destination * 6 + 2, (rem % d2345) // d345)
        tl.store(workspace_ptr + destination * 6 + 3, (rem % d345) // d45)
        tl.store(workspace_ptr + destination * 6 + 4, (rem % d45) // D5)
        tl.store(workspace_ptr + destination * 6 + 5, rem % D5)

    tl.store(count_ptr, tl.sum(valid.to(tl.int32), axis=0).to(tl.int64))


@libentry()
@triton.jit
def _nonzero_static_fill_tail_kernel(
    workspace_ptr,
    count_ptr,
    fill_value,
    size: tl.constexpr,
    ndim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid_count = tl.minimum(tl.load(count_ptr), size)
    keep = (rows >= valid_count).to(tl.int64)
    destination = (size + (rows % BLOCK_SIZE)) * (1 - keep) + rows * keep
    for column in tl.static_range(0, ndim):
        tl.store(workspace_ptr + destination * ndim + column, fill_value)


@libentry()
@triton.jit
def _nonzero_static_multiblock_count_kernel(
    x_ptr,
    counts_ptr,
    numel: tl.constexpr,
    IS_COMPLEX: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    if IS_COMPLEX:
        real = tl.load(x_ptr + offsets * 2, mask=mask, other=0)
        imag = tl.load(x_ptr + offsets * 2 + 1, mask=mask, other=0)
        flags = (real != 0) | (imag != 0)
    else:
        flags = tl.load(x_ptr + offsets, mask=mask, other=0) != 0
    valid = offsets < numel
    tl.store(
        counts_ptr + pid, tl.sum((flags & valid).to(tl.int32), axis=0).to(tl.int64)
    )


@libentry()
@triton.jit
def _nonzero_static_multiblock_scatter_kernel(
    x_ptr,
    counts_ptr,
    prefix_ptr,
    nz_ptr,
    size: tl.constexpr,
    numel: tl.constexpr,
    IS_COMPLEX: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Compact the linear index of every selected element into ``nz``.

    Only ONE int64 is stored per lane (the linear index) instead of ``ndim``
    coordinate values: on this backend every store with a data-dependent
    address is a per-lane scalar store (~1.15 ns/lane), while a store whose
    address the compiler can prove is stride-1 is block-DMA (~0.012 ns/lane).
    The store *count* therefore has to be minimised, and the coordinate
    expansion is deferred to the fully affine assemble kernel.

    Store masks are NOT honoured by this backend (a masked store lowers to an
    address-select, all lanes still store), so instead of masking, every
    non-selected lane writes its own index into a unique per-(program, lane)
    sink slot in ``[size, size + num_blocks * BLOCK_SIZE)``.  The slots must
    stay unique -- colliding scatter addresses cost an order of magnitude more
    than distinct ones.
    """
    pid = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    linear = pid * BLOCK_SIZE + offsets
    mask = linear < numel
    if IS_COMPLEX:
        real = tl.load(x_ptr + linear * 2, mask=mask, other=0)
        imag = tl.load(x_ptr + linear * 2 + 1, mask=mask, other=0)
        flags = (real != 0) | (imag != 0)
    else:
        flags = tl.load(x_ptr + linear, mask=mask, other=0) != 0
    valid = mask & flags
    local_rank = tl.cumsum(valid.to(tl.int32), axis=0) - 1
    base = tl.load(prefix_ptr + pid) - tl.load(counts_ptr + pid)
    selected = valid & (base + local_rank < size)
    destination = tl.where(
        selected,
        base + local_rank,
        size + pid * BLOCK_SIZE + offsets,
    ).to(tl.int64)
    tl.store(nz_ptr + destination, linear.to(tl.int64))


@libentry()
@triton.jit
def _nonzero_static_multiblock_assemble_kernel(
    nz_ptr,
    count_ptr,
    out_ptr,
    fill_value,
    size: tl.constexpr,
    ndim: tl.constexpr,
    D0: tl.constexpr,
    D1: tl.constexpr,
    D2: tl.constexpr,
    D3: tl.constexpr,
    D4: tl.constexpr,
    D5: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Expand the compacted linear indices into ``[size, ndim]`` coordinates.

    Both the load and the stores are affine in ``tl.arange`` (rows outside
    ``[0, size)`` are redirected to a trailing sink), so this kernel is block
    DMA traffic; rows past the last nonzero are filled with ``fill_value``.
    """
    pid = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    rows = pid * BLOCK_SIZE + offsets
    valid_count = tl.minimum(tl.load(count_ptr), size)
    keep = rows < valid_count
    destination = tl.where(rows < size, rows, size + offsets).to(tl.int64)
    linear = tl.load(nz_ptr + tl.where(keep, rows, 0))

    if ndim == 1:
        tl.store(out_ptr + destination, tl.where(keep, linear, fill_value))
    if ndim == 2:
        tl.store(out_ptr + destination * 2, tl.where(keep, linear // D1, fill_value))
        tl.store(out_ptr + destination * 2 + 1, tl.where(keep, linear % D1, fill_value))
    if ndim == 3:
        d12 = D1 * D2
        rem = linear % d12
        tl.store(out_ptr + destination * 3, tl.where(keep, linear // d12, fill_value))
        tl.store(out_ptr + destination * 3 + 1, tl.where(keep, rem // D2, fill_value))
        tl.store(out_ptr + destination * 3 + 2, tl.where(keep, rem % D2, fill_value))
    if ndim == 4:
        d123 = D1 * D2 * D3
        d23 = D2 * D3
        rem = linear % d123
        tl.store(out_ptr + destination * 4, tl.where(keep, linear // d123, fill_value))
        tl.store(out_ptr + destination * 4 + 1, tl.where(keep, rem // d23, fill_value))
        tl.store(
            out_ptr + destination * 4 + 2,
            tl.where(keep, (rem % d23) // D3, fill_value),
        )
        tl.store(out_ptr + destination * 4 + 3, tl.where(keep, rem % D3, fill_value))
    if ndim == 5:
        d1234 = D1 * D2 * D3 * D4
        d234 = D2 * D3 * D4
        d34 = D3 * D4
        rem = linear % d1234
        tl.store(out_ptr + destination * 5, tl.where(keep, linear // d1234, fill_value))
        tl.store(out_ptr + destination * 5 + 1, tl.where(keep, rem // d234, fill_value))
        tl.store(
            out_ptr + destination * 5 + 2,
            tl.where(keep, (rem % d234) // d34, fill_value),
        )
        tl.store(
            out_ptr + destination * 5 + 3,
            tl.where(keep, (rem % d34) // D4, fill_value),
        )
        tl.store(out_ptr + destination * 5 + 4, tl.where(keep, rem % D4, fill_value))
    if ndim == 6:
        d12345 = D1 * D2 * D3 * D4 * D5
        d2345 = D2 * D3 * D4 * D5
        d345 = D3 * D4 * D5
        d45 = D4 * D5
        rem = linear % d12345
        tl.store(
            out_ptr + destination * 6, tl.where(keep, linear // d12345, fill_value)
        )
        tl.store(
            out_ptr + destination * 6 + 1, tl.where(keep, rem // d2345, fill_value)
        )
        tl.store(
            out_ptr + destination * 6 + 2,
            tl.where(keep, (rem % d2345) // d345, fill_value),
        )
        tl.store(
            out_ptr + destination * 6 + 3,
            tl.where(keep, (rem % d345) // d45, fill_value),
        )
        tl.store(
            out_ptr + destination * 6 + 4,
            tl.where(keep, (rem % d45) // D5, fill_value),
        )
        tl.store(out_ptr + destination * 6 + 5, tl.where(keep, rem % D5, fill_value))


def _multiblock_nonzero_static(input, size, fill_value, out):
    ndim = input.dim()
    numel = input.numel()
    if ndim == 0 or ndim > 6 or numel <= _SMALL_INPUT_MAX_NUMEL:
        return None
    num_blocks = triton.cdiv(numel, _MULTI_BLOCK_TILE_SIZE)

    source = input.contiguous()
    if source.is_complex():
        x = torch.view_as_real(source).reshape(-1)
    else:
        x = source
    shape = tuple(input.shape) + (1,) * (6 - ndim)

    # nz[:size] holds the compacted linear index of every selected element and
    # each non-selected lane parks in its own unique sink slot in
    # [size, size + num_blocks * TILE).  Only rows [0, size) of `workspace` are
    # returned; the trailing _FILL_BLOCK_SIZE rows are the assemble sink.
    nz = torch.empty(
        (size + num_blocks * _MULTI_BLOCK_TILE_SIZE,),
        device=input.device,
        dtype=torch.int64,
    )
    counts = torch.empty((num_blocks,), device=input.device, dtype=torch.int64)
    workspace = torch.empty(
        (size + _FILL_BLOCK_SIZE, ndim), device=input.device, dtype=torch.int64
    )
    with torch_device_fn.device(input.device):
        _nonzero_static_multiblock_count_kernel[(num_blocks,)](
            x,
            counts,
            numel,
            IS_COMPLEX=source.is_complex(),
            BLOCK_SIZE=_MULTI_BLOCK_TILE_SIZE,
        )
        prefix = flag_gems.cumsum(counts, dim=0)
        _nonzero_static_multiblock_scatter_kernel[(num_blocks,)](
            x,
            counts,
            prefix,
            nz,
            size,
            numel,
            IS_COMPLEX=source.is_complex(),
            BLOCK_SIZE=_MULTI_BLOCK_TILE_SIZE,
        )
        if size:
            _nonzero_static_multiblock_assemble_kernel[
                (triton.cdiv(size, _FILL_BLOCK_SIZE),)
            ](
                nz,
                prefix[-1],
                workspace,
                fill_value,
                size,
                ndim,
                *shape,
                BLOCK_SIZE=_FILL_BLOCK_SIZE,
            )
    result = workspace[:size]
    if out is None:
        return result
    out.resize_((size, ndim))
    out.copy_(result)
    return out


def _small_nonzero_static(input, size, fill_value, out):
    ndim = input.dim()
    numel = input.numel()
    if ndim == 0 or ndim > 6 or numel > _SMALL_INPUT_MAX_NUMEL:
        return None

    if numel == 0:
        if out is None:
            return torch.full(
                (size, ndim), fill_value, dtype=torch.int64, device=input.device
            )
        out.resize_((size, ndim))
        out.fill_(fill_value)
        return out

    block_size = triton.next_power_of_2(max(numel, 1))
    source = input.contiguous()
    if source.is_complex():
        x = torch.view_as_real(source).reshape(-1)
    else:
        x = source

    workspace = torch.empty(
        (size + max(block_size, _FILL_BLOCK_SIZE), ndim),
        device=input.device,
        dtype=torch.int64,
    )
    count = torch.empty((), device=input.device, dtype=torch.int64)
    shape = tuple(input.shape) + (1,) * (6 - ndim)
    with torch_device_fn.device(input.device):
        _nonzero_static_small_kernel[(1,)](
            x,
            workspace,
            count,
            size,
            numel,
            ndim,
            *shape,
            IS_COMPLEX=source.is_complex(),
            BLOCK_SIZE=block_size,
        )
        if size:
            _nonzero_static_fill_tail_kernel[(triton.cdiv(size, _FILL_BLOCK_SIZE),)](
                workspace, count, fill_value, size, ndim, BLOCK_SIZE=_FILL_BLOCK_SIZE
            )

    result = workspace[:size]
    if out is None:
        return result
    out.resize_((size, ndim))
    out.copy_(result)
    return out


def nonzero_static(input: torch.Tensor, *, size: int, fill_value: int = -1):
    size = _check_int_arg(size, "size")
    fill_value = _check_int_arg(fill_value, "fill_value")
    if size < 0:
        raise RuntimeError("nonzero_static: size must be non-negative")
    result = _small_nonzero_static(input, size, fill_value, out=None)
    if result is not None:
        return result
    result = _multiblock_nonzero_static(input, size, fill_value, out=None)
    if result is not None:
        return result
    return _nonzero_static(input, size=size, fill_value=fill_value)


def nonzero_static_out(
    input: torch.Tensor,
    *,
    size: int,
    fill_value: int = -1,
    out: torch.Tensor,
):
    if out.dtype != torch.int64:
        raise RuntimeError(
            f"Expected out tensor to have dtype torch.int64, but got {out.dtype} instead"
        )
    if out.device != input.device:
        raise RuntimeError(
            f"Expected out tensor to be on {input.device}, but got {out.device} instead"
        )

    size = _check_int_arg(size, "size")
    fill_value = _check_int_arg(fill_value, "fill_value")
    if size < 0:
        raise RuntimeError("nonzero_static: size must be non-negative")
    result = _small_nonzero_static(input, size, fill_value, out=out)
    if result is not None:
        return result
    result = _multiblock_nonzero_static(input, size, fill_value, out=out)
    if result is not None:
        return result
    result = _nonzero_static(input, size=size, fill_value=fill_value)
    out.resize_((size, input.dim()))
    out.copy_(result)
    return out
