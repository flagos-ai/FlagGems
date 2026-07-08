import logging
import math
from numbers import Number

import torch
import triton
import triton.language as tl

from flag_gems.ops.full import check_dtype, full as _fallback_full
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)

_SOPHGO_GRID_CAP = 64


@libentry()
@triton.jit(do_not_specialize=["fill_value"])
def _full_scalar_contig_kernel(
    out,
    n_elements,
    fill_value,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    num_jobs = tle.num_programs(0)
    block_start = (pid * BLOCK_SIZE).to(tl.int64)
    step = num_jobs * BLOCK_SIZE
    for block_start_offset in range(block_start, n_elements, step):
        offsets = block_start_offset + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        tl.store(out + offsets, fill_value, mask=mask)


@libentry()
@triton.jit(do_not_specialize=["fill_value"])
def _full_scalar_contig_nomask_kernel(
    out,
    n_elements,
    fill_value,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    num_jobs = tle.num_programs(0)
    block_start = (pid * BLOCK_SIZE).to(tl.int64)
    step = num_jobs * BLOCK_SIZE
    for block_start_offset in range(block_start, n_elements, step):
        offsets = block_start_offset + tl.arange(0, BLOCK_SIZE)
        tl.store(out + offsets, fill_value)


@libentry()
@triton.jit
def _full_tensor_contig_kernel(
    out,
    n_elements,
    fill_value,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    num_jobs = tle.num_programs(0)
    block_start = (pid * BLOCK_SIZE).to(tl.int64)
    step = num_jobs * BLOCK_SIZE
    value_scalar = tl.load(fill_value)
    for block_start_offset in range(block_start, n_elements, step):
        offsets = block_start_offset + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        tl.store(out + offsets, value_scalar, mask=mask)


@libentry()
@triton.jit
def _full_tensor_contig_nomask_kernel(
    out,
    n_elements,
    fill_value,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    num_jobs = tle.num_programs(0)
    block_start = (pid * BLOCK_SIZE).to(tl.int64)
    step = num_jobs * BLOCK_SIZE
    value_scalar = tl.load(fill_value)
    for block_start_offset in range(block_start, n_elements, step):
        offsets = block_start_offset + tl.arange(0, BLOCK_SIZE)
        tl.store(out + offsets, value_scalar)


def _is_tpu_device(device):
    if device is None:
        return False
    device = torch.device(device)
    return device.type in ("tpu", "sophgo")


def _is_tpu_tensor(x):
    return isinstance(x, torch.Tensor) and x.device.type in ("tpu", "sophgo")


def _numel(size):
    if isinstance(size, int):
        return size
    return math.prod(size)


def _normalize_dtype(fill_value, dtype):
    if dtype is not None:
        return dtype
    if isinstance(fill_value, bool):
        return torch.bool
    if isinstance(fill_value, int):
        return torch.int32
    return torch.get_default_dtype()


def _choose_block_size(n_elements):
    return 4096 if n_elements <= 4096 else 8192


def _launch_grid(n_elements, block_size):
    return (min(triton.cdiv(n_elements, block_size), _SOPHGO_GRID_CAP),)


def _launch_full_scalar(size, fill_value, dtype, device):
    out = torch.empty(size, device=device, dtype=dtype)
    n_elements = _numel(size)
    if n_elements == 0:
        return out
    block_size = _choose_block_size(n_elements)
    grid = _launch_grid(n_elements, block_size)
    if n_elements % block_size == 0:
        _full_scalar_contig_nomask_kernel[grid](
            out,
            n_elements,
            fill_value,
            BLOCK_SIZE=block_size,
        )
    else:
        _full_scalar_contig_kernel[grid](
            out,
            n_elements,
            fill_value,
            BLOCK_SIZE=block_size,
        )
    return out


def _launch_full_tensor(size, fill_value, dtype, device):
    out = torch.empty(size, device=device, dtype=dtype)
    n_elements = _numel(size)
    if n_elements == 0:
        return out
    block_size = _choose_block_size(n_elements)
    grid = _launch_grid(n_elements, block_size)
    if n_elements % block_size == 0:
        _full_tensor_contig_nomask_kernel[grid](
            out,
            n_elements,
            fill_value,
            BLOCK_SIZE=block_size,
        )
    else:
        _full_tensor_contig_kernel[grid](
            out,
            n_elements,
            fill_value,
            BLOCK_SIZE=block_size,
        )
    return out


def _launch_full_half_scalar(size, fill_value, dtype, device):
    return _launch_full_scalar(size, fill_value, torch.float32, device).to(dtype)


def full(size, fill_value, *, dtype=None, layout=None, device=None, pin_memory=None):
    logger.debug("SOPHGO FULL")
    if device is None:
        return _fallback_full(
            size,
            fill_value,
            dtype=dtype,
            layout=layout,
            device=device,
            pin_memory=pin_memory,
        )

    resolved_dtype = _normalize_dtype(fill_value, dtype)
    if dtype is not None:
        fill_value = check_dtype(fill_value, resolved_dtype, device)

    if (
        layout in (None, torch.strided)
        and pin_memory in (None, False)
        and resolved_dtype
        in (torch.float32, torch.int32, torch.float16, torch.bfloat16)
        and _is_tpu_device(device)
    ):
        if isinstance(fill_value, torch.Tensor):
            if fill_value.ndim != 0:
                return _fallback_full(
                    size,
                    fill_value,
                    dtype=resolved_dtype,
                    layout=layout,
                    device=device,
                    pin_memory=pin_memory,
                )
            if _is_tpu_tensor(fill_value):
                if resolved_dtype in (torch.float16, torch.bfloat16):
                    return _launch_full_tensor(
                        size, fill_value.to(torch.float32), torch.float32, device
                    ).to(resolved_dtype)
                return _launch_full_tensor(size, fill_value, resolved_dtype, device)
            if resolved_dtype in (torch.float16, torch.bfloat16):
                return _launch_full_half_scalar(
                    size, fill_value.item(), resolved_dtype, device
                )
            return _launch_full_scalar(size, fill_value.item(), resolved_dtype, device)
        if isinstance(fill_value, Number):
            if resolved_dtype in (torch.float16, torch.bfloat16):
                return _launch_full_half_scalar(size, fill_value, resolved_dtype, device)
            return _launch_full_scalar(size, fill_value, resolved_dtype, device)

    return _fallback_full(
        size,
        fill_value,
        dtype=resolved_dtype,
        layout=layout,
        device=device,
        pin_memory=pin_memory,
    )
