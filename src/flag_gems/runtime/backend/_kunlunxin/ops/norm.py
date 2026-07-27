import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

from .vector_norm import vector_norm


@libentry()
@triton.jit
def _norm_scalar_l2_partial_kernel(
    x, partials, n_elements, BLOCK_SIZE: tl.constexpr
):
    block = ext.program_id(0).to(tl.int64)
    block_offset = block * BLOCK_SIZE
    total = tl.zeros((), dtype=tl.float32)
    compensation = tl.zeros((), dtype=tl.float32)
    for offset in tl.range(0, BLOCK_SIZE, 8):
        offsets = block_offset + offset + tl.arange(0, 8)
        values = tl.load(
            x + offsets, mask=offsets < n_elements, other=0.0
        ).to(tl.float32)
        value = tl.sum(values * values, axis=0)
        adjusted = value - compensation
        updated = total + adjusted
        compensation = (updated - total) - adjusted
        total = updated
    tl.store(partials + block, total)


@libentry()
@triton.jit
def _norm_scalar_sum_partial_kernel(
    partials, next_partials, n_partials, BLOCK_SIZE: tl.constexpr
):
    block = ext.program_id(0).to(tl.int64)
    block_offset = block * BLOCK_SIZE
    total = tl.zeros((), dtype=tl.float32)
    compensation = tl.zeros((), dtype=tl.float32)
    for offset in tl.range(0, BLOCK_SIZE, 8):
        offsets = block_offset + offset + tl.arange(0, 8)
        values = tl.load(
            partials + offsets, mask=offsets < n_partials, other=0.0
        ).to(tl.float32)
        value = tl.sum(values, axis=0)
        adjusted = value - compensation
        updated = total + adjusted
        compensation = (updated - total) - adjusted
        total = updated
    tl.store(next_partials + block, total)


@libentry()
@triton.jit
def _norm_scalar_l2_finalize_kernel(partials, out):
    tl.store(out, tl.sqrt_rn(tl.load(partials).to(tl.float32)))


def _norm_scalar_l2(x):
    # Keep every device reduction below the XPU lane-count accuracy limit.
    block_size = 8
    n_elements = x.numel()
    n_partials = triton.cdiv(n_elements, block_size)
    partials = torch.empty(n_partials, dtype=torch.float32, device=x.device)
    _norm_scalar_l2_partial_kernel[(n_partials,)](
        x, partials, n_elements, BLOCK_SIZE=block_size
    )

    while n_partials > 1:
        next_n_partials = triton.cdiv(n_partials, block_size)
        next_partials = torch.empty(
            next_n_partials, dtype=torch.float32, device=x.device
        )
        _norm_scalar_sum_partial_kernel[(next_n_partials,)](
            partials, next_partials, n_partials, BLOCK_SIZE=block_size
        )
        partials = next_partials
        n_partials = next_n_partials

    out = torch.empty((), dtype=x.dtype, device=x.device)
    _norm_scalar_l2_finalize_kernel[(1,)](partials, out)
    return out


def norm(x, p=2, dim=None, keepdim=False):
    return vector_norm(x, ord=2 if p is None else p, dim=dim, keepdim=keepdim)


def norm_scalar(x, p=2):
    if (
        p in (None, 2)
        and x.is_contiguous()
        and x.numel() > 0
        and x.dtype in (torch.float16, torch.float32, torch.bfloat16)
    ):
        return _norm_scalar_l2(x)
    return norm(x, p=p, dim=None, keepdim=False)


def norm_scalaropt_dim(x, p, dim, keepdim=False):
    return norm(x, p=p, dim=dim, keepdim=keepdim)
