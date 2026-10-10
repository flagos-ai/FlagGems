import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)

_SMALL_N_BOUNDARIES = 8
_boundary_cache = {}


def _host_boundaries(boundaries):
    """Host-side f32 values of `boundaries` with a version-guarded memo."""
    key = (boundaries.data_ptr(), boundaries.numel(), boundaries.dtype)
    version = boundaries._version
    entry = _boundary_cache.get(key)
    if entry is not None and entry[0] == version:
        return entry[1]
    values = [float(x) for x in boundaries.reshape(-1).cpu().tolist()]
    _boundary_cache[key] = (version, values, boundaries)
    if len(_boundary_cache) > 64:
        _boundary_cache.pop(next(iter(_boundary_cache)))
    return values


@libentry()
@triton.jit
def bucketize_kernel(
    inp_ptr,
    boundaries_ptr,
    out_ptr,
    n_elements,
    N_BOUNDARIES: tl.constexpr,
    right: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    v = tl.load(inp_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    idx = tl.zeros([BLOCK_SIZE], dtype=tl.int32)
    for i in tl.static_range(N_BOUNDARIES):
        b = tl.load(boundaries_ptr + i).to(tl.float32)
        if right:
            cond = b <= v
        else:
            cond = b < v
        idx = tl.where(cond, i + 1, idx)

    tl.store(out_ptr + offsets, idx.to(tl.int64), mask=mask)


@libentry()
@triton.jit
def bucketize_kernel_small(
    inp_ptr,
    out_ptr,
    n_elements,
    b0,
    b1,
    b2,
    b3,
    b4,
    b5,
    b6,
    b7,
    N_BOUNDARIES: tl.constexpr,
    right: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    pid = tle.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    if NEED_MASK:
        mask = offsets < n_elements
        v = tl.load(inp_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    else:
        v = tl.load(inp_ptr + offsets).to(tl.float32)

    idx = tl.zeros([BLOCK_SIZE], dtype=tl.int32)
    if N_BOUNDARIES > 0:
        idx += (b0 <= v).to(tl.int32) if right else (b0 < v).to(tl.int32)
    if N_BOUNDARIES > 1:
        idx += (b1 <= v).to(tl.int32) if right else (b1 < v).to(tl.int32)
    if N_BOUNDARIES > 2:
        idx += (b2 <= v).to(tl.int32) if right else (b2 < v).to(tl.int32)
    if N_BOUNDARIES > 3:
        idx += (b3 <= v).to(tl.int32) if right else (b3 < v).to(tl.int32)
    if N_BOUNDARIES > 4:
        idx += (b4 <= v).to(tl.int32) if right else (b4 < v).to(tl.int32)
    if N_BOUNDARIES > 5:
        idx += (b5 <= v).to(tl.int32) if right else (b5 < v).to(tl.int32)
    if N_BOUNDARIES > 6:
        idx += (b6 <= v).to(tl.int32) if right else (b6 < v).to(tl.int32)
    if N_BOUNDARIES > 7:
        idx += (b7 <= v).to(tl.int32) if right else (b7 < v).to(tl.int32)

    if NEED_MASK:
        tl.store(out_ptr + offsets, idx.to(tl.int64), mask=mask)
    else:
        tl.store(out_ptr + offsets, idx.to(tl.int64))


def bucketize(input, boundaries, *, out_int32=False, right=False):
    logger.debug("GEMS_KUNLUNXIN BUCKETIZE")
    output_dtype = torch.int32 if out_int32 else torch.int64

    if boundaries.numel() == 0:
        return torch.zeros_like(input, dtype=output_dtype)

    output = torch.empty_like(input, dtype=torch.int64)

    n_elements = input.numel()
    n_boundaries = boundaries.numel()

    input_flat = input.contiguous().flatten()
    output_flat = output.flatten()
    boundaries = boundaries.contiguous()

    _SMALL_SHAPE_THRESHOLD = 65536
    if n_boundaries <= _SMALL_N_BOUNDARIES and n_elements > _SMALL_SHAPE_THRESHOLD:
        values = _host_boundaries(boundaries)
        padded = values + [0.0] * (_SMALL_N_BOUNDARIES - n_boundaries)
        BLOCK_SIZE = 4096 if n_elements <= 4 * 1024 * 1024 else 16384
        grid = (triton.cdiv(n_elements, BLOCK_SIZE), 1, 1)
        bucketize_kernel_small[grid](
            input_flat,
            output_flat,
            n_elements,
            *padded,
            n_boundaries,
            right,
            BLOCK_SIZE,
            (n_elements % BLOCK_SIZE) != 0,
        )
    else:
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n_elements, BLOCK_SIZE), 1, 1)
        bucketize_kernel[grid](
            input_flat,
            boundaries,
            output_flat,
            n_elements,
            n_boundaries,
            right,
            BLOCK_SIZE,
            num_warps=1,
            buffer_size_limit=2048,
            isCloseVectorization=True,
        )

    output = output.reshape(input.shape)
    if out_int32:
        output = output.to(torch.int32)
    return output
