import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils.random_utils import (
    philox_backend_seed_offset,
    uint_to_uniform_float,
)

logger = logging.getLogger(__name__)

NUM_SIPS = 24


@triton.jit
def transform_exponential_f32(u, inv_lambd, eps_minus):
    log = tl.where(u >= 1.0 + eps_minus, eps_minus, tl.math.log(u))
    return -inv_lambd * log


@libentry()
@triton.jit(do_not_specialize=["philox_seed", "philox_offset", "N"])
def fused_exponential_kernel_f32(
    out_ptr, N, inv_lambd, eps_minus, philox_seed, philox_offset,
    BLOCK: tl.constexpr,
):
    philox_seed = philox_seed.to(tl.int64)
    philox_offset = philox_offset.to(tl.int64)
    c0 = (philox_offset & 0xFFFFFFFF).to(tl.uint32)
    c1 = ((philox_offset >> 32) & 0xFFFFFFFF).to(tl.uint32)

    pid = tl.program_id(0)
    i = pid * BLOCK + tl.arange(0, BLOCK)
    c0 += i
    z = c0 * 0
    r0, r1, r2, r3 = tl.philox(philox_seed, c0, c1, z, z)

    y0 = transform_exponential_f32(uint_to_uniform_float(r0), inv_lambd, eps_minus)
    y1 = transform_exponential_f32(uint_to_uniform_float(r1), inv_lambd, eps_minus)
    y2 = transform_exponential_f32(uint_to_uniform_float(r2), inv_lambd, eps_minus)
    y3 = transform_exponential_f32(uint_to_uniform_float(r3), inv_lambd, eps_minus)

    start = pid.to(tl.uint64) * BLOCK * 4
    off0 = start + tl.arange(0, BLOCK)
    off1 = off0 + BLOCK
    off2 = off1 + BLOCK
    off3 = off2 + BLOCK

    tl.store(out_ptr + off0, y0, mask=off0 < N)
    tl.store(out_ptr + off1, y1, mask=off1 < N)
    tl.store(out_ptr + off2, y2, mask=off2 < N)
    tl.store(out_ptr + off3, y3, mask=off3 < N)


def exponential_(x, lambd: float = 1.0, *, generator=None):
    logger.debug("GEMS EXPONENTIAL_")

    dtype = x.dtype
    device = x.device
    inplace = x.is_contiguous()
    assert dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64)

    N = x.numel()
    inv_lambd = 1.0 / lambd
    eps_minus = -0.5 * torch.finfo(dtype).eps

    out = x if inplace else torch.empty_like(x)

    UNROLL = 4
    if N >= 67108864:
        BLOCK = 32768
    elif N >= 65536:
        BLOCK = 16384
    else:
        BLOCK = min(triton.next_power_of_2(N), 1024)
    increment = triton.cdiv(N, UNROLL)
    philox_seed, philox_offset = philox_backend_seed_offset(
        increment, generator=generator
    )

    use_f32_buffer = dtype in (torch.float16, torch.bfloat16) and N > 65536
    if use_f32_buffer:
        out_buf = torch.empty(N, dtype=torch.float32, device=device)
    else:
        out_buf = out.view(-1) if inplace else out

    grid_size = triton.cdiv(N, BLOCK * UNROLL)
    with torch_device_fn.device(device):
        fused_exponential_kernel_f32[(grid_size,)](
            out_buf, N, inv_lambd, eps_minus, philox_seed, philox_offset,
            BLOCK,
            num_warps=1,
        )

    if use_f32_buffer:
        out.view(-1).copy_(out_buf.to(dtype))

    if not inplace:
        x.copy_(out)
    return x
