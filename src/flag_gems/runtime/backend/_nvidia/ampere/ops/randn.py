import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import device, torch_device_fn
from flag_gems.utils.random_utils import (
    philox_backend_seed_offset,
    uint_to_uniform_float,
)
from flag_gems.utils.shape_utils import volume

device_ = device
logger = logging.getLogger(
    f'flag_gems.runtime.backend._nvidia.ops.{__name__.split(".")[-1]}'
)


@triton.jit
def fast_cos_ptx(x):
    return tl.inline_asm_elementwise(
        "cos.approx.ftz.f32 $0, $1;",
        "=f,f",
        [x],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def fast_sin_ptx(x):
    return tl.inline_asm_elementwise(
        "sin.approx.ftz.f32 $0, $1;",
        "=f,f",
        [x],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def fast_box_muller_ptx(u1, u2):
    u1 = tl.maximum(1.0e-7, u1)
    th = 6.283185307179586 * u2

    r = tl.sqrt(-2.0 * tl.log(u1))

    return r * fast_cos_ptx(th), r * fast_sin_ptx(th)


pair_uniform_to_normal = fast_box_muller_ptx


@triton.heuristics(runtime.get_heuristic_config("randn"))
@triton.jit(do_not_specialize=["philox_seed", "philox_offset"])
def randn_kernel(
    out_ptr,
    N,
    philox_seed,
    philox_offset,
    BLOCK: tl.constexpr,
):
    philox_seed = philox_seed.to(tl.int64)
    philox_offset = philox_offset.to(tl.int64)
    c0 = (philox_offset & 0xFFFFFFFF).to(tl.uint32)
    c1 = ((philox_offset >> 32) & 0xFFFFFFFF).to(tl.uint32)

    i4 = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    c0 += i4
    _O = c0 * 0

    r0, r1, r2, r3 = tl.philox(philox_seed, c0, c1, _O, _O)

    r0 = uint_to_uniform_float(r0)
    r1 = uint_to_uniform_float(r1)
    r2 = uint_to_uniform_float(r2)
    r3 = uint_to_uniform_float(r3)

    n0, n1 = pair_uniform_to_normal(r0, r1)
    n2, n3 = pair_uniform_to_normal(r2, r3)

    n_01 = tl.cat(n0, n1, can_reorder=True)
    n_23 = tl.cat(n2, n3, can_reorder=True)
    n_all = tl.cat(n_01, n_23, can_reorder=True)

    base_offset = tl.program_id(0) * BLOCK * 4
    off_all = base_offset + tl.arange(0, BLOCK * 4)

    tl.store(out_ptr + off_all, n_all, mask=off_all < N, eviction_policy="evict_first")


UNROLL = 4


def randn(size, *, dtype=None, layout=None, device=None, pin_memory=None):
    logger.debug("GEMS_NVIDIA RANDN")
    if dtype is None:
        dtype = torch.get_default_dtype()
    if device is None:
        device = torch.device(device_.name)
    out = torch.empty(size, device=device, dtype=dtype)
    N = volume(size)

    grid_fn = lambda meta: (triton.cdiv(N, meta["BLOCK"] * UNROLL),)

    increment = triton.cdiv(N, UNROLL)
    philox_seed, philox_offset = philox_backend_seed_offset(increment)

    with torch_device_fn.device(device):
        randn_kernel[grid_fn](out, N, philox_seed, philox_offset)

    return out
