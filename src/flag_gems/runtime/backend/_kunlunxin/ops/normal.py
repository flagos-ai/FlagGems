import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils.random_utils import (
    philox_backend_seed_offset,
    uint_to_uniform_float,
)
from flag_gems.utils.shape_utils import broadcast_shapes, volume

from ..utils.pointwise_dynamic import pointwise_dynamic
from .randn import pair_uniform_to_normal, randn_kernel

logger = logging.getLogger(__name__)


@pointwise_dynamic(
    is_tensor=[True, True, True], promotion_methods=[(0, 1, 2, "DEFAULT")]
)
@triton.jit
def transform_func_tensor_tensor(val, std, mean):
    return val * std + mean


@pointwise_dynamic(
    is_tensor=[True, False, True], promotion_methods=[(0, 1, 2, "DEFAULT")]
)
@triton.jit
def transform_func_tensor_float(val, std, mean):
    return val * std + mean


@pointwise_dynamic(
    is_tensor=[True, True, False], promotion_methods=[(0, 1, 2, "DEFAULT")]
)
@triton.jit
def transform_func_float_tensor(val, std, mean):
    return val * std + mean


@pointwise_dynamic(
    is_tensor=[True, False, False], promotion_methods=[(0, 1, 2, "DEFAULT")]
)
@triton.jit
def transform_func_float_float(val, std, mean):
    return val * std + mean


UNROLL = 4


def normal_distribution(shape, device, *, generator=None, out=None):
    if out is None:
        out = torch.empty(shape, device=device, dtype=torch.float32)
    N = volume(shape)
    cluster_num = 12
    BLOCK_SIZE = min(triton.next_power_of_2(triton.cdiv(N, cluster_num * UNROLL)), 1024)
    grid_fn = triton.cdiv(N, BLOCK_SIZE * UNROLL)

    increment = triton.cdiv(N, UNROLL)
    philox_seed, philox_offset = philox_backend_seed_offset(
        increment, generator=generator
    )
    with torch_device_fn.device(device):
        randn_kernel[(grid_fn,)](out, N, philox_seed, philox_offset, BLOCK_SIZE)
    return out


def normal_tensor_tensor(mean, std, *, generator=None):
    logger.debug("GEMS_KUNLUNXIN NORMAL_TENSOR_TENSOR")
    shape = broadcast_shapes([mean.shape, std.shape])
    device = mean.device
    out = normal_distribution(shape, device)
    return transform_func_tensor_tensor(out, std, mean)


def normal_tensor_float(mean, std, *, generator=None):
    logger.debug("GEMS_KUNLUNXIN NORMAL_TENSOR_FLOAT")
    shape = mean.shape
    device = mean.device
    out = normal_distribution(shape, device)
    return transform_func_tensor_float(out, std, mean)


def normal_float_tensor(mean, std, *, generator=None):
    logger.debug("GEMS_KUNLUNXIN NORMAL_FLOAT_TENSOR")
    shape = std.shape
    device = std.device
    out = normal_distribution(shape, device)
    return transform_func_float_tensor(out, std, mean)


def normal_(self, mean=0, std=1, *, generator=None):
    logger.debug("GEMS_KUNLUNXIN NORMAL_")
    shape = self.shape
    device = self.device
    N = volume(shape)
    cluster_num = 12
    BLOCK_SIZE = min(triton.next_power_of_2(triton.cdiv(N, cluster_num * UNROLL)), 1024)
    grid_fn = triton.cdiv(N, BLOCK_SIZE * UNROLL)
    FULL = N % (BLOCK_SIZE * UNROLL) == 0
    increment = triton.cdiv(N, UNROLL)
    philox_seed, philox_offset = philox_backend_seed_offset(
        increment, generator=generator
    )
    with torch_device_fn.device(device):
        normal_fill_kernel[(grid_fn,)](
            self, N, philox_seed, philox_offset, std, mean, BLOCK_SIZE, FULL
        )
    return self


@triton.jit(do_not_specialize=["philox_seed", "philox_offset", "std", "mean"])
def normal_fill_kernel(
    out_ptr,
    N,
    philox_seed,
    philox_offset,
    std,
    mean,
    BLOCK: tl.constexpr,
    FULL: tl.constexpr,
):
    """Generate standard normal samples (same Philox layout as randn_kernel)
    and store ``n * std + mean`` directly into the output buffer.  When FULL
    is true the whole grid covers the buffer exactly, so plain unmasked stores
    are emitted; otherwise the four stores carry a boundary mask."""
    philox_seed = philox_seed.to(tl.int64)
    philox_offset = philox_offset.to(tl.int64)
    c0 = (philox_offset & 0xFFFFFFFF).to(tl.uint32)
    c1 = ((philox_offset >> 32) & 0xFFFFFFFF).to(tl.uint32)
    i4 = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    c0 += i4
    _O = c0 * 0
    r0, r1, r2, r3 = tl.philox(philox_seed, c0, c1, _O, _O, 5)
    r0 = uint_to_uniform_float(r0)
    r1 = uint_to_uniform_float(r1)
    r2 = uint_to_uniform_float(r2)
    r3 = uint_to_uniform_float(r3)
    n0, n1 = pair_uniform_to_normal(r0, r1)
    n2, n3 = pair_uniform_to_normal(r2, r3)
    off_0 = tl.program_id(0) * BLOCK * 4 + tl.arange(0, BLOCK)
    off_1 = off_0 + BLOCK
    off_2 = off_1 + BLOCK
    off_3 = off_2 + BLOCK
    if FULL:
        tl.store(out_ptr + off_0, n0 * std + mean)
        tl.store(out_ptr + off_1, n1 * std + mean)
        tl.store(out_ptr + off_2, n2 * std + mean)
        tl.store(out_ptr + off_3, n3 * std + mean)
    else:
        tl.store(out_ptr + off_0, n0 * std + mean, mask=off_0 < N)
        tl.store(out_ptr + off_1, n1 * std + mean, mask=off_1 < N)
        tl.store(out_ptr + off_2, n2 * std + mean, mask=off_2 < N)
        tl.store(out_ptr + off_3, n3 * std + mean, mask=off_3 < N)
