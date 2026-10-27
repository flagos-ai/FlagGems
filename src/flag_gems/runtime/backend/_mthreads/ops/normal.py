# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging

import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle

from flag_gems.ops.randn import high_precision_fast_sin_cos, randn_kernel
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import pointwise_dynamic
from flag_gems.utils.random_utils import (
    philox_backend_seed_offset,
    uint_to_uniform_float,
)
from flag_gems.utils.shape_utils import broadcast_shapes, volume

logger = logging.getLogger(__name__)

UNROLL = 4


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


def normal_distribution(shape, device, *, generator=None, out=None):
    if out is None:
        out = torch.empty(shape, device=device, dtype=torch.float32)
    N = volume(shape)
    grid_fn = lambda meta: (triton.cdiv(N, meta["BLOCK"] * UNROLL),)

    increment = triton.cdiv(N, UNROLL)
    philox_seed, philox_offset = philox_backend_seed_offset(
        increment, generator=generator
    )
    with torch_device_fn.device(device):
        randn_kernel[grid_fn](out, N, philox_seed, philox_offset)
    return out


def normal_tensor_tensor(mean, std, *, generator=None):
    logger.debug("GEMS NORMAL_TENSOR_TENSOR")
    shape = broadcast_shapes([mean.shape, std.shape])
    device = mean.device
    out = normal_distribution(shape, device)
    return transform_func_tensor_tensor(out, std, mean)


def normal_float_tensor(mean, std, *, generator=None):
    logger.debug("GEMS NORMAL_FLOAT_TENSOR")
    shape = std.shape
    device = std.device
    out = normal_distribution(shape, device)
    return transform_func_float_tensor(out, std, mean)


def normal_(self, mean=0, std=1, *, generator=None):
    logger.debug("GEMS NORMAL_")
    shape = self.shape
    device = self.device
    self = normal_distribution(shape, device, generator=None, out=self)
    transform_func_float_float(self, std, mean, out0=self)
    return self


# ===========================================================================
# OPTIMIZED: normal_tensor_float (Fused + TLE Prefetch + Adaptive Tile)
# ===========================================================================

_DTYPE_MAP = {
    torch.float32: tl.float32,
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
}


@triton.jit
def _pair_uniform_to_normal_fast(u1, u2):
    u1 = tl.maximum(1.0e-7, u1)
    theta = 6.283185307179586 * u2
    r = tl.sqrt(-2.0 * tl.log(u1))
    sin_t, cos_t = high_precision_fast_sin_cos(theta)
    return r * cos_t, r * sin_t


@triton.jit(do_not_specialize=["philox_seed", "philox_offset", "std_val"])
def _normal_tensor_float_fused_kernel(
    out_ptr,
    mean_ptr,
    std_val,
    N,
    philox_seed,
    philox_offset,
    OUT_DTYPE: tl.constexpr,
    IS_BCAST: tl.constexpr,
    BLOCK: tl.constexpr,
):
    base = tl.program_id(0) * (BLOCK * 4)
    off_0 = base + tl.arange(0, BLOCK)
    off_1 = off_0 + BLOCK
    off_2 = off_1 + BLOCK
    off_3 = off_2 + BLOCK
    mask_0 = off_0 < N
    mask_1 = off_1 < N
    mask_2 = off_2 < N
    mask_3 = off_3 < N

    # 1. 消除向量乘法，优先发射异步全局访存 (TLE is_async=True)
    if IS_BCAST:
        m_val = tl.load(mean_ptr)
        m0 = m_val
        m1 = m_val
        m2 = m_val
        m3 = m_val
    else:
        m0 = tle.load(mean_ptr + off_0, mask=mask_0, other=0.0, is_async=True)
        m1 = tle.load(mean_ptr + off_1, mask=mask_1, other=0.0, is_async=True)
        m2 = tle.load(mean_ptr + off_2, mask=mask_2, other=0.0, is_async=True)
        m3 = tle.load(mean_ptr + off_3, mask=mask_3, other=0.0, is_async=True)

    # 2. Philox 伪随机数生成（计算掩盖访存）
    philox_seed = philox_seed.to(tl.int64)
    philox_offset = philox_offset.to(tl.int64)
    c0 = (philox_offset & 0xFFFFFFFF).to(tl.uint32)
    c1 = ((philox_offset >> 32) & 0xFFFFFFFF).to(tl.uint32)
    i4 = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    c0 = c0 + i4
    _O = c0 * 0
    r0, r1, r2, r3 = tl.philox(philox_seed, c0, c1, _O, _O)
    r0 = uint_to_uniform_float(r0)
    r1 = uint_to_uniform_float(r1)
    r2 = uint_to_uniform_float(r2)
    r3 = uint_to_uniform_float(r3)

    # 3. Box-Muller 变换
    n0, n1 = _pair_uniform_to_normal_fast(r0, r1)
    n2, n3 = _pair_uniform_to_normal_fast(r2, r3)

    # 4. 融合仿射变换并写回
    r0f = n0 * std_val + m0.to(tl.float32)
    r1f = n1 * std_val + m1.to(tl.float32)
    r2f = n2 * std_val + m2.to(tl.float32)
    r3f = n3 * std_val + m3.to(tl.float32)

    tl.store(out_ptr + off_0, r0f.to(OUT_DTYPE), mask=mask_0)
    tl.store(out_ptr + off_1, r1f.to(OUT_DTYPE), mask=mask_1)
    tl.store(out_ptr + off_2, r2f.to(OUT_DTYPE), mask=mask_2)
    tl.store(out_ptr + off_3, r3f.to(OUT_DTYPE), mask=mask_3)


def normal_tensor_float(mean, std, *, generator=None):
    N = mean.numel()
    if N == 0:
        return torch.empty_like(mean)

    out = torch.empty_like(mean)
    out_dtype = _DTYPE_MAP[mean.dtype]

    if N == 1:
        is_bcast = True
    else:
        if not mean.is_contiguous():
            mean = mean.contiguous()
        is_bcast = False

    # 小规模张量自适应减小 Block，激活更多 MPC 并消除尾块闲置
    if N <= 16384:
        cur_block = 256
    else:
        cur_block = 512

    tile_size = cur_block * UNROLL

    increment = (N + 3) // UNROLL
    philox_seed, philox_offset = philox_backend_seed_offset(
        increment, generator=generator
    )

    grid = ((N + (tile_size - 1)) // tile_size,)
    with torch_device_fn.device(mean.device):
        _normal_tensor_float_fused_kernel[grid](
            out,
            mean,
            float(std),
            N,
            philox_seed,
            philox_offset,
            OUT_DTYPE=out_dtype,
            IS_BCAST=is_bcast,
            BLOCK=cur_block,
            num_warps=4,
        )
    return out