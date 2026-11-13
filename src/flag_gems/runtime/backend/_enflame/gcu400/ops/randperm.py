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
import math

import torch
import triton
import triton.language as tl

from flag_gems.runtime import device, torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils.random_utils import philox_backend_seed_offset

logger = logging.getLogger(__name__)
device_ = device

# GCU notes (measured on L600 / gcu400):
# - tl.philox has an inconsistent vectorized lowering across kernels on this
#   backend (results land on even lanes only), so keys are generated with a
#   splitmix32-style bijection hash expressed in plain integer ops instead.
# - The generic bitonic argsort costs ~1-6 ms per program on GCU and does not
#   scale with grid parallelism; the previous counting/partition kernels also
#   need O(n^2) comparisons or 5 radix passes (3 kernels each). The single
#   kernel PRP below needs no sort, histogram or cross-kernel synchronization.
# - The GCU compiler toolchain crashes intermittently under machine load
#   (subprocess segfault during compile, not cached on failure), so kernel
#   launches are retried a couple of times.


@triton.jit
def _rand_key(seed, offset, idx):
    # splitmix32-style bijection hash: deterministic per element id
    seed = seed.to(tl.int64)
    offset = offset.to(tl.int64)
    lo = (offset & 0xFFFFFFFF).to(tl.uint32)
    hi = ((offset >> 32) & 0xFFFFFFFF).to(tl.uint32)
    slo = (seed & 0xFFFFFFFF).to(tl.uint32)
    x = idx.to(tl.uint32) ^ lo
    x = x + hi
    x = x ^ slo
    x = x ^ (x >> 16)
    x = tl.mul(x, 0x21F0AAAD, sanitize_overflow=False)
    x = x ^ (x >> 15)
    x = tl.mul(x, 0x735A2D97, sanitize_overflow=False)
    x = x ^ (x >> 15)
    return x


# 单 kernel 伪随机置换：先用与 n 互质的线性同余映射打散全局顺序，再叠加若干层
# 「块内旋转」。LCG 与每层旋转都是 [0, n) 上的双射，叠加后仍是双射，因此输出
# 天然是 0..n-1 的排列，不需要排序、直方图、前缀和或跨 kernel 同步。每个元素
# 只依赖自己的下标与 seed，可完全并行；块大小取 2 的幂以便用位运算定位，尾块
# 用编译期常量单独旋转，避免越界或改变双射性。


@libentry()
@triton.jit(do_not_specialize=["philox_seed", "philox_offset"])
def randperm_prp_kernel(
    out_ptr,
    philox_seed,
    philox_offset,
    slope,
    bias,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
    NUM_CTAS: tl.constexpr,
    LAYERS: tl.constexpr,
):
    pid = tl.program_id(0)
    for base in range(pid * BLOCK, N, NUM_CTAS * BLOCK):
        idx = base + tl.arange(0, BLOCK)
        mask = idx < N
        x = ((idx.to(tl.int64) * slope + bias) % N).to(tl.int32)
        for layer in tl.static_range(LAYERS):
            log_block = 3 + 3 * layer
            block_size = 1 << log_block
            last_id = N >> log_block
            last_size = N - (last_id << log_block)
            block_id = x >> log_block
            block_start = block_id << log_block
            h = _rand_key(philox_seed, philox_offset + layer * 0x1F4A7C15, block_id).to(
                tl.int64
            )
            rot_full = (h & (block_size - 1)).to(tl.int32)
            if last_size > 0:
                rot_last = (h % last_size).to(tl.int32)
                rot = tl.where(block_id == last_id, rot_last, rot_full)
                delta = x - block_start + rot
                rotated = tl.where(
                    block_id == last_id,
                    delta % last_size,
                    delta & (block_size - 1),
                )
            else:
                rotated = (x - block_start + rot_full) & (block_size - 1)
            x = block_start + rotated
        tl.store(out_ptr + idx, x, mask=mask)


def _prp_params(n, seed, offset):
    """由 philox seed/offset 导出与 n 互质的奇数斜率与任意偏置。"""
    mixed = (int(seed) ^ (int(offset) * 0x9E3779B97F4A7C15)) & 0xFFFFFFFF
    slope = (mixed % max(n - 1, 1)) + 1
    if slope % 2 == 0:
        slope += 1
    while math.gcd(slope, n) > 1:
        slope += 2
        if slope >= n:
            slope = 1
    bias = ((mixed * 0x85EBCA6B) & 0xFFFFFFFF) % n
    return slope, bias


def _retry(fn, attempts=3):
    for i in range(attempts):
        try:
            return fn()
        except Exception:
            if i == attempts - 1:
                raise


def randperm(
    n,
    *,
    generator=None,
    out=None,
    dtype=torch.int64,
    layout=torch.strided,
    device=None,
    requires_grad=False,
    pin_memory=False,
):
    logger.debug("GEMS_ENFLAME RANDPERM")
    assert dtype == torch.int16 or dtype == torch.int32 or dtype == torch.int64
    # 元素 id 在 int32 域内解码，输出 dtype 只影响最终 store
    assert n <= 2**31, "n exceeds maximum int32"
    if device is None:
        device = torch.device(device_.name)
    if n <= 1:
        return torch.zeros(n, dtype=dtype, device=device)

    seed, offset = philox_backend_seed_offset(max(n, 1), generator=generator)
    slope, bias = _prp_params(n, seed, offset)
    # 直接按目标 dtype 分配并写出，省去 int32 结果再转换的一次 kernel launch
    result = torch.empty(n, dtype=dtype, device=device)
    block = 1024
    num_ctas = min(96, triton.cdiv(n, block))
    with torch_device_fn.device(device):
        _retry(
            lambda: randperm_prp_kernel[(num_ctas,)](
                result,
                seed,
                offset,
                slope,
                bias,
                n,
                block,
                num_ctas,
                3,
                num_warps=4,
            )
        )
    return result
