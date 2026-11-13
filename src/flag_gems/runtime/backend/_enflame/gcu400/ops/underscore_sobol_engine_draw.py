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

"""gcu400 sobol_engine_draw: 闭式 Gray-code 并行实现。

原实现对样本维做 n 次串行循环、每次再扫描 30 个 bit，且 ``n`` 是
``tl.constexpr``，fp64 输出的 kernel 在 GCU400 的 ``make_gcuir`` 阶段会
稳定触发编译器段错误。这里把递推改成闭式：

    Q(k) = Q0 ^ (G(num_generated + k) ^ G(num_generated))
    G(k) = XOR_{j: gray(k)_j = 1} v_j

其中 ``v_j`` 是 ``sobolstate[:, j]``，``gray(k) = k ^ (k >> 1)``。
这样每个样本可在各自的 program 中独立计算，样本维与维度维都能并行；
输出统一以 fp32 计算，fp64 请求在 host 侧做一次 dtype 转换，避开
设备上不被支持的 fp64 store。
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)


@triton.jit
def sobol_draw_parallel_kernel(
    result_ptr,
    quasi_out_ptr,
    quasi_ptr,
    sobolstate_ptr,
    n,
    dimension,
    num_generated,
    recipd: tl.constexpr,
    MAXBIT: tl.constexpr,
    BLOCK_SAMPLES: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
):
    """按 (样本块, 维度块) 二维网格并行生成 Sobol 点。

    ``delta_bits = gray(num_generated + sample + 1) ^ gray(num_generated)``
    的每个置位对应一个需要 XOR 的方向向量；30 轮 ``static_range`` 在编译期
    展开，运行期不再有样本间的数据依赖。
    """
    samples = tl.program_id(0) * BLOCK_SAMPLES + tl.arange(0, BLOCK_SAMPLES)
    dims = tl.program_id(1) * BLOCK_DIM + tl.arange(0, BLOCK_DIM)
    sample_mask = samples < n
    dim_mask = dims < dimension

    quasi_initial = tl.load(quasi_ptr + dims, mask=dim_mask, other=0).to(tl.int64)
    base_index = num_generated.to(tl.int64)
    gray_base = base_index ^ (base_index >> 1)
    gray_next = (base_index + samples.to(tl.int64) + 1) ^ (
        (base_index + samples.to(tl.int64) + 1) >> 1
    )
    delta_bits = gray_next ^ gray_base

    quasi = tl.broadcast_to(quasi_initial[None, :], (BLOCK_SAMPLES, BLOCK_DIM)).to(
        tl.int64
    )
    for bit_index in tl.static_range(MAXBIT):
        bit = ((delta_bits >> bit_index) & 1).to(tl.int64)
        direction = tl.load(
            sobolstate_ptr + dims * MAXBIT + bit_index,
            mask=dim_mask,
            other=0,
        ).to(tl.int64)
        quasi = quasi ^ (bit[:, None] * direction[None, :])

    result_vals = quasi.to(tl.float32) * recipd
    tl.store(
        result_ptr + samples[:, None] * dimension + dims[None, :],
        result_vals,
        mask=sample_mask[:, None] & dim_mask[None, :],
    )

    # 最后一个样本块顺带写回最终状态：Q(n) 正是最后一个样本所在行。
    is_last_sample = samples == (n - 1)
    final_quasi = tl.sum(tl.where(is_last_sample[:, None], quasi, 0), axis=0)
    if tl.program_id(0) == tl.num_programs(0) - 1:
        tl.store(quasi_out_ptr + dims, final_quasi, mask=dim_mask)


def _select_sobol_config(n, dimension):
    """返回 (样本块、维度块、num_warps)，10 轮扫描后的定稿配置。

    32 样本 × 4 维度 × 4 warps 在 4 档 benchmark 上合计 2.855 ms，优于
    32×8（4.866 ms）、128×8（2.905 ms）与按 n 动态分档（3.074 ms）。维度块
    取 4 让 grid.y 覆盖更多 SIP；样本块继续放大会受 dsm 内存限制，16 维度块、
    8 warps 以及更小的样本块都会明显劣化。
    """
    return 32, 4, 4


def underscore_sobol_engine_draw(
    quasi, n, sobolstate, dimension, num_generated, *, dtype=torch.float32
):
    """生成 n 个 Sobol 序列点，返回 (结果, 更新后的 quasi 状态)。"""
    logger.debug("GEMS UNDERSCORE_SOBOL_ENGINE_DRAW")

    MAXBIT = 30
    recipd = 1.0 / (2**MAXBIT)
    # GCU400 不支持 fp64 store；fp64 请求先按 fp32 计算再做一次 dtype 转换。
    compute_dtype = torch.float32 if dtype == torch.float64 else dtype
    result = torch.empty((n, dimension), dtype=compute_dtype, device=quasi.device)
    quasi_out = torch.empty_like(quasi)

    if n == 0:
        quasi_out.copy_(quasi)
        return result.to(dtype) if compute_dtype != dtype else result, quasi_out

    block_samples, block_dim, num_warps = _select_sobol_config(n, dimension)
    grid = (triton.cdiv(n, block_samples), triton.cdiv(dimension, block_dim))
    with torch_device_fn.device(quasi.device):
        sobol_draw_parallel_kernel[grid](
            result,
            quasi_out,
            quasi,
            sobolstate,
            n,
            dimension,
            num_generated,
            recipd,
            MAXBIT,
            block_samples,
            block_dim,
            num_warps=num_warps,
        )
    if compute_dtype != dtype:
        result = result.to(dtype)
    return result, quasi_out
