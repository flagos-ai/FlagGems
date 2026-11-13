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

"""GCU400 专属 ``_foreach_lgamma``：逐 tensor 一维 flat kernel + 纯 Triton Lanczos。

公共 ``foreach_unary`` 执行器的 lgamma 行落到 GCU libdevice ``lgamma`` extern。
该例程体量大（级数 + 反射 + 本地内存操作），在共享 executor 的大形状 launch
形态下（16×(1024,1024)，≥512 并发 CTA）触发设备级 Sip exception → KMD abort
（P16，见 compiler_problems/P16_flat_reshape_longloop_sip_abort/sources/
flaggems_enflame_foreach_lgamma_extern_concurrency_sip.md）；128 CTA 以内可跑。
这里改成逐 tensor 的一维固定 grid + 核内跨步（每 launch ≤ FIXED_GRID 个 CTA），
函数体换成带反射的 Lanczos 近似（只用 ``tl.log``/``tl.sin``，无 extern），
计算统一提升 fp32。整型/bool/非连续输入回退公共 ``apply_unary``（小形状下
extern 并发安全，且语义/异常行为与公共实现完全一致）。
"""

import logging
from contextlib import nullcontext

import torch
import triton
import triton.language as tl

from flag_gems.ops._foreach_unary import apply_unary
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

# --- TUNING PARAMS START ---


# 由优化迭代驱动脚本按轮次生成；编译期/启动期配置，不改变算子语义。
FIXED_GRID = 384
BLOCK_SIZE = 1024
NUM_WARPS = 4
NUM_STAGES = 1
USE_DEVICE_CONTEXT = 1
REVERSE_LAUNCH_ORDER = 0
FALLBACK_TO_PUBLIC = 0
# --- TUNING PARAMS END ---

_FLOAT_DTYPES = (torch.float16, torch.float32, torch.bfloat16)


@triton.jit
def _lgamma_pos_lanczos(y):
    """``lgamma(y)`` for ``y >= 0.5``：Lanczos g=7/N=9（mvlgamma_ 的 fallback 体）。

    系数与 ``gcu400/ops/mvlgamma_.py`` 的 ``_fallback_lgamma`` 相同；只在
    ``y >= 0.5`` 上调用，``y < 0.5`` 由外层反射公式处理。
    """
    z = y - 1.0
    a = 0.9999999999998099
    a += 676.5203681218851 / (z + 1.0)
    a += -1259.1392167224028 / (z + 2.0)
    a += 771.3234287776531 / (z + 3.0)
    a += -176.6150291621406 / (z + 4.0)
    a += 12.507343278686905 / (z + 5.0)
    a += -0.13857109526572012 / (z + 6.0)
    a += 9.984369578019572e-6 / (z + 7.0)
    a += 1.5056327351493116e-7 / (z + 8.0)
    t = z + 7.5
    return 0.9189385332046727 + (z + 0.5) * tl.log(t) - t + tl.log(a)


@triton.jit
def _lgamma_full(x):
    """全定义域 ``lgamma``：``x < 0.5`` 走反射，非正整数极点与 ±inf 给 +inf。

    ``lgamma(x) = log(pi/|sin(pi x)|) - lgamma(1-x)``（x < 0.5，含负数）。
    两支共用同一个 Lanczos 参数：``y = where(x < 0.5, 1-x, x)`` 恒落在
    ``>= 0.5``——反射支要算的正是 ``lgamma(1-x)``，正支要算 ``lgamma(x)``，
    一次求值两个分支都够用（避免 where 双侧求值把 Lanczos 的 9 个除法
    付两遍）。``sin(pi*x)`` 的参数先归约到 ``[-0.5, 0.5]`` 再乘 pi——直接
    乘时大 ``|x|`` 的参数舍入误差随 ``|x|`` 无界增长；``|reduced| < 1e-7``
    时 ``sin(z) == z`` 到 fp32 以下，线性分支精确（与公共
    ``_approx_lgamma_func`` 同构）。NaN 经比较全 False 走正支后自然传播；
    0 与负整数是极点（+inf），与 ATen 的 gammaln 语义一致。
    """
    small = x < 0.5
    y = tl.where(small, 1.0 - x, x)
    value = _lgamma_pos_lanczos(y)
    reduced = x - tl.floor(x + 0.5)
    z = 3.141592653589793 * reduced
    sin_pi = tl.where(tl.abs(reduced) < 1.0e-7, z, tl.sin(z))
    refl = tl.log(3.141592653589793 / tl.abs(sin_pi)) - value
    res = tl.where(small, refl, value)
    pole = (x <= 0.0) & (x == tl.floor(x))
    return tl.where(pole | (tl.abs(x) == float("inf")), float("inf"), res)


@libentry()
@triton.jit(do_not_specialize=["N"])
def _foreach_lgamma_flat_kernel(
    in_ptr, out_ptr, N, BLOCK: tl.constexpr, FIXED_GRID: tl.constexpr
):
    """一维固定 grid 跨步执行逐元素 lgamma（纯 Triton 函数体，无 extern）。"""
    arange = tl.arange(0, BLOCK)
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    for block_id in tl.range(pid, tl.cdiv(N, BLOCK), num_programs):
        offs = block_id * BLOCK + arange
        mask = offs < N
        value = tl.load(in_ptr + offs, mask=mask).to(tl.float32)
        result = _lgamma_full(value)
        tl.store(out_ptr + offs, result, mask=mask)


def _can_use_flat_kernel(self):
    """仅接受非空的连续浮点 tensor 列表。"""
    if len(self) == 0:
        return False
    try:
        for tensor in self:
            if tensor.dtype not in _FLOAT_DTYPES or tensor.dtype.is_complex:
                return False
            if not tensor.is_contiguous():
                return False
    except AttributeError:
        # 非 Tensor 元素：返回 False 让调用方回退公共 apply_unary 对齐异常。
        return False
    return True


def _run_flat_kernel(self, inplace):
    if FALLBACK_TO_PUBLIC or not _can_use_flat_kernel(self):
        return apply_unary("lgamma", self, inplace)
    tensors = list(self)
    if not tensors:
        return [] if not inplace else None

    # 反序只改变 launch 次序，out-of-place 的输出必须保持与原列表一致。
    launch_indices = list(range(len(tensors)))
    if REVERSE_LAUNCH_ORDER:
        launch_indices.reverse()
    outputs = [None] * len(tensors) if not inplace else None
    device_context = (
        torch_device_fn.device(tensors[0].device)
        if USE_DEVICE_CONTEXT
        else nullcontext()
    )
    with device_context:
        for index in launch_indices:
            tensor = tensors[index]
            num_elems = tensor.numel()
            out_tensor = tensor if inplace else torch.empty_like(tensor)
            # 空 tensor 无元素可算，跳过 kernel launch；但输出列表必须仍保留
            # 这个空结果，保持与输入一一对应（foreach 的返回长度等于输入长度）。
            if num_elems == 0:
                if not inplace:
                    outputs[index] = out_tensor
                continue
            grid = (min(triton.cdiv(num_elems, BLOCK_SIZE), FIXED_GRID),)
            _foreach_lgamma_flat_kernel[grid](
                tensor,
                out_tensor,
                num_elems,
                BLOCK_SIZE,
                FIXED_GRID,
                num_warps=NUM_WARPS,
                num_stages=NUM_STAGES,
            )
            if not inplace:
                outputs[index] = out_tensor
    return None if inplace else outputs


def _foreach_lgamma(self):
    logger.debug("GEMS_GCU400 _FOREACH_LGAMMA")
    return _run_flat_kernel(self, inplace=False)


def _foreach_lgamma_(self):
    logger.debug("GEMS_GCU400 _FOREACH_LGAMMA_")
    return _run_flat_kernel(self, inplace=True)
