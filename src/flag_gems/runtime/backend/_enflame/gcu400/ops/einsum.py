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

"""GCU400 einsum 特化：在 use_gems() 上下文内把常见 equation 分派到 Triton 路径。

aten 的通用 einsum 在 enflame 后端会拆成一串 pointwise + 通用归约，
`i,i->` 在 65536 元素上编译出的单块归约 kernel 需要 2.6MB DSM（硬件上限
917504B），直接 OutOfResources；`i,j->ij` 落到广播 mul 的 gather 路径。
本模块只在 FlagGems 注册上下文内接管，native 参考仍走原始 aten 实现。
"""

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

_ORIGINAL_EINSUM = torch.einsum
_DOT_BLOCK = 1024


@triton.jit
def _einsum_dot_partial(x_ptr, y_ptr, partial_ptr, N, BLOCK: tl.constexpr):
    # 每个 program 归约一个 BLOCK 段，块内状态向量长度固定，DSM 与 N 解耦
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    xv = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    yv = tl.load(y_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    tl.store(partial_ptr + pid, tl.sum(xv * yv))


@triton.jit
def _einsum_dot_final(partial_ptr, out_ptr, M, BLOCK_M: tl.constexpr):
    offsets = tl.arange(0, BLOCK_M)
    vals = tl.load(partial_ptr + offsets, mask=offsets < M, other=0.0)
    tl.store(out_ptr, tl.sum(vals))


def _einsum_dot(x, y):
    # 延迟导入避免 flag_gems 初始化期的循环依赖；两阶段分块归约与 N 解耦 DSM 占用
    from flag_gems.runtime import torch_device_fn

    num_elements = x.numel()
    block = min(_DOT_BLOCK, triton.next_power_of_2(num_elements))
    num_partials = triton.cdiv(num_elements, block)
    partial = torch.empty((num_partials,), dtype=torch.float32, device=x.device)
    out = torch.empty((), dtype=x.dtype, device=x.device)
    with torch_device_fn.device(x.device):
        _einsum_dot_partial[(num_partials,)](
            x, y, partial, num_elements, BLOCK=block, num_warps=8
        )
        _einsum_dot_final[(1,)](
            partial,
            out,
            num_partials,
            BLOCK_M=triton.next_power_of_2(num_partials),
            num_warps=4,
        )
    return out


def _gems_is_active():
    # use_gems().__enter__ 通过 enable() 设置该全局，退出时删除
    try:
        import flag_gems

        return getattr(flag_gems, "current_work_registrar", None) is not None
    except Exception:
        return False


def _dispatch(equation, operands):
    eq = equation.replace(" ", "")
    if eq == "i,i->":
        return _einsum_dot(operands[0], operands[1])
    if eq == "i,j->ij":
        # 直接调用 enflame 的 row-streaming Triton kernel，避免 aten 广播 mul 路径
        import flag_gems

        return flag_gems.outer(operands[0], operands[1])
    if eq in ("ij,jk->ik", "bij,bjk->bik", "...ij,...jk->...ik"):
        return torch.matmul(operands[0], operands[1])
    if eq == "ii->":
        return torch.diagonal(operands[0]).sum()
    if eq == "ii->i":
        return torch.diagonal(operands[0])
    if eq == "ij->ji":
        return operands[0].transpose(0, 1)
    if eq == "ijk->":
        return operands[0].sum()
    if eq == "ijk->j":
        return operands[0].sum(dim=(0, 2))
    # 未覆盖的 equation 保持 aten 原生行为，避免影响其它算子
    return _ORIGINAL_EINSUM(equation, *operands)


def einsum(equation, *operands):
    if _gems_is_active():
        try:
            return _dispatch(equation, operands)
        except Exception:
            logger.debug("GCU400 einsum fell back for equation %s", equation)
            return _ORIGINAL_EINSUM(equation, *operands)
    return _ORIGINAL_EINSUM(equation, *operands)


torch.einsum = einsum
