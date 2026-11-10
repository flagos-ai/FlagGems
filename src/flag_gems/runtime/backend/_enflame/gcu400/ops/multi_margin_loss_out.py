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
"""GCU400 专属 ``multi_margin_loss.out``。

公共 ``_compute_forward`` 以 ``grid=min(N, 65535)`` 启动逐行 kernel，
在 GCU400（4 DIE x 6 SIP）上会创建数千个 program，调度与 DTE 启动开销
远大于实际计算。这里保持同一套 Triton 计算 kernel，仅把启动策略改为
固定 grid + 核内跨步，并把 ``BLOCK_C`` / ``num_warps`` / 归约块大小
暴露为可调参数；其余输入校验、dtype policy、out 语义与公共实现一致。
"""

import logging

import torch
import triton

from flag_gems.ops.multi_margin_loss import (
    _REDUCE_BLOCK_SIZE,
    _USE_DEVICE_ASSERT,
    _check_inputs,
    _empty_forward,
    _multi_margin_loss_kernel,
    _multi_margin_loss_reduce_kernel,
    _normalize_p,
    _normalize_reduction,
    _output_shape,
)
from flag_gems.ops.multi_margin_loss import (
    multi_margin_loss_out as _public_multi_margin_loss_out,
)
from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)

# --- TUNING PARAMS START ---


# 由优化迭代驱动脚本按轮次覆盖；均为编译期/启动期配置，不改变算子语义。
FIXED_GRID = 48
NUM_WARPS = 4
BLOCK_C_CAP = 1024
REDUCE_BLOCK = 512
FALLBACK_TO_PUBLIC = 0
# --- TUNING PARAMS END ---


def _compute_forward_gcu400(
    input,
    target,
    weight,
    N,
    C,
    is_batched,
    p,
    margin,
    reduction,
    output_shape,
    destination,
):
    """固定 grid 启动逐行 loss kernel，需要归约时再启动一次 reduce kernel。"""
    if N == 0:
        return _empty_forward(input, N, is_batched, reduction)

    output = (
        destination
        if destination is not None
        else torch.empty(output_shape, dtype=input.dtype, device=input.device)
    )
    # 每行一个 program 会超出 SIP 拓扑；固定 program 数后在核内按行跨步。
    grid = (min(N, FIXED_GRID),)
    block_c = min(BLOCK_C_CAP, triton.next_power_of_2(C))
    weight_ptr = input if weight is None else weight
    has_weight = weight is not None

    needs_reduction = is_batched and N > 1 and reduction != 0
    if needs_reduction:
        # 逐行结果用 fp32 累积，避免低精度下二次归约放大误差。
        row_losses = torch.empty((N,), dtype=torch.float32, device=input.device)
    else:
        row_losses = output

    with torch_device_fn.device(input.device):
        _multi_margin_loss_kernel[grid](
            input,
            target,
            weight_ptr,
            row_losses,
            N,
            C,
            margin,
            P=p,
            HAS_WEIGHT=has_weight,
            USE_DEVICE_ASSERT=_USE_DEVICE_ASSERT,
            BLOCK_C=block_c,
            num_warps=NUM_WARPS,
        )
        if needs_reduction:
            _multi_margin_loss_reduce_kernel[(1,)](
                row_losses,
                output,
                N,
                REDUCTION=reduction,
                BLOCK_SIZE=REDUCE_BLOCK or _REDUCE_BLOCK_SIZE,
                num_warps=NUM_WARPS,
            )
    return output


def multi_margin_loss_out(
    input: torch.Tensor,
    target: torch.Tensor,
    p=1,
    margin=1,
    weight=None,
    reduction=1,
    *,
    out: torch.Tensor,
) -> torch.Tensor:
    logger.debug("GEMS_GCU400 MULTI_MARGIN_LOSS_OUT")
    if FALLBACK_TO_PUBLIC:
        return _public_multi_margin_loss_out(
            input, target, p, margin, weight, reduction, out=out
        )
    p = _normalize_p(p)
    reduction = _normalize_reduction(reduction)
    try:
        margin = float(margin)
    except (TypeError, ValueError) as error:
        raise RuntimeError("multi_margin_loss: margin must be a real scalar") from error
    input, target, weight, N, C, is_batched = _check_inputs(input, target, weight)
    if out.device != input.device:
        raise RuntimeError("multi_margin_loss.out: out must be on the input device")
    if out.dtype != input.dtype:
        raise RuntimeError("multi_margin_loss.out: out must have the input dtype")

    shape = _output_shape(N, is_batched, reduction)
    if tuple(out.shape) != shape:
        out.resize_(shape)
    destination = out if out.is_contiguous() else None
    result = _compute_forward_gcu400(
        input,
        target,
        weight,
        N,
        C,
        is_batched,
        p,
        margin,
        reduction,
        shape,
        destination,
    )
    if result is not out:
        out.copy_(result)
    return out
