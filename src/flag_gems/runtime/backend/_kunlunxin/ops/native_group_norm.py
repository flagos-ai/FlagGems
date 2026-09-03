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

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, tl_extra_shim
from flag_gems.utils import triton_lang_extension as ext

# The accuracy test asserts on the generic logger name; native_layer_norm.py
# does the same thing for the same reason.
logger = logging.getLogger("flag_gems.ops.native_group_norm")
rsqrt = tl_extra_shim.rsqrt


# One program per (n, group).  A group is `group_size` channels x HW
# CONTIGUOUS elements, so both the reduction and the affine write can live in
# the SAME program: the flat 1D reduce over [base, base + group_size*HW) keeps
# mean/rstd in registers and the normalize walks the group ONE CHANNEL AT A
# TIME (`GROUP_SIZE` is constexpr so the channel loop unrolls statically),
# loading a SCALAR weight/bias per channel -> contiguous HW block DMA, no
# per-element `idx // HW` gather.
#
# groupnorm.py splits these two phases into two @libentry kernels, which costs
# a second launch of the SAME grid (N*group).  These shapes are launch bound
# (grid=128 costs ~65us of the ~135us total), so folding the phases into one
# kernel removes half of that, and for fp16/bf16 it also removes the fp32
# mean/rstd scratch pair: mean/rstd never round-trip through memory, so there
# is no precision loss on reload and no second store.
@libentry()
@triton.jit(do_not_specialize=["eps"])
def native_group_norm_kernel(
    X,
    Y,
    W,
    B,
    Mean,
    Rstd,
    group_size,
    HW,
    num_groups,
    eps,
    GROUP_SIZE: tl.constexpr,
    BLOCK_HW_SIZE: tl.constexpr,
):
    pid = ext.program_id(0)
    group = pid % num_groups
    num_elements = group_size * HW
    base = pid * num_elements
    ch_base = group * group_size

    sum_acc = tl.zeros([BLOCK_HW_SIZE], dtype=tl.float32)
    sumsq_acc = tl.zeros([BLOCK_HW_SIZE], dtype=tl.float32)
    for off in range(0, num_elements, BLOCK_HW_SIZE):
        idx = off + tl.arange(0, BLOCK_HW_SIZE)
        m = idx < num_elements
        x = tl.load(X + base + idx, mask=m, other=0.0).to(tl.float32)
        sum_acc += x
        sumsq_acc += x * x

    mean = tl.sum(sum_acc) / num_elements
    var = tl.sum(sumsq_acc) / num_elements - mean * mean
    rstd = rsqrt(var + eps)
    tl.store(Mean + pid, mean)
    tl.store(Rstd + pid, rstd)

    for c in range(0, GROUP_SIZE):
        cbase = base + c * HW
        if W is None:
            weight = 1.0
        else:
            weight = tl.load(W + ch_base + c).to(tl.float32)
        if B is None:
            bias = 0.0
        else:
            bias = tl.load(B + ch_base + c).to(tl.float32)
        for off in range(0, HW, BLOCK_HW_SIZE):
            idx = off + tl.arange(0, BLOCK_HW_SIZE)
            m = idx < HW
            x = tl.load(X + cbase + idx, mask=m, other=0.0).to(tl.float32)
            y = (x - mean) * rstd * weight + bias
            tl.store(Y + cbase + idx, y, mask=m)


def native_group_norm(input, weight, bias, N, C, HxW, group, eps=1e-05):
    """aten::native_group_norm on a single fused Kunlunxin kernel.

    The generic flag_gems.ops.native_group_norm binds
    flag_gems.ops.groupnorm.group_norm at import time, so SpecOpRegistrar
    swapping flag_gems.group_norm never reached it and native_group_norm kept
    running the generic single-kernel giant-2D-tile implementation on XPU.
    That path miscompiles on the small tiles used by the accuracy matrix and
    hard-fails with `out of resource: uni_sram` for HxW >= 4096, so bind a
    vendor kernel here explicitly.
    """
    logger.debug("GEMS NATIVE_GROUP_NORM")

    group_size = triton.cdiv(C, group)
    input = input.contiguous()
    weight = None if weight is None else weight.contiguous()
    bias = None if bias is None else bias.contiguous()

    y = torch.empty_like(input)
    mean = torch.empty((N, group), dtype=input.dtype, device=input.device)
    rstd = torch.empty((N, group), dtype=input.dtype, device=input.device)

    grid = (N * group,)
    block_hw = min(triton.next_power_of_2(HxW), 1024)
    with torch_device_fn.device(input.device):
        native_group_norm_kernel[grid](
            input,
            y,
            weight,
            bias,
            mean,
            rstd,
            group_size,
            HxW,
            group,
            eps,
            GROUP_SIZE=group_size,
            BLOCK_HW_SIZE=block_hw,
        )
    return y, mean, rstd
