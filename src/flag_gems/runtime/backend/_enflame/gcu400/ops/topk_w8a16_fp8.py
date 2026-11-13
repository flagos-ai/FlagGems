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

"""GCU400 专属的 W8A16 FP8 TopK（单 kernel 实现）。

原多阶段实现（TLE radix / chunked stage1+stage2 / argsort）在 gcu400 的
benchmark 口径下 gems latency 达到 1.3~33.6 ms。这里改为单 kernel：把 FP8
位模式按 uint8 读入（不依赖设备侧 FP8 指针），按组 scale 解码成 float32 值，
再逐次选极值得到有序的 top-k；一次 launch 完成，不需要临时候选缓冲。
"""

import math

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry


@libentry()
@triton.jit
def topk_w8a16_fp8_kernel(
    x_ptr,
    scale_ptr,
    y_vals_ptr,
    y_idx_ptr,
    N: tl.constexpr,
    K: tl.constexpr,
    GROUPS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    LARGEST: tl.constexpr,
    E4M3: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    valid = offs < N
    bits = tl.load(x_ptr + row * N + offs, mask=valid, other=0).to(tl.int32)

    sign = (bits >> 7) & 1
    if E4M3:
        exp = (bits >> 3) & 0xF
        mantissa = bits & 0x7
        normal_bits = (sign << 31) | ((exp + 120) << 23) | (mantissa << 20)
        subnormal = mantissa.to(tl.float32) * 0.001953125
    else:
        exp = (bits >> 2) & 0x1F
        mantissa = bits & 0x3
        normal_bits = (sign << 31) | ((exp + 112) << 23) | (mantissa << 21)
        subnormal = mantissa.to(tl.float32) * 1.52587890625e-5

    # normal 的 IEEE 位里已含符号位；subnormal 没有位模式，需单独带符号
    normal_value = normal_bits.to(tl.float32, bitcast=True)
    signed_subnormal = tl.where(sign == 1, -subnormal, subnormal)
    decoded = tl.where(exp > 0, normal_value, signed_subnormal)

    group = offs // GROUP_SIZE
    scale = tl.load(scale_ptr + row * GROUPS + group, mask=valid, other=0).to(
        tl.float32
    )
    value = decoded * scale

    # 无效 padding 用极值屏蔽，保证极值/索引只在真实元素中选出
    if LARGEST:
        sentinel = float("-inf")
    else:
        sentinel = float("inf")
    candidates = tl.where(valid, value, sentinel)

    for i in range(K):
        if LARGEST:
            best_value = tl.max(candidates, 0)
            best_index = tl.argmax(candidates, 0)
        else:
            best_value = tl.min(candidates, 0)
            best_index = tl.argmin(candidates, 0)
        tl.store(y_vals_ptr + row * K + i, best_value)
        tl.store(y_idx_ptr + row * K + i, best_index.to(tl.int64))
        candidates = tl.where(offs == best_index, sentinel, candidates)


def topk_w8a16_fp8(
    x_fp8,
    x_scale,
    k,
    dim=-1,
    largest=True,
    sorted=True,
    group_size=128,
    out_dtype=torch.bfloat16,
):
    if dim < 0:
        dim += x_fp8.ndim
    assert dim == x_fp8.ndim - 1, "Currently only support topk in last dimension"
    assert sorted, "Currently only support sorted == True"

    leading_shape = x_fp8.shape[:-1]
    n = x_fp8.shape[-1]
    if k == 0:
        out_shape = list(leading_shape) + [0]
        return (
            torch.empty(out_shape, device=x_fp8.device, dtype=out_dtype),
            torch.empty(out_shape, device=x_fp8.device, dtype=torch.int64),
        )

    batch_size = math.prod(leading_shape) if leading_shape else 1
    num_groups = triton.cdiv(n, group_size)
    expected_scale_shape = tuple(leading_shape) + (num_groups,)
    assert (
        x_scale.shape == expected_scale_shape
    ), f"x_scale shape should be {expected_scale_shape}, got {x_scale.shape}"

    y_vals = torch.empty(
        tuple(leading_shape) + (k,), device=x_fp8.device, dtype=out_dtype
    )
    y_idx = torch.empty(
        tuple(leading_shape) + (k,), device=x_fp8.device, dtype=torch.int64
    )

    # FP8 位模式按 uint8 读取：不依赖设备侧 FP8 指针，也不改变量化数据
    x_bits = x_fp8.contiguous().view(torch.uint8).reshape(batch_size, n)
    scale_2d = x_scale.reshape(batch_size, num_groups)
    is_e4m3 = x_fp8.dtype == torch.float8_e4m3fn
    block_n = triton.next_power_of_2(n)
    with torch_device_fn.device(x_fp8.device):
        topk_w8a16_fp8_kernel[(batch_size,)](
            x_bits,
            scale_2d,
            y_vals.reshape(batch_size, k),
            y_idx.reshape(batch_size, k),
            n,
            k,
            num_groups,
            group_size,
            largest,
            is_e4m3,
            block_n,
            num_warps=1,
        )
    return y_vals, y_idx
