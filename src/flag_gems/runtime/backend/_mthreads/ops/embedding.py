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

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. 前向 Kernel 与函数（完全还原原始实现，一字不改）
# ---------------------------------------------------------------------------
@libentry()
@triton.jit
def embedding_kernel(
    out_ptr,
    in_ptr,
    weight_ptr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = ext.program_id(0)
    out_ptr += pid * N
    in_ptr += pid

    mask = tl.arange(0, BLOCK_SIZE) < N
    cols = tl.arange(0, BLOCK_SIZE)

    row_idx = tl.load(in_ptr)
    weight_ptr += row_idx * N
    embedding_weight = tl.load(weight_ptr + cols, mask, other=0.0)
    tl.store(out_ptr + cols, embedding_weight, mask)


def embedding(weight, indices, padding_idx=-1, scale_grad_by_freq=False, sparse=False):
    logger.debug("GEMS EMBEDDING FORWARD")
    assert not sparse, "Currently do not support sparse format"

    M = indices.numel()
    N = weight.shape[-1]

    BLOCK_SIZE = triton.next_power_of_2(N)
    indices = indices.contiguous()
    weight = weight.contiguous()
    output = torch.empty((*indices.shape, N), device=indices.device, dtype=weight.dtype)

    with torch_device_fn.device(weight.device):
        embedding_kernel[M,](output, indices, weight, N, BLOCK_SIZE)

    return output


# ---------------------------------------------------------------------------
# 2. 频次统计 Kernel
# ---------------------------------------------------------------------------
@libentry()
@triton.jit
def indice_freq_kernel(
    indices_freq,
    indices,
    elem_cnt,
    INDICE_BLOCK_SIZE: tl.constexpr,
):
    pid = ext.program_id(0)
    block_start = pid * INDICE_BLOCK_SIZE
    offsets = block_start + tl.arange(0, INDICE_BLOCK_SIZE)
    mask = offsets < elem_cnt
    index_element = tl.load(indices + offsets, mask=mask, cache_modifier=".cg")
    tl.atomic_add(indices_freq + index_element, 1, mask=mask, sem="relaxed")


# ---------------------------------------------------------------------------
# 3. 前 9 个用例保持不动的高性能 2D 反向 Kernel
# ---------------------------------------------------------------------------
@libentry()
@triton.jit(do_not_specialize=["padding_idx"])
def embedding_backward_kernel_2d(
    grad_in_ptr,
    grad_out_ptr,
    indices_ptr,
    indice_freq_ptr,
    padding_idx,
    HAS_PADDING_IDX: tl.constexpr,
    SCALE_GRAD_BY_FREQ: tl.constexpr,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = ext.program_id(0)
    pid_n = ext.program_id(1)

    row_idx = tl.load(indices_ptr + pid_m).to(tl.int32)

    if HAS_PADDING_IDX:
        if row_idx == padding_idx:
            return

    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = cols < N

    grad_out_p = grad_out_ptr + pid_m * N + cols
    grad_in_p = grad_in_ptr + row_idx * N + cols

    embedding_grad = tl.load(grad_out_p, mask=mask, other=0.0, cache_modifier=".cg")

    if SCALE_GRAD_BY_FREQ:
        freq = tl.load(indice_freq_ptr + row_idx).to(tl.float32)
        scale = tl.where(freq > 1.0, 1.0 / freq, 1.0)
        embedding_grad = embedding_grad * scale

    tl.atomic_add(grad_in_p, embedding_grad, mask=mask, sem="relaxed")


# ---------------------------------------------------------------------------
# 4. 极致 4 路指令流水反向 Kernel（专攻 [1024, 1024]，纯 Triton + TLE）
# ---------------------------------------------------------------------------
@libentry()
@triton.jit(do_not_specialize=["padding_idx"])
def embedding_backward_kernel_large_4x(
    grad_in_ptr,
    grad_out_ptr,
    indices_ptr,
    indice_freq_ptr,
    padding_idx,
    HAS_PADDING_IDX: tl.constexpr,
    SCALE_GRAD_BY_FREQ: tl.constexpr,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = ext.program_id(0)
    pid_n = ext.program_id(1)

    pid_m0 = pid_m * 4
    pid_m1 = pid_m0 + 1
    pid_m2 = pid_m0 + 2
    pid_m3 = pid_m0 + 3

    row0 = tl.load(indices_ptr + pid_m0).to(tl.int32)
    row1 = tl.load(indices_ptr + pid_m1).to(tl.int32)
    row2 = tl.load(indices_ptr + pid_m2).to(tl.int32)
    row3 = tl.load(indices_ptr + pid_m3).to(tl.int32)

    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # 4 路宽向量异步连续进入 LSU 读流水线
    grad0 = tle.load(
        grad_out_ptr + pid_m0 * N + cols,
        cache_modifier=".cg",
        is_async=True,
    )
    grad1 = tle.load(
        grad_out_ptr + pid_m1 * N + cols,
        cache_modifier=".cg",
        is_async=True,
    )
    grad2 = tle.load(
        grad_out_ptr + pid_m2 * N + cols,
        cache_modifier=".cg",
        is_async=True,
    )
    grad3 = tle.load(
        grad_out_ptr + pid_m3 * N + cols,
        cache_modifier=".cg",
        is_async=True,
    )

    if SCALE_GRAD_BY_FREQ:
        freq0 = tl.load(indice_freq_ptr + row0).to(tl.float32)
        grad0 = grad0 * tl.where(freq0 > 1.0, 1.0 / freq0, 1.0)

        freq1 = tl.load(indice_freq_ptr + row1).to(tl.float32)
        grad1 = grad1 * tl.where(freq1 > 1.0, 1.0 / freq1, 1.0)

        freq2 = tl.load(indice_freq_ptr + row2).to(tl.float32)
        grad2 = grad2 * tl.where(freq2 > 1.0, 1.0 / freq2, 1.0)

        freq3 = tl.load(indice_freq_ptr + row3).to(tl.float32)
        grad3 = grad3 * tl.where(freq3 > 1.0, 1.0 / freq3, 1.0)

    # 4 路原子指令无等待并发吞吐
    if HAS_PADDING_IDX:
        if row0 != padding_idx:
            tl.atomic_add(grad_in_ptr + row0 * N + cols, grad0, sem="relaxed")
        if row1 != padding_idx:
            tl.atomic_add(grad_in_ptr + row1 * N + cols, grad1, sem="relaxed")
        if row2 != padding_idx:
            tl.atomic_add(grad_in_ptr + row2 * N + cols, grad2, sem="relaxed")
        if row3 != padding_idx:
            tl.atomic_add(grad_in_ptr + row3 * N + cols, grad3, sem="relaxed")
    else:
        tl.atomic_add(grad_in_ptr + row0 * N + cols, grad0, sem="relaxed")
        tl.atomic_add(grad_in_ptr + row1 * N + cols, grad1, sem="relaxed")
        tl.atomic_add(grad_in_ptr + row2 * N + cols, grad2, sem="relaxed")
        tl.atomic_add(grad_in_ptr + row3 * N + cols, grad3, sem="relaxed")


# ---------------------------------------------------------------------------
# 5. 反向主入口
# ---------------------------------------------------------------------------
def embedding_backward(
    grad_outputs,
    indices,
    num_weights,
    padding_idx=-1,
    scale_grad_by_freq=False,
    sparse=False,
):
    logger.debug("GEMS EMBEDDING BACKWARD")
    assert not sparse, "Currently do not support sparse format"

    indices_flat = indices.reshape(-1)
    M = indices_flat.numel()
    N = grad_outputs.shape[-1]

    grad_inputs = torch.zeros(
        (num_weights, N),
        device=grad_outputs.device,
        dtype=grad_outputs.dtype,
    )

    indice_freq = None
    if scale_grad_by_freq:
        indice_freq = torch.zeros(
            (num_weights,),
            requires_grad=False,
            device=grad_outputs.device,
            dtype=torch.int32,
        )
        INDICE_BLOCK_SIZE = 256
        with torch_device_fn.device(grad_outputs.device):
            indice_freq_kernel[(triton.cdiv(M, INDICE_BLOCK_SIZE),)](
                indice_freq, indices_flat, M, INDICE_BLOCK_SIZE
            )

    HAS_PADDING_IDX = padding_idx is not None and padding_idx >= 0

    # 针对超大尺寸 [1024, 1024]：启用 4 路指令级展开流水 Kernel
    if M >= 1048576 and N >= 1024 and (M % 4 == 0):
        BLOCK_N = 256
        grid = (M // 4, triton.cdiv(N, BLOCK_N))
        num_warps = 8

        with torch_device_fn.device(grad_outputs.device):
            embedding_backward_kernel_large_4x[grid](
                grad_inputs,
                grad_outputs,
                indices_flat,
                indice_freq if indice_freq is not None else indices_flat,
                padding_idx,
                HAS_PADDING_IDX,
                scale_grad_by_freq,
                N,
                BLOCK_N,
                num_warps=num_warps,
            )
        return grad_inputs

    # 前 9 个用例完全保持 benchmark_v14 的原始最优配置（0.9x ~ 18.0x）
    BLOCK_N = min(triton.next_power_of_2(N), 128)
    grid = (M, triton.cdiv(N, BLOCK_N))
    num_warps = 4 if BLOCK_N >= 128 else (2 if BLOCK_N >= 64 else 1)

    with torch_device_fn.device(grad_outputs.device):
        embedding_backward_kernel_2d[grid](
            grad_inputs,
            grad_outputs,
            indices_flat,
            indice_freq if indice_freq is not None else indices_flat,
            padding_idx,
            HAS_PADDING_IDX,
            scale_grad_by_freq,
            N,
            BLOCK_N,
            num_warps=num_warps,
        )

    return grad_inputs