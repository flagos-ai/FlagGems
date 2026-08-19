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
from triton.tools.tensor_descriptor import TensorDescriptor

from flag_gems.fused.matmul_bias_activation import (
    matmul_bias_activation as generic_matmul_bias_activation,
)
from flag_gems.runtime import torch_device_fn
from flag_gems.runtime.backend._mthreads.fused.matmul_bias_activation_fma import (
    matmul_bias_activation_fma,
)
from flag_gems.runtime.backend._mthreads.ops.mm import (
    SQMMA_ON,
    is_sqmma_compatible,
    is_supported_sqmma_layout,
)
from flag_gems.utils import broadcastable_to, libentry, libtuner
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)


@triton.jit
def prev_multiple_of(a, b):
    return tl.cdiv(a, b) * b - b


def matmul_bias_activation_sqmma_descriptor_pre_hook(nargs):
    nargs["a_desc"].block_shape = [nargs["BLOCK_M"], nargs["BLOCK_K"]]
    nargs["b_desc"].block_shape = [nargs["BLOCK_K"], nargs["BLOCK_N"]]
    nargs["c_desc"].block_shape = [nargs["BLOCK_M"], nargs["BLOCK_N"]]


@libentry()
@libtuner(
    configs=[
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8},
            num_stages=1,
            num_warps=4,
            pre_hook=matmul_bias_activation_sqmma_descriptor_pre_hook,
        )
    ],
    key=["M", "N", "K", "dtype"],
    strategy=["align32", "align32", "align32", "default"],
    warmup=5,
    rep=5,
    flagtune_op_name="matmul_bias_activation",
    flagtune_expand_op_name="matmul_bias_activation_sqmma",
    flagtune_pre_hook=matmul_bias_activation_sqmma_descriptor_pre_hook,
)
@triton.jit
def matmul_bias_activation_sqmma_kernel(
    a_desc,
    b_desc,
    bias_ptr,
    c_desc,
    M,
    N,
    K,
    stride_bias,
    dtype: tl.constexpr,
    GROUP_M: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = ext.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    offs_am = (pid_m * BLOCK_M).to(tl.int32)
    offs_bn = (pid_n * BLOCK_N).to(tl.int32)
    offs_k = tl.full((), 0, dtype=tl.int32)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load_tensor_descriptor(a_desc, [offs_am, offs_k])
        b = tl.load_tensor_descriptor(b_desc, [offs_k, offs_bn])
        accumulator = tl.dot(a, b, acc=accumulator)
        offs_k += BLOCK_K

    bias_offsets = offs_bn + tl.arange(0, BLOCK_N)
    bias = tl.load(
        bias_ptr + bias_offsets * stride_bias, mask=bias_offsets < N, other=0.0
    )
    result = accumulator + bias[None, :]
    result = tl.maximum(result, 0.0)
    tl.store_tensor_descriptor(c_desc, [offs_am, offs_bn], result.to(c_desc.dtype))


def is_matmul_bias_activation_sqmma_compatible(input, weight, bias, N, K):
    if not (SQMMA_ON and is_sqmma_compatible(input, weight, N, K)):
        return False
    if bias.dim() == 1:
        return bias.shape[0] == N and bias.stride(0) == 1
    if bias.dim() == 2:
        return bias.shape == (1, N) and is_supported_sqmma_layout(bias)
    return False


def matmul_bias_activation_sqmma(input, weight, bias, M, N, K):
    logger.debug("GEMS_MTHREADS MATMUL_BIAS_ACTIVATION_SQMMA")
    if not input.is_contiguous():
        input = input.contiguous()
    if not weight.is_contiguous():
        weight = weight.contiguous()
    if bias.dim() > 1:
        bias = bias.reshape(-1)

    out = torch.empty((M, N), dtype=input.dtype, device=input.device)
    desc_a = TensorDescriptor.from_tensor(input, [1, 1])
    desc_b = TensorDescriptor.from_tensor(weight, [1, 1])
    desc_c = TensorDescriptor.from_tensor(out, [1, 1])
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
        1,
        1,
    )
    with torch_device_fn.device(input.device):
        matmul_bias_activation_sqmma_kernel[grid](
            desc_a,
            desc_b,
            bias,
            desc_c,
            M,
            N,
            K,
            bias.stride(0),
            str(input.dtype).split(".")[-1],
        )
    return out


def matmul_bias_activation(input, weight, bias):
    assert input.shape[1] == weight.shape[0], "Incompatible dimensions"
    assert broadcastable_to(
        bias.shape, (input.shape[0], weight.shape[1])
    ), "Incompatible input shape"

    M, K = input.shape
    _, N = weight.shape

    if is_matmul_bias_activation_sqmma_compatible(input, weight, bias, N, K):
        return matmul_bias_activation_sqmma(input, weight, bias, M, N, K)
    if input.dtype in (torch.float16, torch.bfloat16, torch.float32):
        return matmul_bias_activation_fma(input, weight, bias, M, N, K)
    return generic_matmul_bias_activation(input, weight, bias)
