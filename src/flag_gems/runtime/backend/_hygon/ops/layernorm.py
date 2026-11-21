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
from typing import Optional, Sequence, Tuple

import torch
import triton

from flag_gems.ops.layernorm import (
    layer_norm_loop_kernel,
    layer_norm_persistent_kernel,
    layer_norm_persistent_kernel_multiline,
)
from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)


def layer_norm(
    input: torch.Tensor,
    normalized_shape: Sequence[int],
    weight: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    logger.debug("GEMS_HYGON LAYERNORM_FORWARD")

    if not normalized_shape:
        raise RuntimeError("Expected normalized_shape to be at least 1-dimensional")
    if len(normalized_shape) > input.ndim or tuple(
        input.shape[-len(normalized_shape) :]
    ) != tuple(normalized_shape):
        raise RuntimeError(
            "Expected input with trailing dimensions matching normalized_shape"
        )
    for name, tensor in (("weight", weight), ("bias", bias)):
        if tensor is not None and tuple(tensor.shape) != tuple(normalized_shape):
            raise RuntimeError(f"Expected {name} shape to match normalized_shape")

    N = math.prod(normalized_shape)
    M = input.numel() // N
    input = input.contiguous()
    weight = None if weight is None else weight.contiguous()
    bias = None if bias is None else bias.contiguous()
    output = torch.empty_like(input)
    # Native ATen keeps the reduction statistics in FP32 for FP16/BF16 inputs.
    mean = torch.empty(M, dtype=torch.float32, device=input.device)
    rstd = torch.empty_like(mean)
    tile_n = triton.next_power_of_2(N)

    # Scalar constexpr options retain libentry's cached launch binding.
    # Eight Welford streams align with the loaded CPU boxing moments order.
    moment_options = {}
    if input.dtype == torch.float32:
        depth = max(1, (triton.cdiv(N // 8, 16) - 1).bit_length())
        moment_options = {
            "USE_CASCADE": True,
            "MOMENTS_N": N,
            "MOMENTS_DEPTH": depth,
            "MOMENTS_STACK": triton.next_power_of_2(depth),
            "enable_fp_fusion": False,
        }

    with torch_device_fn.device(input.device):
        if N <= 128:
            tile_m = triton.cdiv(1024, tile_n)
            layer_norm_persistent_kernel_multiline[(triton.cdiv(M, tile_m),)](
                input,
                output,
                weight,
                bias,
                mean,
                rstd,
                M,
                N,
                eps,
                tile_m,
                tile_n,
                **moment_options,
            )
        elif N <= 8192:
            layer_norm_persistent_kernel[(M,)](
                input,
                output,
                weight,
                bias,
                mean,
                rstd,
                M,
                N,
                eps,
                tile_n,
                **moment_options,
            )
        else:
            layer_norm_loop_kernel[(M,)](
                input, output, weight, bias, mean, rstd, M, N, eps, **moment_options
            )
    return output, mean, rstd


def native_layer_norm(
    input: torch.Tensor,
    normalized_shape: Sequence[int],
    weight: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    logger.debug("GEMS_HYGON NATIVE_LAYER_NORM")
    output, mean, rstd = layer_norm(input, normalized_shape, weight, bias, eps)
    stats_shape = input.shape[: -len(normalized_shape)] + (1,) * len(normalized_shape)
    return output, mean.view(stats_shape), rstd.view(stats_shape)
