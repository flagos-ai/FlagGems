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
from typing import Sequence, cast

import torch
import triton

from flag_gems import runtime
from flag_gems.ops.layernorm import (
    layer_norm_loop_kernel,
    layer_norm_persistent_kernel_multiline,
)
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)
PERSISTENT_WIDTH_LIMIT = 16384
ELEMENTS_PER_WARP_LIMIT = 2048


def prune_layer_norm_configs(
    configs: list[triton.Config], named_args: dict[str, object], **kwargs: object
) -> list[triton.Config]:
    width = triton.next_power_of_2(cast(int, named_args["N"]))
    # Very wide tiles with few warps inflate live vectors and spend minutes
    # in register allocation. Keep the search within the useful register budget.
    return [
        config
        for config in configs
        if config.kwargs["TILE_M"] * width <= config.num_warps * ELEMENTS_PER_WARP_LIMIT
    ]


# Reuse the generic reduction and affine computation with a backend-specific
# row/warp search. The original entry retains its own autotuning and cache.
layer_norm_rows_kernel = libentry()(
    triton.autotune(
        configs=runtime.get_tuned_config("layer_norm_rows"),
        key=["M", "N"],
        prune_configs_by={"early_config_prune": prune_layer_norm_configs},
    )(layer_norm_persistent_kernel_multiline.jit_function)
)


def layer_norm(
    input: torch.Tensor,
    normalized_shape: Sequence[int],
    weight: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    eps: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    logger.debug("GEMS_THEAD LAYER_NORM")
    normalized_shape = tuple(normalized_shape)
    if (
        not normalized_shape
        or input.shape[-len(normalized_shape) :] != normalized_shape
    ):
        raise ValueError("normalized_shape must match a nonempty suffix of input.shape")
    if input.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError("layer_norm supports float16, bfloat16, and float32 inputs")
    for parameter in (weight, bias):
        if parameter is not None:
            if parameter.shape != normalized_shape or parameter.device != input.device:
                raise ValueError(
                    "affine parameters must match normalized_shape and device"
                )
            if parameter.dtype not in (input.dtype, torch.float32):
                raise TypeError(
                    "affine parameters must have input dtype or float32 dtype"
                )
    width = math.prod(normalized_shape)
    if width == 0:
        raise NotImplementedError("zero-sized normalized dimensions are not supported")
    rows = input.numel() // width
    if rows == 0:
        return (
            torch.empty_like(input),
            torch.empty(0, dtype=torch.float32, device=input.device),
            torch.empty(0, dtype=torch.float32, device=input.device),
        )

    input = input.contiguous()
    weight = None if weight is None else weight.contiguous()
    bias = None if bias is None else bias.contiguous()
    output = torch.empty_like(input)
    # Native LayerNorm exposes FP32 statistics for half-precision inputs.
    mean = torch.empty(rows, dtype=torch.float32, device=input.device)
    rstd = torch.empty_like(mean)
    with torch_device_fn.device(input.device):
        if width <= PERSISTENT_WIDTH_LIMIT:
            layer_norm_rows_kernel[lambda meta: (triton.cdiv(rows, meta["TILE_M"]),)](
                input,
                output,
                weight,
                bias,
                mean,
                rstd,
                rows,
                width,
                eps,
                TILE_N=triton.next_power_of_2(width),
            )
        else:
            layer_norm_loop_kernel[(rows, 1, 1)](
                input, output, weight, bias, mean, rstd, rows, width, eps
            )
    return output, mean, rstd


def native_layer_norm(
    input: torch.Tensor,
    normalized_shape: Sequence[int],
    weight: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    eps: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    logger.debug("GEMS_THEAD NATIVE_LAYER_NORM")
    output, mean, rstd = layer_norm(input, normalized_shape, weight, bias, eps)
    stats_shape = input.shape[: -len(normalized_shape)] + (1,) * len(normalized_shape)
    return output, mean.view(stats_shape), rstd.view(stats_shape)
