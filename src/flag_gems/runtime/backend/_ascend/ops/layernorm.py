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

import torch
import triton

from flag_gems.ops.layernorm import (
    layer_norm_loop_kernel,
    layer_norm_persistent_kernel,
    layer_norm_persistent_kernel_multiline,
)
from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)


def _select_tile_m(M, N, tile_n):
    if N <= 128:
        return triton.cdiv(1024, tile_n)
    if M < 256:
        return 1
    # Pack adjacent rows while keeping the one-pass tile bounded.
    return max(1, min(8, 4096 // tile_n))


def layer_norm(input, normalized_shape, weight=None, bias=None, eps=1e-5):
    logger.debug("GEMS_ASCEND LAYERNORM FORWARD")

    N = math.prod(normalized_shape)
    M = input.numel() // N

    input = input.contiguous()
    weight = None if weight is None else weight.contiguous()
    bias = None if bias is None else bias.contiguous()
    y = torch.empty_like(input)

    # Statistics saved for backward use the input dtype, matching the common path.
    mean = torch.empty(M, dtype=input.dtype, device=input.device)
    rstd = torch.empty(M, dtype=input.dtype, device=input.device)

    with torch_device_fn.device(input.device):
        if N <= 4096:
            tile_n = triton.next_power_of_2(N)
            tile_m = _select_tile_m(M, N, tile_n)
            if tile_m > 1:
                grid = (triton.cdiv(M, tile_m), 1, 1)
                layer_norm_persistent_kernel_multiline[grid](
                    input,
                    y,
                    weight,
                    bias,
                    mean,
                    rstd,
                    M,
                    N,
                    eps,
                    tile_m,
                    tile_n,
                )
            else:
                grid = (M, 1, 1)
                layer_norm_persistent_kernel[grid](
                    input,
                    y,
                    weight,
                    bias,
                    mean,
                    rstd,
                    M,
                    N,
                    eps,
                    tile_n,
                )
        else:
            grid = (M, 1, 1)
            layer_norm_loop_kernel[grid](
                input,
                y,
                weight,
                bias,
                mean,
                rstd,
                M,
                N,
                eps,
            )
    return y, mean, rstd
