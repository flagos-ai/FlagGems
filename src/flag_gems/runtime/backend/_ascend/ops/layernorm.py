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

from flag_gems.ops.layernorm import layer_norm as common_layer_norm
from flag_gems.ops.layernorm import (
    layer_norm_persistent_kernel,
    layer_norm_persistent_kernel_multiline,
)
from flag_gems.runtime import torch_device_fn
from flag_gems.utils.libentry import LibEntry

from .mm import _blockified

logger = logging.getLogger(__name__)


class BlockifiedLayerNormEntry(LibEntry):
    """Enable block mapping for this kernel during its first compile.

    The flagtree Ascend compiler reads TRITON_ALL_BLOCKS_PARALLEL at compile
    time; cached launches reuse the generated launcher, so the flag is scoped
    to the first run. The env var is process-global, so a concurrent first
    compile of another kernel may still observe it; this is the same trade-off
    the other Ascend kernels make, scoped to the narrowest window.
    """

    def _first_run(self, *args, **kwargs):
        with _blockified():
            return super()._first_run(*args, **kwargs)


layer_norm_persistent_kernel = BlockifiedLayerNormEntry(layer_norm_persistent_kernel.fn)
layer_norm_persistent_kernel_multiline = BlockifiedLayerNormEntry(
    layer_norm_persistent_kernel_multiline.fn
)


def _select_tile_m(M, N, tile_n):
    # Narrow rows keep the packed schedule: they already run a whole row per
    # program and the one-row kernel is not exercised at these widths.
    if tile_n <= 128:
        return triton.cdiv(1024, tile_n)
    # Pack medium-width rows to amortize per-program device work. Wider rows
    # stay single-row to avoid the measured 4096-column packing regression.
    if M < 256 or tile_n > 2048:
        return 1
    return max(1, min(16, 8192 // tile_n))


def layer_norm(input, normalized_shape, weight=None, bias=None, eps=1e-5):
    logger.debug("GEMS_ASCEND LAYERNORM_FORWARD")

    N = math.prod(normalized_shape)
    M = input.numel() // N

    # This backend specialization only changes the one-pass schedule. Keep the
    # common streaming implementation for large normalized dimensions.
    if N > 4096:
        return common_layer_norm(input, normalized_shape, weight, bias, eps)

    input = input.contiguous()
    weight = None if weight is None else weight.contiguous()
    bias = None if bias is None else bias.contiguous()
    y = torch.empty_like(input)

    # Statistics saved for backward use the input dtype, matching the common path.
    mean = torch.empty(M, dtype=input.dtype, device=input.device)
    rstd = torch.empty(M, dtype=input.dtype, device=input.device)

    # Explicit block mapping makes scheduling independent of other operator imports.
    with torch_device_fn.device(input.device):
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
                enable_auto_blockify=True,
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
                enable_auto_blockify=True,
            )
    return y, mean, rstd


def native_layer_norm(input, normalized_shape, weight=None, bias=None, eps=1e-5):
    """Route the registered native operator through the Ascend forward path."""
    logger.debug("GEMS_ASCEND NATIVE_LAYER_NORM")
    output, mean, rstd = layer_norm(input, normalized_shape, weight, bias, eps)
    stats_shape = input.shape[: -len(normalized_shape)] + (1,) * len(normalized_shape)
    return output, mean.view(stats_shape), rstd.view(stats_shape)
