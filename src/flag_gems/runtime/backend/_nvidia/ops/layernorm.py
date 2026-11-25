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

"""NVIDIA layer_norm: tile the rows, widen the tile, keep statistics in FP32.

Reuses the generic Triton kernels unchanged -- no reduction or affine
arithmetic is copied -- and changes only the schedule around them, plus the
dtype the statistics are stored in.

Why the schedule. The generic host gives one program one row whenever
``N > 128``, and falls back to the two-pass loop kernel above ``N = 4096``. On
an H800 both choices leave bandwidth on the table: a 256-wide BF16 row is 1 KB
of work for a whole program, so ``[65536, 256]`` runs at 1.34 TB/s out of
3.35 TB/s, and the loop kernel's second sweep re-reads an input that no longer
fits in L2. Tiling several rows into one program, with the warp count tuned per
``(M, N)``, recovers that, and keeping the single-pass persistent kernel up to
``N = 16384`` avoids the extra read entirely.

Above that width the two-pass kernel is the only one that fits, and it gets its
own tile search here: the generic NVIDIA configurations stop at
``TILE_N = 8192``, while 16384 is what these shapes want -- 1.10x at
``[16, 131072]`` and ``[64, 262144]``.

Why FP32 statistics. ATen's ``native_layer_norm`` returns FP32 ``mean`` and
``rstd`` for half-precision inputs -- the repository's own backward benchmark
says as much -- while the generic host allocates them in the input dtype, so
the backward kernel consumes half-precision statistics it was never meant to
see. The entries below always allocate FP32, which is also why the loop kernel
is launched directly instead of through the generic host.

The wider tile is what makes that affordable. An FP32 scalar store instead of a
half-precision one costs 17-23% on latency-bound wide shapes at TILE_N 8192 --
few rows, very wide, where the kernel runs at a tenth of peak bandwidth anyway
-- and only 4-13% at TILE_N 16384. Store position and cache hints change
nothing, and ncu reports identical occupancy and DRAM traffic with lower issue
throughput, so the cost is codegen around the store rather than the store
itself. The remaining gap is covered by the tile search.
"""

import logging
import math
from typing import Optional, Sequence, Tuple

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

# Above this normalized width a persistent tile no longer fits a sensible
# register budget, so the two-pass loop kernel stays in charge. Measured on
# H800: the persistent schedule is 1.08-1.21x the loop kernel at N = 8192 and
# N = 16384, and TILE_N would have to double again beyond that.
PERSISTENT_WIDTH_LIMIT = 16384

# A configuration is worth compiling only when each warp gets a sensible slice
# of the tile. Too few elements per warp wastes the warps outright (one 256-wide
# row across 16 warps measured 0.04x on H800); too many inflate live vectors and
# spend minutes in register allocation.
ELEMENTS_PER_WARP_FLOOR = 128
ELEMENTS_PER_WARP_LIMIT = 2048

SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def prune_layer_norm_configs(configs, named_args, **kwargs):
    width = triton.next_power_of_2(named_args["N"])
    kept = [
        config
        for config in configs
        if ELEMENTS_PER_WARP_FLOOR * config.num_warps
        <= config.kwargs["TILE_M"] * width
        <= ELEMENTS_PER_WARP_LIMIT * config.num_warps
    ]
    # Never hand the autotuner an empty set: a very wide row with one warp
    # falls outside the window above, and still has to run.
    return kept or [min(configs, key=lambda config: config.kwargs["TILE_M"])]


# The generic reduction and affine computation, with an NVIDIA row/warp search.
# The generic entries keep their own autotuning and caches.
layer_norm_rows_kernel = libentry()(
    triton.autotune(
        configs=runtime.get_tuned_config("layer_norm_rows"),
        key=["M", "N"],
        prune_configs_by={"early_config_prune": prune_layer_norm_configs},
    )(layer_norm_persistent_kernel_multiline.jit_function)
)

# The same two-pass kernel the generic host uses above the persistent limit,
# searched over a wider tile.
layer_norm_wide_kernel = libentry()(
    triton.autotune(
        configs=runtime.get_tuned_config("layer_norm_wide"),
        key=["M", "N"],
    )(layer_norm_loop_kernel.jit_function)
)


def _check_arguments(input, normalized_shape, weight, bias):
    if (
        not normalized_shape
        or input.shape[-len(normalized_shape) :] != normalized_shape
    ):
        raise ValueError("normalized_shape must match a nonempty suffix of input.shape")
    if input.dtype not in SUPPORTED_DTYPES:
        raise TypeError("layer_norm supports float16, bfloat16, and float32 inputs")
    for parameter in (weight, bias):
        if parameter is None:
            continue
        if parameter.shape != normalized_shape or parameter.device != input.device:
            raise ValueError("affine parameters must match normalized_shape and device")
        if parameter.dtype not in (input.dtype, torch.float32):
            raise TypeError("affine parameters must have input dtype or float32 dtype")


def layer_norm(
    input: torch.Tensor,
    normalized_shape: Sequence[int],
    weight: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    logger.debug("GEMS_NVIDIA LAYER_NORM")
    normalized_shape = tuple(normalized_shape)
    _check_arguments(input, normalized_shape, weight, bias)

    width = math.prod(normalized_shape)
    if width == 0:
        raise NotImplementedError("zero-sized normalized dimensions are not supported")
    rows = input.numel() // width
    if rows == 0:
        empty_stats = torch.empty(0, dtype=torch.float32, device=input.device)
        return torch.empty_like(input), empty_stats, empty_stats.clone()

    input = input.contiguous()
    weight = None if weight is None else weight.contiguous()
    bias = None if bias is None else bias.contiguous()
    output = torch.empty_like(input)
    # ATen exposes FP32 statistics for half-precision inputs; the backward pass
    # expects them in that precision.
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
            layer_norm_wide_kernel[(rows, 1, 1)](
                input, output, weight, bias, mean, rstd, rows, width, eps
            )
    return output, mean, rstd


def native_layer_norm(
    input: torch.Tensor,
    normalized_shape: Sequence[int],
    weight: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    logger.debug("GEMS_NVIDIA NATIVE_LAYER_NORM")
    output, mean, rstd = layer_norm(input, normalized_shape, weight, bias, eps)
    stats_shape = input.shape[: -len(normalized_shape)] + (1,) * len(normalized_shape)
    return output, mean.view(stats_shape), rstd.view(stats_shape)
