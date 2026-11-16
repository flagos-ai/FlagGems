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
import triton.language as tl
from triton.experimental.tle.language import dsa

from flag_gems.ops.layernorm import layer_norm as common_layer_norm
from flag_gems.ops.layernorm import (
    layer_norm_loop_kernel,
    layer_norm_persistent_kernel,
    layer_norm_persistent_kernel_multiline,
)
from flag_gems.runtime import torch_device_fn
from flag_gems.runtime.backend._ascend.utils import CORE_NUM
from flag_gems.utils import triton_lang_extension as ext
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


layer_norm_loop_kernel = BlockifiedLayerNormEntry(layer_norm_loop_kernel.jit_function)
layer_norm_persistent_kernel = BlockifiedLayerNormEntry(layer_norm_persistent_kernel.fn)
layer_norm_persistent_kernel_multiline = BlockifiedLayerNormEntry(
    layer_norm_persistent_kernel_multiline.fn
)


@BlockifiedLayerNormEntry
@triton.jit(do_not_specialize=["eps"])
def layer_norm_compact_kernel(
    input_ptr,
    output_ptr,
    weight_ptr,
    bias_ptr,
    mean_ptr,
    rstd_ptr,
    M,
    N: tl.constexpr,
    eps,
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
    NUM_PROGRAMS: tl.constexpr,
):
    columns = tl.arange(0, TILE_N)
    if weight_ptr is None:
        weight = 1.0
    else:
        weight = tl.load(weight_ptr + columns, columns < N, 0)
        weight = dsa.extract_slice(weight, [0], [N], [1]).to(tl.float32)[None, :]
    if bias_ptr is None:
        bias = 0.0
    else:
        bias = tl.load(bias_ptr + columns, columns < N, 0)
        bias = dsa.extract_slice(bias, [0], [N], [1]).to(tl.float32)[None, :]

    for block in range(ext.program_id(0), tl.cdiv(M, TILE_M), NUM_PROGRAMS):
        rows = block * TILE_M + tl.arange(0, TILE_M)
        offsets = rows[:, None] * N + columns[None, :]
        mask = (rows[:, None] < M) & (columns[None, :] < N)
        values = tl.load(input_ptr + offsets, mask, 0)
        # Compact arithmetic keeps padded columns out of UB intermediates.
        values = dsa.extract_slice(values, [0, 0], [TILE_M, N], [1, 1]).to(tl.float32)
        mean = tl.sum(values, 1) / N
        centered = values - mean[:, None]
        rstd = tl.rsqrt(tl.sum(centered * centered, 1) / N + eps)
        output = (centered * rstd[:, None] * weight + bias).to(
            output_ptr.dtype.element_ty
        )
        # GM stores retain a power-of-two shape for the Ascend lowering.
        output = dsa.insert_slice(
            tl.full((TILE_M, TILE_N), 0, output_ptr.dtype.element_ty),
            output,
            [0, 0],
            [TILE_M, N],
            [1, 1],
        )
        tl.store(output_ptr + offsets, output, mask)
        tl.store(mean_ptr + rows, mean, rows < M)
        tl.store(rstd_ptr + rows, rstd, rows < M)


def _select_tile_m(M, tile_n):
    # Larger narrow-row groups amortize the per-program copy/reduction cost.
    if tile_n <= 128:
        if M >= 1024:
            return min(64, triton.next_power_of_2(triton.cdiv(M, CORE_NUM)))
        return triton.cdiv(1024, tile_n)
    # Pack medium-width rows to amortize per-program device work. Wider rows
    # stay single-row to avoid the measured 4096-column packing regression.
    if M < 256 or tile_n > 2048:
        return 1
    return max(1, min(32, 8192 // tile_n))


def layer_norm(input, normalized_shape, weight=None, bias=None, eps=1e-5):
    logger.debug("GEMS_ASCEND LAYERNORM_FORWARD")

    N = math.prod(normalized_shape)
    M = input.numel() // N

    is_fixed_stream = N == 16384 and input.dtype in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
    )
    # Keep the common streaming dispatch outside the measured fixed-width path.
    if N > 4096 and not is_fixed_stream:
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
        if is_fixed_stream:
            # Device timing favors 2048-element Welford chunks at this width.
            layer_norm_loop_kernel[(M,)](
                input,
                y,
                weight,
                bias,
                mean,
                rstd,
                M,
                N,
                eps,
                2048,
                num_warps=4,
                enable_auto_blockify=True,
            )
            return y, mean, rstd
        tile_n = triton.next_power_of_2(N)
        # The traced width wastes nearly half of a padded 2048-column tile.
        if (
            N == 1152
            and M >= 128
            and input.dtype in (torch.float16, torch.bfloat16, torch.float32)
        ):
            tile_m = 4 if input.dtype == torch.float32 else 8
            num_programs = min(CORE_NUM, triton.cdiv(M, tile_m))
            layer_norm_compact_kernel[(num_programs,)](
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
                num_programs,
                enable_auto_blockify=True,
            )
            return y, mean, rstd
        tile_m = _select_tile_m(M, tile_n)
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
