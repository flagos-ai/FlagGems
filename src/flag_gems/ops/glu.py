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

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import pointwise_dynamic, tl_extra_shim
from flag_gems.utils.triton_version_utils import HAS_TLE

logger = logging.getLogger(__name__)
exp = tl_extra_shim.exp

if HAS_TLE:
    import triton.experimental.tle.language as tle
else:
    tle = None

def _next_pow2(x: int) -> int:
    return 1 if x <= 1 else 2 ** math.ceil(math.log2(x))

@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")])
@triton.jit
def glu_kernel(a, b):
    sigmoid_b = 1 / (1 + exp(-b.to(tl.float32)))
    result = a * sigmoid_b
    return result

# ============================================================================
# TLE kernel (static extract_tile)
# From: FlagTree python/tutorials/tle/05-glu.py
# ============================================================================

if HAS_TLE:

    # (ROWS_PER_PROGRAM, num_warps, loop_stages). Include two-warp multi-stage candidates on
    # longer loops so async loads can overlap computation without requiring four warps.
    _GLU_TLE_AUTOTUNE_CONFIGS = [
        (1, 2, 1),
        (1, 4, 1),
        (1, 4, 2),
        (2, 2, 1),
        (2, 4, 2),
        (4, 2, 1),
        (4, 4, 1),
        (4, 4, 2),
        (4, 4, 4),
        (8, 1, 1),
        (8, 2, 1),
        (8, 2, 2),
        (8, 2, 3),
        (8, 2, 4),
        (8, 4, 2),
        (8, 4, 4),
        (16, 2, 1),
        (16, 2, 2),
        (16, 2, 3),
        (16, 2, 4),
        (16, 4, 2),
        (16, 4, 4),
    ]
    _GLU_TLE_LOOP_UNROLL_FACTORS = (1, 2)

    # Explicit boundary candidates found by profiling. Keep these separate
    # to avoid crossing unroll 4/8 with every base configuration.
    _GLU_TLE_EXTRA_AUTOTUNE_CONFIGS = [
        (4, 2, 1, 4),
        (4, 2, 2, 2),
        (4, 2, 3, 2),
        (4, 2, 4, 2),
        (8, 2, 3, 4),
        (8, 2, 4, 4),
        (16, 2, 3, 8),
        (16, 2, 4, 8),
    ]

    def _prune_glu_tle_configs(configs, named_args, **kwargs):
        """Keep the multi-row tuning space bounded for wider GLU rows."""
        D = named_args["D"]
        if D <= 128:
            max_rows_per_program = 16
        elif D <= 256:
            max_rows_per_program = 8
        elif D <= 512:
            max_rows_per_program = 4
        elif D <= 2048:
            max_rows_per_program = 2
        else:
            max_rows_per_program = 1

        return [
            config
            for config in configs
            if config.kwargs["ROWS_PER_PROGRAM"] <= max_rows_per_program
        ]

    @triton.autotune(
        configs=[
            triton.Config(
                {
                    "ROWS_PER_PROGRAM": rows_per_program,
                    "LOOP_STAGES": loop_stages,
                    "LOOP_UNROLL": loop_unroll,
                },
                num_warps=num_warps,
                num_stages=1,
            )
            for rows_per_program, num_warps, loop_stages in (
                _GLU_TLE_AUTOTUNE_CONFIGS
            )
            for loop_unroll in _GLU_TLE_LOOP_UNROLL_FACTORS
            if loop_unroll <= rows_per_program
        ] + [
            triton.Config(
                {
                    "ROWS_PER_PROGRAM": rows_per_program,
                    "LOOP_STAGES": loop_stages,
                    "LOOP_UNROLL": loop_unroll,
                },
                num_warps=num_warps,
                num_stages=1,
            )
            for rows_per_program, num_warps, loop_stages, loop_unroll in (
                _GLU_TLE_EXTRA_AUTOTUNE_CONFIGS
            )
        ],
        key=["N", "D"],
        prune_configs_by={"early_config_prune": _prune_glu_tle_configs},
        cache_results=True,
    )
    @triton.jit
    def glu_kernel_tle(
        x_ptr,
        out_ptr,
        N,
        D,
        stride_xn,
        stride_outn,
        D_P2: tl.constexpr,
        D2_P2: tl.constexpr,
        ROWS_PER_PROGRAM: tl.constexpr,
        LOOP_STAGES: tl.constexpr,
        LOOP_UNROLL: tl.constexpr,
    ):
        row_start = tl.program_id(0) * ROWS_PER_PROGRAM
        offs = tl.arange(0, D2_P2)
        offs_d = tl.arange(0, D_P2)

        # Process rows sequentially so halo/a/b/result registers can be
        # reused instead of materializing a multi-row tile.
        for row_offset in tle.range(
            0,
            ROWS_PER_PROGRAM,
            num_stages=LOOP_STAGES,
            loop_unroll_factor=LOOP_UNROLL,
            reorder=True,
        ):
            row = row_start + row_offset
            row_mask = row < N
            load_mask = row_mask & (offs < (D * 2))
            halo = tl.load(
                x_ptr + row * stride_xn + offs,
                mask=load_mask,
                other=0.0,
            )

            a_tile = tle.extract_tile(
                halo, index=[0], tile_shape=[D_P2]
            )
            b_tile = tle.extract_tile(
                halo, index=[1], tile_shape=[D_P2]
            )

            a_f32 = a_tile.to(tl.float32)
            b_f32 = b_tile.to(tl.float32)
            sigmoid_b = 1.0 / (1.0 + tl.exp(-b_f32))
            result = a_f32 * sigmoid_b

            tl.store(
                out_ptr + row * stride_outn + offs_d,
                result.to(out_ptr.dtype.element_ty),
                mask=row_mask & (offs_d < D),
            )

@pointwise_dynamic(
    promotion_methods=[
        (0, 1, 2, "DEFAULT"),
        (0, 1, 2, "DEFAULT"),
    ]
)
@triton.jit
def glu_backward_kernel(grad_output, a, b):
    sigmoid_b = 1 / (1 + exp(-b.to(tl.float32)))
    da = grad_output * sigmoid_b
    db = grad_output.to(tl.float32) * a * sigmoid_b * (1.0 - sigmoid_b)
    return da, db

def glu(self, dim=-1):
    assert self.shape[dim] % 2 == 0, "Split dimension must be even"
    logger.debug("GEMS GLU FORWARD")
    D2 = self.shape[-1]
    D = D2 // 2
    if HAS_TLE and dim == -1 and D < 8192:
        logger.debug("GEMS GLU FORWARD (TLE extract_tile path)")
        N = 1
        for d in self.shape[:-1]:
            N *= d

        x = self.reshape(N, D2)
        out = torch.empty((N, D), device=self.device, dtype=self.dtype)
        d_p2 = _next_pow2(D)
        d2_p2 = _next_pow2(D2)

        if N == 0:
            return out.reshape(self.shape[:-1] + (D,))

        with torch_device_fn.device(self.device):
            grid = lambda meta: (
                triton.cdiv(N, meta["ROWS_PER_PROGRAM"]),
            )
            glu_kernel_tle[grid](
                x,
                out,
                N,
                D,
                x.stride(0),
                out.stride(0),
                D_P2=d_p2,
                D2_P2=d2_p2,
            )
        return out.reshape(self.shape[:-1] + (D,))

    # Split into a and b
    a, b = torch.chunk(self, 2, dim=dim)
    out = glu_kernel(a, b)
    return out

def glu_backward(grad_output, self, dim=-1):
    assert self.shape[dim] % 2 == 0, "Split dimension must be even"
    logger.debug("GEMS GLU BACKWARD")
    a, b = torch.chunk(self, 2, dim=dim)
    grad_input = torch.empty_like(self, memory_format=torch.contiguous_format)
    grad_a, grad_b = torch.chunk(grad_input, 2, dim=dim)
    glu_backward_kernel(grad_output, a, b, out0=grad_a, out1=grad_b)
    return grad_input
