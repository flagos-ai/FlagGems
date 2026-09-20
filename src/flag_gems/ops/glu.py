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
from functools import lru_cache

import torch
import triton
import triton.language as tl

from flag_gems.runtime import device as runtime_device
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import pointwise_dynamic, tl_extra_shim
from flag_gems.utils.triton_version_utils import HAS_TLE

logger = logging.getLogger(__name__)
exp = tl_extra_shim.exp

IS_ASCEND = runtime_device.vendor_name == "ascend"

# Ascend's vendor-wide TLE flag is disabled by default. Check the specific
# DSA primitive locally without enabling unrelated TLE kernels.
if HAS_TLE or IS_ASCEND:
    try:
        import triton.experimental.tle.language as tle
    except ImportError:
        tle = None
else:
    tle = None

HAS_TLE_EXTRACT_TILE = HAS_TLE and not IS_ASCEND and hasattr(tle, "extract_tile")
HAS_TLE_EXTRACT_SLICE = IS_ASCEND and hasattr(
    getattr(tle, "dsa", None), "extract_slice"
)


def _next_pow2(x: int) -> int:
    return 1 if x <= 1 else 2 ** math.ceil(math.log2(x))


@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")])
@triton.jit
def glu_kernel(a, b):
    sigmoid_b = 1 / (1 + exp(-b.to(tl.float32)))
    result = a * sigmoid_b
    return result


if HAS_TLE_EXTRACT_TILE:

    # Each configuration is (ROWS_PER_PROGRAM, num_warps, LOOP_STAGES).
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
        """Bound row tiles by the split width."""
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
            for rows_per_program, num_warps, loop_stages in (_GLU_TLE_AUTOTUNE_CONFIGS)
            for loop_unroll in _GLU_TLE_LOOP_UNROLL_FACTORS
            if loop_unroll <= rows_per_program
        ]
        + [
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
        D: tl.constexpr,
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
        # Pad A and B separately so extract_tile's second tile starts at B.
        # For power-of-two D this reduces to the original contiguous offsets.
        cols = offs % D_P2
        input_offs = tl.where(offs < D_P2, cols, D + cols)

        # Process rows sequentially so halo/a/b/result registers can be
        # reused instead of materializing a multi-row tile.
        # `tle.range` is the TLE `gpu.range` extension and is not supported by
        # the MThreads backend. The loop itself only needs standard Triton
        # range semantics here; `reorder=True` is an optional optimization.
        for row_offset in tl.range(
            0,
            ROWS_PER_PROGRAM,
            num_stages=LOOP_STAGES,
            loop_unroll_factor=LOOP_UNROLL,
        ):
            row = row_start + row_offset
            row_mask = row < N
            load_mask = row_mask & (cols < D)
            halo = tl.load(
                x_ptr + row * stride_xn + input_offs,
                mask=load_mask,
                other=0.0,
            )

            a_tile = tle.extract_tile(halo, index=[0], tile_shape=[D_P2])
            b_tile = tle.extract_tile(halo, index=[1], tile_shape=[D_P2])

            a_f32 = a_tile.to(tl.float32)
            b_f32 = b_tile.to(tl.float32)
            sigmoid_b = 1.0 / (1.0 + tl.exp(-b_f32))
            result = a_f32 * sigmoid_b

            tl.store(
                out_ptr + row * stride_outn + offs_d,
                result.to(out_ptr.dtype.element_ty),
                mask=row_mask & (offs_d < D),
            )


if HAS_TLE_EXTRACT_SLICE:

    @lru_cache(maxsize=None)
    def _glu_ascend_num_cores(device_index):
        try:
            props = triton.runtime.driver.active.utils.get_device_properties(
                device_index
            )
            num_cores = int(props["num_vectorcore"])
            if num_cores <= 0:
                raise ValueError("num_vectorcore must be positive")
            return num_cores
        except (AttributeError, RuntimeError, KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                f"GLU: cannot query vector core count for Ascend device {device_index}"
            ) from exc

    def _prune_glu_ascend_configs(configs, named_args, **kwargs):
        # Beyond the smallest power of two covering N, larger row tiles only
        # add masked rows. Otherwise let device measurements choose the tile.
        max_rows = triton.next_power_of_2(named_args["N"])
        return [config for config in configs if config.kwargs["BLOCK_M"] <= max_rows]

    def _glu_ascend_do_bench(fn, quantiles=None):
        from triton.backends.ascend.testing import do_bench_npu

        # Compile before profiling. Use the same device timer as the fixed
        # configuration sweep, rather than the default Triton event timer.
        fn()
        torch_device_fn.synchronize()
        latency = float(do_bench_npu(fn, warmup=10, active=50, clear_l2_cache=False))
        if not math.isfinite(latency) or latency <= 0:
            raise RuntimeError(f"GLU: invalid Ascend tuning measurement: {latency}")
        if quantiles is None:
            return latency
        # The timer returns one aggregate, not per-launch quantiles. Repeat
        # that score to satisfy the autotuner's requested result shape.
        return [latency for _ in quantiles]

    @triton.autotune(
        configs=[
            triton.Config(
                {"BLOCK_M": rows, "LOOP_STAGES": stages}, num_warps=4, num_stages=1
            )
            for rows in (1, 2, 4, 8, 16, 32, 64, 128, 256)
            for stages in (1, 2)
        ],
        key=["N", "D", "NUM_CORES"],
        prune_configs_by={"early_config_prune": _prune_glu_ascend_configs},
        do_bench=_glu_ascend_do_bench,
        cache_results=False,
    )
    @triton.jit
    def glu_kernel_ascend(
        x_ptr,
        out_ptr,
        N: tl.constexpr,
        D: tl.constexpr,
        BLOCK_N: tl.constexpr,
        NUM_CORES: tl.constexpr,
        BLOCK_M: tl.constexpr,
        LOOP_STAGES: tl.constexpr = 1,
    ):
        col_tiles = tl.cdiv(D, BLOCK_N)
        total_tiles = tl.cdiv(N, BLOCK_M) * col_tiles
        tiles_per_program = tl.cdiv(total_tiles, tl.num_programs(0))
        tile_begin = tl.program_id(0) * tiles_per_program
        tile_end = tl.minimum(tile_begin + tiles_per_program, total_tiles)
        row_offsets = tl.arange(0, BLOCK_M)
        packed_offsets = tl.arange(0, 2 * BLOCK_N)
        out_offsets = tl.arange(0, BLOCK_N)

        # Batch rows into vector tiles; each program owns a contiguous tile
        # range. Keep stage 1 as a baseline against the stage 2 candidate.
        for tile_id in tl.range(tile_begin, tile_end, num_stages=LOOP_STAGES):
            rows = (tile_id // col_tiles) * BLOCK_M + row_offsets
            col_start = (tile_id % col_tiles) * BLOCK_N
            if D == BLOCK_N:
                input_ptrs = x_ptr + rows[:, None] * (2 * D) + packed_offsets[None, :]
                if N % BLOCK_M == 0:
                    packed = tl.load(input_ptrs)
                else:
                    packed = tl.load(input_ptrs, mask=rows[:, None] < N, other=0.0)
            else:
                cols = col_start + packed_offsets % BLOCK_N
                input_cols = cols + tl.where(packed_offsets < BLOCK_N, 0, D)
                packed = tl.load(
                    x_ptr + rows[:, None] * (2 * D) + input_cols[None, :],
                    mask=(rows[:, None] < N) & (cols[None, :] < D),
                    other=0.0,
                )
            a = tle.dsa.extract_slice(
                packed, offsets=[0, 0], sizes=[BLOCK_M, BLOCK_N], strides=[1, 1]
            )
            b = tle.dsa.extract_slice(
                packed,
                offsets=[0, BLOCK_N],
                sizes=[BLOCK_M, BLOCK_N],
                strides=[1, 1],
            )
            a_f32 = a.to(tl.float32)
            b_f32 = b.to(tl.float32)
            result = a_f32 * (1.0 / (1.0 + tl.exp(-b_f32)))
            out_cols = col_start + out_offsets
            output_ptrs = out_ptr + rows[:, None] * D + out_cols[None, :]
            if N % BLOCK_M == 0 and D % BLOCK_N == 0:
                tl.store(output_ptrs, result.to(out_ptr.dtype.element_ty))
            else:
                tl.store(
                    output_ptrs,
                    result.to(out_ptr.dtype.element_ty),
                    mask=(rows[:, None] < N) & (out_cols[None, :] < D),
                )

    # Do not reuse selections measured with the previous timer, even when
    # TRITON_CACHE_AUTOTUNING enables disk caching globally. Compiled kernels
    # and this process's in-memory tuning results remain cached.
    glu_kernel_ascend.cache_results = False

    def _glu_ascend(self):
        D = self.shape[-1] // 2
        shape = self.shape[:-1] + (D,)
        out = torch.empty(shape, device=self.device, dtype=self.dtype)
        if out.numel() == 0:
            return out
        N = out.numel() // D
        with torch_device_fn.device(self.device):
            num_cores = _glu_ascend_num_cores(self.device.index)
            block_n = min(triton.next_power_of_2(D), 256)

            def grid(meta):
                block_m = meta["BLOCK_M"]
                total_tiles = triton.cdiv(N, block_m) * triton.cdiv(D, block_n)
                return (min(num_cores, total_tiles),)

            glu_kernel_ascend[grid](
                self, out, N, D, BLOCK_N=block_n, NUM_CORES=num_cores
            )
        return out


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
    if HAS_TLE_EXTRACT_SLICE:
        if (
            dim in (-1, self.ndim - 1)
            and self.is_contiguous()
            and self.dtype in (torch.float16, torch.bfloat16, torch.float32)
        ):
            logger.debug("GEMS GLU FORWARD (Ascend batched DSA path)")
            return _glu_ascend(self)
        a, b = torch.chunk(self, 2, dim=dim)
        return glu_kernel(a, b)
    D2 = self.shape[-1]
    D = D2 // 2
    if HAS_TLE_EXTRACT_TILE and dim == -1 and D < 8192:
        logger.debug("GEMS GLU FORWARD (TLE extract_tile path)")
        N = 1
        for d in self.shape[:-1]:
            N *= d

        x = self.reshape(N, D2)
        out = torch.empty((N, D), device=self.device, dtype=self.dtype)
        d_p2 = _next_pow2(D)
        d2_p2 = _next_pow2(D2)

        if N == 0 or D == 0:
            return out.reshape(self.shape[:-1] + (D,))

        with torch_device_fn.device(self.device):
            grid = lambda meta: (triton.cdiv(N, meta["ROWS_PER_PROGRAM"]),)
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
