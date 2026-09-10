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

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, tl_extra_shim

logger = logging.getLogger(__name__)
rsqrt = tl_extra_shim.rsqrt


# NOTE (kunlunxin / XPU perf rewrite, 2026-08-17):
# The previous flat kernel indexed per-lane channel stats as
# `channels = (offsets // INNER) % C` and loaded running_mean/running_var/weight/bias
# through that per-lane gather, which the XPU compiler lowered to slow discrete
# accesses (measured ~6-12 ms/call for the small benchmark shapes, ~0.009x speedup).
#
# In the natural [N, C, S] contiguous layout each (n, c) slice is a run of S
# CONTIGUOUS elements sharing ONE channel. So we map one program to each (n, c)
# slice (grid = N*C, same pattern as the batch_norm 3-stage normalize kernel):
# stats/affine are loaded ONCE per program as scalars, and the data tiles are
# contiguous block-DMA (masked only when S % TILE_S != 0). Measured: all benchmark
# cases drop from ~6-12 ms to ~0.06-0.15 ms. TILE_S < 64 is deliberately avoided:
# the XPU compiler miscompiles scalar+small-tile broadcast math for TILE<=32
# (wrong results, verified); TILE_S=4096 with num_warps=4 is the latency optimum.

# TILE_S and the channel-major grid are tuned to the OFFICIAL benchmark core
# shapes (S in 128..704, N in 4..16, C=16). Measured on device (2026-09-05,
# event timing): a fixed tile=1024 beats the "next_pow2(spatial)" rule for
# every official shape (e.g. (16,16,128): tile256=0.067ms -> tile1024=0.036ms;
# (4,16,64,4): grid32/tile256=0.044ms -> grid16/tile1024=0.019ms), because
# sub-1024 tiles trigger the XPU scalarized codegen path (same failure mode as
# the old sub-64 tiles, just at a higher threshold). Cap 16384 avoids VRF
# pressure (measured 32768 regresses). The empty-kernel event floor for a
# 16-32 program launch is ~0.007ms == the official fp32 torch reference, so
# fp32 speedup is structurally capped ~0.4-0.5; fp16/bf16 have headroom.
BNNU_MIN_TILE_S = 1024
BNNU_MAX_TILE_S = 16384
BNNU_MAX_PROGRAMS = 4096
# Backward-compat alias for other vendors/ops that still hardcode a fixed tile
# (e.g. _native_batch_norm_legit_no_training, a different op, unchanged here).
BNNU_TILE_S = 4096


def adaptive_tile_s(spatial: int) -> int:
    """Tile = next_pow2(spatial) clamped to [BNNU_MIN_TILE_S, BNNU_MAX_TILE_S].
    The 1024 floor avoids the XPU sub-1024 scalarized codegen path; 16384 cap
    avoids VRF pressure. Exact fit (no mask) when spatial is a power of two,
    otherwise a masked tail."""
    tile = 1 << (spatial - 1).bit_length() if spatial > 1 else 1
    if tile < BNNU_MIN_TILE_S:
        return BNNU_MIN_TILE_S
    return min(tile, BNNU_MAX_TILE_S)


@libentry()
@triton.jit(do_not_specialize=["eps"])
def _batch_norm_no_update_kernel(
    input_pointer,  # [N*C, S] contiguous, flattened
    weight_pointer,  # [C] or unused
    bias_pointer,  # [C] or unused
    running_mean_pointer,  # [C]
    running_var_pointer,  # [C]
    output_pointer,
    feat_dim,
    spatial_dim,
    eps,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    TILE_S: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    c = pid % feat_dim
    base = pid * spatial_dim

    mean = tl.load(running_mean_pointer + c).to(tl.float32)
    inv_std = rsqrt(tl.load(running_var_pointer + c).to(tl.float32) + eps)
    if HAS_WEIGHT:
        weight = tl.load(weight_pointer + c).to(tl.float32)
    else:
        weight = 1.0
    if HAS_BIAS:
        bias = tl.load(bias_pointer + c).to(tl.float32)
    else:
        bias = 0.0

    for off in range(0, spatial_dim, TILE_S):
        idx = off + tl.arange(0, TILE_S)
        if NEED_MASK:
            mask = idx < spatial_dim
            x = tl.load(input_pointer + base + idx, mask=mask).to(tl.float32)
            y = weight * (x - mean) * inv_std + bias
            tl.store(
                output_pointer + base + idx,
                y.to(output_pointer.dtype.element_ty),
                mask=mask,
            )
        else:
            x = tl.load(input_pointer + base + idx).to(tl.float32)
            y = weight * (x - mean) * inv_std + bias
            tl.store(output_pointer + base + idx, y.to(output_pointer.dtype.element_ty))


# Channel-major kernel (2026-09-05): measured decomposition showed the per-slice
# kernel's 4 scalar stats loads (grid = N*C programs, N*C*4 global scalar loads)
# dominated the runtime (2 loads alone cost ~0.06ms at grid=256, ~40% of total).
# Instead each program owns ONE channel and a chunk of batch slices, so the 4
# scalar stats loads are amortized over CHUNK slices (total C*b_split*4 loads).
# Data stays contiguous per slice (block DMA, masked tail only when S%TILE_S!=0).
# grid = C * b_split (target ~32 programs, measured latency sweet spot); the
# batch loop is a RUNTIME loop (chunk passed as a value) so huge batches do not
# unroll / blow up compile time. Both TILE_S and the batch-split are wrapped up
# so `batch_norm`'s inference path and `_batch_norm_no_update` share them.
@libentry()
@triton.jit(do_not_specialize=["eps"])
def _batch_norm_no_update_kernel_c(
    input_pointer,  # [N, C, S] contiguous, flattened
    weight_pointer,  # [C] or unused
    bias_pointer,  # [C] or unused
    running_mean_pointer,  # [C]
    running_var_pointer,  # [C]
    output_pointer,
    feat_dim,
    spatial_dim,
    batch_dim,
    eps,
    chunk,  # runtime: batch slices handled by this program
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    TILE_S: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    c = pid % feat_dim
    bi = pid // feat_dim
    mean = tl.load(running_mean_pointer + c).to(tl.float32)
    inv_std = rsqrt(tl.load(running_var_pointer + c).to(tl.float32) + eps)
    if HAS_WEIGHT:
        weight = tl.load(weight_pointer + c).to(tl.float32)
    else:
        weight = 1.0
    if HAS_BIAS:
        bias = tl.load(bias_pointer + c).to(tl.float32)
    else:
        bias = 0.0
    n0 = bi * chunk
    for s in range(chunk):  # runtime loop (not unrolled)
        n = n0 + s
        if n < batch_dim:
            base = (n * feat_dim + c) * spatial_dim
            for off in range(0, spatial_dim, TILE_S):
                idx = off + tl.arange(0, TILE_S)
                if NEED_MASK:
                    mask = idx < spatial_dim
                    x = tl.load(input_pointer + base + idx, mask=mask).to(tl.float32)
                    y = weight * (x - mean) * inv_std + bias
                    tl.store(
                        output_pointer + base + idx,
                        y.to(output_pointer.dtype.element_ty),
                        mask=mask,
                    )
                else:
                    x = tl.load(input_pointer + base + idx).to(tl.float32)
                    y = weight * (x - mean) * inv_std + bias
                    tl.store(
                        output_pointer + base + idx, y.to(output_pointer.dtype.element_ty)
                    )


def _channel_grid(batch_dim: int, feat_dim: int, spatial: int):
    """Channel-major grid policy: each program owns ONE channel and ~8
    consecutive batch slices (b_split = ceil(N/8), grid = C*b_split). This
    keeps total scalar stats loads at C*b_split*4 while bounding per-program
    work. Measured: for N=4 (official shape (4,16,64,4)) b_split=1 (grid=16)
    is best; for N=16 b_split=2 (grid=32) is best. Never more than one program
    per slice. Returns (grid, chunk)."""
    b_split = max(1, min(batch_dim, (batch_dim + 7) // 8))
    chunk = (batch_dim + b_split - 1) // b_split
    return feat_dim * b_split, chunk


def _batch_norm_no_update(
    input,
    weight=None,
    bias=None,
    running_mean=None,
    running_var=None,
    momentum=0.1,
    eps=1e-5,
):
    logger.debug("GEMS_KUNLUNXIN _BATCH_NORM_NO_UPDATE")
    if input.ndim < 2:
        raise RuntimeError("batch_norm expects input with at least 2 dimensions")
    if running_mean is None or running_var is None:
        raise RuntimeError(
            "running_mean and running_var are required for no-update batch_norm"
        )

    channels = input.shape[1]
    if running_mean.numel() != channels or running_var.numel() != channels:
        raise RuntimeError("running statistics must contain one value per channel")

    input_contiguous = input.contiguous()
    output = torch.empty_like(input_contiguous)
    n_elements = input_contiguous.numel()
    batch_dim = input.shape[0]
    n_slices = batch_dim * channels
    inner = n_elements // n_slices if n_slices > 0 else 0

    if n_elements > 0:
        input_flat = input_contiguous.reshape(-1)
        output_flat = output.reshape(-1)
        tile_s = adaptive_tile_s(inner)
        need_mask = (inner % tile_s) != 0
        grid, chunk = _channel_grid(batch_dim, channels, inner)
        weight_pointer = input_flat if weight is None else weight
        bias_pointer = input_flat if bias is None else bias
        with torch_device_fn.device(input.device):
            _batch_norm_no_update_kernel_c[(grid,)](
                input_flat,
                weight_pointer,
                bias_pointer,
                running_mean,
                running_var,
                output_flat,
                channels,
                inner,
                batch_dim,
                eps,
                chunk,
                HAS_WEIGHT=weight is not None,
                HAS_BIAS=bias is not None,
                TILE_S=tile_s,
                NEED_MASK=need_mask,
                num_warps=4,
                isCloseVectorization=True,
                buffer_size_limit=2048,
            )

    save_mean = torch.empty((0,), dtype=input.dtype, device=input.device)
    save_var = torch.empty((0,), dtype=input.dtype, device=input.device)
    reserved = torch.empty((0,), dtype=torch.uint8, device=input.device)
    return output.view_as(input), save_mean, save_var, reserved
