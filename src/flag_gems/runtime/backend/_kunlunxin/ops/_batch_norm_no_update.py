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

# Constants for the per-slice kernel above (grid = N*C). It is no longer used by
# `_batch_norm_no_update` (replaced by the flat kernel below, 2026-09-18) but
# remains the workhorse of `_native_batch_norm_legit_no_training`, a different
# op. TILE_S < 64 is deliberately avoided: the XPU compiler miscompiles
# scalar+small-tile broadcast math for TILE<=32 (wrong results, verified);
# TILE_S=4096 with num_warps=4 is the latency optimum for that kernel
# (2026-09-05, event timing).
BNNU_MAX_PROGRAMS = 4096
# Fixed tile for the per-slice kernel (the kernel has no autotune; the value is
# shared with callers that launch it directly).
BNNU_TILE_S = 4096


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


# Flat element-parallel kernel (2026-09-18). Replaces the channel-major kernel
# (each program owning one channel x a chunk of batch slices) whose two costs
# dominated the runtime: the per-program scalar stats loads (~13us at grid=32)
# and the 1.5KB per-slice DMA trips (~20us for bytes a flat copy moves in ~5us).
# Measured on (16,16,8,48) fp32: 36us vs a 6.7us torch reference (speedup 0.19).
#
# Here each program covers one CONTIGUOUS run of BLOCK elements -- a long DMA
# trip like a plain copy -- and re-derives each lane's channel from its element
# index, sid = (gidx // SPATIAL) % C, so the four stats loads have addresses
# that repeat every SPATIAL lanes. Same-shape measurements: 9.6us, and the
# balanced speedup over the official 12 cells goes 0.49 -> 1.17-1.36.
#
# Three constraints are load-bearing, all measured on device (evidence:
# artifacts/op-perf-batch-2026-09/evidence/batch-norm-audit-20260918/):
#   * SPATIAL must stay a constexpr -- `gidx // SPATIAL` then folds to a
#     magic-number multiply at compile time, which is what makes this flat form
#     viable at all: feeding SPATIAL as a runtime value measures 402-1095us on
#     the same shapes. That is the failure mode the 2026-08-17 note above
#     records for an earlier flat kernel (6-12ms/call), and why it was reverted;
#     keeping the divisor compile-time is the difference. Within the constexpr
#     form, re-expressing the division as a shift (`gidx >> log2(S)`) or
#     routing the channel index through an int8 lookup table also collapsed
#     into a ~1400us scalarized path.
#   * BLOCK must stay <= 4096. Larger blocks hit a codegen cliff on some
#     (dtype, SPATIAL) pairs -- 384/fp16 at BLOCK=16384 and 704/fp32 at
#     BLOCK=8192 both jump from ~10us to 34-43us, while the same-geometry pure
#     copy stays flat at 5-7us. 4096 was the largest cap clear of the cliff on
#     every configuration measured.
#   * Vectorization stays ON -- do NOT pass isCloseVectorization=True (skipping
#     the vectorize pass costs ~8% end to end: 1.17 vs 1.36 balanced).
_FLAT_BLOCK_MAX = 4096


@libentry()
@triton.jit(do_not_specialize=["eps"])
def _batch_norm_no_update_kernel_flat(
    input_pointer,  # [N*C*S] contiguous, flattened
    weight_pointer,  # [C] or unused
    bias_pointer,  # [C] or unused
    running_mean_pointer,  # [C]
    running_var_pointer,  # [C]
    output_pointer,
    feat_dim,
    total,
    eps,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK: tl.constexpr,
    SPATIAL: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    gidx = pid * BLOCK + tl.arange(0, BLOCK)
    sid = (gidx // SPATIAL) % feat_dim
    mean = tl.load(running_mean_pointer + sid).to(tl.float32)
    inv_std = 1.0 / tl.sqrt(tl.load(running_var_pointer + sid).to(tl.float32) + eps)
    if HAS_WEIGHT:
        weight = tl.load(weight_pointer + sid).to(tl.float32)
    else:
        weight = 1.0
    if HAS_BIAS:
        bias = tl.load(bias_pointer + sid).to(tl.float32)
    else:
        bias = 0.0
    if NEED_MASK:
        mask = gidx < total
        x = tl.load(input_pointer + gidx, mask=mask, other=0.0).to(tl.float32)
        tl.store(
            output_pointer + gidx,
            (weight * (x - mean) * inv_std + bias).to(output_pointer.dtype.element_ty),
            mask=mask,
        )
    else:
        x = tl.load(input_pointer + gidx).to(tl.float32)
        tl.store(
            output_pointer + gidx,
            (weight * (x - mean) * inv_std + bias).to(output_pointer.dtype.element_ty),
        )


def _flat_block(total: int) -> int:
    """BLOCK for the flat kernel: next_pow2(total) capped at _FLAT_BLOCK_MAX."""
    return min(_FLAT_BLOCK_MAX, max(64, triton.next_power_of_2(total)))


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
        block = _flat_block(n_elements)
        weight_pointer = input_flat if weight is None else weight
        bias_pointer = input_flat if bias is None else bias
        with torch_device_fn.device(input.device):
            _batch_norm_no_update_kernel_flat[(triton.cdiv(n_elements, block),)](
                input_flat,
                weight_pointer,
                bias_pointer,
                running_mean,
                running_var,
                output_flat,
                channels,
                n_elements,
                eps,
                HAS_WEIGHT=weight is not None,
                HAS_BIAS=bias is not None,
                BLOCK=block,
                SPATIAL=inner,
                NEED_MASK=(n_elements % block) != 0,
                num_warps=4,
            )

    save_mean = torch.empty((0,), dtype=input.dtype, device=input.device)
    save_var = torch.empty((0,), dtype=input.dtype, device=input.device)
    reserved = torch.empty((0,), dtype=torch.uint8, device=input.device)
    return output.view_as(input), save_mean, save_var, reserved
