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

# GCU400-specific implementation of linear_backward (cf. the upstream version in src/flag_gems/ops/linear_backward.py).
#
# Three structural overheads of the upstream path (measured on this board in benchmark kernel mode; pre-optimization
# baseline overall average
# 0.546x: fp32 0.871x / fp16 0.222x):
#   1. fp16 inputs are promoted to fp32 throughout: each call chains 5 _to_copy cast kernels,
#      and GEMM accesses memory at fp32 width (doubling bandwidth). Small-shape latency is pinned to the fixed
#      overhead of ~11 dispatches at ~0.17-0.24ms (nearly invariant with shape). Running fp16 directly through this
# backend's
#      mm / this file's kernel: tl.dot always accumulates in fp32 (fp16 products are exact in fp32),
#      giving numerically comparable results to the promoted path while eliminating all cast kernels.
#   2. grad_weight's `grad_output_flat.t().contiguous()` makes a full-tensor transpose copy.
#      Pass the transposed view directly (mm's Triton arm consumes the transposed layout via constexpr stride, and the
# saturated segment
#      delegates to native aten::mm, which natively supports arbitrary strides, matching the native baseline).
#   3. Three separate calls (mm/mm/sum_dim) incur the torch_gcu allocator's serialization tax: for tiny shapes,
#      probes measured mm x1=4.9us / x2=7.2us / x3=22.8us (superlinear); keeping buffers alive without freeing gives
#      x3=4.9us — freeing memory blocks still being written by in-flight kernels stalls the pipeline, and the 3rd
# allocation
#      event incurs the tax (the native triple pays ~23us too). Countermeasure: fuse the wgrad GEMM and grad_bias
#      into a single kernel, with the gw/gb outputs sharing one storage block (only 2
#      allocation/free events per call, within the tax-free zone), while also removing one sum_dim call.
#
# The fused arm is enabled only in segments where mm does not delegate to native (thresholds mirror mm.py: 16-bit
# M*N*K < 2^29, 4-byte < 2^28, K < 2^15; plus a static-unroll cap K/64 <= 64,
# as in mm's _mm_config_prune). The delegated segment keeps the mm + sum_dim combination (native delegation is
# bit-identical, and on large shapes the Python/allocation overhead is hidden by the long kernel, a measured break-even
# region).
# grad_bias is computed in the same kernel as wgrad: the main K-loop does only GEMM (a clean hot loop),
# while the program with pid_i==0 uses a separate small loop to rescan its own GO slice for the bias — ablations
# measured
# that an in-loop tl.sum reduction doubles kernel time ([128,1024] fp16 32.0us vs
# 18.6us for the separate loop); the separate loop costs only +1.7us (amortized 1/grid_i load imbalance).
#
# Hard kernel constraints (same as linear.py / mm.py; violating them yields wrong values or compilation failure):
#   - Unroll the K-loop with tl.static_range (on gcu400 a dynamic loop carrying an fp32 dot accumulator
#     miscompiles; mm already has a workaround);
#   - When K is not divisible by BLOCK_K (i.e., there is a tail block), BLOCK_K <= 64 (BK>=128 causes DSM OOR on the
# tail block);
#   - BLOCK_N=256 combined with any mask produces wrong values for whole rows (an existing case in linear.py), so
#     wide tiles (BLOCK_I=256 / BLOCK_J=256) are enabled only in dispatch tiers with full divisibility (guaranteed by
# FULL_EVEN);
#     tiers with masks cap BLOCK_I at 128;
#   - Do not put dot inside a runtime scalar branch (it degrades to scalar FMA); the bias-write branch contains only
# stores.
import logging

import torch
import triton
import triton.language as tl

import flag_gems
from flag_gems.utils import libentry

from .mm import mm
from .sum import sum_dim

logger = logging.getLogger(__name__)

# Fused-arm dispatch line (mirrors mm.py's native-delegation condition, with M=J=out_features,
# N=I=in_features, K=batch): a hit means mm takes the Triton arm, so fusion is worthwhile.
_WGRAD_NATIVE_WORK_THRESHOLD = 1 << 29  # 16-bit element width
_WGRAD_NATIVE_WORK_THRESHOLD_4BYTE = 1 << 28  # 4-byte element width
_WGRAD_LONG_K_THRESHOLD = 1 << 15
# Static-unroll cap for the K-loop (same cap as mm's _mm_config_prune, to prevent unroll explosion).
_WGRAD_K_BLOCK_CAP = 64
_WGRAD_BLOCK_K = 64


def _wgrad_bias_fused_eligible(dtype, out_features, in_features, batch):
    """True when the grad_weight GEMM sits inside mm's Triton segment and the shape unrolls safely."""
    if dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return False
    if batch >= _WGRAD_LONG_K_THRESHOLD:
        return False
    threshold = (
        _WGRAD_NATIVE_WORK_THRESHOLD_4BYTE
        if dtype.itemsize >= 4
        else _WGRAD_NATIVE_WORK_THRESHOLD
    )
    if out_features * in_features * batch >= threshold:
        return False
    return triton.cdiv(batch, _WGRAD_BLOCK_K) <= _WGRAD_K_BLOCK_CAP


@libentry()
@triton.jit
def linear_backward_wgrad_bias_kernel(
    GO,
    IN,
    GW,
    GB,
    J,
    I,
    K,
    BLOCK_J: tl.constexpr,
    BLOCK_I: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_J: tl.constexpr,
    FULL_EVEN: tl.constexpr,
    K_BLOCKS: tl.constexpr,
):
    # grad_weight (J, I) = GO^T (J, K) @ IN (K, I), K=batch;
    # grad_bias (J,) = sum_k GO[k, j]. GO (K, J) / IN (K, I) / GW (J, I)
    # are all row-major contiguous (guaranteed on the host side).
    pid = tl.program_id(0)
    grid_j = tl.cdiv(J, BLOCK_J)
    grid_i = tl.cdiv(I, BLOCK_I)
    width = GROUP_J * grid_i
    group_id = pid // width
    group_size = min(grid_j - group_id * GROUP_J, GROUP_J)
    pid_j = group_id * GROUP_J + (pid % group_size)
    pid_i = (pid % width) // group_size

    rj = pid_j * BLOCK_J + tl.arange(0, BLOCK_J)
    ri = pid_i * BLOCK_I + tl.arange(0, BLOCK_I)
    rk = tl.arange(0, BLOCK_K)

    # A tile: the transpose of GO, (BLOCK_J, BLOCK_K), with the j dimension contiguous (a column-major tile,
    # structurally identical to mm consuming a transposed view via constexpr stride). BLOCK_J=128 guarantees
    # the contiguous segment length of the transposed load (measured 3x slower for this kernel at 64).
    a_ptrs = GO + rk[None, :] * J + rj[:, None]
    b_ptrs = IN + rk[:, None] * I + ri[None, :]
    acc = tl.zeros((BLOCK_J, BLOCK_I), dtype=tl.float32)
    # Static unroll: on gcu400 a dynamic loop carrying an fp32 dot accumulator miscompiles (same in mm).
    # Keep only load/dot/store in the hot loop — place the bias reduction outside the loop (ablations measured:
    # in-loop tl.sum doubles kernel time, [128,1024] fp16 32.0us vs 18.6us with the separate loop).
    for k in tl.static_range(K_BLOCKS):
        if FULL_EVEN:
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
        else:
            k_remaining = K - k * BLOCK_K
            mask_a = (rj < J)[:, None] & (rk < k_remaining)[None, :]
            mask_b = (rk < k_remaining)[:, None] & (ri < I)[None, :]
            a = tl.load(a_ptrs, mask=mask_a, other=0)
            b = tl.load(b_ptrs, mask=mask_b, other=0)
        acc += tl.dot(a, b, out_dtype=tl.float32, allow_tf32=False)
        a_ptrs += BLOCK_K * J
        b_ptrs += BLOCK_K * I

    gw_ptrs = GW + rj[:, None] * I + ri[None, :]
    if FULL_EVEN:
        tl.store(gw_ptrs, acc.to(GW.dtype.element_ty))
    else:
        mask = (rj < J)[:, None] & (ri < I)[None, :]
        tl.store(gw_ptrs, acc.to(GW.dtype.element_ty), mask=mask)
    # Separate small loop for bias: each (j, i) program of the main K-loop covers the full K,
    # and only the program with pid_i==0 rescans its own GO slice to compute and write the bias;
    # all other programs exit immediately (a 1/grid_i load imbalance, measured to amortize to ~1.7us).
    # The branch contains only load/sum/store, no dot (a dot inside a runtime branch degrades to scalar FMA).
    if pid_i == 0:
        bias = tl.zeros((BLOCK_J,), dtype=tl.float32)
        a_ptrs = GO + rk[None, :] * J + rj[:, None]
        for k in tl.static_range(K_BLOCKS):
            if FULL_EVEN:
                a = tl.load(a_ptrs)
            else:
                k_remaining = K - k * BLOCK_K
                mask_a = (rj < J)[:, None] & (rk < k_remaining)[None, :]
                a = tl.load(a_ptrs, mask=mask_a, other=0)
            bias += tl.sum(a.to(tl.float32), axis=1)
            a_ptrs += BLOCK_K * J
        if FULL_EVEN:
            tl.store(GB + rj, bias.to(GB.dtype.element_ty))
        else:
            tl.store(GB + rj, bias.to(GB.dtype.element_ty), mask=rj < J)


def _launch_wgrad_bias(
    grad_output_flat, input_flat, dtype, out_features, in_features, batch
):
    """Fused arm: one kernel produces grad_weight and grad_bias sharing a single buffer."""
    if out_features == 0 or in_features == 0:
        # Empty output dimension: the result is an empty tensor; do not launch (ATen semantics).
        return (
            torch.empty(
                out_features, in_features, dtype=dtype, device=grad_output_flat.device
            ),
            torch.empty(out_features, dtype=dtype, device=grad_output_flat.device),
        )
    if batch == 0:
        # Empty reduction: grad_weight / grad_bias are both zero (ATen semantics).
        return (
            torch.zeros(
                out_features, in_features, dtype=dtype, device=grad_output_flat.device
            ),
            torch.zeros(out_features, dtype=dtype, device=grad_output_flat.device),
        )
    # gw and gb point to the same storage: only 2 new allocation events per call (the dgrad output +
    # this block), within the torch_gcu allocator's tax-free zone (probe: mm x2=7.2us vs x3=22.8us).
    buf = torch.empty(
        out_features * in_features + out_features,
        dtype=dtype,
        device=grad_output_flat.device,
    )
    grad_weight = buf[: out_features * in_features].view(out_features, in_features)
    grad_bias = buf[out_features * in_features :]

    # BLOCK_J=128: contiguous segment length of the transposed load (ablations measured ~3x slower at BJ=64).
    # BLOCK_K=128 is enabled only when K%128==0 (hard constraint: BK>=128 causes DSM OOR on tail blocks).
    # Wide tiles (BI=256 / BJ=256) are enabled only when all dimensions divide evenly (FULL_EVEN necessarily holds):
    # BN=256 plus any mask produces wrong values for whole rows (an existing case in linear.py); ablations measured
    # [128,1024] fp16 at the wide-N tier: 12.2us vs 15.8us for (128,128,128).
    # DSM budget (largest fp32 single buffer): (128,256,128)=192KB, (256,128,128)=192KB,
    # both <= the 448KB double-occupancy line.
    if batch % 128 == 0 and out_features % 128 == 0 and in_features % 256 == 0:
        BLOCK_J, BLOCK_I, BLOCK_K, num_stages = 128, 256, 128, 1
    elif batch % 128 == 0 and out_features % 256 == 0 and in_features % 128 == 0:
        BLOCK_J, BLOCK_I, BLOCK_K, num_stages = 256, 128, 128, 1
    elif batch % 128 == 0 and out_features % 128 == 0 and in_features % 128 == 0:
        BLOCK_J, BLOCK_I, BLOCK_K, num_stages = 128, 128, 128, 1
    elif batch % 64 == 0 and out_features % 128 == 0 and in_features % 128 == 0:
        BLOCK_J, BLOCK_I, BLOCK_K, num_stages = 128, 128, 64, 2
    else:
        BLOCK_J, BLOCK_I, BLOCK_K, num_stages = (
            128,
            128 if in_features % 128 == 0 else 64,
            64,
            2,
        )
    full_even = (
        batch % BLOCK_K == 0
        and out_features % BLOCK_J == 0
        and in_features % BLOCK_I == 0
    )
    grid = (triton.cdiv(out_features, BLOCK_J) * triton.cdiv(in_features, BLOCK_I),)
    linear_backward_wgrad_bias_kernel[grid](
        grad_output_flat,
        input_flat,
        grad_weight,
        grad_bias,
        out_features,
        in_features,
        batch,
        BLOCK_J=BLOCK_J,
        BLOCK_I=BLOCK_I,
        BLOCK_K=BLOCK_K,
        GROUP_J=8,
        FULL_EVEN=full_even,
        K_BLOCKS=triton.cdiv(batch, BLOCK_K),
        num_warps=2,
        num_stages=num_stages,
    )
    return grad_weight, grad_bias


def linear_backward(
    input: torch.Tensor,
    grad_output: torch.Tensor,
    weight: torch.Tensor,
    output_mask: tuple,
):
    """
    Computes the gradients for a linear layer backward pass.

    Args:
        input: Input tensor of shape (*, in_features)
        grad_output: Gradient of output of shape (*, out_features)
        weight: Weight tensor of shape (out_features, in_features)
        output_mask: Tuple of 3 booleans (grad_input, grad_weight, grad_bias)

    Returns:
        Tuple of (grad_input, grad_weight, grad_bias), where each element is None
        if the corresponding mask is False
    """
    logger.debug("GEMS_ENFLAME LINEAR_BACKWARD")

    # Validate inputs
    assert (
        input.device.type == flag_gems.device
        and grad_output.device.type == flag_gems.device
        and weight.device.type == flag_gems.device
    )
    assert input.device == grad_output.device == weight.device

    # Handle output_mask - ensure it's a tuple
    if not isinstance(output_mask, (tuple, list)):
        output_mask = (output_mask[0], output_mask[1], output_mask[2])

    in_features = weight.shape[1]
    out_features = weight.shape[0]

    # Get batch dimensions (all but the last dim which is features)
    batch_dims = input.shape[:-1]
    batch_size = 1
    for dim in batch_dims:
        batch_size *= dim

    grad_input = None
    grad_weight = None
    grad_bias = None

    # All three gradients consume the flattened grad_output; build the view once.
    grad_output_flat = (
        grad_output.view(batch_size, out_features) if any(output_mask) else None
    )

    # grad_input = grad_output @ weight
    # Shape: (batch, out_features) @ (out_features, in_features) = (batch, in_features)
    if output_mask[0]:
        grad_input = mm(grad_output_flat, weight)
        grad_input = grad_input.view(*batch_dims, in_features)

    # grad_weight = grad_output.T @ input, grad_bias = grad_output.sum(dim=0)
    if output_mask[1]:
        input_flat = input.view(batch_size, in_features)
        if output_mask[2] and _wgrad_bias_fused_eligible(
            grad_output.dtype, out_features, in_features, batch_size
        ):
            # Fused arm: the wgrad GEMM and bias reduction share the same axis (K=batch), with two outputs in one
            # kernel.
            if grad_output_flat.is_contiguous() and input_flat.is_contiguous():
                grad_weight, grad_bias = _launch_wgrad_bias(
                    grad_output_flat,
                    input_flat,
                    grad_output.dtype,
                    out_features,
                    in_features,
                    batch_size,
                )
            else:
                grad_weight = mm(grad_output_flat.t(), input_flat)
                grad_bias = sum_dim(grad_output_flat, dim=(0,))
        else:
            # Transposed view straight into mm (no materialized transpose
            # copy): the Triton arm takes constexpr strides, the delegated
            # native arm takes arbitrary strides.
            grad_weight = mm(grad_output_flat.t(), input_flat)
            if output_mask[2]:
                grad_bias = sum_dim(grad_output_flat, dim=(0,))
    elif output_mask[2]:
        grad_bias = sum_dim(grad_output_flat, dim=(0,))

    return grad_input, grad_weight, grad_bias
