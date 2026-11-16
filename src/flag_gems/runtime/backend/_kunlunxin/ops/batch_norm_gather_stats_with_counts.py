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
from torch import Tensor

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

# XPU3: `grid = (C,)` one-program-per-channel pays ~0.245 us of scalar 4-byte
# DMA per channel (8 scalar loads + fence), making the op linear in C.  The
# statistics tensors are `[num_segments, C]` row-major, so channels are
# contiguous and a whole BLOCK_C channel block can be merged by one program
# (see KernelOpt_Knowledge wiki pattern-xpu3-per-channel-program-scalar-dma-cost
# and technique-xpu3-channel-block-gather-stats-single-program).
# A 128-lane 1-D loop-carried accumulator fails `TritonXPUUnrollControl`
# (`OutOfResources: uni_sram`); 64 lanes is the safe width (compiler trap 2).
# So BLOCK_C is pinned to 64: one program per 64 contiguous channels.
MIN_BLOCK_C = 64


def _pick_block_c(C: int) -> int:
    # Always 64: one program per 64 contiguous channels.  Kept as a helper so
    # the launch site reads the same as the tuning constant.
    return MIN_BLOCK_C


@libentry()
@triton.jit
def batch_norm_gather_stats_with_counts_kernel(
    mean_ptr,
    invstd_ptr,
    counts_ptr,
    running_mean_ptr,
    running_var_ptr,
    out_mean_ptr,
    out_invstd_ptr,
    num_segments,
    C: tl.constexpr,
    momentum,
    eps,
    has_running_mean: tl.constexpr,
    BLOCK_C: tl.constexpr,
    EVEN: tl.constexpr,
):
    pid = tl.program_id(0)
    c = pid * BLOCK_C + tl.arange(0, BLOCK_C)

    if EVEN:
        # C is a multiple of BLOCK_C: every access is affine and unmasked.
        acc_count = 0.0
        acc_cm = tl.zeros([BLOCK_C], tl.float32)
        acc_cv = tl.zeros([BLOCK_C], tl.float32)
        for s in tl.range(0, num_segments):
            cnt = tl.load(counts_ptr + s).to(tl.float32)
            m = tl.load(mean_ptr + s * C + c).to(tl.float32)
            iv = tl.load(invstd_ptr + s * C + c).to(tl.float32)
            v = 1.0 / (iv * iv) - eps
            acc_count += cnt
            acc_cm += cnt * m
            acc_cv += cnt * v

        global_mean = acc_cm / acc_count

        acc_sq = tl.zeros([BLOCK_C], tl.float32)
        for s in tl.range(0, num_segments):
            cnt = tl.load(counts_ptr + s).to(tl.float32)
            m = tl.load(mean_ptr + s * C + c).to(tl.float32)
            d = m - global_mean
            acc_sq += cnt * d * d

        global_var = acc_cv / acc_count + acc_sq / acc_count
        global_invstd = 1.0 / tl.sqrt(global_var + eps)

        tl.store(out_mean_ptr + c, global_mean)
        tl.store(out_invstd_ptr + c, global_invstd)

        if has_running_mean:
            old_rm = tl.load(running_mean_ptr + c).to(tl.float32)
            new_rm = (1.0 - momentum) * old_rm + momentum * global_mean
            tl.store(running_mean_ptr + c, new_rm)

            old_rv = tl.load(running_var_ptr + c).to(tl.float32)
            new_rv = (1.0 - momentum) * old_rv + momentum * global_var
            tl.store(running_var_ptr + c, new_rv)
    else:
        # Masked tail; clamp addresses so tail lanes can never write out of
        # bounds (pattern-xpu3-masked-store-pointer-clamp).
        mask = c < C
        c_safe = tl.where(mask, c, 0)

        acc_count = 0.0
        acc_cm = tl.zeros([BLOCK_C], tl.float32)
        acc_cv = tl.zeros([BLOCK_C], tl.float32)
        for s in tl.range(0, num_segments):
            cnt = tl.load(counts_ptr + s).to(tl.float32)
            m = tl.load(mean_ptr + s * C + c_safe, mask=mask, other=0.0).to(tl.float32)
            iv = tl.load(invstd_ptr + s * C + c_safe, mask=mask, other=0.0).to(tl.float32)
            v = 1.0 / (iv * iv) - eps
            acc_count += cnt
            acc_cm += cnt * m
            acc_cv += cnt * v

        global_mean = acc_cm / acc_count

        acc_sq = tl.zeros([BLOCK_C], tl.float32)
        for s in tl.range(0, num_segments):
            cnt = tl.load(counts_ptr + s).to(tl.float32)
            m = tl.load(mean_ptr + s * C + c_safe, mask=mask, other=0.0).to(tl.float32)
            d = m - global_mean
            acc_sq += cnt * d * d

        global_var = acc_cv / acc_count + acc_sq / acc_count
        global_invstd = 1.0 / tl.sqrt(global_var + eps)

        tl.store(out_mean_ptr + c_safe, global_mean, mask=mask)
        tl.store(out_invstd_ptr + c_safe, global_invstd, mask=mask)

        if has_running_mean:
            old_rm = tl.load(running_mean_ptr + c_safe, mask=mask, other=0.0).to(
                tl.float32
            )
            new_rm = (1.0 - momentum) * old_rm + momentum * global_mean
            tl.store(running_mean_ptr + c_safe, new_rm, mask=mask)

            old_rv = tl.load(running_var_ptr + c_safe, mask=mask, other=0.0).to(
                tl.float32
            )
            new_rv = (1.0 - momentum) * old_rv + momentum * global_var
            tl.store(running_var_ptr + c_safe, new_rv, mask=mask)


@libentry()
@triton.jit
def batch_norm_gather_stats_with_counts_channel_kernel(
    mean_ptr,
    invstd_ptr,
    counts_ptr,
    running_mean_ptr,
    running_var_ptr,
    out_mean_ptr,
    out_invstd_ptr,
    num_segments: tl.constexpr,
    C: tl.constexpr,
    momentum,
    eps,
    has_running_mean: tl.constexpr,
    BLOCK_SEGMENTS: tl.constexpr,
):
    # Small-C path: one program per channel (grid = C) so all channels run in
    # parallel.  For C < 64 the channel-block kernel would leave most of the 64
    # cores idle; this mirrors the generic per-channel formulation.
    c = tl.program_id(0)

    seg_offsets = tl.arange(0, BLOCK_SEGMENTS)
    mask = seg_offsets < num_segments

    mean_vals = tl.load(mean_ptr + seg_offsets * C + c, mask=mask, other=0.0).to(
        tl.float32
    )
    invstd_vals = tl.load(invstd_ptr + seg_offsets * C + c, mask=mask, other=0.0).to(
        tl.float32
    )
    counts_vals = tl.load(counts_ptr + seg_offsets, mask=mask, other=0.0).to(tl.float32)

    total_count = tl.sum(counts_vals, axis=0)

    var_vals = 1.0 / (invstd_vals * invstd_vals) - eps
    global_mean = tl.sum(counts_vals * mean_vals, axis=0) / total_count
    weighted_var = tl.sum(counts_vals * var_vals, axis=0) / total_count
    diff = mean_vals - global_mean
    weighted_sq_diff = tl.sum(counts_vals * diff * diff, axis=0) / total_count
    global_var = weighted_var + weighted_sq_diff
    global_invstd = 1.0 / tl.sqrt(global_var + eps)

    tl.store(out_mean_ptr + c, global_mean)
    tl.store(out_invstd_ptr + c, global_invstd)

    if has_running_mean:
        old_rm = tl.load(running_mean_ptr + c).to(tl.float32)
        new_rm = (1.0 - momentum) * old_rm + momentum * global_mean
        tl.store(running_mean_ptr + c, new_rm)

        old_rv = tl.load(running_var_ptr + c).to(tl.float32)
        new_rv = (1.0 - momentum) * old_rv + momentum * global_var
        tl.store(running_var_ptr + c, new_rv)


def batch_norm_gather_stats_with_counts(
    input: Tensor,
    mean: Tensor,
    invstd: Tensor,
    running_mean,
    running_var,
    momentum: float,
    eps: float,
    counts: Tensor,
):
    logger.debug("GEMS_KUNLUNXIN BATCH_NORM_GATHER_STATS_WITH_COUNTS")

    num_segments = mean.shape[0]
    C = mean.shape[1]

    # Outputs are always float32 (aten contract).
    out_mean = torch.empty(C, dtype=torch.float32, device=mean.device)
    out_invstd = torch.empty(C, dtype=torch.float32, device=mean.device)

    has_running_mean = running_mean is not None and running_var is not None

    # Dummy pointers keep the kernel signature stable; the store block is
    # compiled out when has_running_mean is False.
    running_mean_ptr = running_mean if has_running_mean else out_mean
    running_var_ptr = running_var if has_running_mean else out_invstd

    block_c = _pick_block_c(C)

    with torch_device_fn.device(input.device):
        if C < block_c:
            # Small-C: per-channel programs (grid = C) use every core; the
            # channel-block kernel would run a single, mostly-masked program.
            batch_norm_gather_stats_with_counts_channel_kernel[(C,)](
                mean,
                invstd,
                counts,
                running_mean_ptr,
                running_var_ptr,
                out_mean,
                out_invstd,
                num_segments=num_segments,
                C=C,
                momentum=momentum,
                eps=eps,
                has_running_mean=has_running_mean,
                BLOCK_SEGMENTS=triton.next_power_of_2(num_segments),
            )
        else:
            batch_norm_gather_stats_with_counts_kernel[
                (triton.cdiv(C, block_c),)
            ](
                mean,
                invstd,
                counts,
                running_mean_ptr,
                running_var_ptr,
                out_mean,
                out_invstd,
                num_segments,
                C=C,
                momentum=momentum,
                eps=eps,
                has_running_mean=has_running_mean,
                BLOCK_C=block_c,
                EVEN=(C % block_c) == 0,
            )

    return out_mean, out_invstd
