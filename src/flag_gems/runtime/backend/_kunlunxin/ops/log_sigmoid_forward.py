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

logger = logging.getLogger(__name__)

# Kunlunxin/XPU performance override for aten::log_sigmoid_forward.
#
# Baseline (generic flag_gems/ops kernel): one program per 1024 elements with
# *always-masked* loads/stores. On XPU the masked-memory path serializes DMA
# (fp16 16.7M elts 3.01 ms vs torch 0.36 ms; ~33 GB/s vs ~278 GB/s).
#
# Root cause (probe sweep, /tmp/lsf_xpu2_probe/, 2026-08-19):
#   1. masked loads/stores per-element block: contiguous unmasked chunks are
#      ~4.5x faster than the always-true masked path at the same block size;
#   2. per-program block size matters a lot for launch-bound fan-out: sweep
#      B512..B16384 x warps 4..16 on the official 12-shape matrix shows a
#      per-shape optimum at `{numel <= 16384: B2048/w4, <= 262144: B8192/w8,
#      else B16384/w16}`;
#   3. the two fp32 transcendentals (exp + log) dominate ALU time (~10 ns/elem
#      each); fp16/bf16 inputs are widened to fp32 first as the backend does
#      not provide vectorized fp16 exp/log.
#
# The kernel body is numerically identical to the generic implementation
# (fp32 math, stores quantized back to the input dtype, masked tail kept via
# a constexpr NEED_MASK switch only for non-divisible sizes).


@triton.jit
def log_sigmoid_forward_kernel(
    x_ptr, output_ptr, buffer_ptr, n_elements, BLOCK_SIZE: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    if NEED_MASK:
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    else:
        x = tl.load(x_ptr + offsets)

    # log_sigmoid(x) = -log(1 + exp(-x)), computed as
    # min(x, 0) - log(1 + exp(-|x|)) for numerical stability.
    x_f32 = x.to(tl.float32)
    abs_x = tl.abs(x_f32)
    buffer_val = tl.exp(-abs_x)
    log_sigmoid_val = tl.minimum(x_f32, 0.0) - tl.log(1.0 + buffer_val)

    if NEED_MASK:
        tl.store(output_ptr + offsets, log_sigmoid_val.to(x.dtype), mask=mask)
        tl.store(buffer_ptr + offsets, buffer_val.to(x.dtype), mask=mask)
    else:
        tl.store(output_ptr + offsets, log_sigmoid_val.to(x.dtype))
        tl.store(buffer_ptr + offsets, buffer_val.to(x.dtype))


# (numel_upper_bound, BLOCK_SIZE, num_warps) tuned on the official benchmark
# matrix (12 shapes x fp16/fp32/bf16, do_bench median sweep, 2026-08-18).
_TIERS = (
    (16384, 2048, 4),
    (262144, 8192, 8),
    (None, 16384, 16),
)


def _pick_tier(numel):
    for hi, block, warps in _TIERS:
        if hi is None or numel <= hi:
            return block, warps
    return 16384, 16


def log_sigmoid_forward(A):
    logger.debug("GEMS_KUNLUNXIN LOG_SIGMOID_FORWARD")
    output = torch.empty_like(A)
    buffer = torch.empty_like(A)
    n_elements = A.numel()
    if n_elements == 0:
        return output, buffer

    block, warps = _pick_tier(n_elements)
    need_mask = (n_elements % block) != 0
    grid = (triton.cdiv(n_elements, block),)
    log_sigmoid_forward_kernel[grid](
        A.reshape(-1),
        output,
        buffer,
        n_elements,
        BLOCK_SIZE=block,
        NEED_MASK=need_mask,
        num_warps=warps,
    )
    return output, buffer