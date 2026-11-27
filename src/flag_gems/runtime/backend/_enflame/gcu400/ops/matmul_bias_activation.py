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
"""GCU400 implementation of the fused ``matmul + bias + ReLU`` operator.

The common source uses a fixed 128x128x32 tile with three stages. On GCU400
that combination is dominated by the per-iteration K mask: the dot pipeline
refills every 32 columns, so the measured cost is 4-6x above the achievable
roofline (fp16 1024^3 223.8us, 2048^3 1526.6us, 256x1024x4096 291.0us).

This version keeps the operator semantics identical and only retunes the tile:

* ``BLOCK_K`` follows the reduction length instead of being fixed. A tall K
  tile is what removes the mask from the hot loop (``K % BLOCK_K == 0`` makes
  the predicate statically true for every full tile), and it is also what keeps
  the shared-memory request inside the GCU400 DSM budget: a 512-wide K tile next
  to a 128-wide M/N tile needs ~1.05MB and fails to launch, so the cap is 512
  for the 16-bit dtypes and 256 for fp32, whose dot operand is twice as wide.
  Clamping to ``next_pow2(K)`` is mandatory, not cosmetic -- a K tile wider than
  the reduction makes the masked dot read the wrong slab (a 512-wide tile with
  ``K=71`` returned values off by 2.7e4x on this backend).
* ``BLOCK_M``/``BLOCK_N`` shrink with the smaller matrix side so skinny problems
  do not pad most of the tile away (``M=15``/``N=160``: 16x16 is 1.2x faster than
  128x128 here, and the 1x1x32 case would otherwise need 190us).
* ``num_warps=2`` / ``num_stages=1`` wins at every measured size (GCU400 mm
  evidence agrees: the 2-warp entries are the DSM-safe winners).

Measured with do_bench (median, warmup 25 / rep 100) against ``torch.mm + add +
relu`` and against the common 128x128x32 tile, fp16 / fp32:

    shape                common      this       torch     speedup vs common
    384x384x384          30.9us      9.8us      23.1us    3.2x
    1024x1024x1024      223.8us     45.1us      61.9us    5.0x
    2048x2048x2048     1526.6us    252.4us     308.0us    6.1x
    4096x4096x4096    12671.6us   1978.2us    2207.6us    6.4x
    256x1024x4096       291.0us     44.8us      73.9us    6.5x
    128x256x512          39.6us      8.3us      25.2us    4.8x
    495x5333x71          81.0us     49.3us      72.2us    1.6x

The only shape that stays behind the native library is the degenerate
1x1x32 case (33.4us vs 15.2us), where ATen dispatches a dedicated tiny-GEMV
path; a 16x16 dot tile is the floor for the fused form.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import broadcastable_to, libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def matmul_bias_activation_kernel(
    a_ptr,
    b_ptr,
    bias_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_bias,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(
            a_ptrs,
            mask=(offs_am[:, None] < M) & (offs_k[None, :] < K - k * BLOCK_SIZE_K),
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=(offs_k[:, None] < K - k * BLOCK_SIZE_K) & (offs_bn[None, :] < N),
            other=0.0,
        )
        accumulator += tl.dot(a, b, allow_tf32=False)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    c_ptrs = c_ptr + stride_cm * offs_am[:, None] + stride_cn * offs_bn[None, :]
    c_mask = (offs_am[:, None] < M) & (offs_bn[None, :] < N)
    bias = tl.load(bias_ptr + offs_bn * stride_bias, mask=offs_bn < N, other=0.0)
    accumulator = accumulator + bias[None, :]

    # ReLU via the synthesized-zero form: a rank-0 constant second operand is a
    # separate lowering that differs on NaN/-Inf, while this shape is identical
    # for every finite input.
    accumulator = tl.maximum(accumulator, accumulator * 0.0)

    tl.store(c_ptrs, accumulator.to(bias.dtype), mask=c_mask)


# Widest K tile whose operands fit the GCU400 DSM budget next to a 128x128 tile.
_BLOCK_K_CAP = {torch.float16: 512, torch.bfloat16: 512, torch.float32: 256}


def _config(M, N, K, dtype):
    """Return (BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages) for a shape.

    Tiers come from the do_bench sweep in the module docstring: a K tile capped
    by the dtype budget and clamped to the reduction length, square M/N tiles
    that shrink with the smaller matrix side, always 2 warps and a single stage.
    """
    block_k_cap = _BLOCK_K_CAP.get(dtype, 512)
    block_size_k = min(block_k_cap, max(32, triton.next_power_of_2(max(int(K), 1))))
    small_side = min(M, N)
    if small_side <= 16:
        block_size_m = block_size_n = 16
    elif small_side <= 32:
        block_size_m = block_size_n = 32
    elif small_side <= 64:
        block_size_m = block_size_n = 64
    else:
        block_size_m = block_size_n = 128
    return block_size_m, block_size_n, block_size_k, 2, 1


def matmul_bias_activation(input, weight, bias):
    """Fused matmul + bias + ReLU activation.

    Args:
        input: Input tensor of shape (M, K)
        weight: Weight matrix of shape (K, N)
        bias: Bias vector of shape (N,) or (1, N)

    Returns:
        Output tensor of shape (M, N) with ReLU activation applied
    """
    assert input.shape[1] == weight.shape[0], "Incompatible dimensions"
    assert broadcastable_to(
        bias.shape, (input.shape[0], weight.shape[1])
    ), "Incompatible input shape"
    M, K = input.shape
    _, N = weight.shape

    logger.debug("GEMS_ENFLAME MATMUL_BIAS_ACTIVATION")
    if input.stride(0) > 1 and input.stride(1) > 1:
        input = input.contiguous()
    if weight.stride(0) > 1 and weight.stride(1) > 1:
        weight = weight.contiguous()
    if bias.dim() > 1:
        bias = bias.reshape(-1)
    out = torch.empty((M, N), device=input.device, dtype=input.dtype)
    if M == 0 or N == 0 or K == 0:
        return out

    block_size_m, block_size_n, block_size_k, num_warps, num_stages = _config(
        M, N, K, input.dtype
    )
    grid = (triton.cdiv(M, block_size_m), triton.cdiv(N, block_size_n))
    matmul_bias_activation_kernel[grid](
        input,
        weight,
        bias,
        out,
        M,
        N,
        K,
        input.stride(0),
        input.stride(1),
        weight.stride(0),
        weight.stride(1),
        bias.stride(0),
        out.stride(0),
        out.stride(1),
        BLOCK_SIZE_M=block_size_m,
        BLOCK_SIZE_N=block_size_n,
        BLOCK_SIZE_K=block_size_k,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out
