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

import contextlib
import logging
import os

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, libtuner

logger = logging.getLogger(__name__)


# 910B4 configuration
_CBUF_BUDGET = 512 * 1024
_CUBE_CORES = 20

_MODERATE_TILES = 64
_PIN_MODERATE_KWARGS = {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 256}
_PIN_STARVED_NODIV_KWARGS = {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128}
# fp32 doubles the bytes per tile, so a 64-row A tile keeps the DMA footprint
# down: 4% faster at 384^3, which is this band (N=384 is not a multiple of 256).
_PIN_STARVED_NODIV_FP32_KWARGS = {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 128}
_PIN_STARVED_KWARGS = {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 256}

# Deep-K tile: BK=512 halves the K-loop trip count and the 128-wide N tile
# doubles the program count, which only pays when the 128x256 grid leaves a bad
# wave remainder on the 20 cube cores (2048^3: 128 tiles -> 91% vs 256 -> 98%).
# Short K wastes the second (masked) trip and wide-N / narrow-M shapes lose L0B
# reuse, so gate on all three.
_PIN_DEEP_K_KWARGS = {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 512}
_DEEP_K_MIN_K = 2048
_DEEP_K_MIN_N = 2048
_DEEP_K_MAX_N = 4096

_PIN_FP32_LARGE_KWARGS = {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 256}

# Very large fp32 grids: the 256-row tile halves the program count and wins ~1.6%
# at 4096^3; below ~256 tiles the 128-row tile is still better.
_PIN_FP32_HUGE_KWARGS = {"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 128}
_HUGE_TILES = 256

# Skinny-M band: pin the 16-row tile when M <= 16, where the 64/128-row tiles the
# bands below pin spend their A-tile DMA on masked rows.
_SKINNY_M = 16
_SKINNY_MIN_N = 3072
_PIN_SKINNY_KWARGS = {"BLOCK_M": 16, "BLOCK_N": 256, "BLOCK_K": 256}

_TUNED_CONFIGS = runtime.get_tuned_config("linear")

# M == 1 (decode) streams W through the vector units instead of the cube: faster once
# W is large, slower when it is not.
_GEMV_MIN_WEIGHT_BYTES = 180_000_000
_GEMV_BLOCK_N = 64
_GEMV_BLOCK_K = 256


# Auto-blockify caps the launch at the 20 cube cores and loops the tile body on the core.
@contextlib.contextmanager
def _blockified():
    previous = os.environ.get("TRITON_ALL_BLOCKS_PARALLEL")
    os.environ["TRITON_ALL_BLOCKS_PARALLEL"] = "1"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("TRITON_ALL_BLOCKS_PARALLEL", None)
        else:
            os.environ["TRITON_ALL_BLOCKS_PARALLEL"] = previous


def _prune_configs_by_ub(configs, named_args, **kwargs):
    """Drop configs the compiler cannot lower, or that this shape cannot use."""
    element_size = getattr(
        named_args.get("input_ptr") if named_args else None, "element_size", None
    )
    if not callable(element_size):
        return configs
    budget = _CBUF_BUDGET / element_size()
    pruned = [
        config
        for config in configs
        if (config.kwargs["BLOCK_M"] + config.kwargs["BLOCK_N"])
        * config.kwargs["BLOCK_K"]
        * 2
        <= budget
    ]
    if element_size() == 4:
        pruned = [c for c in pruned if c.kwargs["BLOCK_N"] <= 128]

    def _int(name):
        value = named_args.get(name) if named_args else None
        value = getattr(value, "value", value)
        return value if isinstance(value, int) else None

    M, N = _int("M"), _int("N")
    ref = None if M is None or N is None else triton.cdiv(M, 128) * triton.cdiv(N, 256)
    starved = ref is not None and ref <= _CUBE_CORES
    if (
        element_size() == 2
        and ref is not None
        and M <= _SKINNY_M
        and N >= _SKINNY_MIN_N
        and N % 256 == 0
    ):
        pinned = [c for c in pruned if c.kwargs == _PIN_SKINNY_KWARGS]
        if pinned:
            return pinned
    if starved:
        if N >= 256 and N % 256 != 0:
            want = (
                _PIN_STARVED_NODIV_FP32_KWARGS
                if element_size() == 4
                else _PIN_STARVED_NODIV_KWARGS
            )
            pinned = [c for c in pruned if c.kwargs == want]
            if pinned:
                return pinned
        pinned = [c for c in pruned if c.kwargs == _PIN_STARVED_KWARGS]
        if pinned:
            return pinned
    if element_size() == 4 and ref is not None:
        if ref <= _MODERATE_TILES:
            want = _PIN_STARVED_KWARGS
        elif ref <= _HUGE_TILES:
            want = _PIN_FP32_LARGE_KWARGS
        else:
            want = _PIN_FP32_HUGE_KWARGS
        pinned = [c for c in pruned if c.kwargs == want]
        if pinned:
            return pinned
    pruned = [c for c in pruned if c.kwargs != _PIN_STARVED_KWARGS]
    K = _int("K")
    if (
        element_size() == 2
        and ref is not None
        and _MODERATE_TILES < ref <= 256
        and K is not None
        and K >= _DEEP_K_MIN_K
        and _DEEP_K_MIN_N <= N <= _DEEP_K_MAX_N
    ):
        pinned = [c for c in pruned if c.kwargs == _PIN_DEEP_K_KWARGS]
        if pinned:
            return pinned
    if ref is not None and ref <= _MODERATE_TILES:
        pinned = [c for c in pruned if c.kwargs == _PIN_MODERATE_KWARGS]
        if pinned:
            return pinned
    return pruned or configs


@libentry()
@libtuner(
    configs=_TUNED_CONFIGS,
    key=["M", "N", "K"],
    prune_configs_by={"early_config_prune": _prune_configs_by_ub},
)
@triton.jit
def linear_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """y = x @ W^T (+ b); all operands contiguous, so strides are K and N."""
    pid = tl.program_id(0)
    grid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // grid_n
    pid_n = pid % grid_n

    ram = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rbn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)

    input_ptrs = input_ptr + (ram[:, None] * K + rk[None, :])
    weight_ptrs = weight_ptr + (rk[:, None] + rbn[None, :] * K)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    output_ptrs = output_ptr + (ram[:, None] * N + rbn[None, :])
    output_mask = (ram < M)[:, None] & (rbn < N)[None, :]
    out_ty = output_ptr.dtype.element_ty

    # The BiSheng backend miscompiles the epilogue when a single tl.dot's f32
    # result is consumed by an elementwise bias add: it fuses the add into the
    # Cube/FixPipe epilogue and returns garbage. Only the *fusion* is broken --
    # the bias load itself is fine at any position, and the same add is correct
    # after two accumulations, so anything non-fusable between the dot and the
    # add avoids it. Two workarounds are measured, and which one is cheaper
    # depends only on whether a truncation is available to hide behind:
    #
    #   * no truncation available (fp32 out, or K > BLOCK_K): run the K loop with
    #     a floor of two iterations. The second iteration's mask is all-false so
    #     it contributes zero.
    #   * fp16/bf16 out and K <= BLOCK_K: one dot, then add the bias *after* the
    #     truncation, in the output dtype. This keeps f32->f16 as the only f32
    #     consumer of the accumulator, so the broken fusion never forms, and it
    #     drops the wasted iteration: 1.25x over the loop at (4096,14336,64)
    #     fp16, 1.60x over torch, and 1.46-1.60x across the small-K shapes.
    #
    # The cost is one extra rounding when the sum is truncated, so the worst-case
    # error goes from half an fp16 ulp of the result magnitude to a full one
    # (1.56e-02 -> 3.13e-02 on values in [32,64)). It still passes
    # flag_gems.testing.assert_close(fp16: atol=1e-4*K, rtol=1e-3) on every shape
    # tried, using 0.81-0.87 of the allowance where the loop uses 0.69.
    #
    # Every f32-preserving alternative measured *slower* than the wasted second
    # iteration, so the floor is kept where it is still the cheaper fix: an
    # equivalent no-op barrier (tl.where(acc == acc, acc, 0.0) or -(-acc)), a
    # rank-32 outer-product dot for the bias, and folding the bias into the
    # reduction as an extra unit column all cost 1.13-2.2x the floor. `acc * 1.0`
    # is folded away and does not work; tl.minimum/tl.maximum fail to compile.
    single_dot = (
        BIAS and K <= BLOCK_K and (out_ty == tl.float16 or out_ty == tl.bfloat16)
    )

    if single_dot:
        # Distinct names from the loop arm below: triton joins the two branches
        # as real control flow and would otherwise unify `bias` across them,
        # where it is fp16 here and fp32 there.
        a1 = tl.load(
            input_ptrs,
            mask=(ram < M)[:, None] & (rk < K)[None, :],
            other=0.0,
        )
        b1 = tl.load(
            weight_ptrs,
            mask=(rk < K)[:, None] & (rbn < N)[None, :],
            other=0.0,
        )
        acc1 = accumulator + tl.dot(a1, b1, allow_tf32=False)
        bias1 = tl.load(bias_ptr + rbn, mask=rbn < N, other=0.0).to(out_ty)
        out1 = acc1.to(out_ty) + tl.broadcast_to(bias1[None, :], (BLOCK_M, BLOCK_N))
        tl.store(output_ptrs, out1, mask=output_mask)
    else:
        # A one-trip K loop miscompiles the epilogue, so floor it at 2 (the extra
        # iteration's mask is all-false).
        n_iter = tl.cdiv(K, BLOCK_K)
        if n_iter < 2:
            n_iter = 2

        for k in range(0, n_iter):
            k_remaining = K - k * BLOCK_K
            a = tl.load(
                input_ptrs,
                mask=(ram < M)[:, None] & (rk < k_remaining)[None, :],
                other=0.0,
            )
            b = tl.load(
                weight_ptrs,
                mask=(rk < k_remaining)[:, None] & (rbn < N)[None, :],
                other=0.0,
            )
            accumulator += tl.dot(a, b, allow_tf32=False)
            input_ptrs += BLOCK_K
            weight_ptrs += BLOCK_K

        # Truncate before the bias add: an f32 elementwise op on the Cube result
        # forces a full-precision L0C -> UB round trip that costs more than the
        # half-precision add. The fp16 ulp shift is within the operator's
        # tolerance.
        output = accumulator.to(out_ty)
        if BIAS:
            bias = tl.load(bias_ptr + rbn, mask=rbn < N, other=0.0).to(out_ty)
            output = output + tl.broadcast_to(bias[None, :], (BLOCK_M, BLOCK_N))
        tl.store(output_ptrs, output, mask=output_mask)


@libentry()
@triton.jit
def linear_gemv_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    N: tl.constexpr,
    K: tl.constexpr,
    BIAS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """M == 1: one program per BLOCK_N output rows, reducing over K with the vector
    units. Both operands contiguous (the host enforces it)."""
    pid = tl.program_id(0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    x_ptrs = input_ptr + offs_k
    w_ptrs = weight_ptr + offs_n[:, None] * K + offs_k[None, :]
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    # Floor the K loop at two like linear_kernel, and stay masked whenever that
    # floor extends the loop past K.
    n_iter = tl.cdiv(K, BLOCK_K)
    even_k = (K % BLOCK_K == 0) and n_iter > 1
    if n_iter < 2:
        n_iter = 2

    # Rows past N need a predicate too (a full last block does not): without it the
    # w load reads up to (BLOCK_N - N % BLOCK_N) * K elements past the weight.
    n_mask = offs_n < N
    aligned_n = N % BLOCK_N == 0

    for kb in range(0, n_iter):
        if even_k:
            x = tl.load(x_ptrs)
            if aligned_n:
                w = tl.load(w_ptrs)
            else:
                w = tl.load(w_ptrs, mask=n_mask[:, None], other=0.0)
        else:
            k = kb * BLOCK_K + offs_k
            x = tl.load(x_ptrs, mask=k < K, other=0.0)
            if aligned_n:
                w = tl.load(w_ptrs, mask=k[None, :] < K, other=0.0)
            else:
                w = tl.load(w_ptrs, mask=(k[None, :] < K) & n_mask[:, None], other=0.0)
        acc += tl.sum(w.to(tl.float32) * x.to(tl.float32)[None, :], axis=1)
        x_ptrs += BLOCK_K
        w_ptrs += BLOCK_K

    if BIAS:
        b = tl.load(bias_ptr + offs_n, mask=n_mask, other=0.0)
        acc += b.to(tl.float32)
    tl.store(output_ptr + offs_n, acc.to(output_ptr.dtype.element_ty), mask=n_mask)


def linear(input, weight, bias=None):
    """
    Applies a linear transformation to the incoming data: y = xA^T + b

    Args:
        input: Input tensor of shape (*, in_features) where * means any number of
               additional dimensions, including none.
        weight: Weight tensor of shape (out_features, in_features)
        bias: Bias tensor of shape (out_features), optional

    Returns:
        Output tensor of shape (*, out_features)
    """
    logger.debug("GEMS_ASCEND LINEAR")

    input_dim = input.dim()
    if input_dim == 1:
        # Single 1D input: treat as (1, in_features)
        input = input.unsqueeze(0)
        single_1d = True
    else:
        single_1d = False

    # Flatten batch dimensions: (*, in_features) -> (batch, in_features)
    batch_dims = input.shape[:-1]
    batch_size = 1
    for dim in batch_dims:
        batch_size *= dim
    M = batch_size
    K = input.shape[-1]  # in_features
    N = weight.shape[0]  # out_features

    assert weight.dim() == 2 and weight.shape[1] == K, "incompatible dimensions"

    # An empty product needs no kernel (and no operand copies) -- return first.
    if M == 0 or N == 0:
        output = torch.empty((*batch_dims, N), device=input.device, dtype=input.dtype)
        return output.squeeze(0) if single_1d else output

    bias_is_vector = (
        bias is not None
        and bias.numel() == N
        and (bias.dim() == 1 or all(dim == 1 for dim in bias.shape[:-1]))
    )
    if bias_is_vector and bias.dim() > 0 and bias.stride(-1) != 1:
        bias = bias.contiguous()

    input_flat = input.reshape(M, K)
    if not input_flat.is_contiguous():
        input_flat = input_flat.contiguous()

    weight = weight.contiguous()

    output = torch.empty((M, N), device=input.device, dtype=input.dtype)
    bias_arg = bias if bias_is_vector else weight  # dummy ptr when unfused

    if (
        M == 1
        and input_flat.element_size() == 2
        and N * K * input_flat.element_size() >= _GEMV_MIN_WEIGHT_BYTES
    ):
        grid = (triton.cdiv(N, _GEMV_BLOCK_N),)
        with torch_device_fn.device(input.device), _blockified():
            linear_gemv_kernel[grid](
                input_flat,
                weight,
                bias_arg,
                output,
                N,
                K,
                BIAS=bias_is_vector,
                BLOCK_N=_GEMV_BLOCK_N,
                BLOCK_K=_GEMV_BLOCK_K,
                enable_auto_blockify=True,
                num_warps=4,
                num_stages=2,
            )
    else:
        grid = lambda META: (
            triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
        )
        with torch_device_fn.device(input.device), _blockified():
            linear_kernel[grid](
                input_flat,
                weight,
                bias_arg,
                output,
                M,
                N,
                K,
                BIAS=bias_is_vector,
                enable_auto_blockify=True,
            )

    output = output.view(*batch_dims, N)

    if bias is not None and not bias_is_vector:
        output.add_(bias)

    # If original input was 1D, squeeze the batch dim
    if single_1d:
        output = output.squeeze(0)

    return output
