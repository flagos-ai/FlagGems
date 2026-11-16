# Copyright 2026, The FlagOS Contributors.
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
#
# Kunlunxin XPU3 override for aten::bilinear.
#
# The generic implementation walks in1_features row by row with a 32-lane
# masked vector accumulator per output element: for the benchmark matrix it
# launches M*N programs (up to 524288) that each execute K1*ceil(K2/32)
# masked loads, so it runs at ~0.9 GFLOP/s (0.004-0.045x of torch).
#
# On this backend the arithmetic IS a matmul and `tt.dot` lowers to the
# matrix unit (`llvm.xpu.mm_acc`), so the fast path computes
#     u[m, :] = x1[m, :] @ W[o]        (one tl.dot per (m-block, o))
#     y[m, o] = sum_j u[m, j] * x2[m, j]
# with fully affine, unmasked loads (Kunlunxin miscompiles *any* masked 2-D
# load inside a dot kernel, so the tiles must divide exactly) and one
# axis-1 reduction over a register-resident tile.
#
# Shapes that the dot path cannot cover exactly (odd K1/K2, tiny M) are
# routed to the proven generic kernel below, unchanged.
import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

from .to import _to_copy_func, _to_copy_func_close_interleave

logger = logging.getLogger(__name__)


@triton.jit
def bilinear_kernel(
    input1_ptr,
    input2_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    M,
    N,
    K1,
    K2,
    stride_i1m,
    stride_i1k,
    stride_i2m,
    stride_i2k,
    stride_wn,
    stride_wk1,
    stride_wk2,
    stride_om,
    stride_on,
    BIAS: tl.constexpr,
    BLOCK_K2: tl.constexpr,
):
    """
    Bilinear kernel: y[m, o] = sum_{i, j} x1[m, i] * W[o, i, j] * x2[m, j] + b[o]
    - input1: (M, K1) where M is the flattened batch size, K1 is in1_features
    - input2: (M, K2) where K2 is in2_features
    - weight: (N, K1, K2) where N is out_features
    - bias: (N,) optional
    - output: (M, N)

    Each program computes a single output element (m, o), reducing over the
    in1_features x in2_features grid in tiles to keep memory bounded.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # XPU cannot reduce a 2-D tile here: a full `tl.sum(prod)` or an axis=0
    # reduction silently mis-computes and an axis=1 reduction runs out of
    # shared memory. Reductions fed by gathered loads mis-compute as well, and
    # a `tt.reduce` nested inside two loops fails to legalize. So walk
    # in1_features row by row, accumulate a 1-D vector over in2_features and
    # reduce that vector once, after the loops.
    offs_k2 = tl.arange(0, BLOCK_K2)
    acc = tl.zeros([BLOCK_K2], dtype=tl.float32)

    for i in range(0, K1):
        x1 = tl.load(input1_ptr + pid_m * stride_i1m + i * stride_i1k).to(tl.float32)

        for k2 in range(0, tl.cdiv(K2, BLOCK_K2)):
            offs = k2 * BLOCK_K2 + offs_k2
            mask = offs < K2

            x2 = tl.load(
                input2_ptr + pid_m * stride_i2m + offs * stride_i2k,
                mask=mask,
                other=0.0,
            ).to(tl.float32)
            w = tl.load(
                weight_ptr + pid_n * stride_wn + i * stride_wk1 + offs * stride_wk2,
                mask=mask,
                other=0.0,
            ).to(tl.float32)

            acc += (x1 * w) * x2

    result = tl.sum(acc, axis=0)

    if BIAS:
        result += tl.load(bias_ptr + pid_n).to(tl.float32)

    output = result.to(output_ptr.dtype.element_ty)
    tl.store(output_ptr + pid_m * stride_om + pid_n * stride_on, output)


@triton.jit
def _bilinear_dot_kernel(
    input1_ptr,
    input2_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    stride_i1m,
    stride_i1k,
    stride_i2m,
    stride_i2k,
    stride_wn,
    stride_wk1,
    stride_wk2,
    stride_om,
    stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_K1: tl.constexpr,
    BLOCK_K2: tl.constexpr,
    N_K2: tl.constexpr,
    BIAS: tl.constexpr,
    IEEE: tl.constexpr,
):
    """y[m, o] = sum_j (x1[m, :] @ W[o])[j] * x2[m, j] for BLOCK_M rows and one o.

    All tile extents are exact divisors of the tensor extents (no masks): the
    Kunlunxin dot lowering silently miscompiles masked 2-D loads, even when
    the mask is all-true.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k1 = tl.arange(0, BLOCK_K1)
    offs_k2 = tl.arange(0, BLOCK_K2)

    x1 = tl.load(
        input1_ptr + offs_m[:, None] * stride_i1m + offs_k1[None, :] * stride_i1k
    )
    w_base = weight_ptr + pid_n * stride_wn + offs_k1[:, None] * stride_wk1

    acc = tl.zeros([BLOCK_M], dtype=tl.float32)
    for jc in tl.static_range(N_K2):
        offs = jc * BLOCK_K2 + offs_k2
        x2 = tl.load(
            input2_ptr + offs_m[:, None] * stride_i2m + offs[None, :] * stride_i2k
        )
        w = tl.load(w_base + offs[None, :] * stride_wk2)
        if IEEE:
            u = tl.dot(x1, w, input_precision="ieee")
        else:
            u = tl.dot(x1, w)
        acc += tl.sum(u * x2.to(tl.float32), axis=1)

    if BIAS:
        acc += tl.load(bias_ptr + pid_n).to(tl.float32)

    # A 1-D store of `acc` into a stride-N column silently drops lanes on this
    # backend (probe: MODE0/MODE1/MODE3 leave most lanes unwritten); the same
    # offsets expressed as a 2-D (BLOCK_M, 1) tile store are exact.
    tl.store(
        output_ptr + offs_m[:, None] * stride_om + pid_n * stride_on,
        acc[:, None].to(output_ptr.dtype.element_ty),
    )


def _pick_dot_tiles(M, K1, K2, dtype):
    """Return (BLOCK_M, BLOCK_K1, BLOCK_K2, N_K2) or None when the exact-divisor
    dot path cannot cover this shape."""
    if dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return None
    # tl.dot needs M >= 16, K >= 16, N >= 16 and power-of-two tile extents.
    if K1 < 16 or (K1 & (K1 - 1)) != 0:
        return None
    if K1 > 256:
        return None
    if K2 < 16 or K2 % 16 != 0:
        return None
    block_k2 = K2 if K2 <= 128 else 128
    if K2 % block_k2 != 0:
        return None
    # Largest row block that divides M and keeps the operand tiles bounded
    # (BM=256 with K1=256 measured fastest on the benchmark matrix and still
    # fits uni_sram for fp16 and fp32 alike).
    block_m = 0
    for cand in (256, 128, 64, 32, 16):
        if M % cand == 0 and cand * K1 <= 65536 and cand * block_k2 <= 32768:
            block_m = cand
            break
    if block_m == 0:
        return None
    return block_m, K1, block_k2, K2 // block_k2


def _cast_bf16_to_fp32(x):
    """bf16 -> fp32 with the backend's close-interleave cast kernel.

    In-kernel bf16 loads are silently miscompiled on XPU3 (the same defect that
    forces the generic path in `pow_scalar` / `matmul_bias_activation`), so the
    dot path promotes bf16 inputs to fp32 on the device with a real Triton cast
    kernel instead of loading bf16 inside the dot kernel.
    """
    out = torch.empty(x.shape, dtype=torch.float32, device=x.device)
    return _to_copy_func_close_interleave(x, out0=out)


def _cast_fp32_to(dtype):
    def _cast(x):
        out = torch.empty(x.shape, dtype=dtype, device=x.device)
        return _to_copy_func(x, out0=out)

    return _cast


def bilinear(input1, input2, weight, bias=None):
    """
    Applies a bilinear transformation to the incoming data:
        y = x1^T A x2 + b

    Args:
        input1: Tensor of shape (N, *, in1_features)
        input2: Tensor of shape (N, *, in2_features)
        weight: Tensor of shape (out_features, in1_features, in2_features)
        bias: Tensor of shape (out_features), optional

    Returns:
        Output tensor of shape (N, *, out_features)
    """
    logger.debug("GEMS BILINEAR")

    # All but the last dimension of input1 and input2 must match.
    batch_dims = input1.shape[:-1]
    M = 1
    for dim in batch_dims:
        M *= dim
    K1 = input1.shape[-1]  # in1_features
    K2 = input2.shape[-1]  # in2_features
    N = weight.shape[0]  # out_features

    input1_flat = input1.reshape(M, K1)
    input2_flat = input2.reshape(M, K2)
    weight = weight.contiguous()

    if M == 0:
        return torch.empty(batch_dims + (N,), device=input1.device, dtype=input1.dtype)

    # The dot lowering also needs unit innermost stride on both operands: a
    # stride-2 row (e.g. `base[:, ::2]`) is silently mis-read inside a dot
    # kernel (probe: err ~1e3 on values ~2e2), so such inputs stay on the
    # generic path, which is stride-safe.
    if input1_flat.stride(1) == 1 and input2_flat.stride(1) == 1:
        tiles = _pick_dot_tiles(M, K1, K2, input1.dtype)
    else:
        tiles = None
    kernel_out = None
    if tiles is not None:
        if input1.dtype is torch.bfloat16:
            # In-kernel bf16 loads miscompile (they are lowered with a wrong
            # 2x GEP scale), so promote every operand to fp32 with the device
            # cast kernel and keep the (correct) fp32 dot path. The final
            # result is cast back to bf16 with the same class of cast kernel.
            input1_flat = _cast_bf16_to_fp32(input1_flat)
            input2_flat = _cast_bf16_to_fp32(input2_flat)
            weight = _cast_bf16_to_fp32(weight)
            bias = _cast_bf16_to_fp32(bias) if bias is not None else None
            kernel_out = torch.empty((M, N), device=input1.device, dtype=torch.float32)
        else:
            kernel_out = torch.empty((M, N), device=input1.device, dtype=input1.dtype)

        block_m, block_k1, block_k2, n_k2 = tiles
        # Accumulate in fp32. For fp16/bf16 inputs `tl.dot` already widens to
        # fp32; for fp32 inputs "ieee" avoids the tf32-like ~8e-4 relative loss
        # of the default lowering (measured: 0.03 -> 0 abs err, same latency).
        grid = (M // block_m, N)
        with torch_device_fn.device(input1.device):
            _bilinear_dot_kernel[grid](
                input1_flat,
                input2_flat,
                weight,
                bias if bias is not None else weight,
                kernel_out,
                input1_flat.stride(0),
                input1_flat.stride(1),
                input2_flat.stride(0),
                input2_flat.stride(1),
                weight.stride(0),
                weight.stride(1),
                weight.stride(2),
                kernel_out.stride(0),
                kernel_out.stride(1),
                BLOCK_M=block_m,
                BLOCK_K1=block_k1,
                BLOCK_K2=block_k2,
                N_K2=n_k2,
                BIAS=bias is not None,
                IEEE=input1_flat.dtype is torch.float32,
            )
        if input1.dtype is torch.bfloat16:
            output = _cast_fp32_to(input1.dtype)(kernel_out)
        else:
            output = kernel_out
    else:
        output = torch.empty((M, N), device=input1.device, dtype=input1.dtype)
        # A one-element tile cannot be lowered on XPU, so keep at least 2 lanes.
        BLOCK_K2 = max(min(triton.next_power_of_2(K2), 32), 2)
        grid = (M, N)
        with torch_device_fn.device(input1.device):
            bilinear_kernel[grid](
                input1_flat,
                input2_flat,
                weight,
                bias if bias is not None else weight,
                output,
                M,
                N,
                K1,
                K2,
                input1_flat.stride(0),
                input1_flat.stride(1),
                input2_flat.stride(0),
                input2_flat.stride(1),
                weight.stride(0),
                weight.stride(1),
                weight.stride(2),
                output.stride(0),
                output.stride(1),
                BIAS=bias is not None,
                BLOCK_K2=BLOCK_K2,
            )

    output = output.view(*batch_dims, N)
    return output
