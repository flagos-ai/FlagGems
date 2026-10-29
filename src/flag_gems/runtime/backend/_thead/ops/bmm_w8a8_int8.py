# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

from .mm_w8a8_int8 import _mm_w8a8_aiu_kernel as mm_w8a8_kernel
from .mm_w8a8_int8 import _mm_w8a8_vector_kernel as mm_w8a8_vector_kernel
from .mm_w8a8_int8 import _pick_tiles as pick_tiles
from .mm_w8a8_int8 import mm_w8a8_int8_out

# Reuse the MM body so INT8 AIU lowering and scaling stay identical across APIs.
scaled_mm_kernel = mm_w8a8_kernel.fn
MAX_INT32_REDUCTION = (2**31 - 1) // (128 * 128)


@libentry()
@triton.jit
def bmm_channel_int8_kernel(
    activation,
    weight,
    activation_scale,
    weight_scale,
    output,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    BOUNDARY: tl.constexpr,
    NUM_WARPS: tl.constexpr,
):
    batch = tl.program_id(1).to(tl.int64)
    scaled_mm_kernel(
        activation + batch * M * K,
        weight + batch * N * K,
        activation_scale + batch * M,
        weight_scale + batch * N,
        output + batch * M * N,
        output,
        M,
        N,
        K,
        N,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        GROUP_M,
        BOUNDARY,
        BOUNDARY,
        NUM_WARPS,
        False,
        False,
        False,
        False,
    )


def bmm_w8a8_int8(
    activation: torch.Tensor,
    weight: torch.Tensor,
    activation_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Channel-scaled INT8 A[B,M,K] @ W[B,K,N], with INT32 accumulation.

    A is contiguous; W is a transpose view of contiguous [B,N,K] storage.
    Contiguous FP32 scales have shapes [B,M] / [B,N], optionally [B,M,1]
    / [B,1,N]. Inputs are prequantized; batch broadcasting is unsupported.
    """
    if activation.ndim != 3 or weight.ndim != 3:
        raise ValueError("expected activation[B,M,K] and weight[B,K,N]")
    batch, rows, reduction = activation.shape
    columns = weight.shape[2]
    if weight.shape[:2] != (batch, reduction):
        raise ValueError("batch and reduction dimensions must match")
    if activation.dtype != torch.int8 or weight.dtype != torch.int8:
        raise TypeError("activation and weight must be signed INT8 tensors")
    if activation.device.type != "cuda" or any(
        tensor.device != activation.device
        for tensor in (weight, activation_scale, weight_scale)
    ):
        raise ValueError("inputs and scales must be on the same PPU device")
    if not activation.is_contiguous() or not weight.transpose(1, 2).is_contiguous():
        raise ValueError("expected contiguous activation and column-major weight")
    if activation_scale.shape not in ((batch, rows), (batch, rows, 1)):
        raise ValueError("activation_scale must have shape [B,M] or [B,M,1]")
    if weight_scale.shape not in ((batch, columns), (batch, 1, columns)):
        raise ValueError("weight_scale must have shape [B,N] or [B,1,N]")
    if any(
        scale.dtype != torch.float32 or not scale.is_contiguous()
        for scale in (activation_scale, weight_scale)
    ):
        raise TypeError("scales must be contiguous FP32 tensors")
    if any(scale.requires_grad for scale in (activation_scale, weight_scale)):
        raise NotImplementedError("bmm_w8a8_int8 is forward-only")
    if reduction > MAX_INT32_REDUCTION:
        raise NotImplementedError("reduction may overflow signed INT32 accumulation")
    if out_dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise TypeError("output dtype must be bfloat16, float16 or float32")
    if out is None:
        out = torch.empty(
            (batch, rows, columns), dtype=out_dtype, device=activation.device
        )
    elif (
        out.shape != (batch, rows, columns)
        or out.dtype != out_dtype
        or out.device != activation.device
        or not out.is_contiguous()
        or out.requires_grad
    ):
        raise ValueError(
            "out must have matching shape, dtype and device, be contiguous "
            "and not require gradients"
        )
    if out.numel() and any(
        out.untyped_storage().data_ptr() == tensor.untyped_storage().data_ptr()
        for tensor in (activation, weight, activation_scale, weight_scale)
    ):
        raise ValueError("out must not share storage with an input")
    if out.numel() == 0:
        return out
    if reduction == 0:
        for index in range(batch):
            mm_w8a8_int8_out(
                activation[index],
                weight[index],
                activation_scale[index],
                weight_scale[index],
                out=out[index],
            )
        return out
    if (batch, rows, columns, reduction) == (1, 1, 1024, 4096):
        # A single row cannot amortize the padded MMA tile; reuse the MM GEMV.
        with torch_device_fn.device(activation.device):
            mm_w8a8_vector_kernel[(triton.cdiv(columns, 2),)](
                activation,
                weight,
                activation_scale,
                weight_scale,
                out,
                None,
                columns,
                reduction,
                True,
                BLOCK_R=2,
                BLOCK_K=reduction,
                A_SCALAR=True,
                B_SCALAR=False,
                HAS_BIAS=False,
                num_warps=4,
            )
        return out
    block_m, block_n, block_k, warps, stages, group_m = pick_tiles(
        rows, columns, reduction
    )
    boundary = rows % block_m != 0 or columns % block_n != 0 or reduction % block_k != 0
    with torch_device_fn.device(activation.device):
        bmm_channel_int8_kernel[
            (triton.cdiv(rows, block_m) * triton.cdiv(columns, block_n), batch)
        ](
            activation,
            weight,
            activation_scale,
            weight_scale,
            out,
            rows,
            columns,
            reduction,
            block_m,
            block_n,
            block_k,
            group_m,
            boundary,
            warps,
            num_warps=warps,
            num_stages=stages,
        )
    return out
