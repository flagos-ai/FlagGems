import logging
import math

import paddle
import torch
import triton
import triton.language as tl
from paddle.autograd import PyLayer

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def embedding_kernel(
    out_ptr,  # pointer to the output
    in_ptr,  # pointer to the input
    weight_ptr,  # pointer to the weights
    N: tl.constexpr,  # number of columns in X
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    out_ptr += pid * N
    in_ptr += pid

    mask = tl.arange(0, BLOCK_SIZE) < N
    cols = tl.arange(0, BLOCK_SIZE)

    row_idx = tl.load(in_ptr)
    weight_ptr += row_idx.to(tl.int32) * N
    embedding_weight = tl.load(weight_ptr + cols, mask, other=0.0)
    tl.store(out_ptr + cols, embedding_weight, mask)


@libentry()
@triton.jit
def indice_freq_kernel(
    indices_freq,
    indices,  # pointer to the input
    elem_cnt: tl.constexpr,  # number of columns in X
    INDICE_BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    block_start = pid * INDICE_BLOCK_SIZE

    offsets = block_start + tl.arange(0, INDICE_BLOCK_SIZE)
    mask = offsets < elem_cnt

    index_element = tl.load(indices + offsets, mask=mask)
    tl.atomic_add(indices_freq + index_element, 1, mask=mask)


@libentry()
@triton.jit(do_not_specialize=["padding_idx"])
def embedding_backward_kernel(
    grad_in,  # pointer to the gradient input
    grad_out,  # pointer to the gradient output
    indices,  # pointer to the input
    padding_idx,  # padding_idx
    HAS_PADDING_IDX: tl.constexpr,
    N: tl.constexpr,  # number of columns in X
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    grad_out += pid * N
    indices += pid

    mask = tl.arange(0, BLOCK_SIZE) < N
    cols = tl.arange(0, BLOCK_SIZE)

    row_idx = tl.load(indices).to(tl.int32)
    if not HAS_PADDING_IDX:
        grad_in += row_idx * N
        embedding_grad = tl.load(grad_out + cols, mask, other=0.0)
        # grad_in may be fp32 while grad_out is a half type: the caller widens the
        # accumulator when several index elements collide on the same row, because
        # half-type atomics are emulated and collapse under that contention.
        embedding_grad = embedding_grad.to(grad_in.dtype.element_ty)
        tl.atomic_add(grad_in + cols, embedding_grad, mask=mask)
    else:
        if row_idx != padding_idx:
            grad_in += row_idx * N
            embedding_grad = tl.load(grad_out + cols, mask, other=0.0)
            embedding_grad = embedding_grad.to(grad_in.dtype.element_ty)
            tl.atomic_add(grad_in + cols, embedding_grad, mask=mask)


@libentry()
@triton.jit(do_not_specialize=["n_rows"])
def embedding_grad_scale_kernel(
    grad_out,
    indice_freq,
    n_rows,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    row_start = tle.program_id(0)
    row_step = tle.num_programs(0)

    for row_idx in range(row_start, n_rows, row_step):
        embedding_scale = 1.0
        indice_freq_val = tl.load(indice_freq + row_idx)
        if indice_freq_val > 1:
            embedding_scale = 1.0 / indice_freq_val

        cols = tl.arange(0, BLOCK_SIZE)
        mask = tl.arange(0, BLOCK_SIZE) < N
        embedding_grad = tl.load(grad_out + row_idx * N + cols, mask=mask)
        scaled_embedding_grad = embedding_grad * embedding_scale
        tl.store(grad_out + row_idx * N + cols, scaled_embedding_grad, mask=mask)


def embedding(indices, weight, padding_idx=-1, scale_grad_by_freq=False, sparse=False):
    logger.debug("GEMS EMBEDDING FORWARD")
    assert not sparse, "Currently do not support sparse format"

    M = math.prod(indices.shape)
    N = weight.shape[-1]

    BLOCK_SIZE = triton.next_power_of_2(N)
    # TODO: remove contiguous enforcement
    indices = indices.contiguous()
    weight = weight.contiguous()
    output = torch.empty((*indices.shape, N), device=indices.device, dtype=weight.dtype)

    with torch_device_fn.device(weight.device):
        embedding_kernel[M,](output, indices, weight, N, BLOCK_SIZE)

    return output


def embedding_backward(
    grad_outputs,
    indices,
    num_weights,
    padding_idx=-1,
    scale_grad_by_freq=False,
    sparse=False,
):
    logger.debug("GEMS EMBEDDING BACKWARD")
    assert not sparse, "Currently do not support sparse format"

    # paddle's Tensor.numel() returns a device Tensor, so using it as a grid size
    # forces a device-to-host sync on every call.
    M = math.prod(indices.shape)
    N = grad_outputs.shape[-1]

    # Half-type atomics are emulated with a CAS loop, and their cost blows up
    # superlinearly with collisions per row (~16 collisions still costs 25us, ~128
    # costs 456us). Widening the accumulator to fp32 avoids that but adds a cast
    # pass, which dominates at small sizes, so only widen once contention is heavy.
    acc_in_fp32 = grad_outputs.dtype in (torch.bfloat16, torch.float16) and (
        M > 32 * num_weights
    )
    # `torch.zeros(..., device=...)` costs ~70us per call through the proxy and
    # dominated this op (the kernel launch itself is 0.11us). paddle.zeros
    # allocates on the current place, so do it inside the device guard that the
    # kernel launches need anyway.
    with torch_device_fn.device(grad_outputs.device):
        grad_inputs = paddle.zeros(
            [num_weights, N],
            dtype=paddle.float32 if acc_in_fp32 else grad_outputs.dtype,
        )

        if scale_grad_by_freq:
            indice_freq = paddle.zeros([num_weights], dtype=paddle.int32)
            indice_freq.stop_gradient = True
            INDICE_BLOCK_SIZE = 256
            indice_grid = (triton.cdiv(M, INDICE_BLOCK_SIZE),)
            indice_freq_kernel[indice_grid](indice_freq, indices, M, INDICE_BLOCK_SIZE)
        else:
            indice_freq = None

        BLOCK_SIZE = triton.next_power_of_2(N)
        HAS_PADDING_IDX = padding_idx is not None

        embedding_backward_kernel[M,](
            grad_inputs,
            grad_outputs,
            indices,
            padding_idx,
            HAS_PADDING_IDX,
            N,
            BLOCK_SIZE,
        )

        if scale_grad_by_freq:
            embedding_grad_scale_kernel[M,](
                grad_inputs, indice_freq, num_weights, N, BLOCK_SIZE
            )
    return grad_inputs.to(grad_outputs.dtype) if acc_in_fp32 else grad_inputs


class Embedding(PyLayer):
    @staticmethod
    def forward(
        ctx, indices, weight, padding_idx, scale_grad_by_freq=False, sparse=False
    ):  
        if weight.requires_grad:
            ctx.save_for_backward(indices, weight)
            ctx.padding_idx = padding_idx
            ctx.scale_grad_by_freq = scale_grad_by_freq
            ctx.sparse = sparse

        return embedding(indices, weight, padding_idx, scale_grad_by_freq, sparse)

    @staticmethod
    def backward(ctx, grad_outputs):
        indices, weight = ctx.saved_tensor()
        padding_idx = ctx.padding_idx
        scale_grad_by_freq = ctx.scale_grad_by_freq
        sparse = ctx.sparse

        grad_input = embedding_backward(
            grad_outputs,
            indices,
            weight.shape[0],
            padding_idx,
            scale_grad_by_freq,
            sparse,
        )

        return None, grad_input


embedding_paddle = Embedding.apply
