import logging
import math

import torch
import triton
import triton.language as tl

from ..utils import libentry


@libentry()
@triton.jit
def embedding_kernel(
    Y,  # pointer to the output
    X,  # pointer to the input
    W,  # pointer to the weights
    N: tl.constexpr,  # number of columns in X
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    Y += pid * N
    X += pid

    mask = tl.arange(0, BLOCK_SIZE) < N
    cols = tl.arange(0, BLOCK_SIZE)

    row_idx = tl.load(X).to(tl.int32)
    W += row_idx * N
    embedding_weight = tl.load(W + cols, mask, other=0.0)
    tl.store(Y + cols, embedding_weight, mask)


# @libentry()
@triton.jit
def indice_freq_kernel(
    indices_freq,  # indice frequency
    indices,  # pointer to the input
    elem_cnt: tl.constexpr,  # number of columns in X
    INDICE_BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * INDICE_BLOCK_SIZE

    offsets = block_start + tl.arange(0, INDICE_BLOCK_SIZE)
    mask = offsets < elem_cnt

    index_element = tl.load(indices + offsets, mask=mask).to(tl.int32)
    tl.atomic_add(indices_freq + index_element, 1, mask=mask)


# @libentry()
@triton.jit(do_not_specialize=["padding_idx"])
def embedding_backward_kernel(
    GradIn,  # pointer to the gradient input
    GradOut,  # pointer to the gradient output
    indices,  # pointer to the input
    padding_idx,  # padding_idx
    N: tl.constexpr,  # number of columns in X
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    GradOut += pid * N
    indices += pid

    mask = tl.arange(0, BLOCK_SIZE) < N
    cols = tl.arange(0, BLOCK_SIZE)

    row_idx = tl.load(indices).to(tl.int32)
    if row_idx != padding_idx:
        GradIn += row_idx * N
        embedding_grad = tl.load(GradOut + cols, mask, other=0.0)
        tl.atomic_add(GradIn + cols, embedding_grad, mask=mask)


# @libentry()
@triton.jit(do_not_specialize=["n_rows", "N"])
def embedding_grad_scale_kernel(
    grad_out,  # indice frequency
    indice_freq,  # pointer to the input
    n_rows,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)

    for row_idx in range(row_start, n_rows, row_step):
        embedding_scale = 1.0
        indice_freq_val = tl.load(indice_freq + row_idx).to(tl.int32)
        if indice_freq_val > 1:
            embedding_scale = 1.0 / indice_freq_val

        cols = tl.arange(0, BLOCK_SIZE)
        mask = tl.arange(0, BLOCK_SIZE) < N
        embedding_grad = tl.load(grad_out + row_idx * N + cols, mask=mask)
        scaled_embedding_grad = embedding_grad * embedding_scale
        tl.store(grad_out + row_idx * N + cols, scaled_embedding_grad, mask=mask)


class Embedding(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, weight, indices, padding_idx=-1, scale_grad_by_freq=False, sparse=False
    ):
        logging.debug("GEMS EMBEDDING FORWARD")
        assert not sparse, "Currently do not support sparse format"

        M = math.prod(indices.shape)
        N = weight.shape[-1]

        BLOCK_SIZE = triton.next_power_of_2(N)
        indices = indices.contiguous()
        weight = weight.contiguous()
        output = torch.empty(
            (*indices.shape, N), device=indices.device, dtype=weight.dtype
        )

        embedding_kernel[M,](output, indices, weight, N, BLOCK_SIZE)

        if padding_idx < 0:
            padding_idx = weight.shape[0] + padding_idx

        ctx.M = M
        ctx.N = N
        ctx.num_weights = weight.shape[0]
        ctx.padding_idx = padding_idx
        ctx.scale_grad_by_freq = scale_grad_by_freq
        ctx.sparse = sparse
        ctx.indices = indices

        return output

    @staticmethod
    def backward(ctx, grad_outputs):
        logging.debug("GEMS EMBEDDING BACKWARD")
        assert not ctx.sparse, "Currently do not support sparse format"

        grad_inputs = torch.zeros(
            (ctx.num_weights, grad_outputs.shape[-1]),
            device=grad_outputs.device,
            dtype=grad_outputs.dtype,
        )

        if ctx.scale_grad_by_freq:
            indice_freq = torch.zeros(
                (ctx.num_weights,),
                requires_grad=False,
                device=grad_outputs.device,
                dtype=torch.int32,
            )
            INDICE_BLOCK_SIZE = 256
            indice_grid = lambda meta: (triton.cdiv(ctx.M, INDICE_BLOCK_SIZE),)
            indice_freq_kernel[indice_grid](
                indice_freq, ctx.indices, ctx.M, INDICE_BLOCK_SIZE
            )
        else:
            indice_freq = None

        BLOCK_SIZE = triton.next_power_of_2(ctx.N)

        embedding_backward_kernel[ctx.M,](
            grad_inputs, grad_outputs, ctx.indices, ctx.padding_idx, ctx.N, BLOCK_SIZE
        )

        if ctx.scale_grad_by_freq:
            embedding_grad_scale_kernel[ctx.M,](
                grad_inputs, indice_freq, ctx.num_weights, ctx.N, BLOCK_SIZE
            )
        return grad_inputs, None, None, None, None


def embedding(indices, weight, padding_idx=-1, scale_grad_by_freq=False, sparse=False):
    return Embedding.apply(weight, indices, padding_idx, scale_grad_by_freq, sparse)
