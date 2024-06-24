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
    padding_idx, # padding_idx
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


@libentry()
@triton.jit
def renorm_embedding_kernel(
    X,  # pointer to the input
    W,  # pointer to the weights
    max_norm, 
    norm_type, 
    N: tl.constexpr,  # number of columns in X
    BLOCK_SIZE: tl.constexpr,
): 
    pid = tl.program_id(0)
    X += pid

    mask = tl.arange(0, BLOCK_SIZE) < N 
    cols = tl.arange(0, BLOCK_SIZE)

    row_idx = tl.load(X).to(tl.int32)
    W += row_idx * N 
    embedding_weight = tl.load(W + cols, mask, other=0.0)

    norm_factor = 0.0
    if norm_type == 1.0: 
        norm_factor = tl.sum(tl.abs(embedding_weight))
    else: 
        norm_factor = tl.sum(tl.math.pow(embedding_weight, norm_type))

    norm_factor = tl.math.pow(norm_factor, 1.0 / norm_type)

    if norm_factor > max_norm: 
        factor = (max_norm / norm_factor)
        embedding_weight *= factor

    tl.store(W + cols, embedding_weight, mask)


@triton.jit
def embedding_backward_kernel(
    GradOut,  # pointer to the gradient output
    GradIn,  # pointer to the gradient input
    indices,  # pointer to the input
    padding_idx, # padding_idx
    N: tl.constexpr,  # number of columns in X
    BLOCK_SIZE: tl.constexpr,
): 
    pid = tl.program_id(0)
    GradIn += pid * N
    indices += pid

    mask = tl.arange(0, BLOCK_SIZE) < N 
    cols = tl.arange(0, BLOCK_SIZE)

    row_idx = tl.load(indices).to(tl.int32)
    if (row_idx != padding_idx): 
        GradOut += row_idx * N
        embedding_grad = tl.load(GradIn + cols, mask, other=0.0)
        tl.atomic_add(GradOut + cols, embedding_grad, mask=mask)


@triton.jit
def indice_freq_kernel(
    indices_freq, # indice frequency
    indices,  # pointer to the input
    elem_cnt: tl.constexpr,  # number of columns in X
    BLOCK_SIZE: tl.constexpr,
): 
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE

    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < elem_cnt

    index_element = tl.load(indices + offsets, mask=mask).to(tl.int32)
    tl.atomic_add(indices_freq + index_element, 1, mask=mask)
    

@triton.jit
def embedding_grad_scale_kernel(
    grad_out, # indice frequency
    indice_freq,  # pointer to the input
    n_rows, 
    N, 
    BLOCK_SIZE: tl.constexpr,
): 
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)

    for row_idx in range(row_start, n_rows, row_step):
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
    def forward(ctx, input, weight, padding_idx=None, max_norm=None, norm_type=2.0):
        logging.debug("GEMS EMBEDDING FORWARD")
        dim = x.ndim - len(normalized_shape)
        M = math.prod(x.shape)
        N = weight.shape[-1]

        BLOCK_SIZE = triton.next_power_of_2(N)
        input = input.contiguous()
        weight = weight.contiguous()
        output = torch.empty((*x.shape, N), device=input.device, dtype=weight.dtype)


        if max_norm is not None: 
            renorm_embedding_kernel[M, ](input, weight, max_norm, norm_type, N, BLOCK_SIZE)

        embedding_kernel[M, ](output, input, weight, padding_idx, N, BLOCK_SIZE)

        return output
