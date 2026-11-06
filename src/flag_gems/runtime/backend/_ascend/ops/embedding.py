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
import math

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.runtime.backend._ascend.tensorview import (
    HAS_TENSORVIEW,
    tensorview_call,
    tile_rows_for_ub,
)
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)

if HAS_TENSORVIEW:
    import triton.experimental.tle as tle_exp
    from triton.experimental.tle.language.dsa import tile_load, tile_store

# ---------------------------------------------------------------------------
# TensorView fast path (forward only)
#
# An embedding lookup is a row gather: each index names a contiguous row of the
# weight matrix. A GatherScatterView takes the indices in its sparse dimension
# and moves whole rows as block DMAs, where the generic kernel launches one
# program per token and reads the row element-wise.
#
# Only the forward pass. The backward is a scatter-add over repeated indices
# with optional frequency scaling, which the existing kernels already handle.
# ---------------------------------------------------------------------------
TV_BLOCK_M = 32

if HAS_TENSORVIEW:

    @triton.jit
    def embedding_tv_kernel(
        out,
        indices,
        weight,
        num_tokens,
        vocab_size,
        DIM: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_DIM: tl.constexpr,
    ):
        token_block = tl.program_id(0)
        offsets = token_block * BLOCK_M + tl.arange(0, BLOCK_M)
        rows = tl.load(indices + offsets, mask=offsets < num_tokens, other=-1).to(
            tl.int32
        )

        weight_view = tle_exp.make_gather_scatter_view(
            weight,
            shape=[vocab_size, DIM],
            strides=[DIM, 1],
            tile=[BLOCK_M, BLOCK_DIM],
            sparse_dim=[0],
        )
        gathered = tile_load(
            weight_view, index=(rows, 0), dst_space=tle_exp.language.dsa.ascend.UB
        )

        out_view = tle_exp.make_partition_view(
            out,
            shape=[num_tokens, DIM],
            strides=[DIM, 1],
            tile=[BLOCK_M, BLOCK_DIM],
        )
        tile_store(
            out_view,
            value=gathered,
            index=(token_block, 0),
            src_space=tle_exp.language.dsa.ascend.UB,
        )

    def _embedding_forward_tensorview(weight, indices, block_m=None):
        logger.debug("GEMS_ASCEND EMBEDDING_FORWARD(TENSORVIEW)")
        vocab_size, dim = weight.shape
        if block_m is None:
            # One gathered tile is live.
            block_m = tile_rows_for_ub(
                triton.next_power_of_2(dim) * weight.element_size(), 1, TV_BLOCK_M
            )
        num_tokens = indices.numel()
        out = torch.empty(
            (num_tokens, dim), dtype=weight.dtype, device=weight.device
        )
        grid = (triton.cdiv(num_tokens, block_m),)
        with torch_device_fn.device(weight.device):
            embedding_tv_kernel[grid](
                out,
                indices,
                weight,
                num_tokens,
                vocab_size,
                DIM=dim,
                BLOCK_M=block_m,
                BLOCK_DIM=triton.next_power_of_2(dim),
            )
        return out

    def _embedding_forward_tv_eligible(weight, indices):
        return (
            weight.ndim == 2
            and weight.numel() > 0
            and indices.numel() > 0
            and indices.dtype in (torch.int32, torch.int64)
        )


@libentry()
@triton.jit
def embedding_kernel(
    out_ptr,  # pointer to the output
    in_ptr,  # pointer to the input
    weight_ptr,  # pointer to the weights
    N: tl.constexpr,  # number of columns in X
    BLOCK_SIZE: tl.constexpr,
):
    pid = ext.program_id(0)
    out_ptr += pid * N
    in_ptr += pid

    mask = tl.arange(0, BLOCK_SIZE) < N
    cols = tl.arange(0, BLOCK_SIZE)

    row_idx = tl.load(in_ptr)
    weight_ptr += row_idx * N
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
    pid = ext.program_id(0)
    block_start = pid * INDICE_BLOCK_SIZE

    for i in range(INDICE_BLOCK_SIZE):
        off = block_start + i
        if off < elem_cnt:
            idx = tl.load(indices + off)
            tl.atomic_add(indices_freq + idx, 1)


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
    pid = ext.program_id(0)
    grad_out += pid * N
    indices += pid

    mask = tl.arange(0, BLOCK_SIZE) < N
    cols = tl.arange(0, BLOCK_SIZE)

    row_idx = tl.load(indices).to(tl.int32)
    if not HAS_PADDING_IDX:
        grad_in += row_idx * N
        embedding_grad = tl.load(grad_out + cols, mask, other=0.0)
        if tl.constexpr(embedding_grad.dtype.is_bf16()):
            embedding_grad = embedding_grad.to(tl.float32)
        tl.atomic_add(grad_in + cols, embedding_grad, mask=mask)
    else:
        if row_idx != padding_idx:
            grad_in += row_idx * N
            embedding_grad = tl.load(grad_out + cols, mask, other=0.0)
            if tl.constexpr(embedding_grad.dtype.is_bf16()):
                embedding_grad = embedding_grad.to(tl.float32)
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
    row_start = ext.program_id(0)
    row_step = ext.num_programs(0)

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


class Embedding(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, weight, indices, padding_idx=-1, scale_grad_by_freq=False, sparse=False
    ):
        logger.debug("GEMS_ASCEND EMBEDDING_FORWARD")
        assert not sparse, "Currently do not support sparse format"

        M = math.prod(indices.shape)
        N = weight.shape[-1]

        BLOCK_SIZE = triton.next_power_of_2(N)
        indices = indices.contiguous()
        weight = weight.contiguous()

        def _generic_forward(weight, indices):
            output = torch.empty(
                (*indices.shape, N), device=indices.device, dtype=weight.dtype
            )
            with torch_device_fn.device(weight.device):
                embedding_kernel[M,](output, indices, weight, N, BLOCK_SIZE)
            return output

        if HAS_TENSORVIEW and _embedding_forward_tv_eligible(weight, indices):
            # The fast path produces a flat [num_tokens, N] tile; restore the
            # caller's leading index shape afterwards.
            output = tensorview_call(
                "EMBEDDING_FORWARD",
                _embedding_forward_tensorview,
                _generic_forward,
                weight,
                indices,
            ).reshape(*indices.shape, N)
        else:
            output = _generic_forward(weight, indices)

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
        logger.debug("GEMS_ASCEND EMBEDDING_BACKWARD")
        assert not ctx.sparse, "Currently do not support sparse format"

        grad_inputs = torch.zeros(
            (ctx.num_weights, grad_outputs.shape[-1]),
            device=grad_outputs.device,
            dtype=(
                torch.float32
                if grad_outputs.dtype is torch.bfloat16
                else grad_outputs.dtype
            ),
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

            with torch_device_fn.device(grad_outputs.device):
                indice_freq_kernel[indice_grid](
                    indice_freq, ctx.indices, ctx.M, INDICE_BLOCK_SIZE
                )
        else:
            indice_freq = None

        BLOCK_SIZE = triton.next_power_of_2(ctx.N)

        HAS_PADDING_IDX = ctx.padding_idx is not None

        with torch_device_fn.device(grad_outputs.device):
            embedding_backward_kernel[ctx.M,](
                grad_inputs,
                grad_outputs,
                ctx.indices,
                ctx.padding_idx,
                HAS_PADDING_IDX,
                ctx.N,
                BLOCK_SIZE,
            )

        if ctx.scale_grad_by_freq:
            with torch_device_fn.device(grad_outputs.device):
                embedding_grad_scale_kernel[ctx.M,](
                    grad_inputs, indice_freq, ctx.num_weights, ctx.N, BLOCK_SIZE
                )
        return (
            (
                grad_inputs.to(torch.bfloat16)
                if grad_outputs.dtype is torch.bfloat16
                else grad_inputs
            ),
            None,
            None,
            None,
            None,
        )


def embedding(weight, indices, padding_idx=-1, scale_grad_by_freq=False, sparse=False):
    return Embedding.apply(weight, indices, padding_idx, scale_grad_by_freq, sparse)
