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

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

from .embedding_dense_backward import embedding_dense_backward

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
    pid = tl.program_id(0)
    out_ptr += pid * N
    in_ptr += pid

    mask = tl.arange(0, BLOCK_SIZE) < N
    cols = tl.arange(0, BLOCK_SIZE)

    row_idx = tl.load(in_ptr)
    weight_ptr += row_idx * N
    embedding_weight = tl.load(weight_ptr + cols, mask, other=0.0)
    tl.store(out_ptr + cols, embedding_weight, mask)


def embedding(weight, indices, padding_idx=-1, scale_grad_by_freq=False, sparse=False):
    original_weight_dtype = weight.dtype
    need_restore_int64 = original_weight_dtype == torch.int64
    if indices.dtype == torch.int64:
        indices = indices.to(torch.int32)
    if weight.dtype == torch.float64:
        weight = weight.to(torch.float32)
    logger.debug("GEMS_ENFLAME EMBEDDING")
    assert not sparse, "Currently do not support sparse format"

    if weight.dtype == torch.int64:
        weight = weight.to(torch.int32)
    if indices.dtype == torch.int64:
        indices = indices.to(torch.int32)

    M = indices.numel()
    N = weight.shape[-1]

    BLOCK_SIZE = triton.next_power_of_2(N)
    # TODO: remove contiguous enforcement
    indices = indices.contiguous()
    weight = weight.contiguous()
    output = torch.empty((*indices.shape, N), device=indices.device, dtype=weight.dtype)

    with torch_device_fn.device(weight.device):
        embedding_kernel[M,](output, indices, weight, N, BLOCK_SIZE)

    if need_restore_int64:
        return output.to(torch.int64)
    return output


def embedding_backward(
    grad_outputs,
    indices,
    num_weights,
    padding_idx=-1,
    scale_grad_by_freq=False,
    sparse=False,
):
    logger.debug("GEMS_ENFLAME EMBEDDING_BACKWARD")
    assert not sparse, "Currently do not support sparse format"

    need_restore_int64 = grad_outputs.dtype == torch.int64
    if need_restore_int64:
        grad_outputs = grad_outputs.to(torch.int32)

    # Dense (non-sparse) embedding backward is the same segmented sum as
    # `embedding_dense_backward`; share its atomic-free implementation.
    grad_inputs = embedding_dense_backward(
        grad_outputs, indices, num_weights, padding_idx, scale_grad_by_freq
    )
    if need_restore_int64:
        return grad_inputs.to(torch.int64)
    return grad_inputs
