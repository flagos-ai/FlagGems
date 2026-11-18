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

from ._gemm_tuned import nn as _gemm_nn
from ._gemm_tuned import tn as _gemm_tn
from .sum import sum_dim

logger = logging.getLogger(__name__)


def linear_backward(input, grad_output, weight, output_mask):
    logger.debug("GEMS_KUNLUNXIN LINEAR_BACKWARD")
    batch_dims = input.shape[:-1]
    # numel() // in_features divides by zero for zero-width inputs
    # (e.g. input (batch, 0), weight (out_features, 0)); derive the batch
    # size from the leading dims instead.
    batch_size = math.prod(batch_dims)
    in_features = input.shape[-1]
    out_features = weight.shape[0]
    input_flat = input.reshape(batch_size, in_features).contiguous()
    grad_output_flat = grad_output.reshape(batch_size, out_features).contiguous()

    grad_input = None
    if output_mask[0]:
        # The backend mm kernel always accumulates in fp32 (dot_out_dtype=
        # tl.float32), so fp16/bf16 inputs need no host-side upcast: running the
        # GEMM in the native 16-bit matrix engine is both faster and as accurate
        # as the previous fp32 upcast (which doubled memory traffic and forced
        # the slow fp32 engine).
        # _gemm_nn keeps the same mask-free kernel but with tiles tuned for the
        # (small M, large K) shapes of a linear backward (see _gemm_tuned).
        grad_input = _gemm_nn(grad_output_flat, weight)
        grad_input = grad_input.reshape(*batch_dims, in_features)

    grad_weight = None
    if output_mask[1]:
        # grad_weight = grad_output.T @ input_flat.  ``grad_output_flat`` is
        # already row-major, so the transposed source needs no materialised
        # copy: _gemm_tn reads it K-contiguous and transposes in registers.
        grad_weight = _gemm_tn(grad_output_flat, input_flat)
        if grad_weight is None:
            grad_weight = _gemm_nn(grad_output_flat.t().contiguous(), input_flat)

    grad_bias = None
    if output_mask[2]:
        # Column sum of the gradient.  ``sum_dim`` reduces dim 0 through a
        # host-side transpose + TLE row-reduce; a direct 2-D tile reduce
        # (``tl.sum(axis=0/1)``) does not lower on this backend (uni_sram
        # OutOfResources in TritonXPUVectorize), so this is the faithful path.
        if weight.dtype in (torch.float16, torch.bfloat16):
            grad_bias = sum_dim(grad_output_flat.to(torch.float32), dim=(0,)).to(
                weight.dtype
            )
        else:
            grad_bias = sum_dim(grad_output_flat, dim=(0,))

    return grad_input, grad_weight, grad_bias
