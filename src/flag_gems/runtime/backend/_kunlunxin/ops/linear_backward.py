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

from .mm import mm
from .sum import sum_dim

logger = logging.getLogger(__name__)


def linear_backward(input, grad_output, weight, output_mask):
    logger.debug("GEMS_KUNLUNXIN LINEAR_BACKWARD")
    batch_dims = input.shape[:-1]
    batch_size = input.numel() // input.shape[-1]
    in_features = input.shape[-1]
    out_features = weight.shape[0]
    input_flat = input.reshape(batch_size, in_features).contiguous()
    grad_output_flat = grad_output.reshape(batch_size, out_features).contiguous()

    grad_input = None
    if output_mask[0]:
        if input.dtype == torch.float16:
            grad_input = mm(
                grad_output_flat.to(torch.float32), weight.to(torch.float32)
            ).to(input.dtype)
        else:
            grad_input = mm(grad_output_flat, weight)
        grad_input = grad_input.reshape(*batch_dims, in_features)

    grad_weight = None
    if output_mask[1]:
        grad_output_transposed = grad_output_flat.t().contiguous()
        if weight.dtype == torch.float16:
            grad_weight = mm(
                grad_output_transposed.to(torch.float32), input_flat.to(torch.float32)
            ).to(weight.dtype)
        else:
            grad_weight = mm(grad_output_transposed, input_flat)

    grad_bias = None
    if output_mask[2]:
        if weight.dtype == torch.float16:
            grad_bias = sum_dim(grad_output_flat.to(torch.float32), dim=(0,)).to(
                weight.dtype
            )
        else:
            grad_bias = sum_dim(grad_output_flat, dim=(0,))

    return grad_input, grad_weight, grad_bias
