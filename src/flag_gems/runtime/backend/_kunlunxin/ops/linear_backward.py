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

from .linear import _fused_linear, _linear_fused_kernel_nomask, linear
from .mm import mm
from .sum import sum_dim

logger = logging.getLogger(__name__)

# grad_weight = grad_output^T @ input is a skinny-K GEMM (K = batch).

_GW_BIG_TILE = 256
_GW_FUSED_MAX_K = 256


def _row_major_2d(t):
    return t.stride(1) == 1


def _gw_fused_big(grad_output_flat, input_flat):
    x = grad_output_flat.t()  # (out_features, batch) — A operand
    w = input_flat.t()  # (in_features, batch) — B operand (W of y = x @ W^T)
    M, K = x.shape
    N = w.shape[0]
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)
    grid = (triton.cdiv(M, _GW_BIG_TILE) * triton.cdiv(N, _GW_BIG_TILE),)
    _linear_fused_kernel_nomask[grid](
        x,
        w,
        x,
        out,
        M,
        N,
        K,
        x.stride(0),
        x.stride(1),
        w.stride(0),
        w.stride(1),
        out.stride(0),
        out.stride(1),
        _GW_BIG_TILE,
        _GW_BIG_TILE,
        _GW_BIG_TILE,
        False,
        True,
    )
    return out


def linear_backward(input, grad_output, weight, output_mask):
    logger.debug("GEMS_KUNLUNXIN LINEAR_BACKWARD")
    batch_dims = input.shape[:-1]
    # numel() // in_features divides by zero for zero-width inputs
    # (e.g. input (batch, 0), weight (out_features, 0)); derive the batch
    # size from the leading dims instead.
    batch_size = math.prod(batch_dims)
    in_features = input.shape[-1]
    out_features = weight.shape[0]
    input_flat = input.reshape(batch_size, in_features)
    grad_output_flat = grad_output.reshape(batch_size, out_features)

    grad_input = None
    if output_mask[0]:
        # linear(x, W) == x @ W^T.  For M <= 256 linear() always selects the
        # fused single-launch kernel; calling it directly skips linear()'s
        # wrapper (shape bookkeeping + view) on the host-bound small shapes.
        if batch_size <= 256:
            if _row_major_2d(grad_output_flat) and _row_major_2d(weight):
                grad_input = _fused_linear(grad_output_flat, weight.t(), None)
            else:
                grad_input = mm(grad_output_flat, weight)
        else:
            grad_input = linear(grad_output_flat, weight.t())
        if input.dim() != 2:
            grad_input = grad_input.reshape(*batch_dims, in_features)

    grad_weight = None
    if output_mask[1]:
        fused_ok = _row_major_2d(grad_output_flat) and _row_major_2d(input_flat)
        if batch_size < _GW_FUSED_MAX_K or not fused_ok:
            # linear(g^T, x^T) == g^T @ x via the fused kernel; mm() when the
            # operands are not in the layout the fused kernel can address.
            if fused_ok:
                grad_weight = _fused_linear(grad_output_flat.t(), input_flat.t(), None)
            else:
                grad_weight = mm(grad_output_flat.t(), input_flat)
        elif (
            grad_output_flat.dtype in (torch.float16, torch.float32)
            and out_features % _GW_BIG_TILE == 0
            and in_features % _GW_BIG_TILE == 0
            and batch_size % _GW_BIG_TILE == 0
        ):
            grad_weight = _gw_fused_big(grad_output_flat, input_flat)
        else:
            grad_weight = _fused_linear(grad_output_flat.t(), input_flat.t(), None)

    grad_bias = None
    if output_mask[2]:
        grad_bias = sum_dim(grad_output_flat, dim=(0,))

    return grad_input, grad_weight, grad_bias
