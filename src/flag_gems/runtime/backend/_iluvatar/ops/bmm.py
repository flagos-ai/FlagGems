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

"""
Iluvatar-specific bmm implementation that adds padding for small dimensions.

The Iluvatar Triton compiler requires:
1. All dimensions (M, N, K) >= 32
2. K must be a multiple of 128 (EVEN_K=False crashes the compiler)
"""

import logging

import torch

from flag_gems.ops.bmm import bmm as bmm_default

logger = logging.getLogger(__name__)

# Iluvatar compiler constraints (from mm.py fix #5466)
_MIN_TRITON_DIM = 32
_MAX_BLOCK_K = 128


def bmm(a, b):
    """Batch matrix multiply with Iluvatar-specific padding for small dimensions."""
    logger.debug("GEMS_ILUVATAR BMM")

    batch, M, K = a.shape
    _, K2, N = b.shape
    assert K == K2, f"incompatible dimensions: K={K} vs K2={K2}"

    # Check if padding is needed
    need_pad = (
        M < _MIN_TRITON_DIM
        or N < _MIN_TRITON_DIM
        or K < _MIN_TRITON_DIM
        or K % _MAX_BLOCK_K != 0
    )

    if not need_pad:
        # No padding needed, use default implementation
        return bmm_default(a, b)

    # Calculate padding amounts
    pad_M = max(_MIN_TRITON_DIM - M, 0)
    pad_N = max(_MIN_TRITON_DIM - N, 0)
    new_K = K + max(_MIN_TRITON_DIM - K, 0)
    remainder = new_K % _MAX_BLOCK_K
    if remainder:
        new_K += _MAX_BLOCK_K - remainder
    pad_K = new_K - K

    # Pad inputs
    if pad_M or pad_K:
        # Pad a: (batch, M, K) -> (batch, M+pad_M, K+pad_K)
        a = torch.nn.functional.pad(a, (0, pad_K, 0, pad_M))
    if pad_K or pad_N:
        # Pad b: (batch, K, N) -> (batch, K+pad_K, N+pad_N)
        b = torch.nn.functional.pad(b, (0, pad_N, 0, pad_K))

    # Run bmm on padded inputs
    out_padded = bmm_default(a, b)

    # Slice back to original size
    out = out_padded[:, :M, :N]
    return out


def bmm_out(a, b, *, out):
    """Batch matrix multiply with output tensor, using Iluvatar-specific padding."""
    logger.debug("GEMS_ILUVATAR BMM_OUT")

    result = bmm(a, b)
    out.copy_(result)
    return out
