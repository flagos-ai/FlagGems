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

from flag_gems.ops.linear import linear as _generic_linear

logger = logging.getLogger(__name__)


def linear(input: torch.Tensor, weight: torch.Tensor, bias=None) -> torch.Tensor:
    """Linear transformation y = x @ W^T + b (mthreads/MUSA specialization).

    ``torch.linear`` is exactly ``addmm(bias, x, W^T)``. The generic
    KernelGen linear kernel only reaches ~0.6-0.8 TFLOPS on MUSA (fp32) and
    ~45 TFLOPS (bf16) because its un-tuned blocked GEMM cannot use the SQMMA
    tensor cores efficiently. Delegating to the MTHREADS addmm reuses the
    optimized kernels: SQMMA path for fp16/bf16 (up to ~104 TFLOPS at 4096^3)
    and the FMA path for fp32 (~8.5 TFLOPS, >10x faster than the generic
    kernel).

    The SQMMA path needs a large enough M for its 128-row tiles to fill the
    device; for fp16/bf16 GEMMs with M <= 1024 the generic kernel is faster
    (measured on MTT S5000), so those fall back to it. The MTHREADS fp32
    ``mm`` kernel is much slower than ``addmm`` (~0.4 TFLOPS), so the no-bias
    case also routes through addmm with a scalar zero bias and ``beta=0``
    (the bias term is then ignored by both the SQMMA and FMA kernels).
    """
    logger.debug("GEMS_MTHREADS LINEAR")

    # Flatten batch dimensions: (*, in_features) -> (M, in_features)
    batch_dims = input.shape[:-1]
    M = 1
    for dim in batch_dims:
        M *= dim
    K = input.shape[-1]
    N = weight.shape[0]

    if input.dtype in (torch.float16, torch.bfloat16) and M <= 1024:
        # The SQMMA path needs a large enough M to fill its 128-row tiles;
        # for small fp16/bf16 GEMMs the generic kernel is faster on MUSA.
        return _generic_linear(input, weight, bias)

    if input.dim() == 1:
        input = input.unsqueeze(0)
        single_1d = True
    else:
        single_1d = False

    input_flat = input.reshape(M, K)

    if bias is not None:
        out = torch.addmm(bias, input_flat, weight.t())
    else:
        zero_bias = input_flat.new_zeros(())
        out = torch.addmm(zero_bias, input_flat, weight.t(), beta=0)

    out = out.view(*batch_dims, N)
    if single_1d:
        out = out.squeeze(0)
    return out
